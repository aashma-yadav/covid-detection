#!/usr/bin/env python3
"""
Multimodal Fusion Pipeline for COVID Detection
================================================

Dual-branch architecture that fuses:
  - Branch A (Vision): CNN/ResNet image features from existing models
  - Branch B (Tabular): Patient metadata processed through a small MLP

The outputs are concatenated and passed through a fusion head for
3-class prediction (COVID, Normal, Pneumonia).

Key features:
  - Scikit-learn ColumnTransformer for metadata preprocessing
  - Metadata branch toggleable via `use_metadata` flag
  - Image ↔ CSV synchronization via DataFrame row index
  - Diagnostic comparison function: Image-Only vs Multimodal

Usage:
  # Multimodal training
  python src/multimodal_pipeline.py --backbone pretrained --epochs 20

  # Image-only baseline
  python src/multimodal_pipeline.py --backbone pretrained --epochs 20 --no-metadata

  # Head-to-head comparison
  python src/multimodal_pipeline.py --backbone basiccnn --epochs 10 --compare

  # Include high-missingness features
  python src/multimodal_pipeline.py --backbone pretrained --epochs 20 --include-sparse-features
"""

import argparse
import copy
import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ── Project path setup ──────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import BasicCNN, MiniVGG, MiniResNet, PretrainedResNet
from src.data_loader import (
    load_data,
    clean_data,
    split_data,
    get_train_transforms,
    get_eval_transforms,
    get_weighted_sampler,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.training import EarlyStopping


# =====================================================================
#  1. METADATA PREPROCESSING
# =====================================================================

# Core features: low missingness, clinically relevant
CORE_NUMERICAL_FEATURES = ["age", "offset"]
CORE_CATEGORICAL_FEATURES = ["sex", "view", "modality"]

# Sparse features: high missingness (>60%), opt-in only
SPARSE_CATEGORICAL_FEATURES = ["survival", "intubated", "went_icu"]


class MetadataPreprocessor:
    """
    Scikit-learn pipeline for tabular metadata preprocessing.

    Handles:
      - Numerical features: median imputation + standard scaling
      - Categorical features: most-frequent imputation + one-hot encoding
      - Optional sparse features: constant 'missing' imputation + one-hot

    Usage:
        preprocessor = MetadataPreprocessor(include_sparse=False)
        preprocessor.fit(train_df)
        X_train = preprocessor.transform(train_df)  # returns np.ndarray
        X_val = preprocessor.transform(val_df)
    """

    def __init__(self, include_sparse=False):
        self.include_sparse = include_sparse
        self.pipeline = None
        self.n_features = None
        self.feature_names = None

        # Determine feature sets
        self.numerical_features = CORE_NUMERICAL_FEATURES.copy()
        self.categorical_features = CORE_CATEGORICAL_FEATURES.copy()

        if include_sparse:
            self.categorical_features.extend(SPARSE_CATEGORICAL_FEATURES)

    def fit(self, df):
        """Fit the preprocessing pipeline on training data."""
        transformers = []

        # Numerical pipeline: impute missing → scale to zero-mean unit-variance
        if self.numerical_features:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipeline, self.numerical_features))

        # Categorical pipeline: impute missing → one-hot encode
        if self.categorical_features:
            if self.include_sparse:
                # For sparse features, use constant fill to preserve missingness signal
                cat_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop="if_binary",
                    )),
                ])
            else:
                cat_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop="if_binary",
                    )),
                ])
            transformers.append(("cat", cat_pipeline, self.categorical_features))

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop unused columns
        )

        # Fit and record feature info
        self.pipeline.fit(df)
        self.n_features = self.pipeline.transform(df[:1]).shape[1]
        self.feature_names = self.pipeline.get_feature_names_out().tolist()

        print(f"  📊 MetadataPreprocessor fitted:")
        print(f"     Numerical:   {self.numerical_features}")
        print(f"     Categorical: {self.categorical_features}")
        print(f"     Output dims: {self.n_features} features")
        print(f"     Features:    {self.feature_names}")

        return self

    def transform(self, df):
        """Transform a DataFrame into a numpy array of preprocessed features."""
        if self.pipeline is None:
            raise RuntimeError("MetadataPreprocessor.fit() must be called first.")
        return self.pipeline.transform(df).astype(np.float32)


# =====================================================================
#  2. MULTIMODAL DATASET
# =====================================================================

class MultimodalXrayDataset(Dataset):
    """
    PyTorch Dataset that serves synchronized (image, metadata, label) tuples.

    Synchronization is guaranteed because each DataFrame row contains both
    the `image_path` and all metadata columns. The preprocessor has already
    transformed metadata into a fixed-size numpy array aligned by row index.

    When use_metadata=False, returns (image, label) — same as XrayDataset.
    """

    def __init__(
        self,
        dataframe,
        label_encoder,
        metadata_array=None,
        transform=None,
        use_metadata=True,
    ):
        """
        Args:
            dataframe: DataFrame with 'image_path' and 'label' columns
            label_encoder: fitted sklearn LabelEncoder
            metadata_array: np.ndarray from MetadataPreprocessor.transform()
                           Shape: (len(dataframe), n_features). Must be row-aligned.
            transform: torchvision transform for images
            use_metadata: if False, metadata_array is ignored
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.metadata_array = metadata_array
        self.transform = transform
        self.use_metadata = use_metadata and (metadata_array is not None)

        if self.use_metadata:
            assert len(self.dataframe) == len(self.metadata_array), (
                f"DataFrame ({len(self.dataframe)}) and metadata array "
                f"({len(self.metadata_array)}) must have the same length."
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # ── Image ──
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # ── Label ──
        label = int(self.label_encoder.transform([row["label"]])[0])
        label = torch.tensor(label, dtype=torch.long)

        # ── Metadata ──
        if self.use_metadata:
            metadata = torch.tensor(self.metadata_array[idx], dtype=torch.float32)
            return image, metadata, label
        else:
            return image, label


# =====================================================================
#  3. VISION FEATURE EXTRACTOR
# =====================================================================

class VisionFeatureExtractor(nn.Module):
    """
    Wraps an existing vision model and strips its classification head,
    exposing the penultimate feature vector instead.

    Supported backbones:
      - BasicCNN      → feature_dim = 1024  (64 * 4 * 4)
      - MiniVGG       → feature_dim = 2048  (128 * 4 * 4)
      - MiniResNet    → feature_dim = 256
      - PretrainedResNet → feature_dim = 512

    The original classifier/fc layers are replaced with nn.Identity().
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self._feature_dim = self._extract_feature_dim()

    def _extract_feature_dim(self):
        """Determine feature dim and strip the classification head."""
        model = self.backbone

        if isinstance(model, PretrainedResNet):
            # PretrainedResNet wraps torchvision resnet18
            # base.fc is nn.Sequential(Dropout, Linear(512, num_classes))
            in_features = model.base.fc[1].in_features  # 512
            model.base.fc = nn.Identity()
            return in_features

        elif isinstance(model, MiniResNet):
            # MiniResNet has self.fc = nn.Linear(256, num_classes)
            in_features = model.fc.in_features  # 256
            model.fc = nn.Identity()
            return in_features

        elif isinstance(model, (BasicCNN, MiniVGG)):
            # Both have self.classifier = nn.Sequential(Linear, ReLU, Dropout, Linear)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Identity()
            return in_features

        else:
            raise ValueError(f"Unsupported backbone type: {type(model).__name__}")

    @property
    def feature_dim(self):
        return self._feature_dim

    def forward(self, x):
        return self.backbone(x)


# =====================================================================
#  4. TABULAR MLP
# =====================================================================

class TabularMLP(nn.Module):
    """
    Small Multi-Layer Perceptron for processing tabular metadata features.

    Architecture:
      Linear(n_features, 64) → BatchNorm → ReLU → Dropout(0.3)
      Linear(64, 32)         → BatchNorm → ReLU → Dropout(0.3)

    Output: 32-dimensional feature vector.
    """

    def __init__(self, n_features, hidden_dim=64, output_dim=32, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


# =====================================================================
#  5. MULTIMODAL FUSION MODEL
# =====================================================================

class MultimodalFusionModel(nn.Module):
    """
    Dual-branch fusion model for COVID detection.

    Branch A (Vision): Extracts image features via CNN/ResNet backbone.
    Branch B (Tabular): Processes metadata features via TabularMLP.

    The outputs are concatenated and passed through a fusion head:
      Concat(vision_features, tabular_features)  →  dim = D + 32
      Linear(D+32, 128) → BatchNorm → ReLU → Dropout(0.4)
      Linear(128, num_classes)

    When use_metadata=False, Branch B is bypassed:
      vision_features → Linear(D, 128) → BatchNorm → ReLU → Dropout(0.4)
      Linear(128, num_classes)
    """

    def __init__(
        self,
        backbone,
        n_metadata_features=0,
        num_classes=3,
        use_metadata=True,
        tabular_hidden_dim=64,
        tabular_output_dim=32,
        fusion_hidden_dim=128,
        fusion_dropout=0.4,
    ):
        """
        Args:
            backbone: Vision model instance (BasicCNN, MiniVGG, MiniResNet, PretrainedResNet)
            n_metadata_features: Number of preprocessed metadata features
            num_classes: Number of output classes
            use_metadata: Whether to use the tabular branch
            tabular_hidden_dim: Hidden layer size in TabularMLP
            tabular_output_dim: Output size of TabularMLP
            fusion_hidden_dim: Hidden layer size in the fusion head
            fusion_dropout: Dropout rate in the fusion head
        """
        super().__init__()

        self.use_metadata = use_metadata and (n_metadata_features > 0)

        # Branch A: Vision
        self.vision = VisionFeatureExtractor(backbone)
        vision_dim = self.vision.feature_dim

        # Branch B: Tabular (optional)
        if self.use_metadata:
            self.tabular = TabularMLP(
                n_features=n_metadata_features,
                hidden_dim=tabular_hidden_dim,
                output_dim=tabular_output_dim,
            )
            fusion_input_dim = vision_dim + tabular_output_dim
        else:
            self.tabular = None
            fusion_input_dim = vision_dim

        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        print(f"  🔀 MultimodalFusionModel created:")
        print(f"     Vision dim:      {vision_dim}")
        print(f"     Use metadata:    {self.use_metadata}")
        if self.use_metadata:
            print(f"     Metadata feats:  {n_metadata_features} → {tabular_output_dim}")
        print(f"     Fusion input:    {fusion_input_dim}")
        print(f"     Fusion hidden:   {fusion_hidden_dim}")
        print(f"     Output classes:  {num_classes}")

    def forward(self, image, metadata=None):
        """
        Forward pass.

        Args:
            image: (B, C, H, W) image tensor
            metadata: (B, n_features) metadata tensor (required if use_metadata=True)

        Returns:
            (B, num_classes) logits
        """
        # Vision branch
        vision_features = self.vision(image)

        if self.use_metadata:
            if metadata is None:
                raise ValueError(
                    "Metadata tensor required when use_metadata=True. "
                    "Pass metadata or set use_metadata=False."
                )
            # Tabular branch
            tabular_features = self.tabular(metadata)
            # Concatenate along feature dimension
            fused = torch.cat([vision_features, tabular_features], dim=1)
        else:
            fused = vision_features

        return self.fusion_head(fused)


# =====================================================================
#  6. DATALOADER FACTORY
# =====================================================================

def _multimodal_collate_fn(batch):
    """Custom collate for 3-tuple (image, metadata, label) batches."""
    if len(batch[0]) == 3:
        images, metadata, labels = zip(*batch)
        return (
            torch.stack(images),
            torch.stack(metadata),
            torch.stack(labels),
        )
    else:
        # 2-tuple: image-only mode
        images, labels = zip(*batch)
        return torch.stack(images), torch.stack(labels)


def create_multimodal_dataloaders(
    train_df,
    val_df,
    test_df,
    label_encoder,
    metadata_preprocessor=None,
    use_metadata=True,
    img_size=(224, 224),
    batch_size=16,
    num_workers=4,
    use_randaugment=True,
    use_weighted_sampler=True,
    pin_memory=True,
):
    """
    Create DataLoaders for the multimodal pipeline.

    Fits MetadataPreprocessor on train_df, then transforms all splits.

    Args:
        train_df, val_df, test_df: DataFrames with image_path, label, and metadata
        label_encoder: fitted LabelEncoder
        metadata_preprocessor: MetadataPreprocessor instance (fitted or unfitted)
        use_metadata: whether to include metadata in the dataset
        img_size: image dimensions for transforms
        batch_size: samples per batch
        num_workers: parallel data loading workers
        use_randaugment: include RandAugment in training transforms
        use_weighted_sampler: use WeightedRandomSampler for class balance
        pin_memory: pin data to page-locked memory

    Returns:
        (trainloader, valloader, testloader, metadata_preprocessor)
    """
    train_transform = get_train_transforms(img_size, use_randaugment=use_randaugment)
    eval_transform = get_eval_transforms(img_size)

    # Process metadata
    meta_train = meta_val = meta_test = None
    if use_metadata and metadata_preprocessor is not None:
        if metadata_preprocessor.pipeline is None:
            metadata_preprocessor.fit(train_df)
        meta_train = metadata_preprocessor.transform(train_df)
        meta_val = metadata_preprocessor.transform(val_df)
        meta_test = metadata_preprocessor.transform(test_df)
        print(f"  📊 Metadata shapes — Train: {meta_train.shape}, "
              f"Val: {meta_val.shape}, Test: {meta_test.shape}")

    # Create datasets
    train_dataset = MultimodalXrayDataset(
        train_df, label_encoder, meta_train, train_transform, use_metadata
    )
    val_dataset = MultimodalXrayDataset(
        val_df, label_encoder, meta_val, eval_transform, use_metadata
    )
    test_dataset = MultimodalXrayDataset(
        test_df, label_encoder, meta_test, eval_transform, use_metadata
    )

    # Weighted sampler for class-balanced training
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sampler = get_weighted_sampler(train_df["label"].values, label_encoder)
        shuffle = False

    # DataLoader kwargs
    loader_kwargs = {
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
        "collate_fn": _multimodal_collate_fn,
    }

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        sampler=sampler, drop_last=True, **loader_kwargs,
    )
    valloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )
    testloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )

    print(f"  DataLoaders created — img_size={img_size}, batch_size={batch_size}, "
          f"workers={num_workers}, metadata={use_metadata}")
    print(f"  Train: {len(trainloader)} batches | Val: {len(valloader)} batches | "
          f"Test: {len(testloader)} batches")

    return trainloader, valloader, testloader, metadata_preprocessor


# =====================================================================
#  7. TRAINING / VALIDATION LOOPS
# =====================================================================

def train_multimodal_one_epoch(
    model, dataloader, criterion, optimizer, device, use_metadata=True, max_grad_norm=1.0
):
    """
    Train the multimodal model for one epoch.

    Unpacks (image, metadata, label) or (image, label) from the dataloader
    depending on use_metadata.

    Returns:
        (avg_loss, accuracy_percent)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        if use_metadata and len(batch) == 3:
            images, metadata, labels = batch
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
        else:
            images, labels = batch[0], batch[-1]
            images = images.to(device)
            labels = labels.to(device)
            metadata = None

        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate_multimodal(model, dataloader, criterion, device, use_metadata=True):
    """
    Evaluate the multimodal model on a validation/test set.

    Returns:
        (avg_loss, accuracy_percent, predictions, labels, probabilities)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        if use_metadata and len(batch) == 3:
            images, metadata, labels = batch
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
        else:
            images, labels = batch[0], batch[-1]
            images = images.to(device)
            labels = labels.to(device)
            metadata = None

        outputs = model(images, metadata)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_multimodal_model(
    model,
    model_name,
    trainloader,
    valloader,
    device,
    use_metadata=True,
    num_classes=3,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    scheduler_type="cosine",
    patience=7,
    save_path=None,
):
    """
    Full training loop for the multimodal fusion model.

    Mirrors src/training.py:train_model() but handles the 3-tuple data format.

    Returns:
        (trained_model, history_dict)
    """
    mode_label = "Multimodal" if use_metadata else "Image-Only"
    print(f"\n{'=' * 60}")
    print(f"  Training {model_name} ({mode_label}) on {device}")
    print(f"  Epochs: {epochs} | LR: {lr} | Scheduler: {scheduler_type}")
    print(f"{'=' * 60}\n")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # LR scheduler
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True, min_lr=1e-6
        )

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    history = defaultdict(list)
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_multimodal_one_epoch(
            model, trainloader, criterion, optimizer, device, use_metadata=use_metadata
        )

        val_loss, val_acc, _, _, _ = validate_multimodal(
            model, valloader, criterion, device, use_metadata=use_metadata
        )

        if scheduler is not None:
            if scheduler_type == "cosine":
                scheduler.step()
            elif scheduler_type == "plateau":
                scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"  Epoch [{epoch + 1:3d}/{epochs}]  "
            f"LR: {current_lr:.2e}  |  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:6.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:6.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_model_state, save_path)
                print(f"  💾 Best model saved → {save_path}")

        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\n  Stopped at epoch {epoch + 1}. Best val_loss: {best_val_loss:.4f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n  ✅ Restored best model weights (val_loss: {best_val_loss:.4f})")

    return model, dict(history)


# =====================================================================
#  8. DIAGNOSTIC COMPARISON
# =====================================================================

def run_comparison(
    train_df,
    val_df,
    test_df,
    label_encoder,
    class_names,
    backbone_name="basiccnn",
    include_sparse=False,
    img_size=(224, 224),
    batch_size=16,
    num_workers=4,
    epochs=10,
    lr=1e-3,
    patience=7,
    scheduler_type="cosine",
    device=None,
    results_dir=None,
):
    """
    Train and compare Image-Only vs Multimodal models.

    Trains:
      1. Image-Only model (use_metadata=False)
      2. Multimodal model (use_metadata=True)

    Computes and prints: Accuracy, Macro F1, Per-class Precision/Recall/F1.

    Args:
        train_df, val_df, test_df: DataFrames
        label_encoder: fitted LabelEncoder
        class_names: list of class name strings
        backbone_name: "basiccnn", "minivgg", "miniresnet", or "pretrained"
        include_sparse: include high-missingness features
        img_size: image size tuple
        batch_size: batch size
        num_workers: dataloader workers
        epochs: max training epochs
        lr: learning rate
        patience: early stopping patience
        scheduler_type: LR scheduler type
        device: torch device
        results_dir: directory to save results

    Returns:
        dict with keys "image_only" and "multimodal", each containing metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if results_dir is None:
        results_dir = project_root / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    num_classes = len(class_names)
    results = {}

    # ── Prepare metadata preprocessor ──
    preprocessor = MetadataPreprocessor(include_sparse=include_sparse)
    preprocessor.fit(train_df)

    for mode in ["image_only", "multimodal"]:
        use_metadata = (mode == "multimodal")
        mode_label = "Multimodal" if use_metadata else "Image-Only"

        print(f"\n\n{'#' * 60}")
        print(f"  COMPARISON: {mode_label.upper()}")
        print(f"  Backbone: {backbone_name} | Metadata: {use_metadata}")
        print(f"{'#' * 60}")

        # Create dataloaders
        trainloader, valloader, testloader, _ = create_multimodal_dataloaders(
            train_df, val_df, test_df, label_encoder,
            metadata_preprocessor=preprocessor if use_metadata else None,
            use_metadata=use_metadata,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Create fresh backbone + fusion model
        backbone = _create_backbone(backbone_name, num_classes)
        model = MultimodalFusionModel(
            backbone=backbone,
            n_metadata_features=preprocessor.n_features if use_metadata else 0,
            num_classes=num_classes,
            use_metadata=use_metadata,
        )

        save_path = str(results_dir / f"multimodal_{mode}_{backbone_name}_best.pth")

        # Train
        trained_model, history = train_multimodal_model(
            model=model,
            model_name=f"{backbone_name}_{mode}",
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            use_metadata=use_metadata,
            num_classes=num_classes,
            epochs=epochs,
            lr=lr,
            patience=patience,
            scheduler_type=scheduler_type,
            save_path=save_path,
        )

        # Evaluate on test set
        criterion = nn.CrossEntropyLoss()
        _, test_acc, preds, labels, probs = validate_multimodal(
            trained_model, testloader, criterion, device, use_metadata=use_metadata
        )

        # Compute metrics
        accuracy = accuracy_score(labels, preds) * 100
        f1_macro = f1_score(labels, preds, average="macro") * 100
        f1_per_class = f1_score(labels, preds, average=None) * 100
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0) * 100
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0) * 100

        results[mode] = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_per_class": f1_per_class,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "history": history,
            "preds": preds,
            "labels": labels,
            "probs": probs,
        }

        print(f"\n  {mode_label} Results:")
        print(f"    Accuracy:  {accuracy:.2f}%")
        print(f"    Macro F1:  {f1_macro:.2f}%")
        print(classification_report(
            labels, preds, target_names=class_names, zero_division=0
        ))

    # ── Print comparison table ──
    _print_comparison_table(results, class_names)

    return results


def _print_comparison_table(results, class_names):
    """Pretty-print a comparison table of Image-Only vs Multimodal metrics."""
    img = results["image_only"]
    mm = results["multimodal"]

    print(f"\n{'=' * 70}")
    print(f"  DIAGNOSTIC COMPARISON: Image-Only vs Multimodal")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<30s}  {'Image-Only':>12s}  {'Multimodal':>12s}  {'Delta':>8s}")
    print(f"  {'-' * 66}")

    # Overall metrics
    for metric_name, key in [("Accuracy", "accuracy"), ("Macro F1-Score", "f1_macro")]:
        v_img = img[key]
        v_mm = mm[key]
        delta = v_mm - v_img
        sign = "+" if delta >= 0 else ""
        print(f"  {metric_name:<30s}  {v_img:>11.2f}%  {v_mm:>11.2f}%  {sign}{delta:>6.2f}%")

    print(f"  {'-' * 66}")

    # Per-class F1
    for i, cls_name in enumerate(class_names):
        v_img = img["f1_per_class"][i]
        v_mm = mm["f1_per_class"][i]
        delta = v_mm - v_img
        sign = "+" if delta >= 0 else ""
        print(f"  F1 — {cls_name:<24s}  {v_img:>11.2f}%  {v_mm:>11.2f}%  {sign}{delta:>6.2f}%")

    print(f"{'=' * 70}")

    delta_acc = mm["accuracy"] - img["accuracy"]
    delta_f1 = mm["f1_macro"] - img["f1_macro"]
    if delta_acc > 0 and delta_f1 > 0:
        print(f"  ✅ Multimodal fusion improved Accuracy by {delta_acc:+.2f}% "
              f"and Macro F1 by {delta_f1:+.2f}%")
    elif delta_acc > 0 or delta_f1 > 0:
        print(f"  ⚠️  Mixed results — Accuracy Δ: {delta_acc:+.2f}%, Macro F1 Δ: {delta_f1:+.2f}%")
    else:
        print(f"  ❌ Multimodal fusion did not improve over Image-Only baseline")
        print(f"     Consider: more metadata features, unfreezing backbone, or more data")

    print()


# =====================================================================
#  9. HELPERS
# =====================================================================

def _create_backbone(backbone_name, num_classes):
    """Create a vision backbone by name."""
    backbone_name = backbone_name.lower()
    if backbone_name == "basiccnn":
        return BasicCNN(num_classes=num_classes)
    elif backbone_name == "minivgg":
        return MiniVGG(num_classes=num_classes)
    elif backbone_name == "miniresnet":
        return MiniResNet(num_classes=num_classes)
    elif backbone_name == "pretrained":
        return PretrainedResNet(num_classes=num_classes, freeze_backbone=True)
    else:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. "
            f"Choose from: basiccnn, minivgg, miniresnet, pretrained"
        )


def _get_img_size(backbone_name):
    """Return the appropriate image size for a given backbone."""
    if backbone_name.lower() in ("basiccnn", "minivgg"):
        return (32, 32)
    else:
        return (224, 224)


# =====================================================================
#  10. CLI ENTRY POINT
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal Fusion Pipeline for COVID Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multimodal training with PretrainedResNet
  python src/multimodal_pipeline.py --backbone pretrained --epochs 20

  # Image-only baseline
  python src/multimodal_pipeline.py --backbone pretrained --epochs 20 --no-metadata

  # Head-to-head comparison
  python src/multimodal_pipeline.py --backbone basiccnn --epochs 10 --compare

  # Include high-missingness features
  python src/multimodal_pipeline.py --backbone pretrained --epochs 20 --include-sparse-features
""",
    )
    parser.add_argument(
        "--backbone", type=str, default="pretrained",
        choices=["basiccnn", "minivgg", "miniresnet", "pretrained"],
        help="Vision backbone architecture (default: pretrained)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument(
        "--scheduler", type=str, default="cosine",
        choices=["cosine", "plateau", "none"],
        help="LR scheduler type",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--no-metadata", action="store_true",
        help="Disable metadata branch (image-only mode)",
    )
    parser.add_argument(
        "--include-sparse-features", action="store_true",
        help="Include high-missingness features (survival, intubated, went_icu)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run Image-Only vs Multimodal comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}")

    # ── 1. Data Loading ──
    print(f"\n{'=' * 60}")
    print("  STEP 1: Data Loading & Preprocessing")
    print(f"{'=' * 60}")

    metadata_path = project_root / "data" / "metadata.csv"
    images_dir = project_root / "data" / "images"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    df = load_data(str(metadata_path), str(images_dir))
    df = clean_data(df)

    # ── 2. Split ──
    train_df, val_df, test_df = split_data(df, test_size=0.15, val_size=0.15)

    label_encoder = LabelEncoder()
    label_encoder.fit(df["label"])
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    print(f"\n  Classes: {class_names}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    img_size = _get_img_size(args.backbone)

    # ── 3. Comparison mode ──
    if args.compare:
        results = run_comparison(
            train_df, val_df, test_df,
            label_encoder=label_encoder,
            class_names=class_names,
            backbone_name=args.backbone,
            include_sparse=args.include_sparse_features,
            img_size=img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            scheduler_type=args.scheduler if args.scheduler != "none" else None,
            device=device,
            results_dir=results_dir,
        )
        print(f"\n  All results saved to: {results_dir}/")
        print("  Done! ✅\n")
        return

    # ── 4. Single-mode training ──
    use_metadata = not args.no_metadata
    mode_label = "Multimodal" if use_metadata else "Image-Only"

    print(f"\n  Mode: {mode_label}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Image size: {img_size}")

    # Prep metadata preprocessor
    preprocessor = None
    if use_metadata:
        preprocessor = MetadataPreprocessor(include_sparse=args.include_sparse_features)

    trainloader, valloader, testloader, preprocessor = create_multimodal_dataloaders(
        train_df, val_df, test_df,
        label_encoder=label_encoder,
        metadata_preprocessor=preprocessor,
        use_metadata=use_metadata,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create model
    backbone = _create_backbone(args.backbone, num_classes)
    n_meta = preprocessor.n_features if (use_metadata and preprocessor) else 0

    model = MultimodalFusionModel(
        backbone=backbone,
        n_metadata_features=n_meta,
        num_classes=num_classes,
        use_metadata=use_metadata,
    )

    save_path = str(results_dir / f"multimodal_{args.backbone}_best.pth")

    # Train
    trained_model, history = train_multimodal_model(
        model=model,
        model_name=f"{args.backbone}_{mode_label.lower().replace('-', '_')}",
        trainloader=trainloader,
        valloader=valloader,
        device=device,
        use_metadata=use_metadata,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        scheduler_type=args.scheduler if args.scheduler != "none" else None,
        save_path=save_path,
    )

    # Final evaluation
    criterion = nn.CrossEntropyLoss()
    _, test_acc, preds, labels, probs = validate_multimodal(
        trained_model, testloader, criterion, device, use_metadata=use_metadata
    )

    print(f"\n{'=' * 60}")
    print(f"  FINAL TEST RESULTS — {args.backbone} ({mode_label})")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"{'=' * 60}")
    print(classification_report(
        labels, preds, target_names=class_names, zero_division=0
    ))

    print(f"\n  Model saved to: {save_path}")
    print(f"  Results in: {results_dir}/")
    print("  Done! ✅\n")


if __name__ == "__main__":
    main()
