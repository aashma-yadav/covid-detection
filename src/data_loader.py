"""
Data loading and preprocessing for the COVID Detection pipeline.

Enhancements over original:
  - get_train_transforms / get_eval_transforms: encapsulated, SOTA augmentation
    pipelines with RandAugment and RandomErasing
  - create_dataloaders: factory with num_workers, pin_memory, prefetch_factor
    to eliminate CPU→GPU data-loading bottlenecks
  - get_class_weights / get_weighted_sampler: handle severe class imbalance
    (COVID=287, Pneumonia=61, Normal=3)
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────────────────────────────
# Original data loading functions (preserved)
# ──────────────────────────────────────────────────────────────────────

def load_data(metadata_path, images_dir):

    df = pd.read_csv(metadata_path)

    print("Initial shape:", df.shape)

    # Add image path
    df["image_path"] = df["filename"].apply(
        lambda x: os.path.join(images_dir, x) if isinstance(x, str) else None
    )

    # Remove missing images
    df = df[df["image_path"].apply(lambda x: os.path.exists(x))]

    print("After removing missing images:", df.shape)

    return df

def assign_label(finding):
    if not isinstance(finding, str):
        return None

    f = finding.lower()

    pneumonia_keywords = [
        "pneumonia",
        "sars",
        "mers",
        "streptococcus",
        "pneumocystis",
        "pneumococcal",
        "legionella",
        "klebsiella",
        "chlamydophila",
        "e.coli",
        "aspiration",
        "parapneumonic",
        "ards",
    ]

    if "covid" in f:
        return "COVID"
    elif "normal" in f or "no finding" in f:
        return "Normal"
    elif any(keyword in f for keyword in pneumonia_keywords):
        return "Pneumonia"
    else:
        return None
    
def clean_data(df):
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    df["label"] = df["finding"].apply(assign_label)
    df = df[df["label"].notna()]

    print("After labeling:", df.shape)
    print(df["label"].value_counts())

    return df

def preprocess_metadata(df):
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median())

    df["sex"] = df["sex"].str.upper()
    df["sex_enc"] = (df["sex"] == "M").astype(int)

    df["view"] = df["view"].str.upper()
    df["view_pa"] = (df["view"] == "PA").astype(int)

    features = ["age", "sex_enc", "view_pa"]

    X = df[features]
    y = df["label"]

    return X, y, features

def split_data(df, test_size=0.15, val_size=0.15):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=42
    )

    val_ratio = val_size / (1 - test_size)

    train, val = train_test_split(
        train_val, test_size=val_ratio,
        stratify=train_val["label"], random_state=42
    )

    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    return train, val, test


# ──────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────

class XrayDataset(Dataset):
    """PyTorch Dataset for chest X-ray images with label encoding."""

    def __init__(self, dataframe, label_encoder, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = int(self.label_encoder.transform([row["label"]])[0])
        return image, torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────
# SOTA Transform Pipelines
# ──────────────────────────────────────────────────────────────────────

# ImageNet channel statistics — standard for pretrained model compatibility
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size=(224, 224), use_randaugment=True):
    """
    Training augmentation pipeline with SOTA techniques.

    Key augmentations and why they help:
      - RandAugment (Cubuk et al., 2020): automatically searches the
        augmentation policy space. num_ops=2, magnitude=9 is a proven
        default that works well on medical images without being too aggressive.
      - RandomErasing (Zhong et al., 2020): randomly erases rectangular
        patches, forcing the model to rely on global context rather than
        local discriminative patches. Important for X-rays where pathology
        can appear anywhere in the lung fields.
      - RandomHorizontalFlip: anatomically valid for PA chest X-rays
        (lungs are roughly symmetric).
      - RandomRotation(±15°): accounts for slight patient positioning variation.
      - ColorJitter: simulates X-ray exposure/contrast variation.

    Args:
        img_size: output image dimensions (H, W)
        use_randaugment: whether to include RandAugment

    Returns:
        torchvision.transforms.Compose pipeline
    """
    transform_list = [
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ]

    if use_randaugment:
        # RandAugment: automated augmentation policy search
        # num_ops=2 applies 2 random transforms per image
        # magnitude=9 (out of 30) is moderately aggressive
        transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if use_randaugment:
        # RandomErasing applied post-tensor — acts as an information dropout
        transform_list.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)))

    return transforms.Compose(transform_list)


def get_eval_transforms(img_size=(224, 224)):
    """
    Evaluation/test transform pipeline — deterministic, no augmentation.

    Args:
        img_size: output image dimensions (H, W)

    Returns:
        torchvision.transforms.Compose pipeline
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────────────
# Class Imbalance Handling
# ──────────────────────────────────────────────────────────────────────

def get_class_weights(labels, device="cpu"):
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    This is critical for the COVID dataset where Normal has only 3 samples.
    Without weighting, the model will achieve ~82% accuracy by simply
    predicting everything as COVID (287/351 ≈ 82%).

    Args:
        labels: array-like of string labels (e.g., ["COVID", "Pneumonia", ...])
        device: torch device

    Returns:
        torch.Tensor of per-class weights
    """
    from collections import Counter
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)

    # Sort by class name to match LabelEncoder ordering
    sorted_classes = sorted(counts.keys())
    weights = []
    for cls in sorted_classes:
        w = total / (num_classes * counts[cls])
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_weighted_sampler(labels, label_encoder):
    """
    Create a WeightedRandomSampler for balanced batch sampling.

    This ensures each batch contains roughly equal representation from
    all classes, even when the underlying dataset is severely imbalanced.
    Without this, many training batches would contain zero Normal samples.

    Args:
        labels: array-like of string labels
        label_encoder: fitted LabelEncoder

    Returns:
        WeightedRandomSampler instance
    """
    from collections import Counter
    encoded = label_encoder.transform(labels)
    counts = Counter(encoded.tolist())
    total = len(encoded)

    # Weight per sample = 1 / count_of_its_class
    class_weight = {cls: total / count for cls, count in counts.items()}
    sample_weights = [class_weight[label] for label in encoded]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ──────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ──────────────────────────────────────────────────────────────────────

def create_dataloaders(
    train_df,
    val_df,
    test_df,
    label_encoder,
    img_size=(224, 224),
    batch_size=16,
    num_workers=4,
    use_randaugment=True,
    use_weighted_sampler=True,
    pin_memory=True,
):
    """
    Create optimized DataLoaders for train/val/test splits.

    Performance optimizations:
      - num_workers=4: parallel data loading threads prevent the GPU from
        waiting on CPU image decoding. Rule of thumb: 4× number of GPUs.
      - pin_memory=True: allocates data in page-locked (pinned) memory,
        enabling faster CPU→GPU transfer via DMA.
      - persistent_workers=True: keeps worker processes alive between epochs,
        avoiding the overhead of spawning new processes each epoch.
      - prefetch_factor=2: each worker pre-loads 2 batches in advance.

    Args:
        train_df, val_df, test_df: DataFrames with 'image_path' and 'label'
        label_encoder: fitted LabelEncoder
        img_size: image dimensions for transforms
        batch_size: samples per batch
        num_workers: parallel data loading workers
        use_randaugment: include RandAugment in training transforms
        use_weighted_sampler: use WeightedRandomSampler for class balance
        pin_memory: pin data to page-locked memory

    Returns:
        (trainloader, valloader, testloader)
    """
    train_transform = get_train_transforms(img_size, use_randaugment=use_randaugment)
    eval_transform = get_eval_transforms(img_size)

    train_dataset = XrayDataset(train_df, label_encoder, transform=train_transform)
    val_dataset = XrayDataset(val_df, label_encoder, transform=eval_transform)
    test_dataset = XrayDataset(test_df, label_encoder, transform=eval_transform)

    # Weighted sampler for class-balanced training batches
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sampler = get_weighted_sampler(train_df["label"].values, label_encoder)
        shuffle = False  # sampler and shuffle are mutually exclusive

    # Common DataLoader kwargs for optimal throughput
    loader_kwargs = {
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
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
          f"workers={num_workers}")
    print(f"  Train: {len(trainloader)} batches | Val: {len(valloader)} batches | "
          f"Test: {len(testloader)} batches")

    return trainloader, valloader, testloader
