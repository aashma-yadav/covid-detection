"""
Training utilities for the COVID Detection pipeline.

Implements SOTA training practices:
  - CosineAnnealingWarmRestarts / ReduceLROnPlateau LR scheduling
  - Early stopping with configurable patience
  - Class-weighted CrossEntropyLoss for imbalanced datasets
  - Gradient clipping for training stability
  - Mixup / CutMix data augmentation (batch-level)
  - Full history tracking for downstream visualization
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────
# Class-weighted loss
# ──────────────────────────────────────────────────────────────────────

def compute_class_weights(labels, device="cpu"):
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    With COVID=287, Pneumonia=61, Normal=3 the raw frequencies are wildly
    imbalanced.  Inverse-frequency weighting forces the loss to penalize
    errors on rare classes proportionally more.

    Args:
        labels: array-like of integer-encoded labels
        device: torch device string

    Returns:
        torch.Tensor of shape (num_classes,) with per-class weights
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    counts = np.bincount(labels, minlength=len(classes)).astype(float)

    # Inverse frequency: rarer classes get higher weight
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)  # Normalize so weights sum ≈ num_classes

    return torch.tensor(weights, dtype=torch.float32, device=device)


# ──────────────────────────────────────────────────────────────────────
# Mixup / CutMix  (batch-level augmentation)
# ──────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    """
    Mixup: linearly interpolates pairs of images and their labels.

    From Zhang et al., 2018 — "mixup: Beyond Empirical Risk Minimization".
    This acts as a strong regularizer that smooths decision boundaries and
    reduces overconfident predictions.

    Args:
        x: input batch (B, C, H, W)
        y: label batch (B,)
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix: cuts a rectangular region from one image and pastes onto another.

    From Yun et al., 2019 — "CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features". Forces the model to attend to
    the full spatial extent of the image, not just discriminative local patches.

    Args:
        x: input batch (B, C, H, W)
        y: label batch (B,)
        alpha: Beta distribution parameter

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, h, w = x.shape

    # Sample the bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = np.random.randint(h)
    cx = np.random.randint(w)

    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to the actual area ratio
    lam = 1 - ((y2 - y1) * (x2 - x1)) / (h * w)

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixed loss for Mixup/CutMix: weighted combination of losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss hasn't improved for `patience` epochs.

    This prevents overfitting by detecting when the model starts memorizing
    the training set instead of learning generalizable features.
    """

    def __init__(self, patience=7, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ⏳ EarlyStopping: {self.counter}/{self.patience} "
                      f"(best val_loss: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("  🛑 Early stopping triggered.")


# ──────────────────────────────────────────────────────────────────────
# Single-epoch routines
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    use_mixup=False, use_cutmix=False, max_grad_norm=1.0):
    """
    Train for a single epoch.

    Args:
        model: nn.Module
        dataloader: training DataLoader
        criterion: loss function
        optimizer: optimizer
        device: torch device
        use_mixup: apply Mixup augmentation
        use_cutmix: apply CutMix augmentation (mutually exclusive with Mixup per batch)
        max_grad_norm: gradient clipping max norm (prevents exploding gradients)

    Returns:
        (avg_loss, accuracy_percent)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Optionally apply Mixup or CutMix
        apply_mix = False
        if use_cutmix and np.random.rand() < 0.5:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels)
            apply_mix = True
        elif use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
            apply_mix = True

        optimizer.zero_grad()
        outputs = model(inputs)

        if apply_mix:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping — stabilizes training, especially with small batches
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
def validate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set.

    Returns:
        (avg_loss, accuracy_percent, all_predictions, all_labels, all_probabilities)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
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


# ──────────────────────────────────────────────────────────────────────
# Full training orchestrator
# ──────────────────────────────────────────────────────────────────────

def train_model(
    model,
    model_name,
    trainloader,
    valloader,
    device,
    num_classes=3,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    class_weights=None,
    scheduler_type="cosine",      # "cosine" | "plateau" | None
    patience=7,
    use_mixup=False,
    use_cutmix=False,
    save_path=None,
):
    """
    Full training loop with SOTA practices.

    Args:
        model: nn.Module to train
        model_name: string identifier for logging
        trainloader: training DataLoader
        valloader: validation DataLoader
        device: torch device
        num_classes: number of output classes
        epochs: maximum training epochs
        lr: initial learning rate
        weight_decay: L2 regularization strength
        class_weights: optional tensor of per-class weights for loss
        scheduler_type: LR scheduler type — "cosine", "plateau", or None
        patience: early stopping patience (epochs)
        use_mixup: enable Mixup augmentation
        use_cutmix: enable CutMix augmentation
        save_path: path to save best model checkpoint

    Returns:
        (trained_model, history_dict)
        history_dict keys: train_loss, train_acc, val_loss, val_acc, lr
    """
    print(f"\n{'='*60}")
    print(f"  Training {model_name} on {device}")
    print(f"  Epochs: {epochs} | LR: {lr} | Scheduler: {scheduler_type}")
    print(f"  Mixup: {use_mixup} | CutMix: {use_cutmix}")
    print(f"{'='*60}\n")

    model = model.to(device)

    # --- Loss function with optional class weighting ---
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer ---
    # Only optimize parameters that require gradients
    # (important for PretrainedResNet with frozen backbone)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # --- LR Scheduler ---
    scheduler = None
    if scheduler_type == "cosine":
        # CosineAnnealingWarmRestarts (SGDR): cyclically decays and resets LR.
        # T_0=10 means restart every 10 epochs; T_mult=2 doubles the cycle.
        # This helps escape local minima by periodically increasing LR.
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    elif scheduler_type == "plateau":
        # ReduceLROnPlateau: reduces LR when validation loss plateaus.
        # More conservative than cosine — good when you want monotonic convergence.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True, min_lr=1e-6
        )

    # --- Early Stopping ---
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # --- History tracking ---
    history = defaultdict(list)
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device,
            use_mixup=use_mixup, use_cutmix=use_cutmix
        )

        # Validate
        val_loss, val_acc, _, _, _ = validate(model, valloader, criterion, device)

        # Step scheduler
        if scheduler is not None:
            if scheduler_type == "cosine":
                scheduler.step()
            elif scheduler_type == "plateau":
                scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Print epoch summary
        print(
            f"  Epoch [{epoch+1:3d}/{epochs}]  "
            f"LR: {current_lr:.2e}  |  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:6.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:6.2f}%"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_model_state, save_path)
                print(f"  💾 Best model saved → {save_path}")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\n  Stopped at epoch {epoch+1}. Best val_loss: {best_val_loss:.4f}")
            break

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n  ✅ Restored best model weights (val_loss: {best_val_loss:.4f})")

    return model, dict(history)


def evaluate_model(model, model_name, dataloader, device, class_names, class_weights=None):
    """
    Final evaluation on test set with full metric reporting.

    Args:
        model: trained nn.Module
        model_name: string identifier
        dataloader: test DataLoader
        device: torch device
        class_names: list of class name strings
        class_weights: optional class weights for loss

    Returns:
        (predictions, labels, probabilities)
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    val_loss, val_acc, preds, labels, probs = validate(model, dataloader, criterion, device)

    print(f"\n{'='*60}")
    print(f"  FINAL TEST RESULTS — {model_name}")
    print(f"  Test Loss: {val_loss:.4f}  |  Test Accuracy: {val_acc:.2f}%")
    print(f"{'='*60}\n")

    return preds, labels, probs
