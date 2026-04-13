"""
Publication-quality visualization utilities for the COVID Detection pipeline.

Provides:
  - Training/validation curves with EMA smoothing
  - Confusion matrix with class labels and normalized values
  - Per-class precision/recall/F1 bar charts
  - Grad-CAM heatmap visualization (hook-based, no external deps)
  - Original EDA plots (class distribution, sample images, etc.)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

# ──────────────────────────────────────────────────────────────────────
# Global style configuration — publication-quality defaults
# ──────────────────────────────────────────────────────────────────────

# Use a clean, modern style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Curated color palette
COLORS = {
    "train": "#2196F3",       # Blue
    "val": "#FF5722",         # Deep Orange
    "precision": "#42A5F5",   # Light Blue
    "recall": "#66BB6A",      # Green
    "f1": "#FFA726",          # Orange
    "accent": "#AB47BC",      # Purple
}


# ──────────────────────────────────────────────────────────────────────
# Helper: Exponential Moving Average smoothing
# ──────────────────────────────────────────────────────────────────────

def _ema_smooth(values, alpha=0.3):
    """
    Exponential Moving Average smoothing for noisy training curves.

    EMA reduces epoch-to-epoch noise while preserving trends. This makes
    it much easier to compare training vs. validation curves visually.

    Args:
        values: list of scalar values (one per epoch)
        alpha: smoothing factor (0 = heavy smoothing, 1 = no smoothing)

    Returns:
        list of smoothed values
    """
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = alpha * v + (1 - alpha) * last
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# ──────────────────────────────────────────────────────────────────────
# 1. Training Curves
# ──────────────────────────────────────────────────────────────────────

def plot_training_curves(history, title=None, save_path=None):
    """
    Plot training vs. validation loss and accuracy curves with EMA smoothing.

    Raw values are shown as faint lines for reference, with bold EMA-smoothed
    curves overlaid. The best validation epoch is marked with a dashed line.

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        title: optional figure title
        save_path: optional path to save the figure (300 DPI PNG)
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title or "Training & Validation Curves", fontsize=15, fontweight="bold")

    # --- Loss curves ---
    ax1.plot(epochs, history["train_loss"], color=COLORS["train"], alpha=0.25, linewidth=1)
    ax1.plot(epochs, history["val_loss"], color=COLORS["val"], alpha=0.25, linewidth=1)

    train_loss_smooth = _ema_smooth(history["train_loss"])
    val_loss_smooth = _ema_smooth(history["val_loss"])

    ax1.plot(epochs, train_loss_smooth, color=COLORS["train"], linewidth=2.5,
             label="Train Loss (smoothed)")
    ax1.plot(epochs, val_loss_smooth, color=COLORS["val"], linewidth=2.5,
             label="Val Loss (smoothed)")

    # Mark best validation epoch
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val_loss = min(history["val_loss"])
    ax1.axvline(x=best_epoch, color=COLORS["accent"], linestyle="--", alpha=0.7, linewidth=1.5)
    ax1.annotate(f"Best: {best_val_loss:.4f}\n(epoch {best_epoch})",
                 xy=(best_epoch, best_val_loss),
                 xytext=(best_epoch + 1, best_val_loss * 1.15),
                 fontsize=9, color=COLORS["accent"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.2))

    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_title("Loss")

    # --- Accuracy curves ---
    ax2.plot(epochs, history["train_acc"], color=COLORS["train"], alpha=0.25, linewidth=1)
    ax2.plot(epochs, history["val_acc"], color=COLORS["val"], alpha=0.25, linewidth=1)

    train_acc_smooth = _ema_smooth(history["train_acc"])
    val_acc_smooth = _ema_smooth(history["val_acc"])

    ax2.plot(epochs, train_acc_smooth, color=COLORS["train"], linewidth=2.5,
             label="Train Acc (smoothed)")
    ax2.plot(epochs, val_acc_smooth, color=COLORS["val"], linewidth=2.5,
             label="Val Acc (smoothed)")

    best_val_acc = max(history["val_acc"])
    best_acc_epoch = int(np.argmax(history["val_acc"])) + 1
    ax2.axvline(x=best_acc_epoch, color=COLORS["accent"], linestyle="--",
                alpha=0.7, linewidth=1.5)

    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.set_title("Accuracy")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  📊 Training curves saved → {save_path}")

    plt.show()


# ──────────────────────────────────────────────────────────────────────
# 2. Confusion Matrix
# ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, title=None, save_path=None):
    """
    Plot a confusion matrix heatmap with both raw counts and percentages.

    Each cell shows: count (percentage). The color intensity reflects the
    normalized (row-wise) percentage, making it easy to spot class-specific
    error patterns.

    Args:
        y_true: ground truth labels (integer-encoded)
        y_pred: predicted labels (integer-encoded)
        class_names: list of string class names
        title: optional figure title
        save_path: optional path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Build annotation strings: "count\n(XX.X%)"
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm_normalized, annot=annotations, fmt="",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Percentage (%)"},
        vmin=0, vmax=100, ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title or "Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  📊 Confusion matrix saved → {save_path}")

    plt.show()


# ──────────────────────────────────────────────────────────────────────
# 3. Per-class Metrics Bar Chart
# ──────────────────────────────────────────────────────────────────────

def plot_classwise_metrics(y_true, y_pred, class_names, title=None, save_path=None):
    """
    Grouped bar chart of Precision, Recall, and F1-score per class.

    This reveals class-specific weaknesses. For example, on imbalanced
    datasets, accuracy can look high while recall on minority classes
    is near zero — this chart makes that immediately visible.

    Args:
        y_true: ground truth labels (integer-encoded)
        y_pred: predicted labels (integer-encoded)
        class_names: list of string class names
        title: optional figure title
        save_path: optional path to save figure
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    x = np.arange(len(class_names))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - bar_width, precision, bar_width, label="Precision",
                   color=COLORS["precision"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, recall, bar_width, label="Recall",
                   color=COLORS["recall"], edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + bar_width, f1, bar_width, label="F1-Score",
                   color=COLORS["f1"], edgecolor="white", linewidth=0.5)

    # Add value labels on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    # Add support counts as secondary annotation
    for i, s in enumerate(support):
        ax.annotate(f"n={s}", xy=(x[i], -0.08), ha="center", fontsize=8,
                    color="gray", style="italic")

    ax.set_xlabel("Class", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(title or "Per-Class Precision / Recall / F1-Score",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  📊 Class metrics chart saved → {save_path}")

    plt.show()

    # Also print the full classification report
    print("\n" + classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    ))


# ──────────────────────────────────────────────────────────────────────
# 4. Grad-CAM Visualization
# ──────────────────────────────────────────────────────────────────────

class _GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.

    From Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization".

    Produces a coarse heatmap highlighting the image regions that most
    influenced the model's prediction. Works by:
      1. Hooking into a target convolutional layer
      2. Computing the gradient of the predicted class score w.r.t. that
         layer's activations
      3. Global-average-pooling those gradients to get per-channel importance
      4. Weighting the activations by importance and ReLU'ing

    This is hook-based (no model modification needed) and works with any
    CNN architecture.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: trained nn.Module
            target_layer: the nn.Module to hook (e.g., model.features[-1])
        """
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor (1, C, H, W)
            target_class: class index to visualize (None = predicted class)

        Returns:
            heatmap: numpy array (H, W) with values in [0, 1]
        """
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Global Average Pooling of gradients → channel importance weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only keep positive influences
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def _get_target_layer(model):
    """
    Auto-detect the last convolutional layer for Grad-CAM.

    Tries to find the appropriate target layer based on model architecture.
    """
    # PretrainedResNet → last layer of backbone
    if hasattr(model, "base") and hasattr(model.base, "layer4"):
        return model.base.layer4[-1].conv2

    # MiniResNet → last residual block
    if hasattr(model, "layer3") and isinstance(model.layer3, torch.nn.Module):
        return model.layer3.conv2

    # BasicCNN / MiniVGG → last conv in features Sequential
    if hasattr(model, "features"):
        for layer in reversed(list(model.features.modules())):
            if isinstance(layer, torch.nn.Conv2d):
                return layer

    raise ValueError("Could not auto-detect target layer. Please specify manually.")


def _unnormalize(tensor, mean, std):
    """Reverse ImageNet normalization for display."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)


def plot_gradcam(model, image_tensor, class_names, target_class=None,
                 target_layer=None, title=None, save_path=None):
    """
    Visualize Grad-CAM heatmap for a single image.

    Shows three panels: Original image | Grad-CAM Heatmap | Overlay.
    The heatmap reveals which spatial regions the model focuses on when
    making its prediction — critical for clinical interpretability.

    Args:
        model: trained nn.Module
        image_tensor: preprocessed image tensor (1, C, H, W) or (C, H, W)
        class_names: list of string class names
        target_class: class index to visualize (None = use predicted)
        target_layer: specific layer to hook (None = auto-detect)
        title: optional figure title
        save_path: optional path to save figure
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    # Auto-detect target layer if not specified
    if target_layer is None:
        target_layer = _get_target_layer(model)

    # Generate Grad-CAM
    grad_cam = _GradCAM(model, target_layer)
    heatmap = grad_cam.generate(image_tensor, target_class)

    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()

    # Prepare display image
    img_display = _unnormalize(
        image_tensor.squeeze(0).cpu(),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ).permute(1, 2, 0).numpy()

    # Resize heatmap to image dimensions
    from PIL import Image as PILImage
    heatmap_resized = np.array(
        PILImage.fromarray(np.uint8(heatmap * 255)).resize(
            (img_display.shape[1], img_display.shape[0]),
            PILImage.BILINEAR,
        )
    ).astype(float) / 255.0

    # Create overlay
    import matplotlib.cm as cm
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    overlay = 0.5 * img_display + 0.5 * heatmap_colored

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_display)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontweight="bold")
    axes[2].axis("off")

    pred_label = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"
    fig.suptitle(
        title or f"Grad-CAM — Predicted: {pred_label} ({confidence:.1%})",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  📊 Grad-CAM saved → {save_path}")

    plt.show()


def plot_gradcam_grid(model, dataloader, class_names, n=8,
                      target_layer=None, save_path=None):
    """
    Grid of Grad-CAM visualizations across multiple test samples.

    Shows the overlay (original + heatmap) for `n` samples, with
    prediction labels and confidence scores. Useful for batch-level
    inspection of model attention patterns.

    Args:
        model: trained nn.Module
        dataloader: DataLoader to sample from
        class_names: list of string class names
        n: number of samples to visualize
        target_layer: specific layer to hook (None = auto-detect)
        save_path: optional path to save figure
    """
    device = next(model.parameters()).device
    model.eval()

    if target_layer is None:
        target_layer = _get_target_layer(model)

    grad_cam = _GradCAM(model, target_layer)

    # Collect samples
    images, labels = [], []
    for batch_imgs, batch_labels in dataloader:
        images.append(batch_imgs)
        labels.append(batch_labels)
        if sum(img.size(0) for img in images) >= n:
            break

    images = torch.cat(images)[:n].to(device)
    labels = torch.cat(labels)[:n]

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1)

    import matplotlib.cm as cm

    for idx in range(n):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        img_tensor = images[idx].unsqueeze(0)
        heatmap = grad_cam.generate(img_tensor)

        with torch.no_grad():
            output = model(img_tensor)
            pred_class = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_class].item()

        # Display image
        img_display = _unnormalize(
            images[idx].cpu(),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ).permute(1, 2, 0).numpy()

        from PIL import Image as PILImage
        heatmap_resized = np.array(
            PILImage.fromarray(np.uint8(heatmap * 255)).resize(
                (img_display.shape[1], img_display.shape[0]),
                PILImage.BILINEAR,
            )
        ).astype(float) / 255.0

        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        overlay = 0.5 * img_display + 0.5 * heatmap_colored

        ax.imshow(overlay)
        ax.axis("off")

        true_label = class_names[labels[idx]] if labels[idx] < len(class_names) else "?"
        pred_label = class_names[pred_class] if pred_class < len(class_names) else "?"
        is_correct = pred_class == labels[idx].item()

        color = "green" if is_correct else "red"
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label} ({confidence:.0%})",
            fontsize=10, color=color, fontweight="bold",
        )

    # Hide empty subplots
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    fig.suptitle("Grad-CAM Attention Maps", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  📊 Grad-CAM grid saved → {save_path}")

    plt.show()


# ──────────────────────────────────────────────────────────────────────
# Original EDA plots (preserved and enhanced)
# ──────────────────────────────────────────────────────────────────────

def plot_class_distribution(df, save_path=None):
    """Bar chart of class label distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = df["label"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=[COLORS["train"], COLORS["val"],
                  COLORS["accent"]][:len(counts)],
                  edgecolor="white", linewidth=0.5)

    # Add count labels on bars
    for bar, count in zip(bars, counts.values):
        ax.annotate(
            f"{count}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=12, fontweight="bold",
        )

    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_histograms(df, columns, save_path=None):
    """Histogram grid for metadata feature distributions."""
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        df[col].hist(ax=ax, bins=20, color=COLORS["train"], alpha=0.7, edgecolor="white")
        ax.set_title(col.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def show_sample_images(df, n=5, img_size=(224, 224), save_path=None):
    """Grid of sample images per class."""
    classes = sorted(df["label"].unique())

    fig, axes = plt.subplots(len(classes), n, figsize=(n * 3, len(classes) * 3))

    for i, cls in enumerate(classes):
        subset = df[df["label"] == cls]
        sample_n = min(n, len(subset))
        subset = subset.sample(sample_n)

        for j, (_, row) in enumerate(subset.iterrows()):
            try:
                img = Image.open(row["image_path"]).resize(img_size)
                axes[i, j].imshow(img, cmap="gray")
                axes[i, j].axis("off")
                if j == 0:
                    axes[i, j].set_ylabel(cls, fontsize=12, fontweight="bold", rotation=0,
                                          labelpad=60, va="center")
            except Exception:
                axes[i, j].text(0.5, 0.5, "Error", ha="center", va="center",
                                transform=axes[i, j].transAxes)
                axes[i, j].axis("off")

        # Hide excess subplots if sample_n < n
        for j in range(sample_n, n):
            axes[i, j].axis("off")

    fig.suptitle("Sample Images per Class", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_correlation(df, columns, save_path=None):
    """Feature correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[columns].corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0,
                linewidths=0.5, linecolor="white", ax=ax,
                fmt=".2f", square=True)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_pixel_intensity_histogram(images_batch, title="Normalized Pixel Intensity Distribution",
                                    save_path=None):
    """RGB/grayscale pixel intensity distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    pixels = np.array(images_batch)

    color_map = {"red": "#E53935", "green": "#43A047", "blue": "#1E88E5"}

    if pixels.shape[-1] == 3:  # RGB image
        for i, color in enumerate(["red", "green", "blue"]):
            channel_data = pixels[:, :, :, i].flatten()
            ax.hist(channel_data, bins=50, color=color_map[color], alpha=0.5,
                    label=f"{color.capitalize()} Channel")
    else:  # Grayscale
        channel_data = pixels.flatten()
        ax.hist(channel_data, bins=50, color="gray", alpha=0.7, label="Grayscale Channel")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Pixel Value (Normalized)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()