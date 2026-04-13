# COVID-19 Detection from Chest X-Rays 🫁

A multimodal deep learning pipeline that classifies chest X-ray images into **COVID**, **Pneumonia**, and **Normal** categories. What makes this different from your typical image classifier is the dual-branch fusion architecture — it doesn't just look at pixels, it also factors in patient metadata (age, sex, imaging view, etc.) to make better-informed predictions.

Built with PyTorch + scikit-learn. Runs on both CPU and GPU.

---

## What's Inside

```
Covid_Detection/
├── data/
│   ├── images/              # Chest X-ray images
│   ├── annotations/         # Annotation files
│   └── metadata.csv         # Patient metadata (age, sex, view, modality, etc.)
├── src/
│   ├── models.py            # All backbone architectures (BasicCNN, MiniVGG, MiniResNet, PretrainedResNet)
│   ├── data_loader.py       # Dataset class, transforms, class-imbalance handling
│   ├── training.py          # Training loop, early stopping, Mixup/CutMix, LR schedulers
│   ├── multimodal_pipeline.py   # The main entry point — multimodal fusion pipeline
│   └── visualization.py    # Plotting utils (training curves, confusion matrix, Grad-CAM)
├── results/                 # Saved model weights, plots, and evaluation artifacts
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Set up the environment

Make sure you have Python 3.9+ installed. Then create a virtual environment and install dependencies:

```bash
# Create a virtual env
python3 -m venv .venv

# Activate it
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install everything
pip install -r requirements.txt
```

> **Note**: The `requirements.txt` includes PyTorch with CUDA support. If you're running on a CPU-only machine, it'll still work fine — the pipeline auto-detects the device and falls back to CPU. You might want to install the CPU-only version of torch to save disk space though.

### 2. Get the data

This project uses the [COVID-19 Chest X-ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset). Download it and set up the `data/` directory as follows:

```
data/
├── images/          # Place all chest X-ray images here
└── metadata.csv     # The dataset's metadata file
```

> The `data/` directory is **not** included in this repository and must be populated manually. Make sure `data/images/` contains the X-ray images and `data/metadata.csv` is present at the root of `data/`.

The CSV should contain columns like `filename`, `finding`, `age`, `sex`, `view`, `modality`, etc. The pipeline maps the `finding` column to three labels:

- **COVID** — any finding containing "covid"
- **Normal** — "normal" or "no finding"
- **Pneumonia** — various pneumonia-related keywords (SARS, MERS, streptococcus, etc.)

---

## Usage

The main script is `src/multimodal_pipeline.py`. Here's how to use it:

### Basic training (multimodal, uses metadata by default)

```bash
python3 src/multimodal_pipeline.py --backbone pretrained --epochs 20
```

### Image-only mode (disable metadata)

If you just want a plain image classifier without the metadata branch:

```bash
python3 src/multimodal_pipeline.py --backbone basiccnn --epochs 20 --no-metadata
```

### Head-to-head comparison

This one's handy — it trains both an image-only and a multimodal model back-to-back, then prints a comparison table showing the accuracy and F1 differences:

```bash
python3 src/multimodal_pipeline.py --backbone basiccnn --epochs 10 --compare
```

### Full CLI options

```
usage: multimodal_pipeline.py [-h]
                              [--backbone {basiccnn,minivgg,miniresnet,pretrained}]
                              [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
                              [--patience PATIENCE] [--scheduler {cosine,plateau,none}]
                              [--num-workers NUM_WORKERS] [--no-metadata]
                              [--include-sparse-features] [--compare]
```

| Flag | Default | What it does |
|------|---------|-------------|
| `--backbone` | `pretrained` | Vision backbone to use (see section below) |
| `--epochs` | `20` | Maximum training epochs |
| `--batch-size` | `16` | Batch size for training |
| `--lr` | `0.001` | Initial learning rate |
| `--patience` | `7` | Early stopping patience — stops training if val loss doesn't improve for N epochs |
| `--scheduler` | `cosine` | LR scheduler — `cosine` (warm restarts), `plateau` (reduce on plateau), or `none` |
| `--num-workers` | `4` | DataLoader workers for parallel data loading |
| `--no-metadata` | off | Disables the metadata branch, runs image-only classification |
| `--include-sparse-features` | off | Includes high-missingness features (survival, intubated, went_icu) — these have >60% missing values, so use with caution |
| `--compare` | off | Trains both image-only and multimodal models and prints a side-by-side comparison |

---

## Swapping the Backbone Model 🔄

One of the nice things about this pipeline is you can swap the vision backbone with a single flag. Each backbone has different tradeoffs:

| Backbone | Flag value | Params | Image size | Best for |
|----------|-----------|--------|------------|----------|
| **BasicCNN** | `basiccnn` | ~2M | 32×32 | Quick experiments, debugging |
| **MiniVGG** | `minivgg` | ~5M | 32×32 | Slightly deeper, VGG-style double conv blocks |
| **MiniResNet** | `miniresnet` | ~5M | 224×224 | Custom residual blocks, learns richer features |
| **PretrainedResNet** | `pretrained` | ~11M (mostly frozen) | 224×224 | **Recommended** — ImageNet-pretrained ResNet-18 with frozen backbone, only the head trains |

The pretrained backbone is the way to go for this dataset size (~350 images). Training from scratch with so few samples is basically asking for overfitting. The ImageNet features transfer surprisingly well to X-ray analysis since low-level features like edges and textures are pretty universal.

```bash
# Try different backbones
python3 src/multimodal_pipeline.py --backbone basiccnn
python3 src/multimodal_pipeline.py --backbone minivgg
python3 src/multimodal_pipeline.py --backbone miniresnet
python3 src/multimodal_pipeline.py --backbone pretrained   # recommended
```

---

## Metadata Toggle & Multimodal Fusion 🧬

The pipeline has a dual-branch architecture:

- **Branch A (Vision)**: Extracts image features through the CNN backbone
- **Branch B (Tabular)**: Processes patient metadata through a small MLP

Both branches get concatenated and fed into a fusion head for the final classification. You can toggle the metadata branch on or off:

```bash
# Multimodal (default) — uses both image + metadata
python3 src/multimodal_pipeline.py --backbone pretrained

# Image-only — skips the metadata branch entirely
python3 src/multimodal_pipeline.py --backbone pretrained --no-metadata
```

**Core metadata features** (used by default):

- `age` — patient age (numerical, median-imputed, standardized)
- `offset` — days since symptom onset (numerical)
- `sex` — patient sex (one-hot encoded)
- `view` — X-ray view position like PA, AP, Lateral (one-hot encoded)
- `modality` — imaging modality (one-hot encoded)

**Sparse features** (opt-in with `--include-sparse-features`):

- `survival`, `intubated`, `went_icu` — these have significant missing values (>60%), so they're excluded by default. Including them adds a `"missing"` category during one-hot encoding to preserve the missingness signal.

---

## Training Features Worth Knowing About

A few things happening under the hood that keep the training stable:

- **Early stopping** — monitors validation loss and stops training if it hasn't improved for `patience` epochs. Prevents the model from memorizing the training set.
- **LR scheduling** — cosine annealing with warm restarts by default. The learning rate decays and periodically resets, which helps escape local minima.
- **Class-weighted sampling** — the dataset is heavily imbalanced (COVID: 287, Pneumonia: 61, Normal: 3). A `WeightedRandomSampler` ensures each batch has roughly balanced class representation.
- **Gradient clipping** — clips gradient norms to 1.0 to prevent training instability, especially with small batches.
- **Data augmentation** — RandAugment, random erasing, horizontal flips, rotation, color jitter. All applied during training; eval uses clean resized images.
- **Mixup & CutMix** — available in the training module for additional regularization (batch-level augmentation).

---

## Visualization & Interpretability 📊

The `src/visualization.py` module gives you publication-quality plots out of the box:

- **Training curves** — loss and accuracy over epochs with EMA smoothing, best epoch annotation
- **Confusion matrix** — heatmap with both raw counts and percentages
- **Per-class metrics** — grouped bar chart of precision, recall, and F1 per class
- **Grad-CAM** — gradient-weighted class activation maps that show which regions of the X-ray the model is paying attention to. Works with all four backbones, no extra dependencies needed.

Saved results (model weights, plots) go into the `results/` directory.

---

## Results

After training, you'll find in `results/`:

- `multimodal_{backbone}_best.pth` — best model weights (by validation loss)
- Training curves, confusion matrices, Grad-CAM visualizations (when using the visualization module)

---

## Troubleshooting

**"Invalid choice" error for backbone**: Make sure you're using one of: `basiccnn`, `minivgg`, `miniresnet`, `pretrained`. Partial names like `cnn` or `resnet` won't work.

**Out of memory**: Try reducing `--batch-size` (default 16) or switching to a smaller backbone like `basiccnn`.

**Slow training on CPU**: Expected — the pretrained backbone on 224×224 images is compute-heavy. Either use a GPU, reduce `--epochs`, or try `basiccnn`/`minivgg` which use 32×32 images and train much faster.

**Poor performance on Normal class**: With only 3 normal samples in the dataset, don't expect great per-class metrics there. The weighted sampler helps, but this is fundamentally a data scarcity issue.

---

## Dependencies

Core stack: PyTorch, torchvision, scikit-learn, pandas, matplotlib, seaborn, Pillow. Full list in `requirements.txt`.

---

## License

This project is for educational and research purposes.