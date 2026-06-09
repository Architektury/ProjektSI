# ANN Author Classifier — HPT Dataset

This project trains an **Artificial Neural Network (ANN) classifier** using
**backpropagation** to recognise the author of Polish handwritten text from the
HPT (Handwritten Polish Text) dataset.

## Pipeline

```
Word crops (PIL)
      │
      ├─ [augment, training only]  rotation ±10°, brightness, contrast, noise
      │
      ▼
HOG feature extraction  (orientations=9, 8×8 cells, 2×2 blocks)
      │
      ▼
MLP  512 → 256 → 128 → 64 → softmax(8)
  LeakyReLU · BatchNorm · Dropout · class-weighted cross-entropy
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick start (dry run — no training)

```bash
python train.py --dry-run
```

## Train (default: MLP + HOG + augmentation + class weights)

```bash
python train.py --epochs 30 --max-samples-per-author 400
```

## Common options

| Flag | Default | Effect |
|---|---|---|
| `--model mlp\|cnn` | `mlp` | Plain ANN (topic) or CNN (alternative) |
| `--use-hog` / `--no-hog` | on | HOG features vs raw pixels |
| `--augment` / `--no-augment` | on | Data augmentation on training set |
| `--augment-factor N` | `3` | Extra augmented copies per sample (×4 total) |
| `--epochs N` | `30` | Max epochs (early stopping applies) |
| `--patience N` | `7` | Early stopping patience |
| `--num-authors N` | `8` | Authors to include (1–8) |
| `--image-size H W` | `64 64` | Word crop resolution |
| `--max-samples-per-author N` | `400` | Cap samples per author |
| `--binary-authors A B` | — | Restrict to two-author binary task |

## Notes

- Data is split **by page** to prevent words from the same scan appearing in
  both train and test sets.
- **HOG** (Histogram of Oriented Gradients) captures pen stroke directions —
  much more discriminative than raw pixels for handwriting.
- **Data augmentation** multiplies training samples and makes the network
  robust to small variations in angle, brightness, and noise.
- **Class weights** force the model to pay more attention to hard/rare authors
  instead of defaulting to the most common one.
- Backpropagation is handled by TensorFlow/Keras (Adam optimiser).
