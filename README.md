# HPT Author Classifier (Basic)

This project trains a simple CNN to recognize authors from the HPT handwritten
Polish text dataset by cropping word images from the provided word places files.

## Setup

```bash
pip install -r requirements.txt
```

## Quick start (dry run)

```bash
python train.py --dry-run
```

## Train

```bash
python train.py --epochs 5 --max-samples-per-author 400
```

## Speed + progress tips

```bash
python train.py --max-samples-per-author 200 --image-size 48 48 --progress-every 100
```

- Use `--progress-every N` to print loading progress; set `0` to disable.
- Smaller `--image-size` and fewer samples speed up loading and training.

## Notes

- The loader splits data by page (image file) to avoid mixing words from the same
  page between train and test.
- Use `--image-size HEIGHT WIDTH` to change the input resolution.
- Increase `--max-samples-per-author` for more data if training is stable.
