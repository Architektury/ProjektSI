import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras import layers

print("Dostępne GPU:", len(tf.config.list_physical_devices('GPU')))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def parse_word_places(file_path: Path):
    entries = []
    with file_path.open("r", encoding="latin-1", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            image_token = parts[0].strip('"')
            coords = parts[-4:]
            try:
                row1, col1, row2, col2 = (int(value) for value in coords)
            except ValueError:
                continue
            image_rel = Path(image_token.replace("\\", os.sep).replace("/", os.sep))
            entries.append((image_rel, (row1, col1, row2, col2)))
    return entries


def crop_pil(image: Image.Image, bbox, image_size):
    """Crop a word region and resize to image_size; returns PIL Image or None."""
    row1, col1, row2, col2 = bbox
    left  = max(0, col1)
    top   = max(0, row1)
    right = min(col2, image.width)
    bottom = min(row2, image.height)
    if right <= left or bottom <= top:
        return None
    target_height, target_width = image_size
    return image.crop((left, top, right, bottom)).resize(
        (target_width, target_height), Image.BILINEAR
    )


def pil_to_array(crop: Image.Image) -> np.ndarray:
    """Convert a PIL image to a normalised float32 numpy array (H, W, C)."""
    array = np.asarray(crop, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., np.newaxis]
    return array


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply random photometric and geometric augmentations to a word crop."""
    fill = 255 if image.mode == "L" else (255, 255, 255)

    # Random rotation ±10°
    angle = rng.uniform(-10, 10)
    image = image.rotate(angle, fillcolor=fill)

    # Random brightness ±25%
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))

    # Random contrast ±15%
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))

    # Mild Gaussian noise
    arr = np.asarray(image, dtype=np.float32)
    noise = np.random.normal(0, 5, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=image.mode)


# ---------------------------------------------------------------------------
# HOG feature extraction
# ---------------------------------------------------------------------------

def extract_hog(array: np.ndarray) -> np.ndarray:
    """Extract a HOG descriptor from a (H, W, C) float32 image array.

    HOG captures oriented edge histograms — far more discriminative for
    handwriting style than raw pixel intensities.
    """
    from skimage.feature import hog

    # Always work on grayscale
    img = array[:, :, 0] if array.shape[-1] == 1 else np.mean(array, axis=-1)
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    return features.astype(np.float32)


def prepare_features(X: np.ndarray, use_hog: bool, desc: str = "") -> np.ndarray:
    """Turn image arrays into flat feature vectors (HOG or raw pixels)."""
    if not use_hog:
        return X.reshape(len(X), -1)
    tag = f" ({desc})" if desc else ""
    print(f"Extracting HOG features{tag} …")
    feats = np.array([extract_hog(x) for x in X], dtype=np.float32)
    if desc == "train":
        print(f"  HOG feature dimension: {feats.shape[1]}")
    return feats


# ---------------------------------------------------------------------------
# Class-weight computation
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray) -> dict:
    """Inverse-frequency class weights to counter per-author sample imbalance."""
    counts = np.bincount(y)
    total = len(y)
    n_classes = len(counts)
    return {i: total / (n_classes * max(c, 1)) for i, c in enumerate(counts)}


# ---------------------------------------------------------------------------
# Dataset loading  (with optional augmentation)
# ---------------------------------------------------------------------------

def build_dataset(
    dataset_dir: Path,
    num_authors: int,
    image_size,
    max_samples_per_author: int | None,
    test_split: float,
    val_split: float,
    seed: int,
    progress_every: int,
    input_mode: str,
    augment: bool,
    augment_factor: int,
):
    rng = random.Random(seed)
    train_images, train_labels = [], []
    val_images,   val_labels   = [], []
    test_images,  test_labels  = [], []
    convert_mode = "L" if input_mode == "gray" else "RGB"

    for author_index in range(num_authors):
        author_dir      = dataset_dir / f"author{author_index + 1}"
        word_places_path = author_dir / "word_places.txt"
        if not word_places_path.exists():
            continue

        entries = parse_word_places(word_places_path)
        if not entries:
            continue

        rng.shuffle(entries)
        if max_samples_per_author:
            entries = entries[:max_samples_per_author]

        # Split pages (not words) into train / val / test to avoid leakage
        pages = list({entry[0] for entry in entries})
        rng.shuffle(pages)
        test_count = max(1, int(len(pages) * test_split))
        if len(pages) > 1:
            test_count = min(test_count, len(pages) - 1)
        test_pages = set(pages[:test_count])

        remaining_pages = pages[test_count:]
        val_count = 0
        if len(remaining_pages) > 1:
            val_count = max(1, int(len(remaining_pages) * val_split))
            val_count = min(val_count, len(remaining_pages) - 1)
        val_pages   = set(remaining_pages[:val_count])
        train_pages = set(remaining_pages[val_count:])

        if not train_pages and val_pages:
            moved = next(iter(val_pages))
            val_pages.remove(moved)
            train_pages.add(moved)
        if not train_pages and not val_pages and remaining_pages:
            train_pages.add(remaining_pages[0])

        entries_by_page: dict = {}
        for image_rel, bbox in entries:
            entries_by_page.setdefault(image_rel, []).append(bbox)

        total_entries = sum(len(b) for b in entries_by_page.values())
        if progress_every > 0:
            print(
                f"Author {author_index + 1}: loading {total_entries} samples "
                f"from {len(entries_by_page)} pages"
            )

        processed    = 0
        author_train = 0
        author_val   = 0
        author_test  = 0

        for image_rel, bboxes in entries_by_page.items():
            image_path = author_dir / image_rel
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                image = image.convert(convert_mode)
                for bbox in bboxes:
                    crop = crop_pil(image, bbox, image_size)
                    if crop is None:
                        continue

                    is_train = image_rel in train_pages
                    is_val   = image_rel in val_pages

                    sample = pil_to_array(crop)

                    if is_train:
                        train_images.append(sample)
                        train_labels.append(author_index)
                        author_train += 1
                        # Data augmentation — training only
                        if augment:
                            for _ in range(augment_factor):
                                aug = augment_image(crop, rng)
                                train_images.append(pil_to_array(aug))
                                train_labels.append(author_index)
                                author_train += 1
                    elif is_val:
                        val_images.append(sample)
                        val_labels.append(author_index)
                        author_val += 1
                    else:
                        test_images.append(sample)
                        test_labels.append(author_index)
                        author_test += 1

                    processed += 1
                    if progress_every > 0 and processed % progress_every == 0:
                        print(
                            f"  Author {author_index + 1}: "
                            f"{processed}/{total_entries}"
                        )

        if progress_every > 0:
            print(
                f"Author {author_index + 1}: "
                f"train={author_train}, val={author_val}, test={author_test}"
            )

    if not train_images or not test_images:
        raise RuntimeError("Not enough samples to build train/val/test splits.")

    X_train = np.array(train_images, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int64)
    X_val   = np.array(val_images,   dtype=np.float32)
    y_val   = np.array(val_labels,   dtype=np.int64)
    X_test  = np.array(test_images,  dtype=np.float32)
    y_test  = np.array(test_labels,  dtype=np.int64)

    # Shuffle each split — use ONE permutation applied to both X and y
    perm_train = np.random.default_rng(seed).permutation(len(X_train))
    X_train, y_train = X_train[perm_train], y_train[perm_train]

    if len(X_val) > 0:
        perm_val = np.random.default_rng(seed + 1).permutation(len(X_val))
        X_val, y_val = X_val[perm_val], y_val[perm_val]

    perm_test = np.random.default_rng(seed + 2).permutation(len(X_test))
    X_test, y_test = X_test[perm_test], y_test[perm_test]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_mlp_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    """Plain ANN (MLP) classifier trained by backpropagation.

    Input: 1-D feature vector (HOG descriptors or flattened pixels).
    Uses LeakyReLU to avoid dying-ReLU neurons, BatchNorm for stable
    training, and Dropout for regularisation.
    """
    reg = tf.keras.regularizers.l2(1e-4)

    x_input = layers.Input(shape=(input_dim,), name="features")

    x = layers.Dense(512, kernel_regularizer=reg)(x_input)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(128, kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(0.1)(x)

    output = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=x_input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model




# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def report_test_results(y_true, y_pred, num_authors: int):
    confusion = np.zeros((num_authors, num_authors), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[true_label, pred_label] += 1

    print("\nPer-author accuracy (test):")
    for author_index in range(num_authors):
        total   = confusion[author_index].sum()
        correct = confusion[author_index, author_index]
        acc = 0.0 if total == 0 else correct / total
        print(f"  author{author_index + 1}: {acc * 100:.2f}% ({correct}/{total})")

    overall = np.trace(confusion) / max(confusion.sum(), 1)
    print(f"\nOverall test accuracy: {overall * 100:.2f}%")

    header = "     " + " ".join([f"a{i+1:02d}" for i in range(num_authors)])
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(header)
    for idx in range(num_authors):
        row = " ".join([f"{v:4d}" for v in confusion[idx]])
        print(f"a{idx+1:02d} {row}")


def filter_authors(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    authors,
):
    keep = sorted({author - 1 for author in authors})
    if len(keep) != 2:
        raise ValueError("--binary-authors expects exactly two distinct authors.")

    train_mask = np.isin(y_train, keep)
    val_mask   = np.isin(y_val, keep) if len(y_val) else np.array([], dtype=bool)
    test_mask  = np.isin(y_test, keep)
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError("Not enough samples for the selected authors.")

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    if len(y_val):
        X_val, y_val = X_val[val_mask], y_val[val_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    remap  = {old: new for new, old in enumerate(keep)}
    y_train = np.array([remap[l] for l in y_train], dtype=np.int64)
    if len(y_val):
        y_val = np.array([remap[l] for l in y_val], dtype=np.int64)
    y_test = np.array([remap[l] for l in y_test], dtype=np.int64)
    return X_train, y_train, X_val, y_val, X_test, y_test, len(keep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train an ANN author classifier on the HPT dataset."
    )
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("HPT_handwritten_polish_text_dataset"))
    parser.add_argument("--num-authors",            type=int,   default=8)
    parser.add_argument("--image-size",             type=int,   nargs=2, default=[64, 64])
    parser.add_argument("--max-samples-per-author", type=int,   default=400)
    parser.add_argument("--test-split",             type=float, default=0.3)
    parser.add_argument("--val-split",              type=float, default=0.2)
    parser.add_argument("--epochs",                 type=int,   default=30)
    parser.add_argument("--batch-size",             type=int,   default=32)
    parser.add_argument("--seed",                   type=int,   default=123)
    parser.add_argument("--input-mode", choices=["rgb_only", "gray"], default="gray")

    # HOG
    parser.add_argument(
        "--use-hog", action="store_true", default=True,
        help="Extract HOG features before MLP (default: on).",
    )
    parser.add_argument(
        "--no-hog", action="store_false", dest="use_hog",
        help="Disable HOG; feed raw flattened pixels to the MLP.",
    )

    # Augmentation
    parser.add_argument(
        "--augment", action="store_true", default=True,
        help="Apply data augmentation to the training set (default: on).",
    )
    parser.add_argument(
        "--no-augment", action="store_false", dest="augment",
        help="Disable data augmentation.",
    )
    parser.add_argument(
        "--augment-factor", type=int, default=3,
        help="Number of augmented copies generated per original training sample "
             "(default 3 → training set size ×4).",
    )

    parser.add_argument("--shuffle-labels",    action="store_true")
    parser.add_argument("--binary-authors",    type=int, nargs=2, metavar=("A", "B"))
    parser.add_argument("--progress-every",    type=int, default=200,
                        help="Print load progress every N samples (0 = off).")
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--patience",          type=int, default=7)
    parser.add_argument("--dry-run",           action="store_true")
    args = parser.parse_args(argv)

    base_dir    = Path(__file__).resolve().parent
    dataset_dir = args.dataset_dir
    if not dataset_dir.is_absolute():
        dataset_dir = (base_dir / dataset_dir).resolve()

    image_size = (args.image_size[0], args.image_size[1])

    do_augment = args.augment

    X_train, y_train, X_val, y_val, X_test, y_test = build_dataset(
        dataset_dir=dataset_dir,
        num_authors=args.num_authors,
        image_size=image_size,
        max_samples_per_author=args.max_samples_per_author,
        test_split=args.test_split,
        val_split=args.val_split,
        seed=args.seed,
        progress_every=max(0, args.progress_every),
        input_mode=args.input_mode,
        augment=do_augment,
        augment_factor=args.augment_factor,
    )

    if args.binary_authors:
        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            args.num_authors,
        ) = filter_authors(
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            args.binary_authors,
        )
        print(f"Binary test: authors {args.binary_authors[0]} vs {args.binary_authors[1]}")

    if args.shuffle_labels:
        np.random.default_rng(args.seed).shuffle(y_train)
        print("Shuffled training labels (diagnostic run).")

    aug_info = (f"augment=ON ×{args.augment_factor}" if do_augment else "augment=OFF")
    print(
        f"Dataset loaded — train={X_train.shape}, val={X_val.shape}, "
        f"test={X_test.shape}, authors={args.num_authors}"
    )
    print(f"Model=MLP  HOG={args.use_hog}  {aug_info}")

    if args.dry_run:
        return 0

    # Compute class weights to penalise poor-performing authors more
    class_weights = compute_class_weights(y_train)
    print("Class weights:", {f"a{k+1}": round(v, 3) for k, v in class_weights.items()})

    input_shape = (
        (image_size[0], image_size[1], 1)
        if args.input_mode == "gray"
        else (image_size[0], image_size[1], 3)
    )

    # Prepare HOG features and build MLP model
    F_train = prepare_features(X_train, args.use_hog, "train")
    F_val   = prepare_features(X_val,   args.use_hog, "val")  if len(X_val) else X_val
    F_test  = prepare_features(X_test,  args.use_hog, "test")
    model   = build_mlp_model(F_train.shape[1], args.num_authors)
    train_X, val_X, test_X = F_train, F_val, F_test

    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        )
    ]
    if not args.no_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience,
                restore_best_weights=True, verbose=1,
            )
        )

    # Train
    history = model.fit(
        train_X, y_train,
        validation_data=(val_X, y_val) if len(val_X) > 0 else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    if history.history.get("val_loss"):
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        print(f"Best epoch (val_loss): {best_epoch} / {len(history.history['val_loss'])}")

    # Evaluate
    loss, accuracy = model.evaluate(test_X, y_test, verbose=0)
    print(f"\nTest loss:     {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    y_pred = np.argmax(model.predict(test_X, verbose=0), axis=1)
    report_test_results(y_test, y_pred, args.num_authors)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())