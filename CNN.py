import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers


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


def crop_from_image(image: Image.Image, bbox, image_size):
    row1, col1, row2, col2 = bbox
    left = max(0, col1)
    top = max(0, row1)
    right = min(col2, image.width)
    bottom = min(row2, image.height)
    if right <= left or bottom <= top:
        return None
    target_height, target_width = image_size
    crop = image.crop((left, top, right, bottom)).resize(
        (target_width, target_height), Image.BILINEAR
    )
    array = np.asarray(crop, dtype=np.float32) / 255.0
    return array


def build_dataset(
    dataset_dir: Path,
    num_authors: int,
    image_size,
    max_samples_per_author: int | None,
    test_split: float,
    seed: int,
    progress_every: int,
):
    rng = random.Random(seed)
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for author_index in range(num_authors):
        author_dir = dataset_dir / f"author{author_index + 1}"
        word_places_path = author_dir / "word_places.txt"
        if not word_places_path.exists():
            continue

        entries = parse_word_places(word_places_path)
        if not entries:
            continue

        rng.shuffle(entries)
        if max_samples_per_author:
            entries = entries[:max_samples_per_author]

        pages = list({entry[0] for entry in entries})
        rng.shuffle(pages)
        split_index = max(1, int(len(pages) * (1.0 - test_split)))
        train_pages = set(pages[:split_index])

        entries_by_page = {}
        for image_rel, bbox in entries:
            entries_by_page.setdefault(image_rel, []).append(bbox)

        total_entries = sum(len(bboxes) for bboxes in entries_by_page.values())
        if progress_every > 0:
            print(
                f"Author {author_index + 1}: loading {total_entries} samples "
                f"from {len(entries_by_page)} pages"
            )
        processed = 0
        author_train = 0
        author_test = 0

        for image_rel, bboxes in entries_by_page.items():
            image_path = author_dir / image_rel
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                image = image.convert("L")
                for bbox in bboxes:
                    sample = crop_from_image(image, bbox, image_size)
                    if sample is None:
                        continue
                    if image_rel in train_pages:
                        train_images.append(sample)
                        train_labels.append(author_index)
                        author_train += 1
                    else:
                        test_images.append(sample)
                        test_labels.append(author_index)
                        author_test += 1
                    processed += 1
                    if progress_every > 0 and processed % progress_every == 0:
                        print(
                            f"Author {author_index + 1}: processed "
                            f"{processed}/{total_entries}"
                        )

        if progress_every > 0:
            print(
                f"Author {author_index + 1}: train={author_train}, test={author_test}"
            )

    if not train_images or not test_images:
        raise RuntimeError("Not enough samples to build train/test splits.")

    X_train = np.expand_dims(np.array(train_images), axis=-1)
    y_train = np.array(train_labels, dtype=np.int64)
    X_test = np.expand_dims(np.array(test_images), axis=-1)
    y_test = np.array(test_labels, dtype=np.int64)

    train_order = np.arange(len(X_train))
    rng.shuffle(train_order.tolist())
    X_train = X_train[train_order]
    y_train = y_train[train_order]

    test_order = np.arange(len(X_test))
    rng.shuffle(test_order.tolist())
    X_test = X_test[test_order]
    y_test = y_test[test_order]

    return X_train, y_train, X_test, y_test


def build_model(input_shape, num_classes: int):
    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a CNN author classifier on the HPT dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("HPT_handwritten_polish_text_dataset"),
    )
    parser.add_argument("--num-authors", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--max-samples-per-author", type=int, default=400)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print load progress every N samples (0 disables).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    base_dir = Path(__file__).resolve().parent
    dataset_dir = args.dataset_dir
    if not dataset_dir.is_absolute():
        dataset_dir = (base_dir / dataset_dir).resolve()

    image_size = (args.image_size[0], args.image_size[1])
    X_train, y_train, X_test, y_test = build_dataset(
        dataset_dir=dataset_dir,
        num_authors=args.num_authors,
        image_size=image_size,
        max_samples_per_author=args.max_samples_per_author,
        test_split=args.test_split,
        seed=args.seed,
        progress_every=max(0, args.progress_every),
    )

    print(
        "Loaded dataset:",
        f"train={X_train.shape}, test={X_test.shape}, authors={args.num_authors}",
    )

    if args.dry_run:
        return 0

    model = build_model(
        input_shape=(image_size[0], image_size[1], 1),
        num_classes=args.num_authors,
    )
    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())