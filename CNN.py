import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers

print("Dostępne GPU:", len(tf.config.list_physical_devices('GPU')))

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
    if array.ndim == 2:
        array = array[..., np.newaxis]
    return array


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
):
    rng = random.Random(seed)
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    test_images, test_labels = [], []
    convert_mode = "L" if input_mode == "gray" else "RGB"

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
        test_count = max(1, int(len(pages) * test_split))
        if len(pages) > 1:
            test_count = min(test_count, len(pages) - 1)
        test_pages = set(pages[:test_count])

        remaining_pages = pages[test_count:]
        val_count = 0
        if len(remaining_pages) > 1:
            val_count = max(1, int(len(remaining_pages) * val_split))
            val_count = min(val_count, len(remaining_pages) - 1)
        val_pages = set(remaining_pages[:val_count])
        train_pages = set(remaining_pages[val_count:])

        if not train_pages and val_pages:
            moved_page = next(iter(val_pages))
            val_pages.remove(moved_page)
            train_pages.add(moved_page)
        if not train_pages and not val_pages and remaining_pages:
            train_pages.add(remaining_pages[0])

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
        author_val = 0
        author_test = 0

        for image_rel, bboxes in entries_by_page.items():
            image_path = author_dir / image_rel
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                image = image.convert(convert_mode)
                for bbox in bboxes:
                    sample = crop_from_image(image, bbox, image_size)
                    if sample is None:
                        continue
                    if image_rel in train_pages:
                        train_images.append(sample)
                        train_labels.append(author_index)
                        author_train += 1
                    elif image_rel in val_pages:
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
                            f"Author {author_index + 1}: processed "
                            f"{processed}/{total_entries}"
                        )

        if progress_every > 0:
            print(
                f"Author {author_index + 1}: train={author_train}, val={author_val}, test={author_test}"
            )

    if not train_images or not test_images:
        raise RuntimeError("Not enough samples to build train/val/test splits.")

    X_train = np.array(train_images, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int64)
    X_val = np.array(val_images, dtype=np.float32)
    y_val = np.array(val_labels, dtype=np.int64)
    X_test = np.array(test_images, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.int64)

    train_order = np.arange(len(X_train))
    rng.shuffle(train_order.tolist())
    X_train = X_train[train_order]
    y_train = y_train[train_order]

    if len(X_val) > 0:
        val_order = np.arange(len(X_val))
        rng.shuffle(val_order.tolist())
        X_val = X_val[val_order]
        y_val = y_val[val_order]

    test_order = np.arange(len(X_test))
    rng.shuffle(test_order.tolist())
    X_test = X_test[test_order]
    y_test = y_test[test_order]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(input_shape, num_classes: int):
    image_input = layers.Input(shape=input_shape, name="image")
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=image_input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_encoder(input_shape, embedding_dim: int):
    image_input = layers.Input(shape=input_shape, name="image")
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        embedding_dim,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(embedding_dim, activation=None)(x)
    output = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

    return tf.keras.Model(inputs=image_input, outputs=output, name="encoder")


def contrastive_loss(margin: float):
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        square_distance = tf.square(y_pred)
        margin_distance = tf.square(tf.maximum(margin - y_pred, 0.0))
        return tf.reduce_mean(y_true * square_distance + (1.0 - y_true) * margin_distance)

    return _loss


def build_siamese_model(input_shape, embedding_dim: int, margin: float):
    encoder = build_encoder(input_shape, embedding_dim)

    left_input = layers.Input(shape=input_shape, name="left_image")
    right_input = layers.Input(shape=input_shape, name="right_image")

    left_embedding = encoder(left_input)
    right_embedding = encoder(right_input)

    distance = layers.Lambda(
        lambda tensors: tf.sqrt(
            tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)
            + 1e-9
        )
    )([left_embedding, right_embedding])

    model = tf.keras.Model(inputs=[left_input, right_input], outputs=distance, name="siamese")
    
    def accuracy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # 1.0 oznacza tę samą klasę, dystans mniejszy niż połowa marginesu to pozytywne dopasowanie
        pred_match = tf.cast(y_pred < (margin / 2.0), tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, pred_match), tf.float32))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=contrastive_loss(margin),
        metrics=[accuracy]
    )
    return model, encoder


class SiamesePairSequence(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batch_size: int, steps_per_epoch: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.rng = random.Random(seed)
        self.class_to_indices = {
            int(label): np.where(self.labels == label)[0].tolist()
            for label in np.unique(self.labels)
        }
        self.classes = [label for label, indices in self.class_to_indices.items() if indices]
        self.positive_classes = [label for label, indices in self.class_to_indices.items() if len(indices) >= 2]

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        half = self.batch_size // 2
        left_batch = []
        right_batch = []
        y_batch = []

        for _ in range(half):
            if not self.positive_classes:
                break
            label = self.rng.choice(self.positive_classes)
            i1, i2 = self.rng.sample(self.class_to_indices[label], 2)
            left_batch.append(self.images[i1])
            right_batch.append(self.images[i2])
            y_batch.append(1.0)

        remaining = self.batch_size - len(left_batch)
        for _ in range(remaining):
            if len(self.classes) < 2:
                break
            label1, label2 = self.rng.sample(self.classes, 2)
            i1 = self.rng.choice(self.class_to_indices[label1])
            i2 = self.rng.choice(self.class_to_indices[label2])
            left_batch.append(self.images[i1])
            right_batch.append(self.images[i2])
            y_batch.append(0.0)

        left_batch = np.asarray(left_batch, dtype=np.float32)
        right_batch = np.asarray(right_batch, dtype=np.float32)
        y_batch = np.asarray(y_batch, dtype=np.float32).reshape(-1, 1)
        return (left_batch, right_batch), y_batch


def make_pair_sequence(images, labels, batch_size: int, pairs_per_epoch: int, seed: int):
    steps = max(1, pairs_per_epoch // batch_size)
    return SiamesePairSequence(images, labels, batch_size=batch_size, steps_per_epoch=steps, seed=seed)


def predict_with_nearest_centroid(encoder, train_images, train_labels, test_images):
    train_embeddings = encoder.predict(train_images, verbose=0)
    test_embeddings = encoder.predict(test_images, verbose=0)

    classes = np.unique(train_labels)
    centroids = []
    for label in classes:
        centroids.append(train_embeddings[train_labels == label].mean(axis=0))
    centroids = np.asarray(centroids, dtype=np.float32)

    distances = np.linalg.norm(test_embeddings[:, None, :] - centroids[None, :, :], axis=2)
    predicted = classes[np.argmin(distances, axis=1)]
    return predicted


def report_test_results(y_true, y_pred, num_authors: int):
    confusion = np.zeros((num_authors, num_authors), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[true_label, pred_label] += 1

    print("Per-author accuracy (test):")
    for author_index in range(num_authors):
        total = confusion[author_index].sum()
        correct = confusion[author_index, author_index]
        acc = 0.0 if total == 0 else correct / total
        print(
            f"  author{author_index + 1}: {acc * 100:.2f}%"
            f" ({correct}/{total})"
        )

    header = "     " + " ".join([f"a{idx+1:02d}" for idx in range(num_authors)])
    print("Confusion matrix (rows=true, cols=pred):")
    print(header)
    for idx in range(num_authors):
        row = " ".join([f"{value:4d}" for value in confusion[idx]])
        print(f"a{idx+1:02d} {row}")


def filter_authors(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    authors,
):
    keep = sorted({author - 1 for author in authors})
    if len(keep) != 2:
        raise ValueError("--binary-authors expects exactly two distinct authors.")

    train_mask = np.isin(y_train, keep)
    val_mask = np.isin(y_val, keep) if len(y_val) else np.array([], dtype=bool)
    test_mask = np.isin(y_test, keep)
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError("Not enough samples for the selected authors.")

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    if len(y_val):
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    remap = {old: new for new, old in enumerate(keep)}
    y_train = np.array([remap[label] for label in y_train], dtype=np.int64)
    if len(y_val):
        y_val = np.array([remap[label] for label in y_val], dtype=np.int64)
    y_test = np.array([remap[label] for label in y_test], dtype=np.int64)
    return X_train, y_train, X_val, y_val, X_test, y_test, len(keep)


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
    parser.add_argument("--test-split", type=float, default=0.3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--training-mode",
        choices=["siamese", "cr"],
        default="siamese",
    )
    parser.add_argument(
        "--input-mode",
        choices=["rgb_only", "gray"],
        default="gray",
    )
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument(
        "--binary-authors",
        type=int,
        nargs=2,
        metavar=("A", "B"),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print load progress every N samples (0 disables).",
    )
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--pairs-per-epoch", type=int, default=4000)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    base_dir = Path(__file__).resolve().parent
    dataset_dir = args.dataset_dir
    if not dataset_dir.is_absolute():
        dataset_dir = (base_dir / dataset_dir).resolve()

    image_size = (args.image_size[0], args.image_size[1])
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
    )

    if args.binary_authors:
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            args.num_authors,
        ) = filter_authors(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            args.binary_authors,
        )
        print(
            f"Binary test: authors {args.binary_authors[0]} vs"
            f" {args.binary_authors[1]}"
        )

    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(y_train)
        print("Shuffled training labels (diagnostic run).")

    print(
        "Loaded dataset:",
        f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}, authors={args.num_authors}",
    )
    print(f"Training mode: {args.training_mode}")
    print(f"Input mode: {args.input_mode}")

    if args.dry_run:
        return 0

    input_shape = (
        (image_size[0], image_size[1], 1)
        if args.input_mode == "gray"
        else (image_size[0], image_size[1], 3)
    )
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        )
    ]
    if not args.no_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

    if args.training_mode == "classifier":
        model = build_model(
            input_shape=input_shape,
            num_classes=args.num_authors,
        )
        model.summary()

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
        )

        if history.history.get("val_loss"):
            best_epoch = int(np.argmin(history.history["val_loss"])) + 1
            print(
                f"Best epoch (val_loss): {best_epoch} / {len(history.history['val_loss'])}"
            )

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {accuracy * 100:.2f}%")
        return 0

    siamese_model, encoder = build_siamese_model(
        input_shape=input_shape,
        embedding_dim=args.embedding_dim,
        margin=args.margin,
    )
    siamese_model.summary()

    train_pairs = make_pair_sequence(
        X_train,
        y_train,
        batch_size=args.batch_size,
        pairs_per_epoch=args.pairs_per_epoch,
        seed=args.seed,
    )

    val_pairs = None
    if len(X_val) > 0:
        val_pairs = make_pair_sequence(
            X_val,
            y_val,
            batch_size=args.batch_size,
            pairs_per_epoch=max(args.batch_size, args.pairs_per_epoch // 4),
            seed=args.seed + 1,
        )

    history = siamese_model.fit(
        train_pairs,
        validation_data=val_pairs,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if history.history.get("val_loss"):
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        print(
            f"Best epoch (val_loss): {best_epoch} / {len(history.history['val_loss'])}"
        )

    y_pred = predict_with_nearest_centroid(encoder, X_train, y_train, X_test)
    print(f"Test accuracy: {np.mean(y_pred == y_test) * 100:.2f}%")
    report_test_results(y_test, y_pred, args.num_authors)
    
   
    return 0


if __name__ == "__main__":
    raise SystemExit(main())