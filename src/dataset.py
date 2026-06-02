"""Stratified dataset splitting and tf.data pipeline construction."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import INPUT_SHAPE, NUM_CLASSES


def load_labels(spec_dir: Path) -> pd.DataFrame:
    """
    Read the labels manifest produced by ``features.save_spectrograms``.

    Args:
        spec_dir: Directory containing ``labels.csv``.

    Returns:
        DataFrame with columns ``path``, ``class_label``, ``class_id``,
        ``is_augmented``.
    """
    csv_path = spec_dir / "labels.csv"
    df = pd.read_csv(csv_path)
    df["is_augmented"] = df["is_augmented"].astype(bool)
    return df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produce stratified train / val / test splits from a labels DataFrame.

    The split is computed on **non-augmented** samples only to prevent data
    leakage.  Augmented samples are then appended to the training split.

    Args:
        df: Labels DataFrame as returned by ``load_labels``.
        train_ratio: Fraction of non-augmented samples assigned to training.
        val_ratio: Fraction of non-augmented samples assigned to validation.
        seed: Random seed for reproducible splits.

    Returns:
        Three DataFrames: (train, val, test).  Only the train DataFrame may
        contain rows where ``is_augmented`` is True.
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    non_aug = df[~df["is_augmented"]].copy()
    aug = df[df["is_augmented"]].copy()

    train_val, test = train_test_split(
        non_aug,
        test_size=test_ratio,
        stratify=non_aug["class_id"],
        random_state=seed,
    )

    # Adjust val ratio relative to the remaining train_val pool.
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=train_val["class_id"],
        random_state=seed,
    )

    train = pd.concat([train, aug], ignore_index=True)

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _load_and_normalize(path: str) -> np.ndarray:
    """Load a .npy spectrogram, add channel dim, and normalize to [-1, 1]."""
    spec = np.load(path).astype(np.float32)
    spec = spec[..., np.newaxis]  # (n_mels, T, 1)
    s_min, s_max = spec.min(), spec.max()
    if s_max > s_min:
        spec = 2.0 * (spec - s_min) / (s_max - s_min) - 1.0
    return spec


def make_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = 32,
    shuffle: bool = False,
    augment: bool = False,
) -> "tf.data.Dataset":
    """
    Build a ``tf.data.Dataset`` that streams spectrograms from disk.

    Each element is a ``(spectrogram, one_hot_label)`` pair.  Spectrograms are
    normalized to ``[-1, 1]`` and shaped ``INPUT_SHAPE`` = ``(128, 173, 1)``.
    Labels are one-hot vectors of length ``NUM_CLASSES`` = 5.

    The ``augment`` parameter is reserved for future online augmentation; it
    has no effect in the current implementation because augmented .npy files
    are already written to disk by ``features.save_spectrograms``.

    Args:
        df: Split DataFrame (train, val, or test) from ``split_dataset``.
        batch_size: Number of samples per batch.
        shuffle: If True, shuffle the dataset before batching (use for train).
        augment: Reserved — currently unused.

    Returns:
        Batched and prefetched ``tf.data.Dataset``.
    """
    import tensorflow as tf  # lazy import — not available on Python 3.14

    paths = df["path"].values
    labels = tf.keras.utils.to_categorical(
        df["class_id"].values, num_classes=NUM_CLASSES
    ).astype(np.float32)

    def _load_fn(path: tf.Tensor, label: tf.Tensor):
        spec = tf.numpy_function(
            func=lambda p: _load_and_normalize(p.decode()),
            inp=[path],
            Tout=tf.float32,
        )
        spec.set_shape(INPUT_SHAPE)
        return spec, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)
    ds = ds.map(_load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset splits.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory.",
    )
    args = parser.parse_args()

    spec_dir = args.data_dir / "spectrograms"
    df = load_labels(spec_dir)
    train, val, test = split_dataset(df)
    total = len(train) + len(val) + len(test)
    print(f"total={total}  train={len(train)}  val={len(val)}  test={len(test)}")
    aug_in_val = val["is_augmented"].sum()
    aug_in_test = test["is_augmented"].sum()
    print(f"augmented in val={aug_in_val}  augmented in test={aug_in_test}")


if __name__ == "__main__":
    main()
