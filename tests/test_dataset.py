"""Unit tests for src/dataset.py — all tests use synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.config import CLASS_LABELS, NUM_CLASSES
from src.dataset import load_labels, split_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_PER_CLASS = 40  # enough for stratified splits to work cleanly


def _make_labels_df(n_per_class: int = N_PER_CLASS, include_augmented: bool = False) -> pd.DataFrame:
    """Build a synthetic labels DataFrame matching the features.save_spectrograms schema."""
    records = []
    for class_id, class_label in CLASS_LABELS.items():
        for i in range(n_per_class):
            records.append(
                {
                    "path": f"dummy/{class_label}/clip_{i:03d}.npy",
                    "class_label": class_label,
                    "class_id": class_id,
                    "is_augmented": False,
                }
            )
            if include_augmented:
                # 3 augmented copies per original (matches features.py output)
                for aug_tag in ("aug0", "aug1", "specaug0"):
                    records.append(
                        {
                            "path": f"dummy/{class_label}/clip_{i:03d}_{aug_tag}.npy",
                            "class_label": class_label,
                            "class_id": class_id,
                            "is_augmented": True,
                        }
                    )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# load_labels
# ---------------------------------------------------------------------------


def test_load_labels_reads_csv(tmp_path):
    df = _make_labels_df()
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_labels(tmp_path)
    assert list(loaded.columns) == ["path", "class_label", "class_id", "is_augmented"]
    assert len(loaded) == len(df)


def test_load_labels_is_augmented_is_bool(tmp_path):
    df = _make_labels_df(include_augmented=True)
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_labels(tmp_path)
    assert loaded["is_augmented"].dtype == bool


# ---------------------------------------------------------------------------
# split_dataset — split ratios
# ---------------------------------------------------------------------------


def test_split_ratios_non_augmented_only():
    """Splits should sum to total non-augmented count."""
    df = _make_labels_df(include_augmented=False)
    train, val, test = split_dataset(df, train_ratio=0.70, val_ratio=0.15)

    non_aug_total = len(df[~df["is_augmented"]])
    assert len(train) + len(val) + len(test) == non_aug_total


def test_split_ratios_with_augmented():
    """Non-aug totals stay correct; augmented rows all land in train."""
    df = _make_labels_df(include_augmented=True)
    train, val, test = split_dataset(df, train_ratio=0.70, val_ratio=0.15)

    non_aug_total = len(df[~df["is_augmented"]])
    aug_total = len(df[df["is_augmented"]])

    assert len(val) + len(test) == pytest.approx(
        int(non_aug_total * 0.30), abs=5
    ), "val+test should be ~30% of non-augmented samples"
    assert len(train) == len(df) - len(val) - len(test)
    assert train["is_augmented"].sum() == aug_total


def test_val_ratio_within_tolerance():
    df = _make_labels_df(include_augmented=False)
    _, val, _ = split_dataset(df, train_ratio=0.70, val_ratio=0.15)

    non_aug_total = len(df)
    actual_ratio = len(val) / non_aug_total
    assert abs(actual_ratio - 0.15) < 0.02, (
        f"val ratio {actual_ratio:.3f} is outside the 2% tolerance of 0.15"
    )


def test_test_ratio_within_tolerance():
    df = _make_labels_df(include_augmented=False)
    _, _, test = split_dataset(df, train_ratio=0.70, val_ratio=0.15)

    non_aug_total = len(df)
    actual_ratio = len(test) / non_aug_total
    assert abs(actual_ratio - 0.15) < 0.02, (
        f"test ratio {actual_ratio:.3f} is outside the 2% tolerance of 0.15"
    )


# ---------------------------------------------------------------------------
# split_dataset — no-leakage (most critical correctness constraint)
# ---------------------------------------------------------------------------


def test_no_augmented_rows_in_val():
    df = _make_labels_df(include_augmented=True)
    _, val, _ = split_dataset(df)
    assert val["is_augmented"].sum() == 0, (
        "Val split must never contain augmented samples"
    )


def test_no_augmented_rows_in_test():
    df = _make_labels_df(include_augmented=True)
    _, _, test = split_dataset(df)
    assert test["is_augmented"].sum() == 0, (
        "Test split must never contain augmented samples"
    )


def test_augmented_rows_only_in_train():
    df = _make_labels_df(include_augmented=True)
    train, val, test = split_dataset(df)

    aug_total = len(df[df["is_augmented"]])
    assert train["is_augmented"].sum() == aug_total, (
        "All augmented rows must appear in the train split"
    )


# ---------------------------------------------------------------------------
# split_dataset — stratification
# ---------------------------------------------------------------------------


def test_split_is_stratified_in_val():
    df = _make_labels_df(include_augmented=False)
    _, val, _ = split_dataset(df)

    counts = val["class_id"].value_counts()
    assert counts.max() - counts.min() <= 2, (
        "Val split should be approximately balanced across classes"
    )


def test_split_is_stratified_in_test():
    df = _make_labels_df(include_augmented=False)
    _, _, test = split_dataset(df)

    counts = test["class_id"].value_counts()
    assert counts.max() - counts.min() <= 2


def test_split_reproducible_with_same_seed():
    df = _make_labels_df(include_augmented=False)
    train_a, val_a, test_a = split_dataset(df, seed=0)
    train_b, val_b, test_b = split_dataset(df, seed=0)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(val_a, val_b)
    pd.testing.assert_frame_equal(test_a, test_b)


# ---------------------------------------------------------------------------
# make_tf_dataset — shapes (skipped when TensorFlow is not installed)
# ---------------------------------------------------------------------------


def test_make_tf_dataset_output_shapes(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from src.dataset import make_tf_dataset

    # Write tiny synthetic .npy files so the dataset can actually load them.
    df_records = []
    for class_id in range(NUM_CLASSES):
        class_label = CLASS_LABELS[class_id]
        class_dir = tmp_path / class_label
        class_dir.mkdir(parents=True)
        for i in range(4):
            spec = np.random.randn(128, 173).astype(np.float32)
            npy_path = class_dir / f"clip_{i}.npy"
            np.save(npy_path, spec)
            df_records.append(
                {
                    "path": str(npy_path),
                    "class_label": class_label,
                    "class_id": class_id,
                    "is_augmented": False,
                }
            )

    df = pd.DataFrame(df_records)
    ds = make_tf_dataset(df, batch_size=4, shuffle=False)

    for specs, labels in ds.take(1):
        assert specs.shape == (4, 128, 173, 1), f"unexpected spec shape {specs.shape}"
        assert labels.shape == (4, NUM_CLASSES), f"unexpected label shape {labels.shape}"


def test_make_tf_dataset_normalized_range(tmp_path):
    pytest.importorskip("tensorflow")
    from src.dataset import make_tf_dataset

    df_records = []
    class_dir = tmp_path / "normal_operation"
    class_dir.mkdir()
    for i in range(2):
        spec = np.random.randn(128, 173).astype(np.float32)
        npy_path = class_dir / f"clip_{i}.npy"
        np.save(npy_path, spec)
        df_records.append(
            {"path": str(npy_path), "class_label": "normal_operation",
             "class_id": 0, "is_augmented": False}
        )

    df = pd.DataFrame(df_records)
    ds = make_tf_dataset(df, batch_size=2)

    for specs, _ in ds.take(1):
        arr = specs.numpy()
        assert arr.min() >= -1.0 - 1e-5, "spectrogram values should be ≥ -1"
        assert arr.max() <= 1.0 + 1e-5, "spectrogram values should be ≤ 1"
