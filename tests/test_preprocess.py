"""Unit tests for the preprocessing and feature extraction pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import compute_melspectrogram
from preprocess import normalize, resample, segment, TARGET_SAMPLES


# --- preprocess tests ---

def test_resample_changes_length():
    audio = np.random.randn(44100).astype(np.float32)
    resampled = resample(audio, orig_sr=44100, target_sr=22050)
    assert len(resampled) == pytest.approx(22050, abs=10)


def test_resample_noop_when_same_sr():
    audio = np.random.randn(22050).astype(np.float32)
    result = resample(audio, orig_sr=22050, target_sr=22050)
    np.testing.assert_array_equal(audio, result)


def test_normalize_peak_is_one():
    audio = np.array([0.0, 0.5, -2.0, 1.0], dtype=np.float32)
    result = normalize(audio)
    assert np.max(np.abs(result)) == pytest.approx(1.0)


def test_normalize_silent_audio():
    audio = np.zeros(100, dtype=np.float32)
    result = normalize(audio)
    np.testing.assert_array_equal(result, audio)


def test_segment_pads_short_audio():
    audio = np.ones(1000, dtype=np.float32)
    result = segment(audio)
    assert len(result) == TARGET_SAMPLES
    assert result[999] == 1.0
    assert result[1000] == 0.0


def test_segment_trims_long_audio():
    audio = np.ones(TARGET_SAMPLES + 5000, dtype=np.float32)
    result = segment(audio)
    assert len(result) == TARGET_SAMPLES


# --- feature tests ---

def test_melspectrogram_shape():
    audio = np.random.randn(TARGET_SAMPLES).astype(np.float32)
    spec = compute_melspectrogram(audio)
    assert spec.shape == (128, 173), f"unexpected shape {spec.shape}"


def test_melspectrogram_no_nan():
    audio = np.random.randn(TARGET_SAMPLES).astype(np.float32)
    spec = compute_melspectrogram(audio)
    assert not np.any(np.isnan(spec)), "spectrogram contains NaN values"


def test_melspectrogram_no_inf():
    audio = np.random.randn(TARGET_SAMPLES).astype(np.float32)
    spec = compute_melspectrogram(audio)
    assert not np.any(np.isinf(spec)), "spectrogram contains Inf values"


# --- dataset split tests ---

def test_split_ratios(tmp_path):
    """Verify that AcousticDataset produces splits close to the requested ratios."""
    from dataset import AcousticDataset

    # Build a synthetic labels.csv with 200 samples (40 per class)
    records = []
    for class_id in range(5):
        for i in range(40):
            records.append({"path": f"dummy_{class_id}_{i}.npy", "class_label": f"class_{class_id}", "class_id": class_id})

    import pandas as pd
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    ds = AcousticDataset(csv_path, val_ratio=0.15, test_ratio=0.15)
    total = len(ds.train_df) + len(ds.val_df) + len(ds.test_df)
    assert total == 200
    assert abs(len(ds.test_df) / total - 0.15) < 0.02
    assert abs(len(ds.val_df) / total - 0.15) < 0.02
