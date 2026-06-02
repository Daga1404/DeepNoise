"""Unit tests for src/features.py — all tests use synthetic data."""

import numpy as np
import pytest
import soundfile as sf

from src.config import DURATION, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE
from src.features import augment_audio, augment_spectrogram, compute_melspectrogram, save_spectrograms

TARGET = int(SAMPLE_RATE * DURATION)  # 88200


# ---------------------------------------------------------------------------
# compute_melspectrogram
# ---------------------------------------------------------------------------


def test_melspectrogram_output_shape():
    audio = np.random.randn(TARGET).astype(np.float32)
    spec = compute_melspectrogram(audio)
    assert spec.shape == (N_MELS, 173), f"unexpected shape {spec.shape}"


def test_melspectrogram_no_nan():
    audio = np.random.randn(TARGET).astype(np.float32)
    spec = compute_melspectrogram(audio)
    assert not np.any(np.isnan(spec))


def test_melspectrogram_no_inf():
    audio = np.random.randn(TARGET).astype(np.float32)
    spec = compute_melspectrogram(audio)
    assert not np.any(np.isinf(spec))


def test_melspectrogram_silent_audio_no_crash():
    audio = np.zeros(TARGET, dtype=np.float32)
    spec = compute_melspectrogram(audio)
    assert spec.shape == (N_MELS, 173)


def test_melspectrogram_log_scaling_applied():
    # Without the +1e-6 floor, log of zero would be -inf.
    # With it, all values should be finite even for silence.
    audio = np.zeros(TARGET, dtype=np.float32)
    spec = compute_melspectrogram(audio)
    assert np.all(np.isfinite(spec))


def test_melspectrogram_kwargs_override():
    audio = np.random.randn(TARGET).astype(np.float32)
    spec = compute_melspectrogram(audio, n_mels=64)
    assert spec.shape[0] == 64


# ---------------------------------------------------------------------------
# augment_audio
# ---------------------------------------------------------------------------


def test_augment_audio_returns_at_least_two_arrays():
    audio = np.random.randn(TARGET).astype(np.float32)
    results = augment_audio(audio)
    assert len(results) >= 2


def test_augment_audio_arrays_are_distinct():
    audio = np.random.randn(TARGET).astype(np.float32)
    results = augment_audio(audio)
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert not np.array_equal(results[i], results[j]), (
                f"augment_audio[{i}] and [{j}] are identical"
            )


def test_augment_audio_preserves_length():
    audio = np.random.randn(TARGET).astype(np.float32)
    for aug in augment_audio(audio):
        assert len(aug) == TARGET


def test_augment_audio_gaussian_noise_differs_from_original():
    audio = np.random.randn(TARGET).astype(np.float32)
    noisy = augment_audio(audio)[0]
    assert not np.allclose(noisy, audio)


# ---------------------------------------------------------------------------
# augment_spectrogram
# ---------------------------------------------------------------------------


def test_augment_spectrogram_returns_at_least_one_array():
    log_S = np.random.randn(N_MELS, 173).astype(np.float32)
    results = augment_spectrogram(log_S)
    assert len(results) >= 1


def test_augment_spectrogram_output_shape_unchanged():
    log_S = np.random.randn(N_MELS, 173).astype(np.float32)
    for aug in augment_spectrogram(log_S):
        assert aug.shape == log_S.shape


def test_augment_spectrogram_has_zeroed_bands():
    # With freq_mask=20 and time_mask=30 there must be at least one zero row/column.
    rng = np.random.default_rng(0)
    log_S = rng.random((N_MELS, 173)).astype(np.float32) + 1.0  # all positive
    aug = augment_spectrogram(log_S)[0]
    assert np.any(aug == 0.0), "SpecAugment should zero out at least one element"


def test_augment_spectrogram_does_not_modify_original():
    log_S = np.random.randn(N_MELS, 173).astype(np.float32)
    original = log_S.copy()
    augment_spectrogram(log_S)
    np.testing.assert_array_equal(log_S, original)


# ---------------------------------------------------------------------------
# save_spectrograms (integration: write synthetic WAVs, run save, check output)
# ---------------------------------------------------------------------------


def _write_synthetic_class(root: object, class_name: str, n_files: int = 3) -> None:
    class_dir = root / class_name
    class_dir.mkdir(parents=True)
    for i in range(n_files):
        audio = np.random.randn(TARGET).astype(np.float32)
        sf.write(class_dir / f"clip_{i:03d}.wav", audio, SAMPLE_RATE)


def test_save_spectrograms_creates_npy_files(tmp_path):
    processed = tmp_path / "processed"
    spec_dir = tmp_path / "spectrograms"
    _write_synthetic_class(processed, "normal_operation", n_files=2)

    save_spectrograms(processed, spec_dir, augment_train=False)

    npy_files = list((spec_dir / "normal_operation").glob("*.npy"))
    assert len(npy_files) == 2


def test_save_spectrograms_labels_csv_columns(tmp_path):
    import pandas as pd

    processed = tmp_path / "processed"
    spec_dir = tmp_path / "spectrograms"
    _write_synthetic_class(processed, "alarm_tone", n_files=2)

    save_spectrograms(processed, spec_dir, augment_train=False)

    df = pd.read_csv(spec_dir / "labels.csv")
    assert set(df.columns) == {"path", "class_label", "class_id", "is_augmented"}


def test_save_spectrograms_augmented_copies_tagged(tmp_path):
    import pandas as pd

    processed = tmp_path / "processed"
    spec_dir = tmp_path / "spectrograms"
    _write_synthetic_class(processed, "metallic_impact", n_files=2)

    save_spectrograms(processed, spec_dir, augment_train=True)

    df = pd.read_csv(spec_dir / "labels.csv")
    assert df["is_augmented"].any(), "augmented rows should be present when augment_train=True"
    assert (~df["is_augmented"]).any(), "original (non-augmented) rows should also be present"


def test_save_spectrograms_augmented_count(tmp_path):
    import pandas as pd

    processed = tmp_path / "processed"
    spec_dir = tmp_path / "spectrograms"
    n_files = 3
    _write_synthetic_class(processed, "friction_squeal", n_files=n_files)

    save_spectrograms(processed, spec_dir, augment_train=True)

    df = pd.read_csv(spec_dir / "labels.csv")
    originals = df[~df["is_augmented"]]
    augmented = df[df["is_augmented"]]
    assert len(originals) == n_files
    # Each file produces 2 audio augs + 1 specaug = 3 augmented copies.
    assert len(augmented) == n_files * 3
