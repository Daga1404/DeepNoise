"""Unit tests for src/preprocess.py — all tests use synthetic data."""

import numpy as np
import pytest
import soundfile as sf

from src.config import DURATION, SAMPLE_RATE
from src.preprocess import load_audio, normalize, segment


TARGET = int(SAMPLE_RATE * DURATION)  # 88200


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_peak_equals_one():
    audio = np.array([0.0, 0.5, -2.0, 1.0], dtype=np.float32)
    result = normalize(audio)
    assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-6)


def test_normalize_all_zero_unchanged():
    audio = np.zeros(1000, dtype=np.float32)
    result = normalize(audio)
    np.testing.assert_array_equal(result, audio)


def test_normalize_preserves_shape():
    audio = np.random.randn(TARGET).astype(np.float32)
    result = normalize(audio)
    assert result.shape == audio.shape


# ---------------------------------------------------------------------------
# segment — too-long clip (center-crop)
# ---------------------------------------------------------------------------


def test_segment_trims_long_clip_to_target_length():
    audio = np.ones(TARGET + 10000, dtype=np.float32)
    result = segment(audio)
    assert len(result) == TARGET


def test_segment_center_crop_not_head_crop():
    # Mark the first and last 100 samples; center crop must discard both ends equally.
    audio = np.zeros(TARGET + 200, dtype=np.float32)
    audio[:100] = -1.0  # leading sentinel
    audio[-100:] = -1.0  # trailing sentinel
    result = segment(audio)
    assert len(result) == TARGET
    # Neither leading nor trailing sentinel should dominate the result.
    assert np.sum(result < 0) < 100, "center-crop should trim both ends, not just the tail"


# ---------------------------------------------------------------------------
# segment — too-short clip (zero-pad)
# ---------------------------------------------------------------------------


def test_segment_pads_short_clip_to_target_length():
    audio = np.ones(1000, dtype=np.float32)
    result = segment(audio)
    assert len(result) == TARGET


def test_segment_short_clip_symmetric_padding():
    # Content should sit roughly in the middle; leading and trailing zeros present.
    audio = np.ones(1000, dtype=np.float32)
    result = segment(audio)
    pad_total = TARGET - 1000
    pad_left = pad_total // 2
    # Verify the left-pad region is zero and the content starts where expected.
    assert np.all(result[:pad_left] == 0.0)
    assert result[pad_left] == 1.0


# ---------------------------------------------------------------------------
# segment — exact length (no-op)
# ---------------------------------------------------------------------------


def test_segment_exact_length_unchanged():
    audio = np.random.randn(TARGET).astype(np.float32)
    result = segment(audio)
    np.testing.assert_array_equal(result, audio)


# ---------------------------------------------------------------------------
# load_audio (integration: write synthetic WAV, then load it)
# ---------------------------------------------------------------------------


def test_load_audio_returns_float32_mono(tmp_path):
    wav_path = tmp_path / "test.wav"
    data = np.random.randn(TARGET).astype(np.float32)
    sf.write(wav_path, data, SAMPLE_RATE)
    result = load_audio(wav_path)
    assert result.dtype == np.float32
    assert result.ndim == 1


def test_load_audio_resamples_to_target_sr(tmp_path):
    wav_path = tmp_path / "test_44k.wav"
    # Write at 44100 Hz; load_audio should resample to SAMPLE_RATE.
    data = np.random.randn(44100 * 2).astype(np.float32)
    sf.write(wav_path, data, 44100)
    result = load_audio(wav_path, sr=SAMPLE_RATE)
    # Expected length: 2 s × 22050 = 44100 ± small rounding.
    assert abs(len(result) - SAMPLE_RATE * 2) < 50
