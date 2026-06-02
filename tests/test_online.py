"""Tests for src/online.py — microphone tests are excluded (hardware-dependent).

The --file mode and the shared preprocessing helpers are fully testable
without a microphone. TF-dependent tests skip on Python 3.14.
"""

import numpy as np
import pytest
import soundfile as sf

from src.config import CLASS_LABELS, DURATION, NUM_CLASSES, SAMPLE_RATE
from src.online import _format_result, _preprocess


FIXTURE_WAV = "tests/fixtures/sample.wav"


# ---------------------------------------------------------------------------
# _preprocess (no TF required)
# ---------------------------------------------------------------------------


def test_preprocess_output_shape():
    """_preprocess must return a (1, 128, 173, 1) tensor regardless of input length."""
    audio = np.random.randn(int(SAMPLE_RATE * DURATION)).astype(np.float32)
    tensor = _preprocess(audio)
    assert tensor.shape == (1, 128, 173, 1), f"unexpected shape {tensor.shape}"


def test_preprocess_short_clip_padded():
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32)  # 1 s — shorter than 4 s
    tensor = _preprocess(audio)
    assert tensor.shape == (1, 128, 173, 1)


def test_preprocess_long_clip_cropped():
    audio = np.random.randn(SAMPLE_RATE * 8).astype(np.float32)  # 8 s — longer than 4 s
    tensor = _preprocess(audio)
    assert tensor.shape == (1, 128, 173, 1)


def test_preprocess_silent_clip_no_crash():
    audio = np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)
    tensor = _preprocess(audio)
    assert tensor.shape == (1, 128, 173, 1)
    assert np.all(np.isfinite(tensor))


def test_preprocess_output_dtype_float32():
    audio = np.random.randn(int(SAMPLE_RATE * DURATION)).astype(np.float32)
    tensor = _preprocess(audio)
    assert tensor.dtype == np.float32


def test_preprocess_no_nan_or_inf():
    audio = np.random.randn(int(SAMPLE_RATE * DURATION)).astype(np.float32)
    tensor = _preprocess(audio)
    assert not np.any(np.isnan(tensor))
    assert not np.any(np.isinf(tensor))


# ---------------------------------------------------------------------------
# _format_result (no TF required)
# ---------------------------------------------------------------------------


def test_format_result_contains_class_name():
    line = _format_result("alarm_tone", 0.87)
    assert "alarm_tone" in line


def test_format_result_contains_confidence():
    line = _format_result("normal_operation", 0.92)
    assert "0.92" in line


def test_format_result_contains_timestamp_brackets():
    line = _format_result("metallic_impact", 0.75)
    assert line.startswith("[")
    assert "]" in line


def test_format_result_structure():
    """Output must match the documented format: [HH:MM:SS] Predicted: <name>  (confidence: X.XX)"""
    line = _format_result("friction_squeal", 0.65)
    assert "Predicted:" in line
    assert "confidence:" in line


# ---------------------------------------------------------------------------
# fixture WAV existence
# ---------------------------------------------------------------------------


def test_fixture_wav_exists():
    from pathlib import Path
    assert Path(FIXTURE_WAV).exists(), (
        f"Fixture file {FIXTURE_WAV} must exist for file-mode tests"
    )


def test_fixture_wav_is_readable():
    data, sr = sf.read(FIXTURE_WAV)
    assert sr > 0
    assert len(data) > 0


# ---------------------------------------------------------------------------
# classify_audio and run_file_mode (require TensorFlow)
# ---------------------------------------------------------------------------


def test_classify_audio_returns_valid_class(tmp_path):
    """classify_audio must return a class name from CLASS_LABELS and a confidence in [0,1]."""
    tf = pytest.importorskip("tensorflow")
    from src.model import build_cnn
    from src.online import classify_audio

    model = build_cnn()
    audio = np.random.randn(int(SAMPLE_RATE * DURATION)).astype(np.float32)
    class_name, confidence = classify_audio(model, audio)

    assert class_name in CLASS_LABELS.values(), f"unexpected class '{class_name}'"
    assert 0.0 <= confidence <= 1.0


def test_classify_audio_confidence_is_softmax_max(tmp_path):
    """Confidence must equal the maximum softmax probability."""
    tf = pytest.importorskip("tensorflow")
    from src.model import build_cnn
    from src.online import classify_audio, _preprocess

    model = build_cnn()
    audio = np.random.randn(int(SAMPLE_RATE * DURATION)).astype(np.float32)

    tensor = _preprocess(audio)
    probs = model.predict(tensor, verbose=0)[0]
    expected_confidence = float(probs.max())

    _, confidence = classify_audio(model, audio)
    assert abs(confidence - expected_confidence) < 1e-5


def test_run_file_mode_with_fixture(tmp_path):
    """run_file_mode must load the fixture WAV, run inference, and return a result."""
    pytest.importorskip("tensorflow")
    from pathlib import Path
    from src.model import build_cnn
    from src.online import run_file_mode

    # Save a tiny untrained model for the test.
    model = build_cnn()
    model_path = tmp_path / "cnn_best.keras"
    model.save(model_path)

    class_name, confidence = run_file_mode(model_path, Path(FIXTURE_WAV))
    assert class_name in CLASS_LABELS.values()
    assert 0.0 <= confidence <= 1.0
