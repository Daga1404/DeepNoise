"""Unit tests for src/evaluate.py — most tests run without TensorFlow."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import CLASS_LABELS, NUM_CLASSES
from src.evaluate import (
    _save_augmentation_comparison,
    _save_confusion_matrix,
    _save_error_analysis,
    _save_report,
    _save_training_curves,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS_NAMES = [CLASS_LABELS[i] for i in range(NUM_CLASSES)]


def _fake_preds(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, probs) with some deliberate misclassifications."""
    rng = np.random.default_rng(seed)
    y_true = (np.arange(n) % NUM_CLASSES).astype(np.int32)
    # Introduce ~30 % errors
    y_pred = y_true.copy()
    error_mask = rng.random(n) < 0.3
    y_pred[error_mask] = (y_pred[error_mask] + 1) % NUM_CLASSES
    # Fake softmax probabilities
    probs = rng.dirichlet(np.ones(NUM_CLASSES), size=n).astype(np.float32)
    return y_true, y_pred, probs


def _fake_test_df(n: int) -> pd.DataFrame:
    y_true, _, _ = _fake_preds(n)
    return pd.DataFrame(
        {
            "path": [f"dummy/clip_{i}.npy" for i in range(n)],
            "class_label": [CLASS_LABELS[c] for c in y_true],
            "class_id": y_true,
            "is_augmented": [False] * n,
        }
    )


# ---------------------------------------------------------------------------
# _save_report
# ---------------------------------------------------------------------------


def test_save_report_creates_file(tmp_path):
    _save_report(0.82, 0.80, "some report text", tmp_path / "cnn_report.txt")
    assert (tmp_path / "cnn_report.txt").exists()


def test_save_report_contains_macro_f1_first(tmp_path):
    out = tmp_path / "cnn_report.txt"
    _save_report(0.82, 0.80, "report body", out)
    content = out.read_text()
    macro_pos = content.index("Macro F1")
    accuracy_pos = content.index("Accuracy")
    assert macro_pos < accuracy_pos, "Macro F1 must appear before Accuracy in the report"


def test_save_report_contains_expected_values(tmp_path):
    out = tmp_path / "cnn_report.txt"
    _save_report(0.8200, 0.7950, "body text", out)
    content = out.read_text()
    assert "0.8200" in content
    assert "0.7950" in content


# ---------------------------------------------------------------------------
# _save_confusion_matrix
# ---------------------------------------------------------------------------


def test_save_confusion_matrix_creates_png(tmp_path):
    y_true, y_pred, _ = _fake_preds(50)
    _save_confusion_matrix(y_true, y_pred, CLASS_NAMES, tmp_path / "cnn_confusion.png")
    assert (tmp_path / "cnn_confusion.png").exists()


def test_save_confusion_matrix_png_is_nonempty(tmp_path):
    y_true, y_pred, _ = _fake_preds(50)
    out = tmp_path / "cnn_confusion.png"
    _save_confusion_matrix(y_true, y_pred, CLASS_NAMES, out)
    assert out.stat().st_size > 1000, "PNG file seems too small to be a real plot"


# ---------------------------------------------------------------------------
# _save_training_curves
# ---------------------------------------------------------------------------


def _write_history(path, n_epochs: int = 10):
    hist = {
        "loss": [1.0 / (i + 1) for i in range(n_epochs)],
        "val_loss": [1.1 / (i + 1) for i in range(n_epochs)],
        "accuracy": [0.5 + 0.04 * i for i in range(n_epochs)],
        "val_accuracy": [0.48 + 0.04 * i for i in range(n_epochs)],
    }
    path.write_text(json.dumps(hist))


def test_save_training_curves_creates_png(tmp_path):
    hist_path = tmp_path / "history.json"
    _write_history(hist_path)
    _save_training_curves(hist_path, tmp_path / "training_curves.png")
    assert (tmp_path / "training_curves.png").exists()


def test_save_training_curves_skips_if_missing(tmp_path):
    out = tmp_path / "training_curves.png"
    _save_training_curves(tmp_path / "nonexistent.json", out)
    assert not out.exists(), "Should not create output when history file is absent"


def test_save_training_curves_png_is_nonempty(tmp_path):
    hist_path = tmp_path / "history.json"
    _write_history(hist_path)
    out = tmp_path / "training_curves.png"
    _save_training_curves(hist_path, out)
    assert out.stat().st_size > 1000


# ---------------------------------------------------------------------------
# _save_error_analysis
# ---------------------------------------------------------------------------


def test_save_error_analysis_creates_csv(tmp_path):
    n = 100
    y_true, y_pred, probs = _fake_preds(n)
    df = _fake_test_df(n)
    _save_error_analysis(df, y_true, y_pred, probs, tmp_path / "error_analysis.csv")
    assert (tmp_path / "error_analysis.csv").exists()


def test_save_error_analysis_correct_columns(tmp_path):
    n = 100
    y_true, y_pred, probs = _fake_preds(n)
    df = _fake_test_df(n)
    out = tmp_path / "error_analysis.csv"
    _save_error_analysis(df, y_true, y_pred, probs, out)
    result = pd.read_csv(out)
    assert list(result.columns) == ["path", "true_label", "predicted_label", "confidence"]


def test_save_error_analysis_only_misclassified(tmp_path):
    n = 100
    y_true, y_pred, probs = _fake_preds(n)
    df = _fake_test_df(n)
    out = tmp_path / "error_analysis.csv"
    _save_error_analysis(df, y_true, y_pred, probs, out)
    result = pd.read_csv(out)
    assert all(result["true_label"] != result["predicted_label"])


def test_save_error_analysis_confidence_in_range(tmp_path):
    n = 100
    y_true, y_pred, probs = _fake_preds(n)
    df = _fake_test_df(n)
    out = tmp_path / "error_analysis.csv"
    _save_error_analysis(df, y_true, y_pred, probs, out)
    result = pd.read_csv(out)
    if len(result) > 0:
        assert result["confidence"].between(0.0, 1.0).all()


def test_save_error_analysis_labels_are_class_names(tmp_path):
    n = 100
    y_true, y_pred, probs = _fake_preds(n)
    df = _fake_test_df(n)
    out = tmp_path / "error_analysis.csv"
    _save_error_analysis(df, y_true, y_pred, probs, out)
    result = pd.read_csv(out)
    valid_names = set(CLASS_LABELS.values())
    assert set(result["true_label"]).issubset(valid_names)
    assert set(result["predicted_label"]).issubset(valid_names)


# ---------------------------------------------------------------------------
# _save_augmentation_comparison
# ---------------------------------------------------------------------------


def _write_metrics(path: object, accuracy: float, macro_f1: float) -> None:
    path.write_text(json.dumps({"accuracy": accuracy, "macro_f1": macro_f1}))


def test_save_augmentation_comparison_creates_md(tmp_path):
    (tmp_path / "noaug").mkdir()
    (tmp_path / "aug").mkdir()
    _write_metrics(tmp_path / "noaug" / "test_metrics.json", 0.72, 0.70)
    _write_metrics(tmp_path / "aug" / "test_metrics.json", 0.81, 0.79)

    out = tmp_path / "augmentation_comparison.md"
    _save_augmentation_comparison(tmp_path, out)
    assert out.exists()


def test_save_augmentation_comparison_contains_metrics(tmp_path):
    (tmp_path / "noaug").mkdir()
    (tmp_path / "aug").mkdir()
    _write_metrics(tmp_path / "noaug" / "test_metrics.json", 0.7200, 0.7000)
    _write_metrics(tmp_path / "aug" / "test_metrics.json", 0.8100, 0.7900)

    out = tmp_path / "augmentation_comparison.md"
    _save_augmentation_comparison(tmp_path, out)
    content = out.read_text()
    assert "0.7200" in content
    assert "0.8100" in content
    assert "0.7000" in content
    assert "0.7900" in content


def test_save_augmentation_comparison_skips_if_no_metrics(tmp_path):
    out = tmp_path / "augmentation_comparison.md"
    _save_augmentation_comparison(tmp_path, out)
    assert not out.exists(), "Should not create file when no metrics are present"


def test_save_augmentation_comparison_partial_data(tmp_path):
    """Comparison should still write if only one run's metrics are available."""
    (tmp_path / "aug").mkdir()
    _write_metrics(tmp_path / "aug" / "test_metrics.json", 0.81, 0.79)

    out = tmp_path / "augmentation_comparison.md"
    _save_augmentation_comparison(tmp_path, out)
    assert out.exists()
    content = out.read_text()
    assert "0.8100" in content
    assert "—" in content  # noaug values should show placeholder


# ---------------------------------------------------------------------------
# evaluate_cnn (full pipeline — requires TensorFlow)
# ---------------------------------------------------------------------------


def test_evaluate_cnn_produces_all_output_files(tmp_path):
    """Integration test: mock model writes all required artefacts."""
    tf = pytest.importorskip("tensorflow")
    import soundfile as sf

    from src.config import INPUT_SHAPE, SAMPLE_RATE
    from src.dataset import load_labels, split_dataset
    from src.evaluate import evaluate_cnn
    from src.features import save_spectrograms

    # --- build a tiny fake dataset on disk ---
    raw_dir = tmp_path / "data" / "processed"
    spec_dir = tmp_path / "data" / "spectrograms"
    n_files = 10

    for class_id, class_name in CLASS_LABELS.items():
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(n_files):
            audio = np.random.randn(int(SAMPLE_RATE * 4.0)).astype(np.float32)
            sf.write(class_dir / f"clip_{i:03d}.wav", audio, SAMPLE_RATE)

    save_spectrograms(raw_dir, spec_dir, augment_train=False)

    # --- build and save a tiny untrained model ---
    from src.model import build_cnn
    model = build_cnn()
    model_dir = tmp_path / "models" / "aug"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "cnn_best.keras"
    model.save(model_path)

    # Fake history.json
    hist = {"loss": [1.0, 0.8], "val_loss": [1.1, 0.9],
            "accuracy": [0.5, 0.6], "val_accuracy": [0.48, 0.58]}
    (model_dir / "history.json").write_text(json.dumps(hist))

    # --- evaluate ---
    output_dir = tmp_path / "results"
    metrics = evaluate_cnn(model_path, tmp_path / "data", output_dir)

    assert (output_dir / "cnn_report.txt").exists()
    assert (output_dir / "cnn_confusion.png").exists()
    assert (output_dir / "training_curves.png").exists()
    assert (output_dir / "error_analysis.csv").exists()
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["macro_f1"], float)
