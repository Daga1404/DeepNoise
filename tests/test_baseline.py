"""Unit tests for src/baseline.py — all tests use synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.config import CLASS_LABELS, NUM_CLASSES
from src.baseline import evaluate, load_features, train_svm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FEATURES = 128 * 173  # 22144 — full flattened spectrogram size
N_TRAIN = 50
N_TEST = 20


def _synthetic_features(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.random((n, N_FEATURES)).astype(np.float32)
    y = np.arange(n, dtype=np.int32) % NUM_CLASSES
    return X, y


# ---------------------------------------------------------------------------
# load_features
# ---------------------------------------------------------------------------


def test_load_features_returns_correct_shapes(tmp_path):
    """load_features should flatten (128, 173) spectrograms to 22144-d vectors."""
    df_records = []
    for i in range(10):
        class_id = i % NUM_CLASSES
        spec = np.random.randn(128, 173).astype(np.float32)
        npy_path = tmp_path / f"clip_{i}.npy"
        np.save(npy_path, spec)
        df_records.append(
            {"path": str(npy_path), "class_label": CLASS_LABELS[class_id], "class_id": class_id}
        )

    df = pd.DataFrame(df_records)
    X, y = load_features(df)

    assert X.shape == (10, N_FEATURES), f"unexpected X shape {X.shape}"
    assert y.shape == (10,)
    assert X.dtype == np.float32
    assert y.dtype == np.int32


def test_load_features_class_ids_match(tmp_path):
    """y values returned by load_features must equal df['class_id']."""
    df_records = []
    for class_id in range(NUM_CLASSES):
        spec = np.zeros((128, 173), dtype=np.float32)
        npy_path = tmp_path / f"clip_{class_id}.npy"
        np.save(npy_path, spec)
        df_records.append(
            {"path": str(npy_path), "class_label": CLASS_LABELS[class_id], "class_id": class_id}
        )

    df = pd.DataFrame(df_records)
    _, y = load_features(df)

    np.testing.assert_array_equal(y, df["class_id"].values)


# ---------------------------------------------------------------------------
# train_svm
# ---------------------------------------------------------------------------


def test_train_svm_returns_model_with_predict():
    X, y = _synthetic_features(N_TRAIN)
    model = train_svm(X, y)
    assert hasattr(model, "predict"), "train_svm must return an object with a predict method"


def test_train_svm_predict_output_shape():
    X_train, y_train = _synthetic_features(N_TRAIN, seed=0)
    X_test, _ = _synthetic_features(N_TEST, seed=1)
    model = train_svm(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (N_TEST,)


def test_train_svm_predict_valid_class_ids():
    X_train, y_train = _synthetic_features(N_TRAIN, seed=0)
    X_test, _ = _synthetic_features(N_TEST, seed=1)
    model = train_svm(X_train, y_train)
    preds = model.predict(X_test)
    assert set(preds).issubset(set(range(NUM_CLASSES)))


def test_train_svm_full_feature_size_no_error():
    """SVM must handle the full 22144-feature vectors without memory or type errors."""
    X, y = _synthetic_features(N_TRAIN)
    assert X.shape[1] == N_FEATURES
    model = train_svm(X, y)
    assert model is not None


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_required_keys(tmp_path):
    X_train, y_train = _synthetic_features(N_TRAIN, seed=0)
    X_test, y_test = _synthetic_features(N_TEST, seed=1)
    model = train_svm(X_train, y_train)

    result = evaluate(model, X_test, y_test, tmp_path)

    assert "accuracy" in result
    assert "macro_f1" in result
    assert "report" in result


def test_evaluate_accuracy_is_float_in_range(tmp_path):
    X_train, y_train = _synthetic_features(N_TRAIN, seed=0)
    X_test, y_test = _synthetic_features(N_TEST, seed=1)
    model = train_svm(X_train, y_train)

    result = evaluate(model, X_test, y_test, tmp_path)

    assert isinstance(result["accuracy"], float)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_evaluate_macro_f1_is_float_in_range(tmp_path):
    X_train, y_train = _synthetic_features(N_TRAIN, seed=0)
    X_test, y_test = _synthetic_features(N_TEST, seed=1)
    model = train_svm(X_train, y_train)

    result = evaluate(model, X_test, y_test, tmp_path)

    assert isinstance(result["macro_f1"], float)
    assert 0.0 <= result["macro_f1"] <= 1.0


def test_evaluate_saves_confusion_matrix_png(tmp_path):
    X_train, y_train = _synthetic_features(N_TRAIN, seed=0)
    X_test, y_test = _synthetic_features(N_TEST, seed=1)
    model = train_svm(X_train, y_train)

    evaluate(model, X_test, y_test, tmp_path)

    assert (tmp_path / "baseline_confusion.png").exists()


def test_evaluate_saves_report_txt(tmp_path):
    X_train, y_train = _synthetic_features(N_TRAIN, seed=0)
    X_test, y_test = _synthetic_features(N_TEST, seed=1)
    model = train_svm(X_train, y_train)

    evaluate(model, X_test, y_test, tmp_path)

    report_path = tmp_path / "baseline_report.txt"
    assert report_path.exists()
    content = report_path.read_text()
    assert "Macro F1" in content
    assert "Accuracy" in content


def test_evaluate_confusion_matrix_uses_class_names(tmp_path):
    """The report text must contain class-name strings, not just integer IDs."""
    X_train, y_train = _synthetic_features(N_TRAIN, seed=0)
    X_test, y_test = _synthetic_features(N_TEST, seed=1)
    model = train_svm(X_train, y_train)

    result = evaluate(model, X_test, y_test, tmp_path)

    for class_name in CLASS_LABELS.values():
        assert class_name in result["report"], (
            f"class name '{class_name}' not found in classification report"
        )
