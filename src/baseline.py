"""SVM baseline: train on flattened log-Mel spectrograms, report metrics."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import CLASS_LABELS, NUM_CLASSES
from src.dataset import load_labels, split_dataset


def load_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Load spectrograms from disk and flatten each to a 1-D feature vector.

    Args:
        df: Labels DataFrame (subset) with ``path`` and ``class_id`` columns.

    Returns:
        Tuple ``(X, y)`` where X has shape ``(n_samples, n_mels * T)`` and
        y has shape ``(n_samples,)`` with integer class IDs.
    """
    X, y = [], []
    for _, row in df.iterrows():
        spec = np.load(row["path"])
        X.append(spec.flatten())
        y.append(int(row["class_id"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Fit a StandardScaler → SVC(kernel='rbf', C=10) pipeline.

    The pipeline is returned so that ``evaluate`` can call ``.predict``
    directly on raw (unscaled) feature vectors.

    Args:
        X_train: Float32 array of shape ``(n_samples, n_features)``.
        y_train: Integer class-ID array of shape ``(n_samples,)``.

    Returns:
        Fitted sklearn ``Pipeline`` containing a scaler and an RBF SVM.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> dict:
    """
    Evaluate the SVM on the test set and write artefacts to ``output_dir``.

    Saves:
      - ``baseline_confusion.png`` — seaborn heatmap with class-name axes.
      - ``baseline_report.txt`` — per-class precision, recall, F1.

    Args:
        model: Fitted pipeline as returned by ``train_svm``.
        X_test: Float32 feature array of shape ``(n_samples, n_features)``.
        y_test: Integer class-ID array of shape ``(n_samples,)``.
        output_dir: Directory where artefacts are written.

    Returns:
        Dict with keys ``accuracy`` (float), ``macro_f1`` (float), and
        ``report`` (str — the full sklearn classification report).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    class_names = [CLASS_LABELS[i] for i in range(NUM_CLASSES)]
    report = classification_report(y_test, preds, target_names=class_names)

    _save_confusion_matrix(y_test, preds, class_names, output_dir / "baseline_confusion.png")

    report_path = output_dir / "baseline_report.txt"
    report_path.write_text(
        f"Accuracy : {accuracy:.4f}\nMacro F1 : {macro_f1:.4f}\n\n{report}"
    )

    return {"accuracy": accuracy, "macro_f1": macro_f1, "report": report}


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    """Write a labelled seaborn confusion-matrix heatmap to ``out_path``."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Baseline SVM — Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the SVM baseline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory (must contain spectrograms/labels.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for confusion matrix PNG and report TXT.",
    )
    args = parser.parse_args()

    df = load_labels(args.data_dir / "spectrograms")
    train_df, _val_df, test_df = split_dataset(df)

    print("Loading training features …")
    X_train, y_train = load_features(train_df)
    print(f"  X_train shape: {X_train.shape}")

    print("Training SVM …")
    model = train_svm(X_train, y_train)

    print("Loading test features …")
    X_test, y_test = load_features(test_df)

    print("Evaluating …")
    results = evaluate(model, X_test, y_test, args.output_dir)
    print(f"\nMacro F1 : {results['macro_f1']:.4f}")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(results["report"])


if __name__ == "__main__":
    main()
