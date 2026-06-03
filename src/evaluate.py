"""Evaluation suite for the trained acoustic event CNN.

Usage:
    python src/evaluate.py --model models/aug/cnn_best.keras \
                           --data-dir data/ \
                           --output-dir results/
"""

import argparse
import json
import logging
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

from src.config import CLASS_LABELS, NUM_CLASSES
from src.dataset import load_labels, make_tf_dataset, split_dataset

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_cnn(model_path: Path, data_dir: Path, output_dir: Path) -> dict:
    """
    Evaluate a trained CNN on the held-out test set and write all artefacts.

    Outputs written to ``output_dir``:
      - ``cnn_report.txt``          — accuracy, macro F1, per-class metrics.
      - ``cnn_confusion.png``       — seaborn heatmap with class-name axes.
      - ``training_curves.png``     — loss + accuracy curves from history.json.
      - ``error_analysis.csv``      — misclassified samples with confidence.
      - ``augmentation_comparison.md`` — comparison table (if both runs exist).

    Macro F1 is the primary metric and is printed first.

    Args:
        model_path: Path to ``cnn_best.keras``.
        data_dir: Root data directory (contains ``spectrograms/labels.csv``).
        output_dir: Destination directory for all artefacts.

    Returns:
        Dict with ``accuracy`` and ``macro_f1`` keys.
    """
    # Lazy TF import — module must load cleanly without TF on Python 3.14.
    from tensorflow import keras

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path)

    df = load_labels(data_dir / "spectrograms")
    _, _, test_df = split_dataset(df)
    test_ds = make_tf_dataset(test_df, batch_size=32, shuffle=False)

    y_true_list, y_pred_list, probs_list = [], [], []
    for X_batch, y_batch in test_ds:
        batch_probs = model.predict(X_batch, verbose=0)
        y_true_list.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred_list.extend(np.argmax(batch_probs, axis=1))
        probs_list.extend(batch_probs)

    y_true = np.array(y_true_list, dtype=np.int32)
    y_pred = np.array(y_pred_list, dtype=np.int32)
    probs = np.array(probs_list, dtype=np.float32)

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    class_names = [CLASS_LABELS[i] for i in range(NUM_CLASSES)]
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    # Macro F1 first — it is the primary metric.
    _log.info(f"Macro F1 : {macro_f1:.4f}")
    _log.info(f"Accuracy : {accuracy:.4f}")
    _log.info(report)

    _save_report(accuracy, macro_f1, report, output_dir / "cnn_report.txt")
    _save_confusion_matrix(y_true, y_pred, class_names, output_dir / "cnn_confusion.png")
    _save_training_curves(
        model_path.parent / "history.json", output_dir / "training_curves.png"
    )
    _save_error_analysis(test_df, y_true, y_pred, probs, output_dir / "error_analysis.csv")

    # Persist test metrics alongside model so the comparison can pick them up.
    metrics = {"accuracy": accuracy, "macro_f1": macro_f1}
    (model_path.parent / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    _save_augmentation_comparison(
        model_path.parent.parent, output_dir / "augmentation_comparison.md"
    )

    return metrics


# ---------------------------------------------------------------------------
# Private helpers — each testable independently without a real model
# ---------------------------------------------------------------------------


def _save_report(accuracy: float, macro_f1: float, report: str, out_path: Path) -> None:
    """Write accuracy, macro F1, and the full per-class report to a text file."""
    out_path.write_text(
        f"Macro F1 : {macro_f1:.4f}\n"
        f"Accuracy : {accuracy:.4f}\n\n"
        f"{report}"
    )


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    """Save a seaborn confusion-matrix heatmap with class-name axis labels."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
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
    ax.set_title("CNN — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_training_curves(history_json: Path, out_path: Path) -> None:
    """
    Plot loss and accuracy curves from a ``history.json`` file.

    Skips gracefully if the file does not exist.
    """
    if not history_json.exists():
        return

    hist = json.loads(history_json.read_text())
    epochs = range(1, len(hist.get("loss", [])) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, hist.get("loss", []), label="train")
    ax_loss.plot(epochs, hist.get("val_loss", []), label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, hist.get("accuracy", []), label="train")
    ax_acc.plot(epochs, hist.get("val_accuracy", []), label="val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    fig.suptitle("Training Curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_error_analysis(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    out_path: Path,
) -> None:
    """
    Write misclassified test samples to a CSV.

    Columns: ``path``, ``true_label``, ``predicted_label``, ``confidence``.
    All misclassified samples are included; ≥5 per class are present where the
    model made enough errors in that class.
    """
    class_names = [CLASS_LABELS[i] for i in range(NUM_CLASSES)]
    test_df = test_df.reset_index(drop=True)

    rows = []
    for i in range(len(test_df)):
        if y_true[i] != y_pred[i]:
            rows.append(
                {
                    "path": test_df.iloc[i]["path"],
                    "true_label": class_names[y_true[i]],
                    "predicted_label": class_names[y_pred[i]],
                    "confidence": float(probs[i, y_pred[i]]),
                }
            )

    pd.DataFrame(rows, columns=["path", "true_label", "predicted_label", "confidence"]).to_csv(
        out_path, index=False
    )


def _save_augmentation_comparison(models_root: Path, out_path: Path) -> None:
    """
    Write ``augmentation_comparison.md`` if both run artefacts are present.

    Reads ``models_root/noaug/test_metrics.json`` and
    ``models_root/aug/test_metrics.json`` for test accuracy and macro F1,
    and optionally reads ``history.json`` from each run for best val F1.

    Skips quietly if neither metrics file is found.
    """
    noaug_metrics_path = models_root / "noaug" / "test_metrics.json"
    aug_metrics_path = models_root / "aug" / "test_metrics.json"

    if not noaug_metrics_path.exists() and not aug_metrics_path.exists():
        return

    def _load(p: Path) -> dict:
        return json.loads(p.read_text()) if p.exists() else {}

    noaug = _load(noaug_metrics_path)
    aug = _load(aug_metrics_path)

    def _fmt(m: dict, key: str) -> str:
        return f"{m[key]:.4f}" if key in m else "—"

    lines = [
        "# Augmentation Comparison\n",
        "| Run | Test Accuracy | Test Macro F1 |",
        "|-----|--------------|---------------|",
        f"| No augmentation | {_fmt(noaug, 'accuracy')} | {_fmt(noaug, 'macro_f1')} |",
        f"| With augmentation | {_fmt(aug, 'accuracy')} | {_fmt(aug, 'macro_f1')} |",
        "",
        "## Notes",
        "- Primary metric: **Macro F1** (equal weight per class regardless of sample count).",
        "- Augmentation strategies: Gaussian noise (std=0.005), time shift (±10%), SpecAugment.",
    ]

    # Append best validation loss from history if available.
    for run_name, run_dir in [("noaug", models_root / "noaug"), ("aug", models_root / "aug")]:
        hist_path = run_dir / "history.json"
        if hist_path.exists():
            hist = json.loads(hist_path.read_text())
            val_loss = hist.get("val_loss", [])
            if val_loss:
                lines.append(f"- {run_name}: best val_loss = {min(val_loss):.4f}")

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Evaluate a trained acoustic CNN on the held-out test set."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to cnn_best.keras (e.g. models/aug/cnn_best.keras).",
    )
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
        help="Destination for all evaluation artefacts.",
    )
    args = parser.parse_args()

    evaluate_cnn(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
