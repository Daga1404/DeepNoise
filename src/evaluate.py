"""Evaluation suite: confusion matrix, F1, error analysis, training curves."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow import keras

from dataset import AcousticDataset
from preprocess import CLASS_LABELS

BATCH_SIZE = 32


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    out_path = output_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_training_curves(log_csv: Path, output_dir: Path) -> None:
    if not log_csv.exists():
        print(f"Training log not found: {log_csv}")
        return
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["accuracy"], label="train")
    axes[1].plot(df["epoch"], df["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    out_path = output_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def log_misclassified(
    test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path, n: int = 5
) -> None:
    test_df = test_df.copy()
    test_df["y_true"] = y_true
    test_df["y_pred"] = y_pred
    misclassified = test_df[test_df["y_true"] != test_df["y_pred"]]

    lines = []
    for class_id, class_label in enumerate(CLASS_LABELS):
        subset = misclassified[misclassified["y_true"] == class_id].head(n)
        lines.append(f"\n=== {class_label} (true={class_id}) ===")
        for _, row in subset.iterrows():
            lines.append(f"  path={row['path']}  predicted={CLASS_LABELS[row['y_pred']]}")

    report_path = output_dir / "misclassified.txt"
    report_path.write_text("\n".join(lines))
    print(f"Saved {report_path}")


def evaluate(model_path: Path, data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = data_dir / "spectrograms" / "labels.csv"
    ds = AcousticDataset(labels_csv)
    test_ds = ds.get_test(BATCH_SIZE)

    model = keras.models.load_model(model_path)

    y_true, y_pred = [], []
    for X_batch, y_batch in test_ds:
        preds = model.predict(X_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nTest macro F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

    plot_confusion_matrix(y_true, y_pred, output_dir)
    plot_training_curves(model_path.parent / "training_log.csv", output_dir)
    log_misclassified(ds.test_df, y_true, y_pred, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained acoustic CNN.")
    parser.add_argument("--model", type=Path, default=Path("outputs/best_model.keras"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    evaluate(args.model, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
