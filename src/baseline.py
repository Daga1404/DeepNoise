"""SVM baseline: train on flattened Mel-spectrograms, report metrics."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import CLASS_LABELS


def _load_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _, row in df.iterrows():
        spec = np.load(row["path"])
        X.append(spec.flatten())
        y.append(row["class_id"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_svm_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[SVC, StandardScaler]:
    X_train, y_train = _load_features(train_df)
    X_val, y_val = _load_features(val_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    print(f"Validation macro F1: {val_f1:.4f}")
    print(classification_report(y_val, val_preds, target_names=CLASS_LABELS))
    return clf, scaler


def evaluate_baseline(clf: SVC, scaler: StandardScaler, test_df: pd.DataFrame) -> dict:
    X_test, y_test = _load_features(test_df)
    X_test = scaler.transform(X_test)
    preds = clf.predict(X_test)

    macro_f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, target_names=CLASS_LABELS, output_dict=True)
    print(f"\nTest macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, preds, target_names=CLASS_LABELS))
    return {"macro_f1": macro_f1, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate SVM baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    from dataset import AcousticDataset
    labels_csv = args.data_dir / "spectrograms" / "labels.csv"
    ds = AcousticDataset(labels_csv)

    clf, scaler = train_svm_baseline(ds.train_df, ds.val_df)
    evaluate_baseline(clf, scaler, ds.test_df)


if __name__ == "__main__":
    main()
