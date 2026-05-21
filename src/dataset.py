"""tf.data dataset construction and train/val/test splitting."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

NUM_CLASSES = 5
SPEC_SHAPE = (128, 173, 1)


class AcousticDataset:
    def __init__(self, labels_csv: Path, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
        self.df = pd.read_csv(labels_csv)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self._split()

    def _split(self) -> None:
        train_val, test = train_test_split(
            self.df, test_size=self.test_ratio, stratify=self.df["class_id"], random_state=self.seed
        )
        val_ratio_adjusted = self.val_ratio / (1 - self.test_ratio)
        train, val = train_test_split(
            train_val, test_size=val_ratio_adjusted, stratify=train_val["class_id"], random_state=self.seed
        )
        self.train_df = train.reset_index(drop=True)
        self.val_df = val.reset_index(drop=True)
        self.test_df = test.reset_index(drop=True)

    def _load_spectrogram(self, path: str) -> np.ndarray:
        spec = np.load(path)
        spec = spec[..., np.newaxis]  # (128, 173, 1)
        return spec.astype(np.float32)

    def _make_tf_dataset(self, df: pd.DataFrame, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        paths = df["path"].values
        labels = tf.keras.utils.to_categorical(df["class_id"].values, num_classes=NUM_CLASSES)

        def load_fn(path, label):
            spec = tf.numpy_function(
                lambda p: self._load_spectrogram(p.decode()), [path], tf.float32
            )
            spec.set_shape(SPEC_SHAPE)
            return spec, label

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=self.seed)
        ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_train(self, batch_size: int = 32) -> tf.data.Dataset:
        return self._make_tf_dataset(self.train_df, batch_size, shuffle=True)

    def get_val(self, batch_size: int = 32) -> tf.data.Dataset:
        return self._make_tf_dataset(self.val_df, batch_size, shuffle=False)

    def get_test(self, batch_size: int = 32) -> tf.data.Dataset:
        return self._make_tf_dataset(self.test_df, batch_size, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    labels_csv = args.data_dir / "spectrograms" / "labels.csv"
    ds = AcousticDataset(labels_csv)
    print(f"train: {len(ds.train_df)}  val: {len(ds.val_df)}  test: {len(ds.test_df)}")


if __name__ == "__main__":
    main()
