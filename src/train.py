"""CLI training entry point: python src/train.py --data-dir data"""

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from dataset import AcousticDataset
from model import build_cnn

BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10


def train(data_dir: Path, output_dir: Path) -> None:
    labels_csv = data_dir / "spectrograms" / "labels.csv"
    ds = AcousticDataset(labels_csv)

    train_ds = ds.get_train(BATCH_SIZE)
    val_ds = ds.get_val(BATCH_SIZE)

    model = build_cnn()
    model.summary()

    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(output_dir / "final_model.keras")
    print(f"Model saved to {output_dir}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the acoustic CNN.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
