"""Training entry point for the acoustic event CNN.

Two runs are required for the augmentation experiment:

    python src/train.py --data-dir data/ --augment false --output-dir models/noaug/ --epochs 50
    python src/train.py --data-dir data/ --augment true  --output-dir models/aug/   --epochs 50
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.config import NUM_CLASSES
from src.dataset import load_labels, make_tf_dataset, split_dataset

_log = logging.getLogger(__name__)


def _str_to_bool(value: str) -> bool:
    """Convert a 'true'/'false' CLI string to a Python bool."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got '{value}'")


def _compute_class_weights(train_df) -> dict[int, float]:
    """Return a {class_id: weight} dict balanced for the training split."""
    y = train_df["class_id"].values
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y,
    )
    return {int(i): float(w) for i, w in enumerate(weights)}


def _save_history(history, output_dir: Path) -> None:
    """Serialise the Keras History object to JSON, converting numpy floats."""
    hist_dict = {
        key: [float(v) for v in vals]
        for key, vals in history.history.items()
    }
    (output_dir / "history.json").write_text(json.dumps(hist_dict, indent=2))


def run_training(
    data_dir: Path,
    output_dir: Path,
    augment: bool,
    epochs: int,
    batch_size: int = 32,
) -> None:
    """
    Execute one full training pass and write artefacts to ``output_dir``.

    Loads the spectrogram dataset, optionally includes augmented samples,
    computes balanced class weights, builds the CNN, attaches callbacks,
    fits the model, then saves ``cnn_best.keras`` and ``history.json``.

    Args:
        data_dir: Root data directory containing ``spectrograms/labels.csv``.
        output_dir: Destination for ``cnn_best.keras`` and ``history.json``.
        augment: If True, augmented samples are included in the training set.
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
    """
    # Import here so the module can be imported without TF installed.
    from tensorflow import keras

    from src.model import build_cnn

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ----------------------------------------------------------------
    df = load_labels(data_dir / "spectrograms")

    if not augment:
        # Drop augmented rows before splitting so the noaug run uses raw data only.
        df = df[~df["is_augmented"]].reset_index(drop=True)

    train_df, val_df, _test_df = split_dataset(df)

    _log.info(
        f"Split — train: {len(train_df)}  val: {len(val_df)}  "
        f"test: {len(_test_df)}  (augment={augment})"
    )

    train_ds = make_tf_dataset(train_df, batch_size=batch_size, shuffle=True)
    val_ds = make_tf_dataset(val_df, batch_size=batch_size, shuffle=False)

    # --- Class weights ----------------------------------------------------------
    class_weight = _compute_class_weights(train_df)
    _log.info(f"Class weights: {class_weight}")

    # --- Model ------------------------------------------------------------------
    model = build_cnn()
    model.summary()

    # --- Callbacks --------------------------------------------------------------
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "cnn_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # --- Training ---------------------------------------------------------------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    _save_history(history, output_dir)
    _log.info(f"\nArtefacts written to {output_dir}")
    _log.info(f"  cnn_best.keras")
    _log.info(f"  history.json  (keys: {list(history.history.keys())})")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Train the acoustic event CNN (one run of the augmentation experiment)."
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
        default=Path("models/aug"),
        help="Destination for cnn_best.keras and history.json.",
    )
    parser.add_argument(
        "--augment",
        type=_str_to_bool,
        default=True,
        metavar="true|false",
        help="Include augmented training samples (default: true).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size (default: 32).",
    )
    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        augment=args.augment,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
