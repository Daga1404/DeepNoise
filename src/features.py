"""Log-Mel spectrogram computation and data augmentation primitives."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src.config import (
    CLASS_LABELS,
    DURATION,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)


def compute_melspectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """
    Compute a log-scaled Mel-spectrogram from a mono audio array.

    Args:
        audio: Float32 mono audio signal, shape (N,).
        sr: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        n_mels: Number of Mel filter banks.

    Returns:
        Log-scaled Mel-spectrogram of shape (n_mels, T).
    """
    mel_S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_S = np.log(mel_S + 1e-6)
    return log_S.astype(np.float32)


def augment_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list[np.ndarray]:
    """
    Produce augmented variants of a raw audio array.

    Three strategies are applied:
      1. Gaussian noise — simulates sensor noise.
      2. Forward time shift (+10%) — teaches temporal invariance via circular roll.
      3. Backward time shift (−10%) — covers the opposite shift direction.

    Args:
        audio: Float32 mono audio array of any length.
        sr: Sample rate in Hz (reserved for future SR-aware augmentations).

    Returns:
        List of three float32 arrays: [noisy_audio, forward_shifted, backward_shifted].
    """
    noise = audio + np.random.normal(0.0, 0.005, size=len(audio)).astype(np.float32)

    shift_samples = int(0.1 * len(audio))
    forward  = np.roll(audio,  shift_samples).astype(np.float32)  # +10%
    backward = np.roll(audio, -shift_samples).astype(np.float32)  # -10%

    return [noise, forward, backward]


def augment_spectrogram(log_S: np.ndarray) -> list[np.ndarray]:
    """
    Apply SpecAugment to a log-Mel spectrogram.

    Zeros out one random frequency band (20 bins) and one random time window
    (30 frames) to improve robustness to partial occlusions and signal gaps.

    Args:
        log_S: Log-Mel spectrogram of shape (n_mels, T).

    Returns:
        List containing one augmented spectrogram of the same shape.
    """
    freq_mask_size = 20
    time_mask_size = 30

    n_mels, n_frames = log_S.shape
    aug = log_S.copy()

    freq_start = np.random.randint(0, max(1, n_mels - freq_mask_size))
    aug[freq_start : freq_start + freq_mask_size, :] = 0.0

    time_start = np.random.randint(0, max(1, n_frames - time_mask_size))
    aug[:, time_start : time_start + time_mask_size] = 0.0

    return [aug]


def save_spectrograms(
    processed_dir: Path,
    spec_dir: Path,
    augment_train: bool = False,
    **mel_kwargs,
) -> None:
    """
    Convert every preprocessed WAV in ``processed_dir`` to a log-Mel spectrogram
    and write the result to ``spec_dir``, mirroring the class-subfolder structure.

    A ``labels.csv`` index is written at ``spec_dir/labels.csv`` with columns:
    ``path``, ``class_label``, ``class_id``, ``is_augmented``.

    When ``augment_train=True``, three additional augmented copies are produced
    per sample (Gaussian noise, time shift, SpecAugment) and tagged
    ``is_augmented=True``.  ``dataset.py`` is responsible for ensuring these
    copies appear only in the training split.

    Args:
        processed_dir: Root directory containing one subfolder per class.
        spec_dir: Destination root for .npy files and labels.csv.
        augment_train: If True, write augmented copies alongside originals.
        **mel_kwargs: Optional overrides forwarded to ``compute_melspectrogram``
                      (e.g., ``n_fft``, ``hop_length``, ``n_mels``).
    """
    label_lookup = {v: k for k, v in CLASS_LABELS.items()}
    records: list[dict] = []

    for class_dir in sorted(processed_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_label = class_dir.name
        class_id = label_lookup.get(class_label)
        if class_id is None:
            continue

        dest_class_dir = spec_dir / class_label
        dest_class_dir.mkdir(parents=True, exist_ok=True)

        for wav_path in sorted(class_dir.glob("*.wav")):
            audio, _ = sf.read(wav_path, dtype="float32")

            base_spec = compute_melspectrogram(audio, **mel_kwargs)
            out_path = dest_class_dir / (wav_path.stem + ".npy")
            np.save(out_path, base_spec)
            records.append(
                {
                    "path": str(out_path),
                    "class_label": class_label,
                    "class_id": class_id,
                    "is_augmented": False,
                }
            )

            if augment_train:
                for i, aug_audio in enumerate(augment_audio(audio)):
                    aug_spec = compute_melspectrogram(aug_audio, **mel_kwargs)
                    aug_path = dest_class_dir / f"{wav_path.stem}_aug{i}.npy"
                    np.save(aug_path, aug_spec)
                    records.append(
                        {
                            "path": str(aug_path),
                            "class_label": class_label,
                            "class_id": class_id,
                            "is_augmented": True,
                        }
                    )

                for i, aug_spec in enumerate(augment_spectrogram(base_spec)):
                    aug_path = dest_class_dir / f"{wav_path.stem}_specaug{i}.npy"
                    np.save(aug_path, aug_spec)
                    records.append(
                        {
                            "path": str(aug_path),
                            "class_label": class_label,
                            "class_id": class_id,
                            "is_augmented": True,
                        }
                    )

    df = pd.DataFrame(records, columns=["path", "class_label", "class_id", "is_augmented"])
    df.to_csv(spec_dir / "labels.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute log-Mel spectrograms.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Write augmented copies (Gaussian noise, time shift, SpecAugment).",
    )
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    spec_dir = args.data_dir / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)
    save_spectrograms(processed_dir, spec_dir, augment_train=args.augment)


if __name__ == "__main__":
    main()
