"""Audio loading, normalization, and segmentation for the acoustic classifier."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.config import DURATION, SAMPLE_RATE


def load_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load a WAV file as a mono float32 array resampled to the target sample rate.

    Args:
        path: Path to the audio file.
        sr: Target sample rate in Hz.

    Returns:
        Mono float32 array of shape (N,).
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def normalize(audio: np.ndarray) -> np.ndarray:
    """
    Apply peak normalization so the maximum absolute value equals 1.0.

    All-zero inputs are returned unchanged to avoid division by zero.

    Args:
        audio: Float32 mono audio array.

    Returns:
        Peak-normalized array with the same shape.
    """
    peak = np.max(np.abs(audio))
    if peak == 0.0:
        return audio
    return audio / peak


def segment(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
) -> np.ndarray:
    """
    Crop or pad an audio array to exactly ``duration`` seconds.

    Too-long clips are center-cropped; too-short clips receive symmetric
    zero-padding. Both operations preserve the temporal midpoint.

    Args:
        audio: Float32 mono audio array.
        sr: Sample rate in Hz.
        duration: Target duration in seconds.

    Returns:
        Float32 array of length ``int(sr * duration)``.
    """
    target = int(sr * duration)
    length = len(audio)

    if length > target:
        start = (length - target) // 2
        return audio[start : start + target]

    if length < target:
        pad_total = target - length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(audio, (pad_left, pad_right), mode="constant")

    return audio


def process_directory(
    raw_dir: Path,
    out_dir: Path,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
) -> None:
    """
    Preprocess every WAV file under ``raw_dir``, mirroring the class-subfolder
    structure into ``out_dir``.

    Each file is loaded, peak-normalized, and center-cropped/padded to exactly
    ``duration`` seconds, then written as a 16-bit WAV.

    Args:
        raw_dir: Root directory containing one subfolder per class.
        out_dir: Destination root; class subfolders are created automatically.
        sr: Target sample rate in Hz.
        duration: Target clip duration in seconds.
    """
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        dest_class_dir = out_dir / class_dir.name
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        for wav_path in sorted(class_dir.glob("*.wav")):
            audio = load_audio(wav_path, sr=sr)
            audio = normalize(audio)
            audio = segment(audio, sr=sr, duration=duration)
            assert not np.any(np.isnan(audio)), f"NaN in output for {wav_path}"
            out_path = dest_class_dir / wav_path.name
            sf.write(out_path, audio, sr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw audio files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory containing raw/ subfolder.",
    )
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    out_dir = args.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    process_directory(raw_dir, out_dir)


if __name__ == "__main__":
    main()
