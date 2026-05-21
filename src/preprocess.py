"""Audio loading, resampling, normalization, and segmentation."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050
DURATION = 4.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)

CLASS_LABELS = [
    "normal_operation",
    "metallic_impact",
    "friction_squeal",
    "alarm_tone",
    "silence_ambient",
]


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=None, mono=True)
    return audio, sr


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio / peak


def segment(audio: np.ndarray, target_samples: int = TARGET_SAMPLES) -> np.ndarray:
    if len(audio) >= target_samples:
        return audio[:target_samples]
    pad_width = target_samples - len(audio)
    return np.pad(audio, (0, pad_width), mode="constant")


def process_file(src: Path) -> np.ndarray:
    audio, sr = load_audio(src)
    audio = resample(audio, sr)
    audio = normalize(audio)
    audio = segment(audio)
    return audio


def process_dataset(raw_dir: Path, processed_dir: Path) -> None:
    for class_label in CLASS_LABELS:
        src_class_dir = raw_dir / class_label
        dst_class_dir = processed_dir / class_label
        if not src_class_dir.exists():
            continue
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        for audio_path in sorted(src_class_dir.glob("*.wav")):
            audio = process_file(audio_path)
            out_path = dst_class_dir / audio_path.name
            sf.write(out_path, audio, SAMPLE_RATE)
            print(f"processed {audio_path.name} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw audio files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    processed_dir = args.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    process_dataset(raw_dir, processed_dir)


if __name__ == "__main__":
    main()
