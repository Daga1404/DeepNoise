"""Mel-spectrogram computation and saving."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from preprocess import CLASS_LABELS, SAMPLE_RATE

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def compute_melspectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_S = np.log(S + 1e-6)
    return log_S.astype(np.float32)


def save_spectrograms(processed_dir: Path, spectrogram_dir: Path) -> None:
    records = []
    for class_id, class_label in enumerate(CLASS_LABELS):
        src_class_dir = processed_dir / class_label
        dst_class_dir = spectrogram_dir / class_label
        if not src_class_dir.exists():
            continue
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        for wav_path in sorted(src_class_dir.glob("*.wav")):
            import soundfile as sf
            audio, _ = sf.read(wav_path)
            spec = compute_melspectrogram(audio)
            out_path = dst_class_dir / (wav_path.stem + ".npy")
            np.save(out_path, spec)
            records.append({"path": str(out_path), "class_label": class_label, "class_id": class_id})
            print(f"saved {out_path}")

    labels_df = pd.DataFrame(records, columns=["path", "class_label", "class_id"])
    labels_df.to_csv(spectrogram_dir / "labels.csv", index=False)
    print(f"wrote labels.csv with {len(labels_df)} entries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Mel-spectrograms from processed audio.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory")
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    spectrogram_dir = args.data_dir / "spectrograms"
    spectrogram_dir.mkdir(parents=True, exist_ok=True)
    save_spectrograms(processed_dir, spectrogram_dir)


if __name__ == "__main__":
    main()
