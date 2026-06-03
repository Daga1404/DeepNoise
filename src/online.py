"""Real-time acoustic event classification via microphone or WAV file.

Usage:
    # Continuous microphone mode (Ctrl+C to stop)
    python src/online.py --model models/aug/cnn_best.keras

    # File fallback — classifies one clip and exits
    python src/online.py --model models/aug/cnn_best.keras --file path/to/clip.wav
"""

import argparse
import datetime
import logging
from pathlib import Path

import numpy as np

from src.config import CLASS_LABELS, DURATION, INPUT_SHAPE, SAMPLE_RATE
from src.features import compute_melspectrogram
from src.preprocess import load_audio, normalize, segment

_log = logging.getLogger(__name__)


def _preprocess(audio: np.ndarray) -> np.ndarray:
    """
    Apply the shared inference pipeline: normalize → segment → log-Mel spectrogram.

    Returns a (1, 128, 173, 1) float32 tensor ready for ``model.predict``.
    No preprocessing logic is duplicated here — every step delegates to
    ``src.preprocess`` and ``src.features``.

    Args:
        audio: Raw mono float32 array of any length.

    Returns:
        4-D numpy array of shape ``(1, n_mels, T, 1)`` with values in ``[-1, 1]``.
    """
    audio = normalize(audio)
    audio = segment(audio, sr=SAMPLE_RATE, duration=DURATION)
    log_S = compute_melspectrogram(audio)
    # Mirror dataset._load_and_normalize: scale each spectrogram to [-1, 1] per sample
    # so inference sees the same distribution the CNN was trained on (CRIT-01).
    s_min, s_max = log_S.min(), log_S.max()
    if s_max > s_min:
        log_S = 2.0 * (log_S - s_min) / (s_max - s_min) - 1.0
    return log_S[np.newaxis, ..., np.newaxis].astype(np.float32)


def _format_result(class_name: str, confidence: float) -> str:
    """Return a single formatted output line including timestamp."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    return f"[{ts}] Predicted: {class_name:<20}  (confidence: {confidence:.2f})"


def classify_audio(model, audio: np.ndarray) -> tuple[str, float]:
    """
    Classify a raw audio array and return the predicted class and confidence.

    This is the core inference routine shared by both mic and file modes.

    Args:
        model: Loaded Keras model.
        audio: Raw mono float32 array (any length; will be normalised and padded).

    Returns:
        Tuple of ``(class_name, confidence)`` where ``confidence`` is the
        softmax probability of the predicted class.
    """
    tensor = _preprocess(audio)
    probs = model.predict(tensor, verbose=0)[0]
    class_id = int(np.argmax(probs))
    return CLASS_LABELS[class_id], float(probs[class_id])


def run_file_mode(model_path: Path, file_path: Path) -> tuple[str, float]:
    """
    Classify a single WAV file and print the result.

    Args:
        model_path: Path to the trained ``cnn_best.keras`` model.
        file_path: Path to the WAV file to classify.

    Returns:
        Tuple of ``(class_name, confidence)``.
    """
    from tensorflow import keras  # lazy — not available on Python 3.14

    model = keras.models.load_model(model_path)
    audio = load_audio(file_path, sr=SAMPLE_RATE)
    class_name, confidence = classify_audio(model, audio)
    print(_format_result(class_name, confidence))
    return class_name, confidence


def run_mic_mode(model_path: Path) -> None:
    """
    Continuously capture 4-second audio windows from the default microphone
    and classify each window until the user presses Ctrl+C.

    Args:
        model_path: Path to the trained ``cnn_best.keras`` model.
    """
    import sounddevice as sd  # lazy — requires hardware
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    # First predict triggers XLA/JIT graph compilation; warm up now so the
    # demo loop does not appear to hang on the first real recording (MIN-01).
    model.predict(np.zeros((1, *INPUT_SHAPE), dtype="float32"), verbose=0)

    n_samples = int(SAMPLE_RATE * DURATION)
    print("Listening… (Ctrl+C to stop)")
    try:
        while True:
            try:
                recording = sd.rec(
                    n_samples,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
            except sd.PortAudioError as exc:
                _log.error(
                    "Audio device error: %s. Use --file mode for demo without a mic.", exc
                )
                return
            audio = recording.flatten()
            class_name, confidence = classify_audio(model, audio)
            print(_format_result(class_name, confidence))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sd.stop()  # release device on KeyboardInterrupt or PortAudioError return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time acoustic event classification."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to cnn_best.keras (e.g. models/aug/cnn_best.keras).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        metavar="WAV",
        help="Classify a WAV file instead of the microphone and exit.",
    )
    args = parser.parse_args()

    if args.file is not None:
        run_file_mode(args.model, args.file)
    else:
        run_mic_mode(args.model)


if __name__ == "__main__":
    main()
