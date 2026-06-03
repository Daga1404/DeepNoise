# APPROVAL: Full Codebase — Pre-Submission Adversarial Audit
# Approver Agent · Acoustic Event Classification Project
# Date: 2026-06-02

---

**Verdict: REJECTED**

**Modules audited:**
`src/config.py`, `src/preprocess.py`, `src/features.py`, `src/dataset.py`,
`src/baseline.py`, `src/model.py`, `src/train.py`, `src/evaluate.py`, `src/online.py`,
`tests/` (all test files), `notebooks/` (all notebooks)

**Critical findings: 1 — Major findings: 3 — Minor findings: 5**

A REJECTED verdict means the Planner must route Critical findings back to the Coder
before any training run or live demo. Running inference on REJECTED code produces
silently wrong predictions that look plausible.

---

## Critical Findings

### CRIT-01 — `src/online.py:35–38` — Training/inference spectrogram distribution mismatch

**Category:** Silent Correctness Failure (Category A)

**Trigger condition:** Any call to `classify_audio` or `run_file_mode` / `run_mic_mode`
after a model trained with `make_tf_dataset`.

**Failure mode:** The model was trained on spectrograms normalized to **[-1, 1]** per sample
(via `dataset._load_and_normalize`). The online system feeds the model raw log-Mel values
with a typical range of **[-2.3, +2.9]**. The CNN's first-layer weights were tuned for [-1, 1]
inputs. Inference on out-of-distribution inputs produces wrong class predictions on every
single call — silently, with plausible-looking confidence scores.

**Evidence — training path (what the CNN learns from):**

`dataset.py:81–88`
```python
def _load_and_normalize(path: str) -> np.ndarray:
    spec = np.load(path).astype(np.float32)
    spec = spec[..., np.newaxis]
    s_min, s_max = spec.min(), spec.max()
    if s_max > s_min:
        spec = 2.0 * (spec - s_min) / (s_max - s_min) - 1.0  # ← scales to [-1, 1]
    return spec
```

**Evidence — online inference path (what the CNN receives):**

`online.py:35–38`
```python
def _preprocess(audio: np.ndarray) -> np.ndarray:
    audio = normalize(audio)
    audio = segment(audio, sr=SAMPLE_RATE, duration=DURATION)
    log_S = compute_melspectrogram(audio)
    return log_S[np.newaxis, ..., np.newaxis]   # ← NO spectrogram normalization
```

**Programmatic confirmation:**
```
Raw log-Mel range (online sends):     [-2.30, 2.91]
After dataset normalization (trained): [-1.00, 1.00]
MISMATCH: True
```

**Fix required:**
Add per-sample min-max spectrogram normalization to `_preprocess` in `online.py`,
identical to `_load_and_normalize` in `dataset.py`:

```python
log_S = compute_melspectrogram(audio)
s_min, s_max = log_S.min(), log_S.max()
if s_max > s_min:
    log_S = 2.0 * (log_S - s_min) / (s_max - s_min) - 1.0
return log_S[np.newaxis, ..., np.newaxis].astype(np.float32)
```

This must be applied in `_preprocess` so both `run_file_mode` and `run_mic_mode` are fixed
by a single change.

---

## Major Findings

### MAJ-01 — `src/train.py:_compute_class_weights` — Crash when any class is absent from training split

**Category:** Crash Failure (Category B)

**Trigger condition:** The training DataFrame contains fewer than all 5 classes. This can
happen during development with small synthetic datasets, or if a class has fewer samples
than the stratified split requires.

**Failure mode:** `compute_class_weight("balanced", classes=np.arange(5), y=y)` raises
`ValueError: classes should have valid labels that are in y` if any class ID in `0–4`
is absent from the training labels `y`.

**Programmatic confirmation:**
```
y = [0,0,0,0,1,1,1,3,3,4,4]  # class 2 missing
compute_class_weight("balanced", classes=np.arange(5), y=y)
→ CRASH: classes should have valid labels that are in y
```

**Evidence:** `src/train.py`
```python
def _compute_class_weights(train_df) -> dict[int, float]:
    y = train_df["class_id"].values
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),  # ← always [0,1,2,3,4]
        y=y,                              # ← may not contain all 5 classes
    )
```

**Fix required:**
Filter `classes` to only those present in `y`, or use a `try/except` to fall back
to equal weights if a class is missing:

```python
present = np.unique(y)
weights = compute_class_weight("balanced", classes=present, y=y)
weight_dict = {int(c): float(w) for c, w in zip(present, weights)}
# fill missing classes with 1.0 (no up-weighting)
return {i: weight_dict.get(i, 1.0) for i in range(NUM_CLASSES)}
```

---

### MAJ-02 — `src/online.py:run_mic_mode` — `sd.wait()` has no timeout; demo will hang forever

**Category:** Crash Failure (Category B)

**Trigger condition:** The microphone is unavailable, disconnected, or returns a device
error on the demo machine.

**Failure mode:** `sounddevice.wait()` blocks indefinitely. The terminal appears frozen.
Ctrl+C is the only escape, and the demo is ruined.

**Evidence:** `src/online.py`
```python
recording = sd.rec(
    n_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32"
)
sd.wait()   # ← no timeout parameter; blocks forever if device fails
```

**Programmatic confirmation:** `sd.wait with timeout: False`

**Fix required:**
Wrap recording in a try/except and/or use the `sounddevice` stream API with a timeout.
At minimum, add a fallback message:

```python
try:
    recording = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
except sd.PortAudioError as exc:
    _log.error(f"Audio device error: {exc}. Use --file mode for demo without a mic.")
    return
```

---

### MAJ-03 — `src/train.py:main` — `--batch-size 0` and `--epochs -1` accepted without validation

**Category:** Crash Failure (Category B)

**Trigger condition:** User passes `--batch-size 0` (Keras crashes with a division error)
or `--epochs -1` (Keras silently skips training, producing an empty history).

**Evidence:** `src/train.py`
```python
parser.add_argument("--epochs", type=int, default=50)     # accepts -1
parser.add_argument("--batch-size", type=int, default=32) # accepts 0
```

**Programmatic confirmation:**
```
epochs -1 accepted:   -1
batch-size 0 accepted: 0
```

**Fix required:**
Add a `type` helper that rejects non-positive values:

```python
def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"Must be a positive integer, got {value}")
    return n

parser.add_argument("--epochs", type=_positive_int, default=50)
parser.add_argument("--batch-size", type=_positive_int, default=32)
```

---

## Minor Findings

### MIN-01 — `src/online.py:run_mic_mode` — No model warmup before demo loop

**Observation:** The first call to `model.predict` after loading triggers TensorFlow graph
compilation (XLA/JIT), causing a multi-second delay that looks like a hang during the demo.

**Suggestion:**
Add a warmup call immediately after `model = keras.models.load_model(model_path)`:
```python
import numpy as np
from src.config import INPUT_SHAPE
model.predict(np.zeros((1, *INPUT_SHAPE), dtype="float32"), verbose=0)
```

---

### MIN-02 — `src/online.py:run_mic_mode` — `sd.stop()` not called in `finally`

**Observation:** On some systems (particularly Linux with ALSA), a `KeyboardInterrupt`
that interrupts `sd.wait()` may leave the audio device locked. The current code handles
`KeyboardInterrupt` but has no `finally` block to call `sd.stop()`.

**Suggestion:**
```python
try:
    while True:
        ...
except KeyboardInterrupt:
    _log.info("\nStopped.")
finally:
    sd.stop()
```

---

### MIN-03 — `src/preprocess.py:process_directory` — Non-WAV files silently skipped

**Observation:** `class_dir.glob("*.wav")` silently ignores `.ogg`, `.mp3`, and `.flac`
files. UrbanSound8K ships some clips in these formats. Users following `MANUAL_TASKS.md`
who forget to convert will get silently smaller datasets with no warning.

**Suggestion:** After processing, log the count of skipped non-WAV files:
```python
all_audio = list(class_dir.iterdir())
skipped = [f for f in all_audio if f.suffix != ".wav"]
if skipped:
    _log.warning(f"{class_label}: {len(skipped)} non-WAV files skipped: "
                 f"{[f.suffix for f in skipped]}")
```

---

### MIN-04 — `tests/test_features.py` — Augmentation tests not seeded

**Observation:** `test_augment_audio_arrays_are_distinct` and `test_augment_audio_gaussian_noise_differs_from_original`
call `np.random.randn` and `augment_audio` (which calls `np.random.normal`) without
setting a random seed. The tests are practically stable but are technically non-deterministic.

**Suggestion:** Add `np.random.seed(42)` or use a `rng = np.random.default_rng(42)` fixture
at the top of augmentation tests.

---

### MIN-05 — `src/evaluate.py` and `src/baseline.py` — `zero_division` not passed to sklearn metrics

**Observation:** Both `f1_score(y_true, y_pred, average="macro")` and
`classification_report(y_true, y_pred, ...)` use the default `zero_division="warn"`.
If any class has zero predictions or zero true samples in the test set,
`f1_score` returns NaN and `classification_report` emits a `UndefinedMetricWarning`.
The NaN then propagates silently into `cnn_report.txt` and `augmentation_comparison.md`.

**Suggestion:** Pass `zero_division=0` to both calls so missing classes get score 0.0
instead of NaN, with no noisy warnings:
```python
macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
```

---

## Passed Checks

These were verified programmatically or by exhaustive code trace. An audit with only
findings is not an audit.

| Check | Result | Evidence |
|---|---|---|
| `normalize(np.zeros(88200))` — no ZeroDivisionError | ✅ PASS | `peak == 0.0` guard; returns zeros unchanged |
| `segment(np.array([]))` — no crash; output length 88200 | ✅ PASS | `np.pad` handles zero-length array |
| `compute_melspectrogram` output — no NaN or Inf for float32 input | ✅ PASS | `np.log(mel_S + 1e-6)` epsilon prevents log(0); mel_S always ≥ 0 |
| `split_dataset(df, seed=42)` called twice → identical splits | ✅ PASS | `random_state=seed` in both `train_test_split` calls |
| `CLASS_LABELS[np.argmax(prediction)]` — no KeyError for any prediction | ✅ PASS | `argmax` on a (5,) array returns 0–4; all keys present |
| Augmented samples never appear in val or test | ✅ PASS | `split_dataset` splits only `non_aug` subset, appends `aug` to train only |
| No double log scaling | ✅ PASS | `compute_melspectrogram` applies log once; `_load_and_normalize` does min-max only |
| Class weights computed from training split, not full dataset | ✅ PASS | `_compute_class_weights(train_df)` — `train_df` only |
| Confusion matrix axes: rows=true, columns=predicted | ✅ PASS | `confusion_matrix(y_true, y_pred)` sklearn convention; `ax.set_ylabel("True")`, `ax.set_xlabel("Predicted")` |
| `model.predict` receives `argmax` output for F1 computation | ✅ PASS | `y_pred = np.argmax(batch_probs, axis=1)` before `f1_score` |
| Spectrogram normalization is per-sample, not global (no leakage) | ✅ PASS | `_load_and_normalize` uses each spectrogram's own `s_min/s_max` |
| `save_spectrograms` directory iteration is deterministic | ✅ PASS | Both loops use `sorted()` |
| Augmented `.npy` filenames don't collide with original filenames | ✅ PASS | Suffixes `_aug0`, `_aug1`, `_aug2`, `_specaug0` appended to stem |
| `output_dir.mkdir(parents=True, exist_ok=True)` in `train.py` | ✅ PASS | Line 68 of `run_training` |
| `pathlib.Path` throughout; no `os.path` joins | ✅ PASS | No `os.path` found in any `src/` file |
| Constants never redeclared outside `config.py` | ✅ PASS | No re-declaration of `SAMPLE_RATE`, `DURATION`, etc. in any module |
| `online.py` imports `normalize`, `segment`, `compute_melspectrogram` — no duplication | ✅ PASS | Lines 18–20 of `online.py` |
| `segment` center-crop, not head-crop | ✅ PASS | `start = (length - target) // 2` |
| `normalize` applies peak normalization before `compute_melspectrogram` in both training and online | ✅ PASS | Order: `normalize → segment → compute_melspectrogram` in both paths |
| `KeyboardInterrupt` caught in mic loop | ✅ PASS | `except KeyboardInterrupt: print("\nStopped.")` |
| `augment_audio` returns 3 distinct float32 arrays (post FIX-02) | ✅ PASS | `len=3`, `forward != backward` confirmed |
| `labels.csv` columns exactly: `path, class_label, class_id, is_augmented` | ✅ PASS | `pd.DataFrame(records, columns=[...])` with explicit column list |
| `split_dataset` no-leakage test in test suite | ✅ PASS | `test_no_augmented_rows_in_val` and `test_no_augmented_rows_in_test` pass |
| Spectrogram shape `(128, 173)` for 88200-sample clip | ✅ PASS | Test passes; librosa uses `center=True` by default giving 173 frames |

---

## Overall Assessment

The codebase is architecturally sound and passes all correctness checks for the offline
training pipeline. The no-leakage guarantee, log-scaling, constant sourcing, and augmentation
discipline are all correct.

**However, the online system (CRIT-01) contains a training/inference distribution mismatch
that will cause silent incorrect predictions on every live demo call.** The CNN is trained on
spectrograms normalized to [-1, 1] per sample; `online.py` feeds it raw log-Mel values
ranging from approximately [-3, +3]. This is not a rare edge case — it affects every inference.
The fix is a four-line addition to `_preprocess` in `online.py`.

The two highest-risk areas before the first training run are:
1. Fix CRIT-01 in `online.py` — without this, the demo produces wrong predictions.
2. Fix MAJ-01 in `train.py` (`_compute_class_weights` crash on missing class) and
   MAJ-02 in `online.py` (`sd.wait()` timeout) — both can be addressed in under 10 lines each.

Do not run the live demo against a trained model until CRIT-01 is resolved.

---

## Verdict Summary

| Finding | ID | File | Severity |
|---|---|---|---|
| Online spectrogram not normalized to [-1, 1] — training/inference mismatch | CRIT-01 | `src/online.py:35` | **CRITICAL** |
| `_compute_class_weights` crashes if any class absent from training | MAJ-01 | `src/train.py` | MAJOR |
| `sd.wait()` has no timeout — mic demo hangs forever | MAJ-02 | `src/online.py` | MAJOR |
| `--batch-size 0` / `--epochs -1` accepted without validation | MAJ-03 | `src/train.py` | MAJOR |
| No model warmup call before mic demo loop | MIN-01 | `src/online.py` | MINOR |
| `sd.stop()` not called in `finally` after Ctrl+C | MIN-02 | `src/online.py` | MINOR |
| Non-WAV files silently skipped, no warning logged | MIN-03 | `src/preprocess.py` | MINOR |
| Augmentation tests not seeded — technically non-deterministic | MIN-04 | `tests/test_features.py` | MINOR |
| `zero_division` not passed to sklearn metrics — NaN on missing class | MIN-05 | `src/evaluate.py`, `src/baseline.py` | MINOR |
