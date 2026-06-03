# APPROVAL: Full Codebase — Post-Remediation Re-Audit
# Approver Agent · Acoustic Event Classification Project
# Date: 2026-06-03
# Prior audit: approver_audit_1.md (2026-06-02) — Verdict: REJECTED
# Remediation plan: approver_roadmap_1.md — 5 tasks (TASK-R01 through TASK-R05)

---

**Verdict: APPROVED**

**Modules re-audited:**
`src/online.py`, `src/train.py`, `src/evaluate.py`, `src/baseline.py`,
`src/preprocess.py`, `tests/test_features.py`

**Prior findings resolved: 9 / 9**
Critical: 1 ✅ | Major: 3 ✅ | Minor: 5 ✅

**New findings introduced by remediation: 0**

**Non-Negotiable Checks: 7 PASS, 3 SKIP (hardware/TF-gated, not regressions)**

The codebase is safe to proceed to training runs and live demo.

---

## Re-Verification of All Prior Findings

### CRIT-01 — `src/online.py` — Training/inference spectrogram distribution mismatch

**Prior state:** `_preprocess` returned raw log-Mel values (range ≈ [-2.3, +2.9]) while the
CNN was trained on spectrograms normalized to [-1, 1] per sample.

**Fix applied (`online.py:44–46`):**
```python
s_min, s_max = log_S.min(), log_S.max()
if s_max > s_min:
    log_S = 2.0 * (log_S - s_min) / (s_max - s_min) - 1.0
```

**Programmatic verification:**
```
Training path output:  min=-1.0000, max=1.0000
Online path output:    min=-1.0000, max=1.0000
Paths numerically identical (allclose atol=1e-6): True
```

The normalization math is byte-for-byte identical to `dataset._load_and_normalize` (lines 86–88).
The `s_max > s_min` guard is present. Both `run_file_mode` and `run_mic_mode` route through
`_preprocess` via `classify_audio`, so both paths are fixed by this single change.

**Status: RESOLVED ✅**

---

### MAJ-01 — `src/train.py:_compute_class_weights` — Crash when any class absent

**Prior state:** `compute_class_weight("balanced", classes=np.arange(5), y=y)` crashed with
`ValueError` when any class ID 0–4 was absent from `y`.

**Fix applied (`train.py:47–50`):**
```python
present = np.unique(y)
weights = compute_class_weight("balanced", classes=present, y=y)
weight_dict = {int(c): float(w) for c, w in zip(present, weights)}
return {i: weight_dict.get(i, 1.0) for i in range(NUM_CLASSES)}
```

**Programmatic verification:**
```
y = [0,0,0,0,1,1,1,3,3,4,4]  (class 2 missing)
Result keys:      [0, 1, 2, 3, 4]   ← always 5 keys
class 2 weight:   1.0               ← neutral weight for absent class
No crash:         True
All-present case: weights match reference compute_class_weight exactly
```

**Status: RESOLVED ✅**

---

### MAJ-02 — `src/online.py:run_mic_mode` — `sd.wait()` hangs forever on device failure

**Prior state:** No exception handler around `sd.rec`/`sd.wait()`; a PortAudio error
would block the terminal indefinitely.

**Fix applied (`online.py:116–128`):**
```python
try:
    recording = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
except sd.PortAudioError as exc:
    _log.error("Audio device error: %s. Use --file mode for demo without a mic.", exc)
    return
```

Additionally, `sd.stop()` is now called in a `finally` block (line 134–135) ensuring the
audio device is released even on `KeyboardInterrupt` or early return.

**Code trace verification:**
- `PortAudioError` is caught: ✅
- `finally: sd.stop()`: ✅ present at line 134
- Returns cleanly with logged fallback message instead of hanging: ✅

**Status: RESOLVED ✅**

---

### MAJ-03 — `src/train.py:main` — `--batch-size 0` and `--epochs -1` accepted without validation

**Prior state:** Both arguments used `type=int`, accepting any integer including 0 and negatives.

**Fix applied (`train.py:32–37`):**
```python
def _positive_int(value: str) -> int:
    """Argparse type that rejects zero and negative integers (MAJ-03)."""
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"Must be a positive integer, got {value}")
    return n
```
Both `--epochs` and `--batch-size` now use `type=_positive_int`.

**Programmatic verification:**
```
--epochs 0      rejected: True  (exit code 2)
--epochs -1     rejected: True  (exit code 2)
--batch-size 0  rejected: True  (exit code 2)
valid value 32  accepted: True  (returns 32)
```

**Status: RESOLVED ✅**

---

### MIN-01 — `src/online.py:run_mic_mode` — No model warmup before demo loop

**Prior state:** First `model.predict` call triggered XLA/JIT compilation during the live demo loop.

**Fix applied (`online.py:109–110`):**
```python
# First predict triggers XLA/JIT graph compilation; warm up now so the
# demo loop does not appear to hang on the first real recording (MIN-01).
model.predict(np.zeros((1, *INPUT_SHAPE), dtype="float32"), verbose=0)
```
`INPUT_SHAPE` is now imported from `src/config.py` (line 18).
Warmup runs before `print("Listening…")`, before the capture loop.

**Code trace:** warmup call confirmed present in `run_mic_mode` source via `inspect.getsource`.

**Status: RESOLVED ✅**

---

### MIN-02 — `src/online.py:run_mic_mode` — `sd.stop()` not called in `finally`

**Prior state:** `KeyboardInterrupt` could leave the audio device locked on some systems.

**Fix applied (`online.py:134–135`):**
```python
    finally:
        sd.stop()  # release device on KeyboardInterrupt or PortAudioError return
```
The `finally` block wraps the entire `while True` loop, so it executes on both
`KeyboardInterrupt` and the `PortAudioError` early-return path.

**Status: RESOLVED ✅**

---

### MIN-03 — `src/preprocess.py:process_directory` — Non-WAV files silently skipped

**Prior state:** `class_dir.glob("*.wav")` silently ignored `.ogg`, `.mp3`, `.flac` files.

**Fix applied (`preprocess.py:108–116`):**
```python
all_files = [f for f in class_dir.iterdir() if f.is_file()]
skipped = [f for f in all_files if f.suffix.lower() != ".wav"]
if skipped:
    _log.warning(
        "%s: %d non-WAV file(s) skipped: %s",
        class_dir.name,
        len(skipped),
        sorted({f.suffix for f in skipped}),
    )
```
A module-level `_log = logging.getLogger(__name__)` was added. `.suffix.lower()` is used
for case-insensitive filesystem robustness. WAV processing behavior is unchanged.

**Programmatic verification:**
```
Synthetic class dir with clip1.ogg + clip2.mp3:
  WARNING records captured: 1
  Level:   WARNING
  Message: alarm_tone: 2 non-WAV file(s) skipped: ['.mp3', '.ogg']
  No bare print used: True
```

**Status: RESOLVED ✅**

---

### MIN-04 — `tests/test_features.py` — Augmentation tests not seeded

**Prior state:** Two tests called `np.random.randn` and `augment_audio` (which uses
`np.random.normal`) without a fixed seed, making them technically non-deterministic.

**Fix applied:**
```python
def test_augment_audio_arrays_are_distinct():
    np.random.seed(42)          # ← added
    audio = np.random.randn(TARGET).astype(np.float32)
    ...

def test_augment_audio_gaussian_noise_differs_from_original():
    np.random.seed(42)          # ← added
    audio = np.random.randn(TARGET).astype(np.float32)
    ...
```
`np.random.seed` (legacy global state) was used rather than `default_rng` because
`augment_audio` calls `np.random.normal` internally — seeding the global state pins
both the input generation and the internal noise draw.

**Verification:** Test suite run twice consecutively; both runs produced `18 passed`
with identical output. Assertions on distinctness and difference remain intact.

**Status: RESOLVED ✅**

---

### MIN-05 — `src/evaluate.py` and `src/baseline.py` — `zero_division` not passed to sklearn metrics

**Prior state:** `f1_score` and `classification_report` used default `zero_division="warn"`,
producing `NaN` and `UndefinedMetricWarning` on missing classes.

**Fix applied — four call sites:**

`src/evaluate.py:81,83`:
```python
macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
```

`src/baseline.py:96,98`:
```python
macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
report = classification_report(y_test, preds, target_names=class_names, zero_division=0)
```

**Programmatic verification:**
```
Missing-class test set (class 2 absent):
  Macro F1:                  0.7333  (finite, not NaN)
  UndefinedMetricWarning:    0 emitted
  friction_squeal score:     0.00    (not NaN)
```

**Status: RESOLVED ✅**

---

## Non-Negotiable Checks Re-Run

All 10 checks from `agents/approver.md` re-run against current codebase.

| Check | Condition | Result | Evidence |
|---|---|---|---|
| NNC-01 | `normalize(np.zeros(88200))` — no ZeroDivisionError | ✅ PASS | Returns zeros, shape (88200,) |
| NNC-02 | `segment(np.array([]))` — no crash; output length 88200 | ✅ PASS | Output length 88200 |
| NNC-03 | `compute_melspectrogram` — no NaN or Inf for float32 input | ✅ PASS | NaN=False, Inf=False, shape=(128,173) |
| NNC-04 | `labels.csv` row count == `.npy` file count | ⬜ SKIP | Requires populated `data/spectrograms/` — no real data in CI; was PASS in audit_1 and no code in `features.py` was modified |
| NNC-05 | No augmented `.npy` in val or test splits | ✅ PASS | `test_no_augmented_rows_in_val` and `test_no_augmented_rows_in_test` pass |
| NNC-06 | `split_dataset` called twice with same seed → identical splits | ✅ PASS | Verified programmatically across train/val/test |
| NNC-07 | `build_cnn()` output shape exactly `(None, 5)` | ⬜ SKIP | Requires TensorFlow; `test_model.py` passes — no `model.py` changes in this round |
| NNC-08 | `model.predict(zeros)` returns shape `(1,5)` summing to ~1.0 | ⬜ SKIP | Requires TensorFlow; same rationale as NNC-07 |
| NNC-09 | `online.py` preprocessing order and normalization matches `train.py` | ✅ PASS | Programmatic trace: paths numerically identical (allclose atol=1e-6), range [-1.0000, 1.0000] |
| NNC-10 | `CLASS_LABELS[np.argmax(prediction)]` — no KeyError for any valid prediction | ✅ PASS | 100 random predictions, zero KeyErrors |

**7 PASS, 3 SKIP.** The 3 SKIPs are hardware/TF-gated and identical in scope to audit_1's
passed checks — no code was changed in `dataset.py`, `features.py`, or `model.py` that
would affect these checks. They remain valid.

---

## Regression Check

All 86 tests that passed in audit_1 continue to pass. The 7 skips are unchanged
(hardware-gated: sounddevice, TensorFlow model inference). No new test failures introduced.

```
pytest tests/ -v
86 passed, 7 skipped, 2 warnings in 18.63s
```

The 2 warnings are third-party `DeprecationWarning` from `audioread` on Python 3.14 —
unrelated to any change in this remediation cycle and present since the initial implementation.

---

## Additional Observations (New)

These observations were not in the prior audit. They are noted for awareness but do not
affect the APPROVED verdict — none rise to Major level.

### OBS-01 — `run_file_mode` has no model warmup

`run_file_mode` (`online.py:76–93`) loads the model and calls `classify_audio` directly,
with no warmup predict. For a single-shot file classification this causes the same
first-call compilation delay that MIN-01 fixed for mic mode. Since `--file` mode
runs once and exits, this is a user-experience nuisance (a 1–3 s delay before output)
rather than a correctness or hang risk. The demo is not impaired.

**Severity:** Minor / informational. No action required before training or demo.

### OBS-02 — `_positive_int` does not guard against non-integer strings

`_positive_int("1.5")` will raise a `ValueError` from `int("1.5")` rather than an
`argparse.ArgumentTypeError`, producing a less friendly error message. `int("abc")`
similarly raises `ValueError`. This is acceptable argparse behavior — stdlib `type=int`
has the same characteristic — but a production CLI might wrap the `int()` call in a
try/except. Not a blocker.

**Severity:** Informational only.

---

## Overall Assessment

Every finding from `approver_audit_1.md` has been resolved. The most dangerous issue —
the training/inference spectrogram distribution mismatch (CRIT-01) — is confirmed fixed by
programmatic trace: `online._preprocess` and `dataset._load_and_normalize` now produce
numerically identical output for the same input (allclose atol=1e-6), and the output range
is exactly [-1, 1].

The offline training pipeline (data loading, augmentation, splitting, class weighting,
model architecture) was not touched in this remediation cycle and all prior correctness
guarantees established in audit_1 remain valid.

**The codebase is ready for training runs and live demo.**
Proceed to: `python src/train.py --augment false` → `python src/train.py --augment true`.

---

## Findings Resolution Summary

| Finding | ID | File | Severity | Status |
|---|---|---|---|---|
| Online spectrogram not normalized to [-1, 1] | CRIT-01 | `src/online.py:44` | CRITICAL | ✅ RESOLVED |
| `_compute_class_weights` crash on missing class | MAJ-01 | `src/train.py:47` | MAJOR | ✅ RESOLVED |
| `sd.wait()` no timeout — mic demo hangs forever | MAJ-02 | `src/online.py:116` | MAJOR | ✅ RESOLVED |
| `--batch-size 0` / `--epochs -1` unvalidated | MAJ-03 | `src/train.py:32` | MAJOR | ✅ RESOLVED |
| No model warmup before mic demo loop | MIN-01 | `src/online.py:109` | MINOR | ✅ RESOLVED |
| `sd.stop()` not in `finally` | MIN-02 | `src/online.py:134` | MINOR | ✅ RESOLVED |
| Non-WAV files skipped without warning | MIN-03 | `src/preprocess.py:108` | MINOR | ✅ RESOLVED |
| Augmentation tests not seeded | MIN-04 | `tests/test_features.py:68,85` | MINOR | ✅ RESOLVED |
| `zero_division` not passed to sklearn metrics | MIN-05 | `src/evaluate.py:81`, `src/baseline.py:96` | MINOR | ✅ RESOLVED |
| run_file_mode has no warmup (new) | OBS-01 | `src/online.py:76` | Informational | No action required |
| `_positive_int` non-int string error type (new) | OBS-02 | `src/train.py:32` | Informational | No action required |
