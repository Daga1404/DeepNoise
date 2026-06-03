# Approver Roadmap 1 — Remediation Workplan

**Planner agent · Acoustic Event Classification Project**
**Source audit:** `approver_audit_1.md` (Approver, 2026-06-02) — Verdict: **REJECTED**
**Date issued:** 2026-06-03

---

## Purpose

The Approver returned **REJECTED**: 1 Critical, 3 Major, 5 Minor findings.
Per the workflow in `CLAUDE.md`, a REJECTED verdict means **all Critical findings must be
routed to the Coder and fixed before any training run or live demo**. This roadmap
decomposes every finding into Coder task cards, sequences them, and defines the gate
conditions for re-approval.

**Hard gate:** No training run and no live demo may proceed until **TASK-R01 (CRIT-01)** is
DONE and the Approver has re-audited `online.py`.

---

## Severity → Action Mapping

| Severity | Policy | Blocks training/demo? |
|---|---|---|
| Critical | Fix immediately, re-audit before any pipeline run | **YES** |
| Major | Fix before submission | Demo: MAJ-02 yes; training: MAJ-01/03 yes |
| Minor | Fix if time allows; bundle with same-file Major work | No |

---

## Execution Order (grouped by file to minimize churn)

The audit touches 5 source files + 1 test file. To avoid re-opening the same file
repeatedly, tasks are batched **by file**, ordered by the highest severity in each batch.

```
TASK-R01  online.py     CRIT-01 + MAJ-02 + MIN-01 + MIN-02   ← Critical: do first
TASK-R02  train.py      MAJ-01  + MAJ-03                      ← Major: blocks training
TASK-R03  evaluate.py / baseline.py   MIN-05                  ← Minor: metric robustness
TASK-R04  preprocess.py MIN-03                                ← Minor: skip warning
TASK-R05  test_features.py  MIN-04                            ← Minor: seed tests
```

**Dependency note:** All five tasks are independent (different files, no shared interfaces
changed). They may be implemented in any order, but R01 is the release-gating Critical and
must be delivered, reviewed, and re-audited first.

---

## Task Board

| Task | File(s) | Findings | Severity | Status | Depends on |
|---|---|---|---|---|---|
| TASK-R01 | `src/online.py` | CRIT-01, MAJ-02, MIN-01, MIN-02 | CRITICAL | TODO | none |
| TASK-R02 | `src/train.py` | MAJ-01, MAJ-03 | MAJOR | TODO | none |
| TASK-R03 | `src/evaluate.py`, `src/baseline.py` | MIN-05 | MINOR | TODO | none |
| TASK-R04 | `src/preprocess.py` | MIN-03 | MINOR | TODO | none |
| TASK-R05 | `tests/test_features.py` | MIN-04 | MINOR | TODO | none |

---

## TASK-R01: `online.py` — fix inference distribution mismatch + mic robustness

**Status:** TODO
**Depends on:** none
**Assigned to:** Coder
**Findings:** CRIT-01 (Critical), MAJ-02 (Major), MIN-01 (Minor), MIN-02 (Minor)

### What to build
Harden `src/online.py` so that live inference is (a) numerically consistent with training
and (b) survives a missing/failed microphone during the demo. Four changes, all isolated
to this file.

1. **CRIT-01 — spectrogram normalization.** Add per-sample min-max normalization to
   `[-1, 1]` inside `_preprocess`, identical to `dataset._load_and_normalize`
   (`dataset.py:82–89`). Because both `run_file_mode` and `run_mic_mode` route through
   `_preprocess` (via `classify_audio`), this single change fixes both paths.
2. **MAJ-02 — mic timeout/guard.** Wrap `sd.rec` + `sd.wait()` in `run_mic_mode` in a
   `try/except sd.PortAudioError`, log a clear fallback message pointing to `--file` mode,
   and return instead of hanging.
3. **MIN-01 — model warmup.** After `keras.models.load_model(...)` in `run_mic_mode`, issue
   one warmup `model.predict(np.zeros((1, *INPUT_SHAPE), dtype="float32"), verbose=0)` so the
   first real prediction does not appear to hang during graph compilation.
4. **MIN-02 — device cleanup.** Add a `finally:` block to the mic capture loop that calls
   `sd.stop()` so a `KeyboardInterrupt` cannot leave the audio device locked.

### Acceptance criteria
- [ ] `_preprocess` normalizes `log_S` to `[-1, 1]` per sample using its own `min`/`max`,
      with the `s_max > s_min` guard, **before** adding batch/channel dims; returns float32.
- [ ] Normalization math is byte-for-byte equivalent to `dataset._load_and_normalize`
      (`2.0 * (x - min) / (max - min) - 1.0`).
- [ ] A spectrogram fed through `online._preprocess` and through `dataset._load_and_normalize`
      (same input) produces identical arrays (add a test or assert this manually).
- [ ] `run_mic_mode` catches `sd.PortAudioError`, logs the `--file` fallback hint, and returns
      cleanly (no traceback, no hang).
- [ ] `run_mic_mode` performs exactly one warmup predict before the loop.
- [ ] `run_mic_mode` calls `sd.stop()` in a `finally` block.
- [ ] `--file` mode still works without a microphone:
      `python src/online.py --model models/aug/cnn_best.keras --file tests/fixtures/sample.wav`.
- [ ] No preprocessing logic is duplicated — `normalize`, `segment`, `compute_melspectrogram`
      remain imported (CLAUDE.md online-system rule preserved).

### Files to create or modify
- `src/online.py`

### Notes for Coder
- `INPUT_SHAPE` and `CLASS_LABELS` come from `src/config.py` — import `INPUT_SHAPE`
  (currently not imported) for the warmup call.
- Reference fix from the audit (CRIT-01):
  ```python
  log_S = compute_melspectrogram(audio)
  s_min, s_max = log_S.min(), log_S.max()
  if s_max > s_min:
      log_S = 2.0 * (log_S - s_min) / (s_max - s_min) - 1.0
  return log_S[np.newaxis, ..., np.newaxis].astype(np.float32)
  ```
- This is the **release-gating** task. It must be DONE and re-audited by the Approver
  before any `train.py`/`online.py` run against a real model.

---

## TASK-R02: `train.py` — class-weight robustness + CLI argument validation

**Status:** TODO
**Depends on:** none
**Assigned to:** Coder
**Findings:** MAJ-01 (Major), MAJ-03 (Major)

### What to build
Two independent hardening changes to `src/train.py`.

1. **MAJ-01 — missing-class class weights.** `_compute_class_weights` currently calls
   `compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y)`, which raises
   `ValueError` whenever any class 0–4 is absent from the training split (common with small
   synthetic datasets). Compute weights only over classes present in `y`, then backfill any
   missing class with weight `1.0`.
2. **MAJ-03 — positive-int CLI validation.** `--epochs` and `--batch-size` currently accept
   `0` and negative values, which causes a Keras division error (`--batch-size 0`) or silently
   empty training (`--epochs -1`). Add a `_positive_int` argparse type that rejects values `<= 0`.

### Acceptance criteria
- [ ] `_compute_class_weights` returns a dict with **all** keys `0..NUM_CLASSES-1`, never
      raising when a class is absent; missing classes map to `1.0`.
- [ ] Verified: `y = [0,0,0,0,1,1,1,3,3,4,4]` (class 2 missing) returns a 5-key dict, no crash.
- [ ] Present-class weights still match `compute_class_weight("balanced", ...)` over the
      present subset (no change to balanced behavior when all classes are present).
- [ ] `--epochs 0`, `--epochs -1`, `--batch-size 0` are rejected by argparse with a clear
      `ArgumentTypeError`; valid positive values still accepted.
- [ ] Smoke test still passes:
      `python src/train.py --epochs 2 --augment false --output-dir /tmp/smoke` (< 60 s).

### Files to create or modify
- `src/train.py`

### Notes for Coder
- Reference fixes from the audit:
  ```python
  present = np.unique(y)
  weights = compute_class_weight("balanced", classes=present, y=y)
  weight_dict = {int(c): float(w) for c, w in zip(present, weights)}
  return {i: weight_dict.get(i, 1.0) for i in range(NUM_CLASSES)}
  ```
  ```python
  def _positive_int(value: str) -> int:
      n = int(value)
      if n <= 0:
          raise argparse.ArgumentTypeError(f"Must be a positive integer, got {value}")
      return n
  parser.add_argument("--epochs", type=_positive_int, default=50)
  parser.add_argument("--batch-size", type=_positive_int, default=32)
  ```
- Both Major findings block the training runs (TASK-07 in the main board), so this task
  should land before the noaug/aug experiment is executed.

---

## TASK-R03: `evaluate.py` + `baseline.py` — `zero_division=0` on sklearn metrics

**Status:** TODO
**Depends on:** none
**Assigned to:** Coder
**Findings:** MIN-05 (Minor)

### What to build
Pass `zero_division=0` to every `f1_score` and `classification_report` call so a class with
zero predictions or zero true samples yields `0.0` instead of `NaN` (which would propagate
silently into `cnn_report.txt` and `augmentation_comparison.md`).

Call sites confirmed:
- `src/baseline.py:96` `f1_score(...)`, `:98` `classification_report(...)`
- `src/evaluate.py:81` `f1_score(...)`, `:83` `classification_report(...)`

### Acceptance criteria
- [ ] All four calls pass `zero_division=0`.
- [ ] Macro F1 is a finite float (never `NaN`) even when a class is missing from the test set.
- [ ] No `UndefinedMetricWarning` emitted during evaluation.
- [ ] `pytest tests/test_evaluate.py` and `tests/test_baseline.py` still pass.

### Files to create or modify
- `src/evaluate.py`
- `src/baseline.py`

### Notes for Coder
```python
macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
```

---

## TASK-R04: `preprocess.py` — warn on skipped non-WAV files

**Status:** TODO
**Depends on:** none
**Assigned to:** Coder
**Findings:** MIN-03 (Minor)

### What to build
`process_directory` globs only `*.wav` (`preprocess.py:105`), silently ignoring `.ogg`,
`.mp3`, `.flac` clips that ship with sources like UrbanSound8K. Add a per-class `WARNING` log
listing the count and extensions of skipped non-WAV files so users notice a silently smaller
dataset.

### Acceptance criteria
- [ ] After enumerating a class directory, non-`.wav` files are counted and logged at
      `WARNING` level with their count and the distinct extensions skipped.
- [ ] WAV processing behavior is unchanged (still only `.wav` files are processed).
- [ ] A module-level logger is used (no bare `print`).

### Files to create or modify
- `src/preprocess.py`

### Notes for Coder
```python
all_audio = list(class_dir.iterdir())
skipped = [f for f in all_audio if f.suffix.lower() != ".wav"]
if skipped:
    _log.warning(f"{class_label}: {len(skipped)} non-WAV files skipped: "
                 f"{sorted({f.suffix for f in skipped})}")
```
Use `.lower()` on the suffix to stay robust on case-insensitive filesystems
(consistent with the Approver's Category D concern).

---

## TASK-R05: `tests/test_features.py` — seed augmentation tests

**Status:** TODO
**Depends on:** none
**Assigned to:** Coder
**Findings:** MIN-04 (Minor)

### What to build
`test_augment_audio_arrays_are_distinct` and
`test_augment_audio_gaussian_noise_differs_from_original` call `np.random.randn` /
`augment_audio` (→ `np.random.normal`) without a fixed seed, making them technically
non-deterministic. Seed the RNG so the tests are reproducible.

### Acceptance criteria
- [ ] Both augmentation tests set a fixed seed (`np.random.seed(42)` or a
      `rng = np.random.default_rng(42)` fixture) before generating random data.
- [ ] Tests remain meaningful (still assert distinctness / difference, not just shape).
- [ ] `pytest tests/test_features.py` passes deterministically across repeated runs.

### Files to create or modify
- `tests/test_features.py`

---

## Re-Approval Gate

After the Coder delivers and the Reviewer PASSes the tasks, the Planner re-invokes the
**Approver** for a re-audit. Per the Approver charter, a REJECTED codebase is not approved
after a single fix without re-running all Non-Negotiable Checks.

**Minimum to lift the training/demo block:**
1. TASK-R01 (CRIT-01) DONE + Reviewer PASS.
2. Approver re-audits `online.py` and confirms Non-Negotiable Check #9
   (`online.py` preprocessing matches training path) now passes, including the
   `[-1, 1]` normalization step.

**For full sign-off (APPROVED):**
- TASK-R01, TASK-R02, TASK-R03 DONE (the Critical + all Major findings).
- TASK-R04, TASK-R05 DONE or explicitly deferred with documented risk.
- Approver re-runs all 10 Non-Negotiable Checks and issues a fresh verdict.

---

## Findings Coverage Checklist

| Finding | Severity | Task | Status |
|---|---|---|---|
| CRIT-01 — online spectrogram not normalized to [-1,1] | CRITICAL | TASK-R01 | TODO |
| MAJ-01 — `_compute_class_weights` crash on missing class | MAJOR | TASK-R02 | TODO |
| MAJ-02 — `sd.wait()` no timeout, mic demo hangs | MAJOR | TASK-R01 | TODO |
| MAJ-03 — `--batch-size 0` / `--epochs -1` unvalidated | MAJOR | TASK-R02 | TODO |
| MIN-01 — no model warmup before mic loop | MINOR | TASK-R01 | TODO |
| MIN-02 — `sd.stop()` not in `finally` | MINOR | TASK-R01 | TODO |
| MIN-03 — non-WAV files skipped without warning | MINOR | TASK-R04 | TODO |
| MIN-04 — augmentation tests not seeded | MINOR | TASK-R05 | TODO |
| MIN-05 — `zero_division` not passed to sklearn metrics | MINOR | TASK-R03 | TODO |

All 9 findings are routed. No finding is unaddressed.
