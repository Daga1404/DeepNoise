# Roadmap 2 — Audit Fix Workplan
# Planner Agent · Acoustic Event Classification Project

> Produced from `audit_1.md` findings.
> All items are assigned to the **Coder**. The **Reviewer** re-audits each fix before
> it is marked DONE. The Planner marks the original roadmap step DONE only after
> every applicable fix in this document reaches DONE.

---

## Source → Fix mapping

| Audit finding | Severity | Fix task |
|---|---|---|
| `sounddevice` absent from `requirements.txt` | MAJOR | FIX-01 |
| `models/` directory missing | MAJOR | FIX-01 |
| `results/` directory missing | MAJOR | FIX-01 |
| `augment_audio` shifts in one direction only (+10%) | MINOR | FIX-02 |
| `train_svm` return type annotation says `SVC`, returns `Pipeline` | MINOR | FIX-03 |
| `print()` in library function `run_training` (train.py) | MINOR | FIX-04 |
| `print()` in library function `evaluate_cnn` (evaluate.py) | MINOR | FIX-04 |
| Unused `compute_melspectrogram` import in `01_eda.ipynb` cell-1 | WARNING | FIX-05 |

---

## Dependency graph

```
FIX-01  scaffolding (blocking — must be done first)
FIX-02  features.py augment_audio (independent)
FIX-03  baseline.py return type (independent)
FIX-04  logging discipline in train.py + evaluate.py (independent)
FIX-05  notebook unused import (independent)
```

FIX-02 through FIX-05 have no inter-dependencies and may be implemented in any order
after FIX-01 is merged.

---

## FIX-01 — Scaffolding completion (STEP 0 re-queue)

**Status:** TODO
**Depends on:** none
**Assigned to:** Coder
**Priority:** BLOCKING — no milestone can be closed until this passes

### What to build
Complete the three unfinished items from STEP 0 that the Reviewer flagged as MAJOR.
No source logic changes; this is purely filesystem and dependency metadata.

1. Add `sounddevice` to `requirements.txt`.
2. Create `models/.gitkeep` so the directory is tracked by git.
3. Create `results/.gitkeep` so the directory is tracked by git.

### Acceptance criteria
- [ ] `requirements.txt` contains `sounddevice>=0.4.0`
- [ ] `Test-Path models` → `True` from the repo root
- [ ] `Test-Path results` → `True` from the repo root
- [ ] `models/.gitkeep` and `results/.gitkeep` exist and are committed
- [ ] `python -c "import sounddevice"` succeeds after `pip install -r requirements.txt`
- [ ] All existing tests still pass (86 passed, 6 skipped — no regressions)

### Files to create or modify
- `requirements.txt` — add `sounddevice>=0.4.0`
- `models/.gitkeep` — new empty file
- `results/.gitkeep` — new empty file

### Notes for Coder
`sounddevice` is only imported lazily inside `run_mic_mode` in `src/online.py`, so its
absence does not cause import failures at the test stage — but it will break any user
running `pip install -r requirements.txt` followed by `python src/online.py`.
Do NOT add `sounddevice` to `.gitignore`. The `.gitkeep` files must be committed.
`models/` is no longer in `.gitignore` (that was already fixed), so the `.gitkeep` will
be tracked normally.

---

## FIX-02 — `augment_audio` bidirectional time shift

**Status:** TODO
**Depends on:** none
**Assigned to:** Coder
**Priority:** MINOR (optional — Planner recommends fixing for spec compliance)

### What to build
Modify `augment_audio` in `src/features.py` so it produces **both** a forward (+10%) and
a backward (−10%) time shift, making the return list `[noisy, shifted_forward, shifted_backward]`
(3 arrays). Update `test_features.py` and `test_save_spectrograms_augmented_count` to
reflect the new count.

### Current behaviour
```python
def augment_audio(audio, sr=SAMPLE_RATE) -> list[np.ndarray]:
    noise   = audio + np.random.normal(0.0, 0.005, size=len(audio)).astype(np.float32)
    shifted = np.roll(audio, int(0.1 * len(audio))).astype(np.float32)  # +10% only
    return [noise, shifted]
```

### Required behaviour
```python
def augment_audio(audio, sr=SAMPLE_RATE) -> list[np.ndarray]:
    noise    = audio + np.random.normal(0.0, 0.005, size=len(audio)).astype(np.float32)
    shift_n  = int(0.1 * len(audio))
    forward  = np.roll(audio,  shift_n).astype(np.float32)   # +10%
    backward = np.roll(audio, -shift_n).astype(np.float32)   # -10%
    return [noise, forward, backward]
```

### Acceptance criteria
- [ ] `augment_audio` returns a list of exactly 3 distinct float32 arrays
- [ ] `result[1]` is a forward (+10%) roll of the original
- [ ] `result[2]` is a backward (−10%) roll of the original
- [ ] `result[1]` and `result[2]` are not equal to each other
- [ ] `test_augment_audio_returns_at_least_two_arrays` still passes (≥2 condition, now satisfied by 3)
- [ ] `test_save_spectrograms_augmented_count` updated: expected augmented count per file is now
      **4** (3 audio augs + 1 specaug), not 3 — update the assertion `n_files * 4`
- [ ] All other features tests still pass

### Files to create or modify
- `src/features.py` — `augment_audio` function body only
- `tests/test_features.py` — `test_save_spectrograms_augmented_count`: change `n_files * 3`
  to `n_files * 4`

### Notes for Coder
The `save_spectrograms` function iterates `augment_audio(audio)` and writes one `.npy` per
returned array, so it automatically handles 3 audio-aug copies without needing changes.
Only the test assertion for the total augmented count must be updated. No other test needs
to change; all other assertions use `≥` comparisons or check only the presence of augmented
rows.

---

## FIX-03 — `train_svm` return type annotation

**Status:** TODO
**Depends on:** none
**Assigned to:** Coder
**Priority:** MINOR (optional)

### What to build
Correct the return type annotation of `train_svm` in `src/baseline.py` from `-> SVC`
(which was the roadmap spec) to `-> Pipeline`, matching what the function actually returns.
Add a one-line comment explaining the design rationale so the Reviewer does not flag it again.

### Current code
```python
def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
```
The annotation already says `Pipeline` in the actual code (the roadmap said `-> SVC`).
The fix is to add a comment explaining that `Pipeline` is intentionally used over bare `SVC`.

### Acceptance criteria
- [ ] `train_svm` signature includes `-> Pipeline` annotation (already present — confirm it is there)
- [ ] A single inline comment in the function docstring explains that `Pipeline` is used
      instead of bare `SVC` so the caller passes raw (unscaled) features and the scaler
      is applied transparently
- [ ] All 12 `test_baseline.py` tests still pass

### Files to create or modify
- `src/baseline.py` — docstring of `train_svm` only; no logic changes

### Notes for Coder
The fix is documentation-only. Do not change the `Pipeline` structure or re-introduce a
bare `SVC` return — the scaler is essential for RBF kernel performance.

---

## FIX-04 — Print discipline: `run_training` and `evaluate_cnn`

**Status:** TODO
**Depends on:** none
**Assigned to:** Coder
**Priority:** MINOR (optional)

### What to build
Replace bare `print()` calls inside the public library functions `run_training`
(`src/train.py`) and `evaluate_cnn` (`src/evaluate.py`) with `logging.info()` calls.
The `main()` entry point in each file must configure a basic logging handler so that
CLI users still see the output. No other behavioural change.

### Current violations

`src/train.py` — inside `run_training` (public function):
```python
print(f"Split — train: {len(train_df)}  val: {len(val_df)}  ...")
print(f"Class weights: {class_weight}")
print(f"\nArtefacts written to {output_dir}")
print(f"  cnn_best.keras")
print(f"  history.json  ...")
```

`src/evaluate.py` — inside `evaluate_cnn` (public function):
```python
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Accuracy : {accuracy:.4f}")
print(report)
```

### Required behaviour

In both files, replace every `print(...)` call inside the named functions with
`logging.getLogger(__name__).info(...)`.

In both `main()` functions, add **before** any other call:
```python
logging.basicConfig(level=logging.INFO, format="%(message)s")
```

This ensures:
- CLI users see the same output as before.
- Programmatic callers (e.g. tests, notebooks) can suppress the output with
  `logging.disable(logging.INFO)` if needed.
- The "no print in library functions" rule is satisfied.

### Acceptance criteria
- [ ] No `print()` calls remain inside `run_training` or `evaluate_cnn`
- [ ] All replaced calls use `logging.getLogger(__name__).info(...)`
- [ ] Both `main()` functions call `logging.basicConfig(level=logging.INFO, format="%(message)s")`
- [ ] `logging` is added to the imports in both files (stdlib — goes in the first import group)
- [ ] Output seen when running `python src/train.py --help` or the main block is unchanged
      in terms of what is displayed
- [ ] All existing tests still pass (tests do not check stdout of library functions)

### Files to create or modify
- `src/train.py` — `run_training` body + `main()` + imports
- `src/evaluate.py` — `evaluate_cnn` body + `main()` + imports

### Notes for Coder
The roadmap acceptance criterion for STEP 8 says "Macro F1 is the first metric printed".
After this fix, the logging output preserves the same order — `Macro F1` is logged before
`Accuracy` — so the criterion is still satisfied.
Do not change anything inside `_save_*` private helpers; those are already print-free.

---

## FIX-05 — Remove unused import in `01_eda.ipynb`

**Status:** TODO
**Depends on:** none
**Assigned to:** Coder
**Priority:** WARNING (optional — cosmetic cleanup)

### What to build
Remove the unused `compute_melspectrogram` import from cell-1 of
`notebooks/01_eda.ipynb`. Spectrograms are loaded from pre-computed `.npy` files in
this notebook, so the function is never called and its import is dead code.

### Current cell-1 (relevant lines)
```python
from src.config import CLASS_LABELS, HOP_LENGTH, SAMPLE_RATE
from src.features import compute_melspectrogram   # ← remove this line
```

### Required cell-1
```python
from src.config import CLASS_LABELS, HOP_LENGTH, SAMPLE_RATE
```

### Acceptance criteria
- [ ] `compute_melspectrogram` no longer appears in cell-1 of `01_eda.ipynb`
- [ ] No other cell in the notebook references `compute_melspectrogram`
- [ ] The notebook reads cleanly from top to bottom without NameError (run a
      dry-parse check: `python -c "import ast; ast.parse(open('notebooks/01_eda.ipynb').read())"`)
      — note: this is a JSON file so the actual check is that the cell source parses
- [ ] All other imports in cell-1 remain intact

### Files to create or modify
- `notebooks/01_eda.ipynb` — cell-1 source only (one line deleted)

### Notes for Coder
Use the `NotebookEdit` tool with `edit_mode=replace` on `cell-1`. Be careful not to
delete any other import line — only the `compute_melspectrogram` line is removed.

---

## Task Board

| Task | Title | Status | Priority | Depends on |
|---|---|---|---|---|
| FIX-01 | Scaffolding completion | TODO | **BLOCKING** | none |
| FIX-02 | `augment_audio` bidirectional shift | TODO | MINOR | none |
| FIX-03 | `train_svm` return type annotation | TODO | MINOR | none |
| FIX-04 | Print → logging in `run_training` / `evaluate_cnn` | TODO | MINOR | none |
| FIX-05 | Remove unused import in `01_eda.ipynb` | TODO | WARNING | none |

---

## Reviewer re-audit checklist after all fixes

Once all five FIX tasks are DONE, the Reviewer must confirm:

- [ ] `requirements.txt` contains `sounddevice>=0.4.0`
- [ ] `models/` and `results/` directories exist and are tracked
- [ ] `augment_audio` returns 3 arrays; `test_save_spectrograms_augmented_count` asserts `n_files * 4`
- [ ] `train_svm` docstring explains `Pipeline` rationale
- [ ] Zero `print()` calls inside `run_training` or `evaluate_cnn`
- [ ] `01_eda.ipynb` cell-1 no longer imports `compute_melspectrogram`
- [ ] Full test suite: ≥86 passed, 0 failed (skips for TF-dependent tests are acceptable)
- [ ] `STEP 0` in `initial_roadmap.md` can be marked **DONE**
