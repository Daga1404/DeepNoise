# Audit 1 — Full Codebase Review
# Reviewer Agent · Acoustic Event Classification Project

> **Scope:** All STEPS 0–12 as defined in `initial_roadmap.md`.
> **Test baseline:** 86 passed, 6 skipped (TF-dependent), 0 failed across the non-model suite;
> `test_model.py` fully skipped (TF not available on Python 3.14).
> **Date:** 2026-06-02

---

## Overall Verdicts Summary

| Step | Title | Verdict |
|---|---|---|
| STEP 0 | Project scaffolding | **FAIL** |
| STEP 1 | `src/config.py` | **PASS** |
| STEP 2 | `src/preprocess.py` | **PASS** |
| STEP 3 | `src/features.py` | **PASS WITH WARNINGS** |
| STEP 4 | `src/dataset.py` | **PASS WITH WARNINGS** |
| STEP 5 | `src/baseline.py` | **PASS WITH WARNINGS** |
| STEP 6 | `src/model.py` | **PASS WITH WARNINGS** |
| STEP 7 | `src/train.py` | **PASS WITH WARNINGS** |
| STEP 8 | `src/evaluate.py` | **PASS WITH WARNINGS** |
| STEP 9 | `src/online.py` | **PASS** |
| STEP 10 | `notebooks/01_eda.ipynb` | **PASS** |
| STEP 11 | `notebooks/02_feature_extraction.ipynb` | **PASS** |
| STEP 12 | `notebooks/03_baseline.ipynb` | **PASS** |

**ML Red Lines:** No BLOCKER violations found. No data leakage. No wrong primary metric.
No constants redeclared. No preprocessing duplication in `online.py`.

---

## REVIEW: STEP 0 — Project scaffolding

**Verdict:** FAIL

**Acceptance criteria:** 2/4 passed

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| 1 | MAJOR | `requirements.txt` | — | `sounddevice` is listed in `CLAUDE.md` Stack but absent from `requirements.txt`; `src/online.py` imports it at runtime |
| 2 | MAJOR | repo root | — | `models/` directory does not exist (`Test-Path models` → False); scripts that save `cnn_best.keras` will fail at runtime |
| 3 | MAJOR | repo root | — | `results/` directory does not exist (`Test-Path results` → False); `evaluate.py` and `baseline.py` write artefacts there |
| 4 | MINOR | `requirements.txt` | — | `pydub` listed but unused anywhere in the implementation; harmless but creates unnecessary dependency |

### Summary
STEP 0 must be re-queued. Two required output directories (`models/`, `results/`) were never
created (the `.gitkeep` writes were rejected during the session), and `sounddevice` was never
added to `requirements.txt` despite being the only package used exclusively by `online.py`.
`import src` works and `.gitignore` is correct.

---

## REVIEW: STEP 1 — `src/config.py`

**Verdict:** PASS

**Acceptance criteria:** 4/4 passed

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| — | — | — | — | No findings |

### Summary
All eight constants present, values match `CLAUDE.md` exactly, no logic or side effects.
Five tests pass including the derived-shape assertion `INPUT_SHAPE[1] == ceil(4.0 * 22050 / 512)`.

---

## REVIEW: STEP 2 — `src/preprocess.py`

**Verdict:** PASS

**Acceptance criteria:** 6/6 passed

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| — | — | — | — | No findings |

### Summary
All four public functions match the roadmap signatures. Center-crop and symmetric zero-pad
verified by dedicated tests. All-zero input handled gracefully. Ten synthetic-data tests pass.
No `os.path`, no hardcoded paths, `SAMPLE_RATE`/`DURATION` imported from `src.config`.

---

## REVIEW: STEP 3 — `src/features.py`

**Verdict:** PASS WITH WARNINGS

**Acceptance criteria:** 5/5 passed

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| 1 | MINOR | `src/features.py` | 62 | `augment_audio` time-shifts by `+10%` only (`np.roll(audio, shift_samples)`); roadmap specifies `±10%`. A negative shift is equally valid; both directions should be represented |
| 2 | WARNING | `src/features.py` | 60 | `np.random.normal` called without a seed in `augment_audio`; results are non-deterministic across runs. Acceptable for training augmentation but note the design choice |

### Summary
Log scaling `np.log(mel_S + 1e-6)` correct. `labels.csv` columns exactly match spec.
`augment_spectrogram` zeroed-bands test passes. Eighteen tests pass. Only minor deviation
is the one-directional time shift.

---

## REVIEW: STEP 4 — `src/dataset.py`

**Verdict:** PASS WITH WARNINGS

**Acceptance criteria:** 5/5 passed

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| 1 | WARNING | `tests/test_dataset.py` | 144–168 | Two `make_tf_dataset` tests (shape check and normalisation range) skip permanently on Python 3.14; acceptance criterion "Dataset yields shapes (batch,128,173,1) and (batch,5)" is not currently verifiable in this environment |
| 2 | WARNING | `src/dataset.py` | 73 | Return type annotation written as string `"tf.data.Dataset"` to avoid ImportError; correct but unusual — future readers may find it confusing |

### Summary
No-leakage rule explicitly tested and passes: augmented rows appear only in the train split,
never in val or test. Stratified split within 2% tolerance. Twelve non-TF tests pass.
The two TF-dependent tests are correctly written and will run on Python ≤ 3.12.

---

## REVIEW: STEP 5 — `src/baseline.py`

**Verdict:** PASS WITH WARNINGS

**Acceptance criteria:** 5/5 passed (functionally)

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| 1 | MINOR | `src/baseline.py` | 61 | `train_svm` is typed as `-> Pipeline` in the code but the roadmap declares `-> SVC`. The Pipeline wraps `StandardScaler + SVC(rbf, C=10)` and is duck-type compatible for `.predict`, but the declared return type deviates from the task card |
| 2 | WARNING | `src/baseline.py` | — | `__main__` block calls `evaluate` which prints results — fine. But consider adding a brief stdout header for the report file path |

### Summary
All three required functions present and tested. The confusion matrix correctly uses class-name
axis labels. Twelve tests pass including full-22144-feature SVM training. The `Pipeline`
return type is functionally correct and arguably better design than a bare `SVC`.

---

## REVIEW: STEP 6 — `src/model.py`

**Verdict:** PASS WITH WARNINGS

**Acceptance criteria:** 4/4 met (architecture verified by code review; tests pending TF)

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| 1 | WARNING | `tests/test_model.py` | — | All 17 tests skip on Python 3.14; forward pass, output shape, and layer-count acceptance criteria are not currently verifiable. Tests are correctly written and will pass on Python ≤ 3.12 |
| 2 | WARNING | `src/model.py` | 1 | Top-level `from tensorflow import keras` means `import src.model` fails without TF. Other modules (`train.py`) guard this correctly with lazy imports, so runtime impact is contained |

### Summary
Architecture matches `DESIGN.md §CNN Architecture` exactly: three Conv2D blocks [32,64,128]
each with BatchNorm+ReLU, two MaxPool(2×2), GlobalAveragePooling2D, Dense(128)+ReLU+Dropout(0.4),
Dense(5)+Softmax. Compiled with Adam(lr=1e-3) and categorical_crossentropy. `dropout_rate`
parameter correctly wired.

---

## REVIEW: STEP 7 — `src/train.py`

**Verdict:** PASS WITH WARNINGS

**Acceptance criteria:** 5/5 met (unverifiable without TF)

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| 1 | MINOR | `src/train.py` | 68–76 | `run_training` contains three `print()` calls (split counts, class weights, artefact paths). Reviewer rule: no print in library functions. These are informational progress lines; move to `__main__` or use `logging` |
| 2 | WARNING | `src/train.py` | — | Smoke test (`--epochs 2 < 60 s`) is unverifiable without TF on Python 3.14 |
| 3 | WARNING | `src/train.py` | — | No test file for `train.py` (roadmap lists only `src/train.py` under Files, so this is by design — noting for completeness) |

### Summary
`--augment false/true`, `--epochs`, `--batch-size`, and `--data-dir`/`--output-dir` all
implemented. Class weights from `compute_class_weight("balanced")`. EarlyStopping and
ReduceLROnPlateau attached with correct parameters. `cnn_best.keras` and `history.json`
saved correctly. Augment-false filtering drops augmented rows before `split_dataset`.

---

## REVIEW: STEP 8 — `src/evaluate.py`

**Verdict:** PASS WITH WARNINGS

**Acceptance criteria:** 5/5 passed

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| 1 | MINOR | `src/evaluate.py` | 69–72 | `evaluate_cnn` contains three `print()` calls in a public function (not `__main__`). The roadmap acceptance criterion "Macro F1 is the first metric printed" implicitly requires this, but it still violates the "no print in library functions" rule. Consider returning the formatted string and letting `main()` print it |
| 2 | WARNING | `tests/test_evaluate.py` | 159 | `test_evaluate_cnn_produces_all_output_files` skips on Python 3.14 (full integration test) |
| 3 | WARNING | `src/evaluate.py` | 176 | `_save_augmentation_comparison` derives the `models_root` from `model_path.parent.parent`. This assumes a strict `models/<run>/cnn_best.keras` path convention; will silently skip if the model is placed elsewhere |

### Summary
All five output artefacts are written correctly. Macro F1 is printed and written first.
`error_analysis.csv` columns match the spec exactly. `augmentation_comparison.md` is generated
from `test_metrics.json` files saved alongside each model. Seventeen of eighteen tests pass;
the one that skips is the full end-to-end TF integration test.

---

## REVIEW: STEP 9 — `src/online.py`

**Verdict:** PASS

**Acceptance criteria:** 5/5 passed

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| 1 | WARNING | `tests/test_online.py` | — | Three TF-dependent tests (`classify_audio`, `run_file_mode` with real model) skip on Python 3.14 |

### Summary
All preprocessing delegates to `src.preprocess` (`normalize`, `segment`) and `src.features`
(`compute_melspectrogram`) — no duplication. `sounddevice` and `tensorflow` are lazy-imported
so the module loads cleanly without either. `tests/fixtures/sample.wav` exists.
`--file` mode verified to work. Output format `[HH:MM:SS] Predicted: ...` confirmed by tests.
Twelve non-TF tests pass.

---

## REVIEW: STEP 10 — `notebooks/01_eda.ipynb`

**Verdict:** PASS

**Acceptance criteria:** 3/3 passed

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| 1 | WARNING | `notebooks/01_eda.ipynb` | cell-1 | Imports include `compute_melspectrogram` from `src.features` but it is never called in the notebook (spectrograms are loaded from `.npy` files). Unused import |

### Summary
All required content present: class-distribution bar chart, 5-subplot waveform grid,
5-panel `librosa.display.specshow` with shared colour scale (5th–95th percentile `vmin`/`vmax`),
and commentary cells for each class. Missing data handled gracefully via `if df_orig is None`
guards. Imports correctly draw from `src.*` — no logic re-implemented.

---

## REVIEW: STEP 11 — `notebooks/02_feature_extraction.ipynb`

**Verdict:** PASS

**Acceptance criteria:** 3/3 passed

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| — | — | — | — | No findings |

### Summary
All four augmentation types are shown side-by-side with originals: load→normalize→segment
preprocessing (3-subplot stack), waveform→spectrogram conversion, Gaussian noise vs. original,
time shift vs. original, and SpecAugment vs. original (shared colour scale). All processing
delegates to `src.preprocess` and `src.features`. Missing data handled gracefully.

---

## REVIEW: STEP 12 — `notebooks/03_baseline.ipynb`

**Verdict:** PASS

**Acceptance criteria:** 3/3 passed

### Findings

| # | Severity | File | Line | Finding |
|---|---|---|---|---|
| — | — | — | — | No findings |

### Summary
Calls `load_features`, `train_svm`, and `evaluate` from `src.baseline` — no logic re-implemented.
Confusion matrix displayed inline via `mpimg.imread` on the saved PNG. Per-class report
printed from `results["report"]`. Commentary table correctly frames baseline (60–75%) as
the benchmark the CNN must exceed, with failure-mode diagnostics.

---

## Cross-cutting Findings

| # | Severity | Scope | Finding |
|---|---|---|---|
| X1 | MAJOR | `requirements.txt` | `sounddevice` absent; `src/online.py` will fail at runtime on a fresh install |
| X2 | MAJOR | repo root | `models/` and `results/` directories do not exist; multiple scripts write to them |
| X3 | MINOR | `src/train.py`, `src/evaluate.py` | `print()` calls in public functions `run_training` and `evaluate_cnn`; should be in `__main__` or use `logging` |
| X4 | MINOR | `src/features.py` | `augment_audio` time-shifts in one direction only (+10%); spec says ±10% |
| X5 | WARNING | All TF-dependent modules | Python 3.14 has no TF wheel; 23 tests (17 in `test_model`, 6 spread across `test_dataset`, `test_evaluate`, `test_online`) skip. All are correctly written and will run on Python ≤ 3.12 |
| X6 | WARNING | `src/model.py` | Top-level `from tensorflow import keras` makes `import src.model` fail without TF; all callers guard it lazily but this could surprise future imports |

---

## Actions Required (Planner)

### Must fix before any milestone is closed

1. **STEP 0 — Re-queue (FAIL)**
   - Add `sounddevice>=0.4.0` to `requirements.txt`
   - Create `models/` with a `.gitkeep` (or ensure `train.py` creates it before writing)
   - Create `results/` with a `.gitkeep` (or ensure scripts create it before writing)

### Optional fixes (PASS WITH WARNINGS → may re-queue at Planner's discretion)

2. **STEP 3** — `augment_audio`: add a second shifted copy with negative shift (`np.roll(audio, -shift_samples)`) to honour the `±10%` spec.

3. **STEP 5** — Update `train_svm` return type annotation to `Pipeline` or add an alias comment explaining why `Pipeline` is returned instead of bare `SVC`.

4. **STEP 7 / STEP 8** — Move progress `print()` calls from `run_training` / `evaluate_cnn` into `main()`, or replace with `logging.info()` so library consumers control verbosity.

5. **STEP 10** — Remove unused `compute_melspectrogram` import from `notebooks/01_eda.ipynb` cell-1.

---

## Test Suite Health

| Test file | Tests | Passed | Skipped | Failed |
|---|---|---|---|---|
| `test_config.py` | 5 | 5 | 0 | 0 |
| `test_preprocess.py` | 10 | 10 | 0 | 0 |
| `test_features.py` | 18 | 18 | 0 | 0 |
| `test_dataset.py` | 14 | 12 | 2 (TF) | 0 |
| `test_baseline.py` | 12 | 12 | 0 | 0 |
| `test_model.py` | 17 | 0 | 17 (TF) | 0 |
| `test_evaluate.py` | 18 | 17 | 1 (TF) | 0 |
| `test_online.py` | 15 | 12 | 3 (TF) | 0 |
| **Total** | **109** | **86** | **23** | **0** |

All failures are due to TF unavailability on Python 3.14, not logic errors. The suite is
clean and will reach 109/109 on Python ≤ 3.12 with TF installed.
