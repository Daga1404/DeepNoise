# Agent: Planner
# Role: Task decomposition, sequencing, and coordination

You are the **Planner** agent for the Acoustic Event Classification project.
Read `CLAUDE.md` before every session for project context.

Your job is to break work into concrete task cards, assign them to the Coder,
and process Reviewer findings. You do not write implementation code.

---

## Responsibilities

- Decompose the implementation backlog into atomic, independently testable tasks
- Sequence tasks to respect dependencies
- Write task cards in the standard format below
- Receive Reviewer reports and decide: mark DONE, re-queue with fixes, or escalate
- Maintain the **Task Board** at the bottom of this file
- Flag blockers that require human input (missing data, unclear requirements)

---

## Task Card Format

```
## TASK-<N>: <short title>

**Status:** TODO | IN PROGRESS | REVIEW | DONE | BLOCKED
**Depends on:** TASK-<M>, ... (or "none")
**Assigned to:** Coder

### What to build
<One paragraph. State the module, its public interface, and what it must do.
 Be specific about function signatures and return types.>

### Acceptance criteria
- [ ] <Concrete, verifiable condition>
- [ ] ...

### Files to create or modify
- `src/<file>.py`
- `tests/test_<file>.py`

### Notes for Coder
<Design constraints, gotchas, references to DESIGN.md sections.>
```

---

## Dependency Graph

```
TASK-01  config.py
    └── TASK-02  preprocess.py
            └── TASK-03  features.py  (includes augmentation)
                    └── TASK-04  dataset.py
                            ├── TASK-05  baseline.py
                            └── TASK-06  model.py
                                    └── TASK-07  train.py  (two runs)
                                            └── TASK-08  evaluate.py
                                                    └── TASK-09  online.py
TASK-03 → TASK-10  notebooks/01_eda.ipynb
TASK-03 → TASK-11  notebooks/02_feature_extraction.ipynb
TASK-05 → TASK-12  notebooks/03_baseline.ipynb
```

Human gate: `data/raw/` must be populated before TASK-02 can run end-to-end.
Unit tests for TASK-02 and TASK-03 use synthetic data and do not require real audio.

---

## Receiving Reviewer Reports

- **PASS** → mark DONE, queue next dependent task
- **FAIL** → annotate task card with `### Reviewer Findings`, set IN PROGRESS, re-queue
- **BLOCKED** → add to Blocked Items table, do not re-queue until resolved

Never fix code yourself. Route all fixes through the Coder.

---

## Blocked Items

| ID | Blocker | Waiting on | Since |
|---|---|---|---|
| — | data/raw/ population | Human (see MANUAL_TASKS.md) | — |

---

## Task Board

| Task | Title | Status | Depends on |
|---|---|---|---|
| TASK-01 | `config.py` — shared constants | TODO | none |
| TASK-02 | `preprocess.py` — load, normalize, segment | TODO | TASK-01 |
| TASK-03 | `features.py` — log-Mel + augmentation | TODO | TASK-02 |
| TASK-04 | `dataset.py` — tf.data pipeline + splits | TODO | TASK-03 |
| TASK-05 | `baseline.py` — SVM | TODO | TASK-04 |
| TASK-06 | `model.py` — CNN | TODO | TASK-04 |
| TASK-07 | `train.py` — two runs (noaug + aug) | TODO | TASK-06 |
| TASK-08 | `evaluate.py` — full evaluation suite | TODO | TASK-07 |
| TASK-09 | `online.py` — real-time inference | TODO | TASK-08 |
| TASK-10 | `01_eda.ipynb` — spectrogram plots | TODO | TASK-03 |
| TASK-11 | `02_feature_extraction.ipynb` | TODO | TASK-03 |
| TASK-12 | `03_baseline.ipynb` | TODO | TASK-05 |

---

## Pre-written Task Cards

Issue one card at a time in dependency order.

---

## TASK-01: `config.py` — shared constants

**Status:** TODO
**Depends on:** none
**Assigned to:** Coder

### What to build
Create `src/config.py` as the single source of truth for all project constants.
No other module may redefine these values.

### Acceptance criteria
- [ ] File exports: `SAMPLE_RATE`, `DURATION`, `N_FFT`, `HOP_LENGTH`, `N_MELS`, `NUM_CLASSES`, `INPUT_SHAPE`, `CLASS_LABELS`
- [ ] `CLASS_LABELS` is a `dict[int, str]` mapping 0–4 to the five class name strings
- [ ] All values match CLAUDE.md §Project Constants exactly
- [ ] No logic — constants only
- [ ] `pytest tests/test_config.py` passes (import test + value assertions)

### Files to create or modify
- `src/config.py`
- `tests/test_config.py`

### Notes for Coder
This is the first task. Every subsequent module imports from here.
`INPUT_SHAPE = (128, 173, 1)` — width 173 is derived from `ceil(4.0 * 22050 / 512)`.

---

## TASK-02: `preprocess.py` — audio loading & normalization

**Status:** TODO
**Depends on:** TASK-01
**Assigned to:** Coder

### What to build
Implement `src/preprocess.py` with four public functions.
`load_audio(path, sr)` loads WAV to float32 mono array resampled to `sr`.
`normalize(audio)` applies peak normalization so max abs value = 1.0.
`segment(audio, sr, duration)` center-crops or zero-pads to exactly `duration` seconds.
`process_directory(raw_dir, out_dir, sr, duration)` walks `data/raw/`, applies all three,
writes to `data/processed/` mirroring the class-subfolder structure.

### Acceptance criteria
- [ ] `load_audio` returns dtype float32, shape (N,), mono
- [ ] `normalize` output has max abs = 1.0 (within 1e-6); handles all-zero gracefully
- [ ] `segment` output length == `int(sr * duration)` exactly
- [ ] Center-crop used for trimming (not head-crop)
- [ ] `process_directory` mirrors subfolder structure in output
- [ ] All functions accept `pathlib.Path` args
- [ ] `pytest tests/test_preprocess.py` passes with ≥ 3 tests using synthetic data

### Files to create or modify
- `src/preprocess.py`
- `tests/test_preprocess.py`

### Notes for Coder
Import `SAMPLE_RATE`, `DURATION` from `src/config.py`. Use `librosa.load`.
Do NOT apply silence trimming — it would corrupt the `silence_ambient` class.

---

## TASK-03: `features.py` — log-Mel spectrogram + augmentation

**Status:** TODO
**Depends on:** TASK-02
**Assigned to:** Coder

### What to build
Implement `src/features.py`.
`compute_melspectrogram(audio, sr, n_fft, hop_length, n_mels)` returns log-scaled 2D
array of shape `(n_mels, T)`.
`augment_audio(audio, sr)` returns a list of augmented audio arrays (Gaussian noise,
time shift). Applied to raw audio before spectrogram computation.
`augment_spectrogram(log_S)` returns a list of augmented spectrograms (SpecAugment:
freq mask + time mask). Applied after spectrogram computation.
`save_spectrograms(processed_dir, spec_dir, augment_train, **mel_kwargs)` iterates
over `data/processed/`, computes spectrograms, writes `.npy` files and `labels.csv`.
When `augment_train=True`, augmented copies are written for training-split files only
(split determined by reading `labels.csv` if it exists, else deferred to `dataset.py`).

### Acceptance criteria
- [ ] Output shape `(128, 173)` for a 4 s clip at 22050 Hz with default params
- [ ] No NaN or Inf values in any output
- [ ] Log scaling applied: `log_S = np.log(mel_S + 1e-6)`
- [ ] `augment_audio` returns ≥ 2 distinct arrays (noise + shift)
- [ ] `augment_spectrogram` returns ≥ 1 array with zeroed freq/time bands
- [ ] `labels.csv` has columns: `path`, `class_label`, `class_id`, `is_augmented`
- [ ] `pytest tests/test_features.py` passes with ≥ 4 tests using synthetic data

### Files to create or modify
- `src/features.py`
- `tests/test_features.py`

### Notes for Coder
Augmentation parameters: Gaussian std=0.005, time shift ±10%, freq mask 20 bins,
time mask 30 frames. Import all constants from `src/config.py`.
The `is_augmented` column in `labels.csv` allows `dataset.py` to enforce the rule
that augmented samples never appear in val or test sets.

---

## TASK-04: `dataset.py` — tf.data pipeline & splits

**Status:** TODO
**Depends on:** TASK-03
**Assigned to:** Coder

### What to build
Implement `src/dataset.py`.
`load_labels(spec_dir)` reads `labels.csv` and returns a DataFrame.
`split_dataset(df, train_ratio, val_ratio, seed)` performs stratified split on
**non-augmented** samples only, then adds augmented samples to the train split.
Returns three DataFrames: train, val, test.
`make_tf_dataset(df, batch_size, shuffle, augment)` returns a `tf.data.Dataset`
yielding `(spectrogram_tensor, one_hot_label)` batches with shape `(128, 173, 1)`.

### Acceptance criteria
- [ ] Augmented samples (`is_augmented=True`) appear only in train split — never val or test
- [ ] Split ratios are computed on non-augmented samples; stratification within 2%
- [ ] Dataset yields shape `(batch, 128, 173, 1)` and `(batch, 5)`
- [ ] Spectrograms normalized to [−1, 1] before yielding
- [ ] `pytest tests/test_dataset.py` passes with ≥ 3 tests

### Files to create or modify
- `src/dataset.py`
- `tests/test_dataset.py`

### Notes for Coder
The no-leakage rule is the most important correctness constraint here. The Reviewer
will flag data leakage as a BLOCKER. Use `sklearn.model_selection.train_test_split`
with `stratify=` on the non-augmented subset, then `pd.concat` augmented rows onto train.

---

## TASK-05: `baseline.py` — SVM baseline

**Status:** TODO
**Depends on:** TASK-04
**Assigned to:** Coder

### What to build
Implement `src/baseline.py`.
`load_features(df)` flattens each spectrogram to a 1D vector; returns `(X, y)` arrays.
`train_svm(X_train, y_train)` fits `SVC(kernel='rbf', C=10)` and returns the model.
`evaluate(model, X_test, y_test, output_dir)` returns dict with `accuracy`, `macro_f1`,
`report` (str), and saves `results/baseline_confusion.png` and `results/baseline_report.txt`.
A `__main__` block runs the full pipeline via `python src/baseline.py --data-dir data/`.

### Acceptance criteria
- [ ] SVM trains on full 22144-feature vectors without error
- [ ] `evaluate` dict has `accuracy`, `macro_f1`, `report`
- [ ] Confusion matrix PNG saved with class name labels (not IDs)
- [ ] `python src/baseline.py --data-dir data/ --output-dir results/` runs end-to-end
- [ ] `pytest tests/test_baseline.py` passes

### Files to create or modify
- `src/baseline.py`
- `tests/test_baseline.py`

---

## TASK-06: `model.py` — CNN architecture

**Status:** TODO
**Depends on:** TASK-04
**Assigned to:** Coder

### What to build
Implement `src/model.py`.
`build_cnn(input_shape, num_classes, dropout_rate)` returns a compiled `keras.Model`
matching DESIGN.md §CNN Architecture exactly.

### Acceptance criteria
- [ ] Layer sequence matches DESIGN.md §CNN Architecture exactly
- [ ] Input shape `(128, 173, 1)`, output `(5,)` with Softmax
- [ ] Compiled with `Adam(lr=1e-3)`, `categorical_crossentropy`, metric `accuracy`
- [ ] `build_cnn()` callable with defaults from `src/config.py`
- [ ] `pytest tests/test_model.py` passes: forward pass, output shape, layer count

### Files to create or modify
- `src/model.py`
- `tests/test_model.py`

---

## TASK-07: `train.py` — training loop, two runs

**Status:** TODO
**Depends on:** TASK-06
**Assigned to:** Coder

### What to build
Implement `src/train.py` as a CLI script that runs one full training pass.
`python src/train.py --data-dir data/ --augment false --output-dir models/noaug/ --epochs 50`
`python src/train.py --data-dir data/ --augment true  --output-dir models/aug/  --epochs 50`
Each run: loads datasets, builds CNN, attaches callbacks, fits, saves
`<output-dir>/cnn_best.keras` and `<output-dir>/history.json`.

### Acceptance criteria
- [ ] `--augment false` and `--augment true` both complete without error
- [ ] `cnn_best.keras` loadable with `keras.models.load_model`
- [ ] `history.json` contains `loss`, `val_loss`, `accuracy`, `val_accuracy` arrays
- [ ] Class weights computed from training set, passed to `model.fit`
- [ ] EarlyStopping + ReduceLROnPlateau attached
- [ ] `python src/train.py --epochs 2 --augment false --output-dir /tmp/smoke` smoke-tests in < 60 s

### Files to create or modify
- `src/train.py`

---

## TASK-08: `evaluate.py` — full evaluation suite

**Status:** TODO
**Depends on:** TASK-07
**Assigned to:** Coder

### What to build
Implement `src/evaluate.py` as a CLI script.
`python src/evaluate.py --model models/aug/cnn_best.keras --data-dir data/ --output-dir results/`
Produces: accuracy + macro F1 to stdout, `results/cnn_confusion.png`,
`results/training_curves.png` (from `history.json`), `results/cnn_report.txt`,
`results/error_analysis.csv` (columns: `path`, `true_label`, `predicted_label`, `confidence`).
Also generates `results/augmentation_comparison.md` by reading both
`models/noaug/history.json` and `models/aug/history.json` if both exist.

### Acceptance criteria
- [ ] All output files produced
- [ ] Confusion matrix axes labelled with class name strings
- [ ] Macro F1 is the first metric printed (primary)
- [ ] Error analysis has ≥ 5 rows per class where possible
- [ ] `augmentation_comparison.md` includes test accuracy + macro F1 for both runs
- [ ] `pytest tests/test_evaluate.py` passes (mock model + check file outputs)

### Files to create or modify
- `src/evaluate.py`
- `tests/test_evaluate.py`

---

## TASK-09: `online.py` — real-time inference

**Status:** TODO
**Depends on:** TASK-08
**Assigned to:** Coder

### What to build
Implement `src/online.py`.
Default mode: capture 4 s from microphone via `sounddevice`, preprocess using
`preprocess.py` and `features.py`, call `model.predict`, print result, repeat.
`--file` mode: load a WAV, run same pipeline once, print result and exit.
Output format: `[HH:MM:SS] Predicted: <class_name>  (confidence: <0.xx>)`
Ctrl+C exits cleanly.

### Acceptance criteria
- [ ] Imports `normalize`, `segment` from `src/preprocess.py` — no duplication
- [ ] Imports `compute_melspectrogram` from `src/features.py` — no duplication
- [ ] `--file` mode works without a microphone attached
- [ ] Predicted class name comes from `CLASS_LABELS` dict in `src/config.py`
- [ ] `python src/online.py --model models/aug/cnn_best.keras --file tests/fixtures/sample.wav` runs
- [ ] Output includes confidence score from `model.predict` softmax output

### Files to create or modify
- `src/online.py`

### Notes for Coder
`sounddevice.rec(frames, samplerate, channels)` captures audio.
No unit tests required for microphone capture (hardware-dependent).
Provide a `--file` mode that IS testable in CI.

---

## TASK-10: `01_eda.ipynb` — EDA and spectrogram visualization

**Status:** TODO
**Depends on:** TASK-03
**Assigned to:** Coder

### What to build
Populate `notebooks/01_eda.ipynb` with:
1. Class distribution bar chart (sample counts per class)
2. One example waveform per class (5 subplots)
3. One log-Mel spectrogram per class using `librosa.display.specshow` (5 subplots, required)
4. Commentary cells explaining what each class looks like and how it differs from others

### Acceptance criteria
- [ ] 5 spectrogram plots, one per class, clearly labelled
- [ ] Each spectrogram uses the same colour scale for fair comparison
- [ ] Notebook runs top-to-bottom without error against `data/spectrograms/`
- [ ] Class distribution plot shows at least approximate balance

### Files to create or modify
- `notebooks/01_eda.ipynb`