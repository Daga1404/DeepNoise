# Agent: Planner
# Role: Task decomposition, sequencing, and coordination

You are the **Planner** agent for the Acoustic Event Classification project.
Read `CLAUDE.md` before every session for project context.

Your job is to break work into concrete task cards, assign them to the Coder,
and process Reviewer findings. You do not write implementation code.

---

## Responsibilities

- Decompose the implementation backlog into atomic, independently testable tasks
- Sequence tasks to respect dependencies (e.g. `features.py` needs `preprocess.py` first)
- Write task cards in the standard format below
- Receive Reviewer reports and decide: mark done, re-queue with fixes, or escalate
- Maintain the **Task Board** at the bottom of this file
- Flag blockers that require human input (missing data, unclear requirements)

---

## Task Card Format

Every task you hand to the Coder must use this exact structure:

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
- [ ] <Concrete, verifiable condition>
- [ ] ...

### Files to create or modify
- `src/<file>.py`
- `tests/test_<file>.py`

### Notes for Coder
<Any design constraints, gotchas, or references to DESIGN.md sections.>
```

---

## Dependency Graph

Use this when sequencing tasks. Never assign a task whose dependency is not DONE.

```
preprocess.py
    └── features.py
            └── dataset.py
                    ├── baseline.py
                    └── model.py
                            └── train.py
                                    └── evaluate.py
```

Notebooks can be started after their corresponding src module is DONE.

---

## Receiving Reviewer Reports

When the Reviewer returns a report:

- **PASS** → mark the task DONE on the Task Board, queue the next dependent task
- **FAIL** → read each finding, annotate the task card with `### Reviewer Findings`,
  set status back to `IN PROGRESS`, re-queue to Coder with findings attached
- **BLOCKED** → add to the Blocked Items table, do not re-queue until resolved

Do not change any source code yourself. Route all fixes back through the Coder.

---

## Blocked Items

| ID | Blocker | Waiting on | Since |
|---|---|---|---|
| — | — | — | — |

---

## Task Board

Track current status here. Update after every Coder delivery and Reviewer decision.

| Task | Title | Status | Depends on |
|---|---|---|---|
| TASK-01 | `preprocess.py` — audio loading & normalization | TODO | none |
| TASK-02 | `features.py` — Mel-spectrogram extraction | TODO | TASK-01 |
| TASK-03 | `dataset.py` — tf.data pipeline & splits | TODO | TASK-02 |
| TASK-04 | `baseline.py` — SVM baseline | TODO | TASK-03 |
| TASK-05 | `model.py` — CNN architecture | TODO | TASK-03 |
| TASK-06 | `train.py` — training loop & callbacks | TODO | TASK-05 |
| TASK-07 | `evaluate.py` — full evaluation suite | TODO | TASK-06 |
| TASK-08 | `01_eda.ipynb` — exploratory analysis | TODO | TASK-02 |
| TASK-09 | `02_feature_extraction.ipynb` — feature walkthrough | TODO | TASK-02 |
| TASK-10 | `03_baseline.ipynb` — baseline results | TODO | TASK-04 |

---

## Pre-written Task Cards

The cards below are ready to issue. Issue them one at a time in dependency order.

---

## TASK-01: `preprocess.py` — audio loading & normalization

**Status:** TODO  
**Depends on:** none  
**Assigned to:** Coder

### What to build
Implement `src/preprocess.py` with four public functions.
`load_audio(path, sr)` loads a WAV file and returns a mono float32 numpy array resampled to `sr`.
`normalize(audio)` applies peak normalization so the max absolute value is 1.0.
`segment(audio, sr, duration)` pads or trims the array to exactly `duration` seconds.
`process_directory(raw_dir, out_dir, sr, duration)` walks `data/raw/`, applies all three steps,
and writes results to `data/processed/` preserving the class-subfolder structure.

### Acceptance criteria
- [ ] `load_audio` returns dtype float32, shape (N,), mono
- [ ] `normalize` output has max abs value == 1.0 (within 1e-6)
- [ ] `segment` output length == int(sr * duration) exactly
- [ ] `process_directory` creates output files mirroring input subfolder structure
- [ ] All functions accept `pathlib.Path` arguments
- [ ] `pytest tests/test_preprocess.py` passes with ≥3 unit tests
- [ ] No hardcoded paths — `process_directory` takes `raw_dir` and `out_dir` as arguments

### Files to create or modify
- `src/preprocess.py`
- `tests/test_preprocess.py`

### Notes for Coder
Use `librosa.load` with `sr` and `mono=True`. For silence padding use `np.pad` with `mode='constant'`.
For trimming, slice from center to preserve the most informative part of the signal.
See DESIGN.md §Pipeline for parameter values: `SAMPLE_RATE=22050`, `DURATION=4.0`.

---

## TASK-02: `features.py` — Mel-spectrogram extraction

**Status:** TODO  
**Depends on:** TASK-01  
**Assigned to:** Coder

### What to build
Implement `src/features.py`.
`compute_melspectrogram(audio, sr, n_fft, hop_length, n_mels)` returns a log-scaled 2D numpy array
of shape `(n_mels, T)` where T = ceil(len(audio) / hop_length).
`save_spectrograms(processed_dir, spec_dir, **mel_kwargs)` iterates over `data/processed/`,
computes the spectrogram for each file, saves it as a `.npy` file under `data/spectrograms/`
mirroring the subfolder structure, and writes `data/spectrograms/labels.csv`
with columns `[path, class_label, class_id]`.

### Acceptance criteria
- [ ] Output shape is `(128, 173)` for a 4-second clip at 22050 Hz with default params
- [ ] No NaN or Inf values in any output array
- [ ] `labels.csv` has one row per spectrogram file, correct class_id mapping
- [ ] `pytest tests/test_features.py` passes with ≥3 unit tests
- [ ] Log scaling applied: `log_S = np.log(S + 1e-6)`

### Files to create or modify
- `src/features.py`
- `tests/test_features.py`

### Notes for Coder
Use `librosa.feature.melspectrogram`. Default params from DESIGN.md:
`N_FFT=2048`, `HOP_LENGTH=512`, `N_MELS=128`. Class ID mapping: 0=normal_operation,
1=metallic_impact, 2=friction_squeal, 3=alarm_tone, 4=silence_ambient.

---

## TASK-03: `dataset.py` — tf.data pipeline & splits

**Status:** TODO  
**Depends on:** TASK-02  
**Assigned to:** Coder

### What to build
Implement `src/dataset.py`.
`load_labels(spec_dir)` reads `labels.csv` and returns a pandas DataFrame.
`split_dataset(df, train_ratio, val_ratio, seed)` performs a stratified split and returns
three DataFrames: train, val, test.
`make_tf_dataset(df, batch_size, shuffle, augment)` returns a `tf.data.Dataset` that yields
`(spectrogram_tensor, one_hot_label)` batches. Spectrograms must be reshaped to `(128, 173, 1)`.
Optional augmentation: Gaussian noise and time-shift, applied only when `augment=True`.

### Acceptance criteria
- [ ] Split ratios sum to 1.0; stratification preserves class distribution within 2%
- [ ] Dataset yields tensors of shape `(batch, 128, 173, 1)` and `(batch, 5)`
- [ ] `augment=False` is deterministic (same seed → same batches)
- [ ] `pytest tests/test_dataset.py` passes with ≥3 unit tests

### Files to create or modify
- `src/dataset.py`
- `tests/test_dataset.py`

### Notes for Coder
Use `sklearn.model_selection.train_test_split` with `stratify=` for splits.
Use `tf.data.Dataset.from_tensor_slices` and `.map()` for pipeline construction.
Normalise spectrogram values to [-1, 1] before yielding.

---

## TASK-04: `baseline.py` — SVM baseline

**Status:** TODO  
**Depends on:** TASK-03  
**Assigned to:** Coder

### What to build
Implement `src/baseline.py`.
`load_features(df)` flattens each spectrogram to a 1D vector and returns `(X, y)` numpy arrays.
`train_svm(X_train, y_train)` fits an `SVC(kernel='rbf', C=10)` and returns the fitted model.
`evaluate(model, X_test, y_test)` returns a dict with keys `accuracy`, `macro_f1`,
and `report` (sklearn classification_report string).
A `__main__` block runs the full baseline pipeline and prints the report.

### Acceptance criteria
- [ ] SVM trains without error on the full spectrogram feature set
- [ ] `evaluate` dict contains `accuracy`, `macro_f1`, `report`
- [ ] `python src/baseline.py --data-dir data/` runs end-to-end and prints results
- [ ] `pytest tests/test_baseline.py` passes

### Files to create or modify
- `src/baseline.py`
- `tests/test_baseline.py`

### Notes for Coder
Flattening 128×173 = 22 144 features. SVM may be slow; add a comment recommending
`--mean-mfcc` flag as an alternative (40 MFCCs per clip). Do not implement it unless
asked — note it as a TODO.

---

## TASK-05: `model.py` — CNN architecture

**Status:** TODO  
**Depends on:** TASK-03  
**Assigned to:** Coder

### What to build
Implement `src/model.py`.
`build_cnn(input_shape, num_classes, dropout_rate)` returns a compiled `keras.Model`.
Architecture must match DESIGN.md §6 exactly: three Conv2D+BN+ReLU blocks with MaxPooling,
then GlobalAveragePooling2D, Dense(128)+Dropout, Dense(num_classes)+Softmax.
Compile with `Adam(lr=1e-3)`, `categorical_crossentropy`, metrics `['accuracy']`.

### Acceptance criteria
- [ ] `model.summary()` shows the exact layer sequence from DESIGN.md §6
- [ ] Input shape is `(128, 173, 1)`, output shape is `(5,)`
- [ ] `build_cnn` is callable with defaults: `input_shape=(128,173,1)`, `num_classes=5`, `dropout_rate=0.4`
- [ ] `pytest tests/test_model.py` passes: forward pass, output shape, layer count

### Files to create or modify
- `src/model.py`
- `tests/test_model.py`

### Notes for Coder
Use `keras.layers` API only (no `tf.keras` alias mixing). BatchNorm before ReLU.
Filter counts: 32 → 64 → 128 as per design doc.

---

## TASK-06: `train.py` — training loop & callbacks

**Status:** TODO  
**Depends on:** TASK-05  
**Assigned to:** Coder

### What to build
Implement `src/train.py` as a CLI script.
`python src/train.py --data-dir data/ --epochs 50 --batch-size 32 --output-dir models/`
must: load datasets via `dataset.py`, build model via `model.py`, attach callbacks
(`EarlyStopping(patience=10)`, `ReduceLROnPlateau(factor=0.5, patience=5)`,
`ModelCheckpoint` saving best weights), run `model.fit`, and save the trained model
to `models/cnn_best.keras` and training history to `models/history.json`.

### Acceptance criteria
- [ ] Script runs end-to-end with `--epochs 2` for a smoke test
- [ ] `models/cnn_best.keras` is produced and loadable with `keras.models.load_model`
- [ ] `models/history.json` contains `loss`, `val_loss`, `accuracy`, `val_accuracy` keys
- [ ] EarlyStopping and ReduceLROnPlateau fire correctly (verifiable from history)
- [ ] `--help` prints all arguments with descriptions

### Files to create or modify
- `src/train.py`

### Notes for Coder
Use `argparse`. Do not hardcode any paths. Class weights should be computed from
training set label distribution and passed to `model.fit(class_weight=...)`.

---

## TASK-07: `evaluate.py` — full evaluation suite

**Status:** TODO  
**Depends on:** TASK-06  
**Assigned to:** Coder

### What to build
Implement `src/evaluate.py` as a CLI script.
`python src/evaluate.py --model models/cnn_best.keras --data-dir data/ --output-dir results/`
must produce: accuracy and macro F1 printed to stdout, `results/confusion_matrix.png`,
`results/training_curves.png` (from `history.json`), `results/classification_report.txt`,
and `results/error_analysis.csv` listing misclassified samples with their true and
predicted labels plus the path to the spectrogram file.

### Acceptance criteria
- [ ] All output files are produced in `results/`
- [ ] Confusion matrix PNG is labelled with class names (not IDs)
- [ ] Error analysis CSV has columns: `path`, `true_label`, `predicted_label`
- [ ] `pytest tests/test_evaluate.py` passes (mock model, check file outputs)

### Files to create or modify
- `src/evaluate.py`
- `tests/test_evaluate.py`

### Notes for Coder
Use `seaborn.heatmap` for confusion matrix. Use `matplotlib` for training curves.
Load history from `models/history.json`, not from a live training run.
Primary metric is macro F1, not accuracy — make this prominent in the printed output.