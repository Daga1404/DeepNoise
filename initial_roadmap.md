# Initial Roadmap — Acoustic Event Classification (Part 2)

> Produced by the **Planner** agent from `CLAUDE.md` and `DESIGN.md`.
> This is the authoritative, ordered workplan. The **Coder** implements one step
> at a time, top to bottom. Each step is self-contained: it names the files to
> create, the public interface to build, and the acceptance criteria that the
> **Reviewer** will check before the step is marked DONE.

---

## How to use this document

1. The Coder picks the **lowest-numbered step that is not DONE** and whose
   dependencies are all DONE.
2. The Coder implements **only that step**, then outputs a Delivery Report.
3. The Reviewer audits against the step's **Definition of Done** and returns
   PASS or FAIL.
4. The Planner marks the step DONE or re-queues it with findings.

**Global rules (apply to every step):**

- Import all constants from `src/config.py`. Never re-declare `SAMPLE_RATE`,
  `DURATION`, `N_FFT`, `HOP_LENGTH`, `N_MELS`, `NUM_CLASSES`, `INPUT_SHAPE`,
  or `CLASS_LABELS` anywhere else.
- Use `pathlib.Path` for all filesystem paths. Never `os.path` string joins.
- Every CLI script accepts `--data-dir` (and `--output-dir` where it writes
  artefacts). No hardcoded paths.
- `data/` is gitignored — never commit audio, `.npy`, `.keras`, or `.json` model files.
- Add any newly used package to `requirements.txt`.
- Tests for preprocessing/feature steps must use **synthetic data** so they run
  without real audio in `data/raw/`.

---

## Dependency overview

```
STEP 0  scaffolding (dirs, requirements, __init__)
   └── STEP 1  config.py
          └── STEP 2  preprocess.py
                 └── STEP 3  features.py (+ augmentation)
                        ├── STEP 4  dataset.py
                        │      ├── STEP 5  baseline.py
                        │      └── STEP 6  model.py
                        │             └── STEP 7  train.py (noaug + aug)
                        │                    └── STEP 8  evaluate.py
                        │                           └── STEP 9  online.py
                        ├── STEP 10 notebooks/01_eda.ipynb
                        └── STEP 11 notebooks/02_feature_extraction.ipynb
                 STEP 5 → STEP 12 notebooks/03_baseline.ipynb
```

**Human gate:** `data/raw/<class>/` must be populated (see `MANUAL_TASKS.md`)
before STEP 2 onward can run *end-to-end* on real data. Unit tests for STEPS 2–3
do not need real audio. STEPS 7–9 and notebooks need real, preprocessed data.

---

## STEP 0 — Project scaffolding

**Depends on:** none
**Status:** TODO

### Goal
Create the directory skeleton and package plumbing so every later import works.

### Do this
- Create empty (gitkept) directories: `data/raw/<each class>/`, `data/processed/`,
  `data/spectrograms/`, `models/`, `results/`.
- Create `src/__init__.py` and `tests/__init__.py` so `from src.config import ...` resolves.
- Create / update `requirements.txt` with the Stack from `CLAUDE.md`
  (`librosa`, `soundfile`, `pydub`, `sounddevice`, `scikit-learn`, `tensorflow`,
  `numpy`, `pandas`, `matplotlib`, `seaborn`, `pytest`, `jupyter`).
- Add a `.gitignore` rule for `data/`, `models/*.keras`, `models/**/*.keras`,
  `*.npy`, and `__pycache__/` if not already present.

### Definition of Done
- [ ] All directories exist and are tracked via `.gitkeep`.
- [ ] `python -c "import src"` succeeds from repo root.
- [ ] `requirements.txt` lists every package in the Stack table.
- [ ] `data/` contents are gitignored.

---

## STEP 1 — `src/config.py` (single source of truth)

**Depends on:** STEP 0
**Status:** TODO

### Goal
One module exporting every project constant. No logic.

### Public interface
Module-level constants:
```python
SAMPLE_RATE = 22050
DURATION    = 4.0
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128
NUM_CLASSES = 5
INPUT_SHAPE = (128, 173, 1)
CLASS_LABELS = {0: "normal_operation", 1: "metallic_impact",
                2: "friction_squeal", 3: "alarm_tone", 4: "silence_ambient"}
```

### Definition of Done
- [ ] All eight names exported with values matching `CLAUDE.md` §Project Constants exactly.
- [ ] `CLASS_LABELS` is `dict[int, str]` covering IDs 0–4.
- [ ] No functions, no side effects — constants only.
- [ ] `tests/test_config.py` passes: import test + value assertions, and asserts
      `INPUT_SHAPE[1] == ceil(DURATION * SAMPLE_RATE / HOP_LENGTH)` (= 173).

### Files
- `src/config.py`
- `tests/test_config.py`

---

## STEP 2 — `src/preprocess.py` (load, normalize, segment)

**Depends on:** STEP 1
**Status:** TODO

### Goal
Standardize any input WAV to mono, 22050 Hz, peak-normalized, exactly 4 s.
This module is shared by training, evaluation, **and** the online system —
no preprocessing logic may be duplicated elsewhere.

### Public interface
```python
def load_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray  # float32, shape (N,), mono
def normalize(audio: np.ndarray) -> np.ndarray                   # peak-normalized, max|x|=1.0
def segment(audio: np.ndarray, sr: int = SAMPLE_RATE,
            duration: float = DURATION) -> np.ndarray            # length == int(sr*duration)
def process_directory(raw_dir: Path, out_dir: Path,
                      sr: int = SAMPLE_RATE, duration: float = DURATION) -> None
```

### Implementation notes
- `load_audio` → `librosa.load(path, sr=sr, mono=True)`.
- `normalize` → `audio / max(abs(audio))`; if the clip is all-zero, return it
  unchanged (no divide-by-zero).
- `segment` → **center**-crop if too long, symmetric zero-pad if too short.
  Target length `= int(sr * duration)` = 88200.
- `process_directory` walks `raw_dir/<class>/*.wav`, applies load→normalize→segment,
  writes to `out_dir/<class>/` mirroring the subfolder structure, and validates
  each output has no NaN and is not all-zero (except legitimately silent clips).
- **Do NOT apply silence trimming** — it would destroy `silence_ambient`.

### Definition of Done
- [ ] `load_audio` returns float32, shape `(N,)`, mono.
- [ ] `normalize` output max abs = 1.0 within 1e-6; all-zero input handled gracefully.
- [ ] `segment` output length == `int(sr*duration)` exactly; center-crop (not head-crop).
- [ ] `process_directory` mirrors the class-subfolder structure in `out_dir`.
- [ ] All functions accept `pathlib.Path` arguments.
- [ ] `tests/test_preprocess.py` passes with ≥3 synthetic-data tests
      (too-long clip, too-short clip, all-zero clip).

### Files
- `src/preprocess.py`
- `tests/test_preprocess.py`

---

## STEP 3 — `src/features.py` (log-Mel + augmentation)

**Depends on:** STEP 2
**Status:** TODO

### Goal
Convert preprocessed audio to log-Mel spectrograms and provide augmentation
primitives. The module computes augmentations but **does not decide** which
samples are training samples — that rule is enforced in `dataset.py`.

### Public interface
```python
def compute_melspectrogram(audio, sr=SAMPLE_RATE, n_fft=N_FFT,
                           hop_length=HOP_LENGTH, n_mels=N_MELS) -> np.ndarray  # (n_mels, T)
def augment_audio(audio, sr=SAMPLE_RATE) -> list[np.ndarray]      # gaussian noise + time shift
def augment_spectrogram(log_S) -> list[np.ndarray]               # SpecAugment freq + time masks
def save_spectrograms(processed_dir: Path, spec_dir: Path,
                      augment_train: bool = False, **mel_kwargs) -> None
```

### Implementation notes
- `compute_melspectrogram`: `librosa.feature.melspectrogram(...)` then
  `log_S = np.log(mel_S + 1e-6)`. Output shape `(128, 173)` for a 4 s clip.
- `augment_audio`: Gaussian noise `N(0, 0.005)` added to raw audio; time shift
  `np.roll` by ±10% of length. Return a list of ≥2 distinct arrays.
- `augment_spectrogram`: SpecAugment — zero a random frequency band (20 bins)
  and a random time window (30 frames). Return ≥1 array.
- `save_spectrograms`: iterate `processed_dir/<class>/*.wav`, compute spectrograms,
  write `.npy` files into `spec_dir/`, and write `labels.csv` with columns
  `path, class_label, class_id, is_augmented`. When `augment_train=True`, write
  augmented copies marked `is_augmented=True`. (Final train-only enforcement is
  `dataset.py`'s job; here just tag them.)

### Definition of Done
- [ ] Output shape `(128, 173)` for a 4 s @ 22050 Hz clip with default params.
- [ ] No NaN/Inf in any output; log scaling `np.log(mel_S + 1e-6)` applied.
- [ ] `augment_audio` returns ≥2 distinct arrays; `augment_spectrogram` returns
      ≥1 array with visibly zeroed freq/time bands.
- [ ] `labels.csv` columns exactly: `path, class_label, class_id, is_augmented`.
- [ ] `tests/test_features.py` passes with ≥4 synthetic-data tests.

### Files
- `src/features.py`
- `tests/test_features.py`

---

## STEP 4 — `src/dataset.py` (splits + tf.data pipeline)

**Depends on:** STEP 3
**Status:** TODO

### Goal
Stratified 70/15/15 split and `tf.data` pipelines. **The no-leakage rule is the
single most important correctness constraint in the project.**

### Public interface
```python
def load_labels(spec_dir: Path) -> pd.DataFrame
def split_dataset(df, train_ratio=0.70, val_ratio=0.15,
                  seed=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]  # train, val, test
def make_tf_dataset(df, batch_size=32, shuffle=False, augment=False) -> tf.data.Dataset
```

### Implementation notes
- `split_dataset`: split on the **non-augmented** subset only, stratified by
  `class_id` (`sklearn.model_selection.train_test_split(..., stratify=...)`),
  then `pd.concat` the augmented rows **onto the train split only**.
- `make_tf_dataset`: yields `(spectrogram_tensor, one_hot_label)` with tensor
  shape `(128, 173, 1)` and label shape `(5,)`. Normalize spectrograms to `[-1, 1]`
  before yielding. `shuffle=True` only for train.

### Definition of Done
- [ ] Rows with `is_augmented=True` appear **only** in the train split — never val/test.
- [ ] Split ratios computed on non-augmented samples; per-class stratification within 2%.
- [ ] Dataset yields shapes `(batch, 128, 173, 1)` and `(batch, 5)`.
- [ ] Spectrograms normalized to `[-1, 1]` before yielding.
- [ ] `tests/test_dataset.py` passes with ≥3 tests, including an explicit
      no-leakage assertion.

### Files
- `src/dataset.py`
- `tests/test_dataset.py`

---

## STEP 5 — `src/baseline.py` (SVM baseline)

**Depends on:** STEP 4
**Status:** TODO

### Goal
SVM (RBF) baseline on flattened spectrograms; produce baseline report + confusion matrix.

### Public interface
```python
def load_features(df) -> tuple[np.ndarray, np.ndarray]   # X (flattened), y
def train_svm(X_train, y_train) -> SVC                    # SVC(kernel='rbf', C=10)
def evaluate(model, X_test, y_test, output_dir: Path) -> dict  # accuracy, macro_f1, report
# __main__: python src/baseline.py --data-dir data/ --output-dir results/
```

### Definition of Done
- [ ] SVM trains on full 22,144-feature vectors without error.
- [ ] `evaluate` returns a dict with `accuracy`, `macro_f1`, `report`.
- [ ] `results/baseline_confusion.png` saved with **class-name** axis labels (not IDs).
- [ ] `results/baseline_report.txt` saved with per-class precision/recall/F1.
- [ ] `python src/baseline.py --data-dir data/ --output-dir results/` runs end-to-end.
- [ ] `tests/test_baseline.py` passes.

### Files
- `src/baseline.py`
- `tests/test_baseline.py`

---

## STEP 6 — `src/model.py` (CNN architecture)

**Depends on:** STEP 4
**Status:** TODO

### Goal
Compiled Keras CNN matching `DESIGN.md` §CNN Architecture exactly.

### Public interface
```python
def build_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
              dropout_rate=0.4) -> keras.Model
```

### Architecture (exact)
```
Input (128,173,1)
→ Conv2D(32,3×3) + BatchNorm + ReLU + MaxPool(2×2)
→ Conv2D(64,3×3) + BatchNorm + ReLU + MaxPool(2×2)
→ Conv2D(128,3×3) + BatchNorm + ReLU
→ GlobalAveragePooling2D
→ Dense(128) + ReLU + Dropout(0.4)
→ Dense(5) + Softmax
```
Compile: `Adam(lr=1e-3)`, `categorical_crossentropy`, metric `accuracy`.

### Definition of Done
- [ ] Layer sequence matches DESIGN.md exactly.
- [ ] Input `(128,173,1)`, output `(5,)` Softmax.
- [ ] Compiled as specified; `build_cnn()` callable with config defaults.
- [ ] `tests/test_model.py` passes: forward pass, output shape, layer count.

### Files
- `src/model.py`
- `tests/test_model.py`

---

## STEP 7 — `src/train.py` (two runs: noaug + aug)

**Depends on:** STEP 6
**Status:** TODO

### Goal
CLI script that runs one full training pass; invoked twice for the augmentation
experiment.

### CLI
```bash
python src/train.py --data-dir data/ --augment false --output-dir models/noaug/ --epochs 50
python src/train.py --data-dir data/ --augment true  --output-dir models/aug/   --epochs 50
```

### Behaviour
Load datasets (via `dataset.py`), build CNN (`model.py`), compute class weights
from the training set, attach `EarlyStopping(patience=10, restore_best_weights=True)`
and `ReduceLROnPlateau(factor=0.5, patience=5)`, fit (batch 32, max 50 epochs),
then save `<output-dir>/cnn_best.keras` and `<output-dir>/history.json`.

### Definition of Done
- [ ] Both `--augment false` and `--augment true` complete without error.
- [ ] `cnn_best.keras` loadable via `keras.models.load_model`.
- [ ] `history.json` contains `loss`, `val_loss`, `accuracy`, `val_accuracy` arrays.
- [ ] Class weights computed and passed to `model.fit`.
- [ ] EarlyStopping + ReduceLROnPlateau attached.
- [ ] `python src/train.py --epochs 2 --augment false --output-dir /tmp/smoke`
      smoke-tests in < 60 s.

### Files
- `src/train.py`

---

## STEP 8 — `src/evaluate.py` (full evaluation suite)

**Depends on:** STEP 7
**Status:** TODO

### Goal
Evaluate a trained CNN on the held-out test set and emit every required artefact.

### CLI
```bash
python src/evaluate.py --model models/aug/cnn_best.keras --data-dir data/ --output-dir results/
```

### Outputs
- Accuracy + **macro F1 (printed first, primary)** to stdout and `results/cnn_report.txt`.
- `results/cnn_confusion.png` — seaborn heatmap, class-name axes.
- `results/training_curves.png` — loss + accuracy subplots from `history.json`.
- `results/error_analysis.csv` — columns `path, true_label, predicted_label, confidence`,
  ≥5 misclassified rows per class where available.
- `results/augmentation_comparison.md` — built from both `models/noaug/history.json`
  and `models/aug/history.json` if present; reports test accuracy + macro F1 per run.

### Definition of Done
- [ ] All output files produced.
- [ ] Confusion matrix axes labelled with class-name strings.
- [ ] Macro F1 is the first metric printed.
- [ ] Error analysis has ≥5 rows per class where possible.
- [ ] `augmentation_comparison.md` includes test accuracy + macro F1 for both runs.
- [ ] `tests/test_evaluate.py` passes (mock model, check file outputs).

### Files
- `src/evaluate.py`
- `tests/test_evaluate.py`

---

## STEP 9 — `src/online.py` (real-time inference)

**Depends on:** STEP 8
**Status:** TODO

### Goal
Real-time mic capture + inference, with a testable file-fallback mode. **Reuses**
preprocessing — no duplicated logic.

### CLI
```bash
python src/online.py --model models/aug/cnn_best.keras                 # mic mode
python src/online.py --model models/aug/cnn_best.keras --file clip.wav # file mode
```

### Behaviour
Mic mode: capture 4 s via `sounddevice.rec`, preprocess (`normalize`, `segment`
from `preprocess.py`; `compute_melspectrogram` from `features.py`), `model.predict`,
print `[HH:MM:SS] Predicted: <class_name>  (confidence: <0.xx>)`, repeat until Ctrl+C.
File mode: run the same pipeline once on a WAV and exit.

### Definition of Done
- [ ] Imports `normalize`, `segment` from `preprocess.py` and `compute_melspectrogram`
      from `features.py` — **no duplicated preprocessing**.
- [ ] `--file` mode works with no microphone attached.
- [ ] Predicted class name comes from `CLASS_LABELS` in `config.py`.
- [ ] Output line matches the format above and includes a softmax confidence.
- [ ] `python src/online.py --model models/aug/cnn_best.keras --file tests/fixtures/sample.wav` runs.

### Files
- `src/online.py`
- `tests/fixtures/sample.wav` (small fixture for file-mode test)

---

## STEP 10 — `notebooks/01_eda.ipynb` (EDA + spectrograms)

**Depends on:** STEP 3
**Status:** TODO

### Contents
1. Class-distribution bar chart (sample counts per class).
2. One example waveform per class (5 subplots).
3. One log-Mel spectrogram per class via `librosa.display.specshow`
   (5 subplots, **required for the report**, shared colour scale).
4. Commentary cells describing each class and how it differs from the others.

### Definition of Done
- [ ] 5 spectrogram plots, one per class, clearly labelled, same colour scale.
- [ ] Notebook runs top-to-bottom without error against `data/spectrograms/`.
- [ ] Class-distribution plot shows approximate balance.

### Files
- `notebooks/01_eda.ipynb`

---

## STEP 11 — `notebooks/02_feature_extraction.ipynb`

**Depends on:** STEP 3
**Status:** TODO

### Contents
Walk through the feature pipeline using `src/features.py`: pick a sample clip,
show preprocessing → log-Mel spectrogram, then visualise each augmentation
(Gaussian noise, time shift, SpecAugment freq/time masks) side-by-side with the
original so the augmentation effects are visible.

### Definition of Done
- [ ] Notebook calls into `src/features.py` (no re-implemented logic).
- [ ] Shows original vs. each augmentation visually.
- [ ] Runs top-to-bottom without error.

### Files
- `notebooks/02_feature_extraction.ipynb`

---

## STEP 12 — `notebooks/03_baseline.ipynb`

**Depends on:** STEP 5
**Status:** TODO

### Contents
Reproduce the SVM baseline interactively using `src/baseline.py`: load features,
train/evaluate, display the confusion matrix and per-class metrics inline, and
add commentary on baseline performance (expected 60–75%) as the bar the CNN must clear.

### Definition of Done
- [ ] Notebook calls into `src/baseline.py` (no re-implemented logic).
- [ ] Confusion matrix + per-class report shown inline.
- [ ] Runs top-to-bottom without error.

### Files
- `notebooks/03_baseline.ipynb`

---

## Milestones

| Milestone | Steps | Outcome |
|---|---|---|
| **M1 — Data pipeline** | 0–4 | Raw audio → spectrograms → leak-free `tf.data` splits |
| **M2 — Baseline** | 5, 12 | SVM baseline metrics + confusion matrix |
| **M3 — CNN trained** | 6–7 | Two trained models (`noaug`, `aug`) |
| **M4 — Evaluation** | 8 | Full metrics, curves, error analysis, aug comparison |
| **M5 — Online + report** | 9–11 | Live demo system + EDA/feature notebooks |

---

## Open blockers (human-owned)

| ID | Blocker | Waiting on | Blocks |
|---|---|---|---|
| B1 | `data/raw/<class>/` not yet populated | Human — see `MANUAL_TASKS.md` | End-to-end runs of STEPS 2–12 (unit tests for 1–6 are unaffected) |
