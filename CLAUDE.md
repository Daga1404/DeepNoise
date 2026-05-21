# CLAUDE.md — Acoustic Event Classification Project

This file provides context for Claude Code sessions working on this project.
Read it at the start of every session before making changes.

---

## Project Summary

An acoustic event classification system using a CNN trained on Mel-spectrogram representations of audio.
This is a university ML project (two-part deliverable): **Part 1 = planning** (done), **Part 2 = implementation**.

**Goal:** Classify short audio clips (≤5s) into one of 5 acoustic event classes.

---

## Multi-Agent Workflow

This project uses a three-agent system. Each agent has its own instruction file under `agents/`.

| Agent | File | Invoked when |
|---|---|---|
| **Planner** | `agents/planner.md` | Starting a new task, decomposing work, resolving blockers |
| **Coder** | `agents/coder.md` | Implementing any file in `src/`, `tests/`, or `notebooks/` |
| **Reviewer** | `agents/reviewer.md` | A module is complete and ready for quality check |

**Standard loop:**
```
Planner → issues task card
    ↓
Coder → implements, outputs file list + notes
    ↓
Reviewer → audits, returns PASS or FAIL with findings
    ↓
Planner → marks done or re-queues with findings attached
```

Always load the relevant agent file before starting that role's work.

---

## Classes

| ID | Label | Description |
|---|---|---|
| 0 | `normal_operation` | Healthy machine hum/rumble |
| 1 | `metallic_impact` | Sharp percussive collision events |
| 2 | `friction_squeal` | High-frequency squealing, worn bearings |
| 3 | `alarm_tone` | Synthetic buzzer/alarm repetitions |
| 4 | `silence_ambient` | Background/idle noise floor |

---

## Stack

- **Language:** Python 3.11
- **Audio:** `librosa`, `soundfile`, `pydub`
- **ML baseline:** `scikit-learn` (SVM)
- **Deep learning:** `tensorflow` + `keras`
- **Data:** `numpy`, `pandas`
- **Visualization:** `matplotlib`, `seaborn`
- **Tests:** `pytest`
- **Notebooks:** `jupyter`

Install everything with:
```bash
pip install -r requirements.txt
```

---

## Repository Layout

```
acoustic-classifier/
├── CLAUDE.md                   ← YOU ARE HERE
├── DESIGN.md                   ← Full technical design document
├── agents/
│   ├── planner.md              ← Planner agent instructions
│   ├── coder.md                ← Coder agent instructions
│   └── reviewer.md             ← Reviewer agent instructions
├── data/
│   ├── raw/                    ← Original audio files, organized by class subfolder
│   ├── processed/              ← Resampled 22050Hz mono WAVs
│   └── spectrograms/           ← Precomputed .npy spectrogram arrays + labels.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_baseline.ipynb
├── src/
│   ├── preprocess.py           ← load_audio(), resample(), normalize(), segment()
│   ├── features.py             ← compute_melspectrogram(), save_spectrograms()
│   ├── dataset.py              ← AcousticDataset class, tf.data pipeline, splits
│   ├── baseline.py             ← train_svm_baseline(), evaluate_baseline()
│   ├── model.py                ← build_cnn() returns compiled keras.Model
│   ├── train.py                ← CLI entry point: python src/train.py --config ...
│   └── evaluate.py             ← confusion matrix, F1, error analysis, plots
└── tests/
    └── test_preprocess.py
```

---

## Audio Pipeline (key parameters)

```python
SAMPLE_RATE   = 22050   # Hz, mono
DURATION      = 4.0     # seconds — pad/trim all clips to this
N_FFT         = 2048
HOP_LENGTH    = 512
N_MELS        = 128
# Output spectrogram shape: (128, 173)  →  fed to CNN as (128, 173, 1)
```

All spectrograms are log-scaled: `log_S = np.log(S + 1e-6)`

---

## CNN Architecture (target)

```
Input (128, 173, 1)
→ Conv2D(32, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
→ Conv2D(64, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
→ Conv2D(128, 3×3) + BatchNorm + ReLU
→ GlobalAveragePooling2D
→ Dense(128) + ReLU + Dropout(0.4)
→ Dense(5) + Softmax
```

Loss: `categorical_crossentropy`  
Optimizer: `Adam(lr=1e-3)` + `ReduceLROnPlateau`  
Primary metric: **macro F1-score** (not accuracy — dataset may be imbalanced)

---

## Baseline Model

SVM with RBF kernel on flattened or mean-reduced Mel-spectrogram features.
Expected accuracy: ~60–75%. The CNN should beat this.

---

## Data Directory Convention

```
data/raw/
    normal_operation/   *.wav
    metallic_impact/    *.wav
    friction_squeal/    *.wav
    alarm_tone/         *.wav
    silence_ambient/    *.wav
```

`preprocess.py` walks this tree → `data/processed/` (same structure).  
`features.py` produces `data/spectrograms/` + `labels.csv`.

---

## Evaluation Checklist

- [ ] Accuracy on test set
- [ ] Macro F1-score on test set
- [ ] Per-class precision, recall, F1
- [ ] Confusion matrix (saved as PNG)
- [ ] Training + validation loss/accuracy curves (saved as PNG)
- [ ] At least 5 misclassified examples per class with audio paths logged

---

## Implementation Order (Part 2)

1. `src/preprocess.py`
2. `src/features.py`
3. `src/dataset.py`
4. `src/baseline.py`
5. `src/model.py`
6. `src/train.py`
7. `src/evaluate.py`
8. Notebooks for EDA and result visualization

---

## Testing

```bash
pytest tests/ -v
```

Each pipeline stage needs at least one unit test: correct output shape, no NaNs, split ratios.

---

## Notes for Claude Code

- **Never commit raw audio files** — `data/` is gitignored
- Prefer `pathlib.Path` over `os.path` throughout
- All scripts accept `--data-dir` so paths are never hardcoded
- Update `requirements.txt` when adding dependencies
- Keep functions small and testable
- Check `python -c "import tensorflow as tf; print(tf.__version__)"` before debugging TF version issues