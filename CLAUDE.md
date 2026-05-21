# CLAUDE.md — Acoustic Event Classification Project

This file provides context for Claude Code sessions working on this project.
Read it at the start of every session before making changes.

---

## Project Summary

An acoustic event classification system using a CNN trained on Mel-spectrogram representations of audio.
This is a university ML project (two-part deliverable): **Part 1 = planning** (done), **Part 2 = implementation**.

**Goal:** Classify short audio clips (≤5s) into one of 5 acoustic event classes.

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
├── CLAUDE.md               ← YOU ARE HERE
├── DESIGN.md               ← Full technical design document
├── data/
│   ├── raw/                ← Original audio files, organized by class subfolder
│   ├── processed/          ← Resampled 22050Hz mono WAVs
│   └── spectrograms/       ← Precomputed .npy spectrogram arrays + labels.csv
├── notebooks/
│   ├── 01_eda.ipynb        ← Exploratory analysis, waveform/spectrogram plots
│   ├── 02_feature_extraction.ipynb
│   └── 03_baseline.ipynb
├── src/
│   ├── preprocess.py       ← load_audio(), resample(), normalize(), segment()
│   ├── features.py         ← compute_melspectrogram(), save_spectrograms()
│   ├── dataset.py          ← AcousticDataset class, tf.data pipeline, splits
│   ├── baseline.py         ← train_svm_baseline(), evaluate_baseline()
│   ├── model.py            ← build_cnn() returns compiled keras.Model
│   ├── train.py            ← CLI entry point: python src/train.py --config ...
│   └── evaluate.py         ← confusion matrix, F1, error analysis, plots
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

Place raw audio files like this:
```
data/raw/
    normal_operation/   *.wav
    metallic_impact/    *.wav
    friction_squeal/    *.wav
    alarm_tone/         *.wav
    silence_ambient/    *.wav
```

`preprocess.py` will walk this tree and build `data/processed/` with the same structure.

`features.py` will produce:
```
data/spectrograms/
    normal_operation/   *.npy
    metallic_impact/    *.npy
    ...
    labels.csv          (path, class_label, class_id)
```

---

## Evaluation Checklist

When evaluating a trained model, always produce:
- [ ] Accuracy on test set
- [ ] Macro F1-score on test set
- [ ] Per-class precision, recall, F1
- [ ] Confusion matrix (saved as PNG)
- [ ] Training + validation loss/accuracy curves (saved as PNG)
- [ ] At least 5 misclassified examples per class with audio paths logged

---

## Implementation Order (Part 2)

1. `src/preprocess.py` — audio loading and normalization
2. `src/features.py` — spectrogram computation and saving
3. `src/dataset.py` — tf.data dataset and splits
4. `src/baseline.py` — SVM baseline, report metrics
5. `src/model.py` — CNN definition
6. `src/train.py` — training loop with callbacks
7. `src/evaluate.py` — full evaluation suite
8. Notebooks for EDA and result visualization

---

## Testing

```bash
pytest tests/ -v
```

Each pipeline stage should have at least one unit test covering:
- Correct output shape
- No NaN values in spectrogram output
- Split sizes match requested ratios

---

## Notes for Claude Code

- **Never commit raw audio files** — keep them in `data/` which is gitignored
- Prefer `pathlib.Path` over `os.path` throughout
- All scripts should accept a `--data-dir` argument so paths are not hardcoded
- When adding dependencies, update `requirements.txt` immediately
- Keep functions small and testable — avoid monolithic scripts
- If a TensorFlow version conflict appears, check `python -c "import tensorflow as tf; print(tf.__version__)"` before debugging further