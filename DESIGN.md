# Acoustic Event Classification System — Design Document
# Part 2: Implementation

> **Status:** Active implementation phase.
> Part 1 (planning) is complete. This document has been updated to reflect
> the full Part 2 requirements: data augmentation, online system, and live testing.

---

## Project Overview

An intelligent acoustic monitoring system that classifies short audio segments into discrete
event categories using a CNN trained on log-Mel spectrogram representations.

**Scenario:** Industrial / predictive maintenance — classifying machine sounds to detect
operational states and early fault signals in real-time.

---

## Acoustic Classes

| ID | Label | Description | Challenge |
|---|---|---|---|
| 0 | `normal_operation` | Steady hum/rumble of healthy equipment | May overlap with low-energy ambient |
| 1 | `metallic_impact` | Sharp percussive collisions, drops, hammering | Short transients, easy to miss |
| 2 | `friction_squeal` | High-frequency squealing from worn bearings or brakes | Confused with alarms |
| 3 | `alarm_tone` | Repetitive synthetic tones, buzzers, warning signals | Varies across devices |
| 4 | `silence_ambient` | Background noise / idle state | Low-information anchor class |

---

## Technology Stack

### Language
**Python 3.11+** — dominant in ML/audio research; best ecosystem coverage.

### Audio Processing
| Library | Role |
|---|---|
| `librosa` | Feature extraction, Mel-spectrograms, MFCCs, resampling |
| `soundfile` | Audio I/O (WAV, FLAC) |
| `pydub` | Segmentation, format conversion |
| `sounddevice` | Real-time microphone capture for the online system |
| `numpy` | Array manipulation throughout |

### Machine Learning
| Library | Role |
|---|---|
| `scikit-learn` | SVM baseline, metrics, stratified splits |
| `tensorflow` + `keras` | CNN architecture, training loop |

### Evaluation & Visualization
| Library | Role |
|---|---|
| `matplotlib` / `seaborn` | Training curves, confusion matrices, spectrogram plots |
| `pandas` | Dataset manifests, metadata CSVs |
| `pytest` | Unit tests for all pipeline stages |

---

## Dataset

### Sources

| Source | Type | Classes used | Samples target |
|---|---|---|---|
| ESC-50 | Public download | `silence_ambient`, `alarm_tone`, `metallic_impact` subsets | ≥ 200/class |
| UrbanSound8K | Public download | `metallic_impact`, `alarm_tone` reinforcement | ≥ 200/class |
| Team recordings | Manual capture | `normal_operation`, `friction_squeal` | ≥ 200/class |

See `MANUAL_TASKS.md` for exact download instructions.

### Dataset Targets

| Class | Min samples | Duration | Format | SR |
|---|---|---|---|---|
| `normal_operation` | 200 | 4 s (fixed) | WAV mono | 22050 Hz |
| `metallic_impact` | 200 | 4 s (fixed) | WAV mono | 22050 Hz |
| `friction_squeal` | 200 | 4 s (fixed) | WAV mono | 22050 Hz |
| `alarm_tone` | 200 | 4 s (fixed) | WAV mono | 22050 Hz |
| `silence_ambient` | 200 | 4 s (fixed) | WAV mono | 22050 Hz |

All raw files are placed in `data/raw/<class_label>/` before any processing.
The preprocessing pipeline standardizes everything to 22050 Hz mono WAV at exactly 4 s.

### Change from Part 1

Part 1 proposed ESC-50 alone. Part 2 adds UrbanSound8K to reinforce impact and alarm classes
where ESC-50 coverage is thin, and requires team recordings for the two most domain-specific
classes (`normal_operation`, `friction_squeal`) to ensure industrial relevance.

---

## Audio Preprocessing

The same preprocessing logic is used for offline training, offline evaluation, and the online
system. No step may differ between these modes.

### Steps applied (in order)

| Step | Function | Why |
|---|---|---|
| Load | `librosa.load(path, sr=22050, mono=True)` | Standardize SR and channels |
| Normalize amplitude | Peak normalization to max abs = 1.0 | Remove recording level variation |
| Trim / pad | Crop from center or zero-pad to exactly 4 s | Fixed input size for CNN |
| Validate | Assert no NaN, no all-zero | Catch corrupted files early |

### What is NOT applied
- Silence trimming (would destroy `silence_ambient` class)
- Stereo-specific processing (always load as mono)

### Online system preprocessing

The online system captures audio via `sounddevice` at 22050 Hz, mono, in 4-second windows,
then applies the same normalize → pad/trim → spectrogram pipeline before calling `model.predict`.
**The preprocessing code is shared** — `src/preprocess.py` is imported by both
`src/train.py` and `src/online.py`.

---

## Data Augmentation

Augmentation is applied **only to the training set**. Validation and test sets receive
raw (unaugmented) preprocessed audio.

### Augmentation strategies implemented

| Strategy | Implementation | Rationale |
|---|---|---|
| Gaussian noise | Add N(0, 0.005) to raw audio | Simulates sensor noise |
| Time shift | Roll array by ±10% of length | Teaches temporal invariance |
| SpecAugment — freq mask | Zero out random frequency bands in spectrogram | Robust to partial occlusion |
| SpecAugment — time mask | Zero out random time windows in spectrogram | Robust to signal gaps |

### Augmentation experiment

Two training runs are required for the report:

| Condition | Label | Notes |
|---|---|---|
| No augmentation | `baseline_noaug` | Raw preprocessed spectrograms only |
| With augmentation | `baseline_aug` | Gaussian noise + time shift + SpecAugment |

Results of both runs must be reported in `results/augmentation_comparison.md` with
test accuracy and macro F1 for each condition.

---

## Audio Representation

**Selected:** Log-Mel spectrogram (identical to Mel-spectrogram with log compression applied).

### Parameters

```python
SAMPLE_RATE  = 22050   # Hz, mono
DURATION     = 4.0     # seconds
N_FFT        = 2048
HOP_LENGTH   = 512
N_MELS       = 128

# Derived output shape fed to CNN
# T = ceil(DURATION * SAMPLE_RATE / HOP_LENGTH) = 173 frames
INPUT_SHAPE  = (128, 173, 1)   # (n_mels, time_frames, channels)

# Log scaling
log_S = np.log(mel_S + 1e-6)
```

### Visualization requirement

`notebooks/01_eda.ipynb` must include one plotted log-Mel spectrogram per class
(5 plots total), clearly labelled, using `librosa.display.specshow`.

### Representation comparison

| Representation | Time | Freq | CNN-ready | Decision |
|---|---|---|---|---|
| Raw waveform | ✅ | ❌ | ❌ | Rejected — no frequency structure |
| STFT spectrogram | ✅ | ✅ | ✅ | Rejected — not perceptual |
| **Log-Mel spectrogram** | ✅ | ✅ | ✅ | **Selected** |
| MFCCs | Partial | ✅ | ⚠️ | Baseline input only |
| Chroma | ❌ | Partial | ⚠️ | Rejected — pitch-focused |

---

## Machine Learning Pipeline

```
data/raw/<class>/           ← raw audio placed manually
        ↓
[1] preprocess.py
    load → mono → 22050 Hz → normalize → trim/pad to 4 s
        ↓
data/processed/<class>/     ← standardized WAVs
        ↓
[2] features.py
    log-Mel spectrogram (128 × 173) → .npy
    augmented copies written for training set only
        ↓
data/spectrograms/          ← .npy arrays + labels.csv
        ↓
[3] dataset.py
    stratified split 70/15/15 → tf.data pipelines
        ↓
        ├── [4] baseline.py    SVM (RBF) on flattened features
        │         ↓
        │   results/baseline_report.txt
        │
        └── [5] model.py       CNN architecture
                  ↓
              [6] train.py     fit — no-aug run + aug run
                  ↓
              models/cnn_best.keras  +  models/history.json
                  ↓
              [7] evaluate.py  accuracy, macro F1, confusion matrix,
                               training curves, error analysis
                  ↓
              results/         ← all evaluation artefacts
                  ↓
              [8] online.py    real-time capture → preprocess → predict
```

---

## Baseline Model

**Model:** SVM with RBF kernel (`sklearn.svm.SVC(kernel='rbf', C=10)`)

**Input:** Flattened log-Mel spectrogram (128 × 173 = 22,144 features)

**Expected accuracy:** 60–75%

**Required outputs:**
- Test accuracy
- Confusion matrix (`results/baseline_confusion.png`)
- Per-class precision, recall, F1 (`results/baseline_report.txt`)

---

## CNN Architecture

```
Input: (128, 173, 1)
    ↓
Conv2D(32, 3×3) + BatchNorm + ReLU
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3) + BatchNorm + ReLU
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(128, 3×3) + BatchNorm + ReLU
    ↓
GlobalAveragePooling2D
    ↓
Dense(128) + ReLU + Dropout(0.4)
    ↓
Dense(5) + Softmax
```

### Training configuration

| Parameter | Value |
|---|---|
| Loss | `categorical_crossentropy` |
| Optimizer | `Adam(lr=1e-3)` |
| LR schedule | `ReduceLROnPlateau(factor=0.5, patience=5)` |
| Early stopping | `EarlyStopping(patience=10, restore_best_weights=True)` |
| Batch size | 32 |
| Max epochs | 50 |
| Class weights | Computed from training set; passed to `model.fit` |

---

## Offline Evaluation

Both the baseline and CNN must be evaluated on the **same held-out test set**.

### Required outputs

| Output | File | Tool |
|---|---|---|
| Test accuracy + macro F1 | stdout + `results/cnn_report.txt` | sklearn |
| Confusion matrix | `results/cnn_confusion.png` | seaborn heatmap |
| Training loss curve | `results/training_curves.png` | matplotlib |
| Training accuracy curve | (same figure, second subplot) | matplotlib |
| Per-class precision/recall/F1 | `results/cnn_report.txt` | sklearn |
| Error analysis table | `results/error_analysis.csv` | pandas |
| Augmentation comparison | `results/augmentation_comparison.md` | manual + metrics |

### Error analysis

`error_analysis.csv` columns: `path`, `true_label`, `predicted_label`, `confidence`.
Must include ≥5 misclassified samples per class where available.

---

## Online System

**File:** `src/online.py`

The online system captures 4-second audio windows from the microphone and classifies
each window using the trained CNN. It runs in a continuous loop until the user presses Ctrl+C.

### Required behaviour

1. Load model from `models/cnn_best.keras`
2. Open microphone stream at 22050 Hz, mono, via `sounddevice`
3. Capture exactly 4 s of audio
4. Apply the same preprocessing as training (normalize → log-Mel spectrogram)
5. Call `model.predict` on the spectrogram
6. Print predicted class name and confidence to stdout
7. Repeat from step 3

### Output format (per window)

```
[00:04] Predicted: normal_operation  (confidence: 0.91)
[00:08] Predicted: metallic_impact   (confidence: 0.74)
```

### Online system testing

During the final demonstration, at least one live example from each class must be
tested and the system output recorded. Results go in `results/online_test_log.txt`.

---

## Validation Strategy

- **Split:** Stratified 70% / 15% / 15% — no overlap, no leakage
- **Primary metric:** Macro F1-score (equal class weight regardless of sample count)
- **Secondary metric:** Accuracy
- **Confusion matrix:** Named axes (class label strings, not IDs)
- **Training curves:** Loss and accuracy for both train and validation sets
- **Error analysis:** ≥5 misclassified samples per class, reviewed by ear

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Overfitting on small dataset | HIGH | Dropout, BatchNorm, augmentation, early stopping |
| Class imbalance | MEDIUM | Stratified splits + class weights in `model.fit` |
| Microphone not available in demo env | MEDIUM | Fallback: `online.py --file` mode accepts a WAV path |
| Augmentation hurts performance | LOW | Compare aug vs no-aug runs; report both |
| Short transients lost in 4 s window | MEDIUM | Center-crop trimming; verify in EDA notebook |

---

## Repository Structure

```
acoustic-classifier/
├── CLAUDE.md                       ← AI assistant context
├── DESIGN.md                       ← This document
├── MANUAL_TASKS.md                 ← What you must do by hand
├── agents/
│   ├── planner.md
│   ├── coder.md
│   └── reviewer.md
├── data/
│   ├── raw/                        ← Place downloaded/recorded audio here
│   │   ├── normal_operation/
│   │   ├── metallic_impact/
│   │   ├── friction_squeal/
│   │   ├── alarm_tone/
│   │   └── silence_ambient/
│   ├── processed/                  ← Generated by preprocess.py
│   └── spectrograms/               ← Generated by features.py
├── models/
│   ├── cnn_best.keras
│   └── history.json
├── notebooks/
│   ├── 01_eda.ipynb                ← Spectrogram plots (1 per class, required)
│   ├── 02_feature_extraction.ipynb
│   └── 03_baseline.ipynb
├── results/
│   ├── baseline_confusion.png
│   ├── baseline_report.txt
│   ├── cnn_confusion.png
│   ├── cnn_report.txt
│   ├── training_curves.png
│   ├── error_analysis.csv
│   ├── augmentation_comparison.md
│   └── online_test_log.txt
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── dataset.py
│   ├── baseline.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── online.py                   ← NEW: real-time inference
├── tests/
│   ├── test_preprocess.py
│   ├── test_features.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_evaluate.py
├── requirements.txt
└── README.md
```

---

## References

- Piczak, K.J. (2015). ESC: Dataset for Environmental Sound Classification. *ACM MM*.
- Salamon, J. & Bello, J.P. (2017). Deep CNNs for Urban Sound Classification. *IEEE Signal Processing Letters*.
- McFee, B. et al. (2015). librosa: Audio and Music Signal Analysis in Python. *SciPy Proceedings*.
- Park, D.S. et al. (2019). SpecAugment: A Simple Data Augmentation Method for ASR. *Interspeech*.
- TensorFlow / Keras documentation: https://keras.io
- sounddevice documentation: https://python-sounddevice.readthedocs.io