# Acoustic Event Classification System — Design Document

## Project Overview

An intelligent acoustic monitoring system that classifies short audio segments into discrete event categories using a convolutional neural network trained on Mel-spectrogram representations.

**Proposed scenario:** Industrial / predictive maintenance — classifying machine sounds to detect operational states and early fault signals.

---

## Acoustic Classes (Proposed)

| Class | Description | Challenge |
|---|---|---|
| `normal_operation` | Steady hum/rumble of healthy equipment | May overlap with other low-energy classes |
| `metallic_impact` | Sharp percussive events (collisions, drops, hammering) | Short transients, easy to miss |
| `friction_squeal` | High-frequency squealing from worn bearings or brakes | Can be confused with alarms |
| `alarm_tone` | Repetitive synthetic tones, buzzers, warning signals | Varies greatly across devices |
| `silence_ambient` | Background noise / idle state | Low-information but important anchor class |

Minimum 5 classes, extendable to 6–7 with recordings.

---

## Technology Stack

### Language
**Python 3.11+**
- Dominant in ML/audio research
- Best ecosystem coverage for every stage of this pipeline
- Strong typing support via `mypy` for production-quality code

### Audio Processing
| Library | Role |
|---|---|
| `librosa` | Feature extraction, Mel-spectrograms, MFCCs, resampling |
| `soundfile` / `audioread` | Audio I/O (WAV, FLAC, MP3) |
| `pydub` | Segmentation, normalization, format conversion |
| `numpy` | Array manipulation throughout the pipeline |

### Machine Learning
| Library | Role |
|---|---|
| `scikit-learn` | Baseline models (SVM, kNN, Random Forest), metrics, splits |
| `tensorflow` + `keras` | CNN architecture, training loop |
| `tensorflow-io` | Optional: streaming audio loading |

**Why TensorFlow/Keras over PyTorch?**
Keras offers a cleaner API for a first CNN implementation. If the team is more comfortable with PyTorch, it is equally valid — the architecture proposed here translates directly.

### Experiment Tracking
| Library | Role |
|---|---|
| `matplotlib` / `seaborn` | Training curves, confusion matrices |
| `mlflow` (optional) | Run tracking, parameter logging |

### Data & Environment
| Tool | Role |
|---|---|
| `pandas` | Dataset manifests, metadata CSVs |
| `conda` or `venv` | Reproducible environment |
| `jupyter` | Exploratory notebooks |
| `pytest` | Unit tests for pipeline stages |

---

## Dataset Plan

**Primary source:** [ESC-50](https://github.com/karolpiczak/ESC-50) (Environmental Sound Classification, 50 classes, 2000 clips, 5s each, 44.1kHz, WAV)

Select a 5-class subset that maps to the industrial scenario above, OR supplement with:
- [UrbanSound8K](https://urbansounddataset.webusc.edu/urbansound8k.html) for alarm/impact sounds
- Custom team recordings for `normal_operation` and `friction_squeal`

**Target per class:** ≥ 200 samples, aiming for balance.

---

## Audio Representation: Mel-Spectrogram (Selected)

**Rationale:**
- Converts 1D audio → 2D image (time × frequency) consumable by a CNN
- Mel scale matches human auditory perception — compresses high-frequency redundancy
- Outperforms raw MFCCs for CNN input because it retains full spectral envelope
- Well-supported in `librosa` with one function call

**Key parameters:**
```
sample_rate    = 22050 Hz
n_fft          = 2048
hop_length     = 512
n_mels         = 128
duration       = 4 seconds (pad/trim to fixed length)
output_shape   = (128, 173, 1)  # height × width × channels
```

### Comparison Table

| Representation | Time info | Freq info | CNN-ready | Notes |
|---|---|---|---|---|
| Raw waveform | ✅ | ❌ | ❌ (1D) | Needs 1D-CNN or RNN |
| STFT spectrogram | ✅ | ✅ | ✅ | High resolution, not perceptual |
| **Mel-spectrogram** | ✅ | ✅ | ✅ | **Selected — perceptual, compact** |
| MFCCs | Partial | ✅ | ⚠️ | Loses spectral envelope; good for baselines |
| Log-Mel spectrogram | ✅ | ✅ | ✅ | Same as Mel with log compression — very close second |

---

## Machine Learning Pipeline

```
Raw audio files (.wav)
        ↓
[1] Preprocessing
    - Resample to 22050 Hz
    - Convert to mono
    - Normalize amplitude
        ↓
[2] Segmentation / Duration Normalization
    - Trim or pad to exactly 4 seconds
    - Optionally split long files into 4s windows with 50% overlap
        ↓
[3] Feature Extraction
    - Compute Mel-spectrogram via librosa
    - Apply log scaling: log(S + 1e-6)
    - Stack as single-channel 2D array (128 × 173)
        ↓
[4] Dataset Split
    - 70% train / 15% validation / 15% test
    - Stratified by class
    - No data leakage across splits
        ↓
[5] Baseline Model
    - Flatten Mel-spectrogram → 1D vector
    - Feed to SVM (RBF kernel)
        ↓
[6] CNN Model
    - Input: (128, 173, 1)
    - Conv blocks → Dense → Softmax
        ↓
[7] Evaluation
    - Accuracy, F1, confusion matrix
    - Training/validation curves
        ↓
[8] Error Analysis
    - Inspect misclassified samples
    - Listen to hard cases
    - Identify class overlap
```

---

## Baseline Model

**Model:** Support Vector Machine with RBF kernel (`sklearn.svm.SVC`)

**Input:** Flattened Mel-spectrogram (128 × 173 = 22,144 features) or mean MFCCs (40 coefficients — faster)

**Why SVM?**
- Strong performance on small/medium datasets
- No gradient training — fast to iterate
- Provides a meaningful upper bound for "non-deep" approaches

**Expected accuracy:** 60–75% depending on class difficulty and dataset quality.

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
GlobalAveragePooling2D         ← avoids over-parameterization vs Flatten
    ↓
Dense(128) + ReLU + Dropout(0.4)
    ↓
Dense(5) + Softmax
```

**Training config:**
- Loss: `categorical_crossentropy`
- Optimizer: `Adam(lr=1e-3)` with `ReduceLROnPlateau`
- Batch size: 32
- Epochs: 50 with `EarlyStopping(patience=10)`
- Data augmentation: time shift, add Gaussian noise (in feature space)

**Why CNN?**
A Mel-spectrogram is a 2D image where local patterns (frequency bands, temporal events) are spatially localized. Conv2D kernels learn these local time-frequency patterns — exactly what distinguishes acoustic classes.

---

## Validation Strategy

- **Split:** Stratified 70/15/15
- **Metrics:** Accuracy, macro F1-score, per-class precision/recall
- **Confusion matrix:** Identify which pairs of classes are confused
- **Curves:** Loss and accuracy vs. epoch to detect overfitting
- **Error analysis:** Manually review ≥10 misclassified samples per class

**Why not accuracy alone?**
If one class dominates the dataset, a model that always predicts it achieves high accuracy while being useless. Macro F1 weights all classes equally and is the primary metric.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Overfitting (small dataset) | Dropout, BatchNorm, data augmentation, early stopping |
| Class imbalance | Stratified splits, class weights in loss function |
| Short recordings | Fixed-length windowing with overlap |
| Similar-sounding classes | Collect diverse samples; inspect confusion matrix early |
| Poor audio quality | Noise floor check during preprocessing; discard corrupted files |

---

## Repository Structure (Suggested)

```
acoustic-classifier/
├── CLAUDE.md               ← AI assistant context file
├── DESIGN.md               ← This document
├── data/
│   ├── raw/                ← Original audio files
│   ├── processed/          ← Resampled, normalized WAVs
│   └── spectrograms/       ← Saved .npy spectrogram arrays
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_baseline.ipynb
├── src/
│   ├── preprocess.py       ← Audio loading, resampling, normalization
│   ├── features.py         ← Mel-spectrogram computation
│   ├── dataset.py          ← Dataset class / tf.data pipeline
│   ├── baseline.py         ← SVM baseline
│   ├── model.py            ← CNN architecture
│   ├── train.py            ← Training loop
│   └── evaluate.py         ← Metrics, confusion matrix, error analysis
├── tests/
│   └── test_preprocess.py
├── requirements.txt
└── README.md
```

---

## References

- Piczak, K.J. (2015). ESC: Dataset for Environmental Sound Classification. *ACM MM*.
- McFee et al. (2015). librosa: Audio and Music Signal Analysis in Python.
- Salamon, J. & Bello, J.P. (2017). Deep Convolutional Neural Networks for Urban Sound Classification. *IEEE Signal Processing Letters*.
- TensorFlow / Keras documentation: https://keras.io