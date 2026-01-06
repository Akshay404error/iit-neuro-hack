# BIMBO AI
## Brain Imaging and Machine-learning Based Observation AI

An intelligent EEG-based emotion recognition platform that combines cutting-edge signal processing, advanced machine learning algorithms, and interactive visualization to decode human emotions from brain activity patterns.

## 🧠 Overview

This project implements a complete pipeline for EEG-based emotion recognition following neuroscience best practices. Every preprocessing step, feature extraction method, and classification approach is scientifically justified and explainable.

### Key Features

- **Scientifically Rigorous**: All methods follow established neuroscience practices
- **Modular Architecture**: Clean separation of preprocessing, features, models, and visualization
- **Reproducible**: Fixed random seeds and documented parameters
- **Interactive Interface**: Streamlit web app for easy experimentation
- **Comprehensive Documentation**: Every function includes scientific rationale

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Scientific Background](#scientific-background)
- [Usage](#usage)
- [Modules](#modules)
- [References](#references)

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
cd eeg-emotion-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
eeg-emotion-recognition/
│
├── data/
│   ├── raw/              # Raw EEG data files
│   └── processed/        # Preprocessed data
│
├── src/
│   ├── preprocessing/
│   │   ├── filtering.py      # Bandpass and notch filters
│   │   ├── ica.py            # Independent Component Analysis
│   │   ├── asr.py            # Artifact Subspace Reconstruction
│   │   └── referencing.py    # Re-referencing strategies
│   │
│   ├── features/
│   │   ├── segmentation.py   # Epoch extraction
│   │   └── psd_features.py   # Power Spectral Density features
│   │
│   ├── models/
│   │   └── classifier.py     # ML classifiers (SVM, Random Forest)
│   │
│   ├── visualization/
│   │   ├── plots.py          # Time-series and spectral plots
│   │   └── topography.py     # Topographic maps
│   │
│   └── utils/
│       └── loader.py         # Data loading utilities
│
├── streamlit_app/
│   └── app.py            # Interactive Streamlit application
│
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🎯 Quick Start

### Run the Streamlit App

```bash
streamlit run streamlit_app/app.py
```

This launches an interactive web interface where you can:
1. Generate synthetic EEG data or load your own
2. Apply preprocessing (filtering, ICA, ASR, re-referencing)
3. Extract PSD features
4. Train emotion classifiers
5. Visualize results

### Python API Example

```python
from src.utils.loader import create_synthetic_emotion_dataset
from src.preprocessing.filtering import preprocess_with_filters
from src.features.segmentation import create_fixed_length_epochs
from src.features.psd_features import extract_all_psd_features
from src.models.classifier import EmotionClassifier

# 1. Load data
raw, labels = create_synthetic_emotion_dataset(
    n_trials_per_emotion=30,
    emotions=['happy', 'sad', 'neutral']
)

# 2. Preprocess
raw = preprocess_with_filters(raw, l_freq=0.5, h_freq=50.0)

# 3. Create epochs
epochs = create_fixed_length_epochs(raw, duration=2.0)

# 4. Extract features
features = extract_all_psd_features(epochs)

# 5. Train classifier
clf = EmotionClassifier(model_type='svm')
clf.fit(features, labels)

# 6. Evaluate
clf.cross_validate(features, labels, cv=5)
```

## 🔬 Scientific Background

### Preprocessing Pipeline

#### 1. Filtering
- **Bandpass (0.5-50 Hz)**: Removes DC drift and high-frequency muscle artifacts
- **Notch (50/60 Hz)**: Removes powerline interference
- **Scientific Basis**: Widmann et al. (2015) - Digital filter design for electrophysiological data

#### 2. Independent Component Analysis (ICA)
- **Purpose**: Separates EEG into independent components
- **Application**: Removes ocular and muscle artifacts
- **Algorithm**: FastICA for robust convergence
- **Scientific Basis**: Jung et al. (2000) - Removing electroencephalographic artifacts by blind source separation

#### 3. Artifact Subspace Reconstruction (ASR)
- **Purpose**: Removes high-amplitude burst artifacts
- **Method**: Sliding window PCA to identify artifact subspaces
- **Scientific Basis**: Mullen et al. (2015) - Real-time neuroimaging using wearable dry EEG

#### 4. Re-referencing
- **Common Average Reference (CAR)**: Most common for emotion recognition
- **Laplacian**: Enhances local activity, reduces volume conduction
- **Scientific Basis**: Nunez & Srinivasan (2006) - Electric fields of the brain

### Feature Extraction

#### Power Spectral Density (PSD)
- **Method**: Welch's method for variance reduction
- **Frequency Bands**:
  - **Delta (0.5-4 Hz)**: Deep sleep, unconscious processes
  - **Theta (4-8 Hz)**: Drowsiness, meditation, memory
  - **Alpha (8-13 Hz)**: Relaxation, calmness (inverse with arousal)
  - **Beta (13-30 Hz)**: Active thinking, focus, anxiety
  - **Gamma (30-50 Hz)**: High-level cognition, attention

#### Asymmetry Features
- **Frontal Alpha Asymmetry**: Emotional valence marker
  - Left > Right: Positive emotions (approach)
  - Right > Left: Negative emotions (withdrawal)
- **Scientific Basis**: Davidson (2004) - Prefrontal cortex and affect

### Classification

#### Support Vector Machine (SVM)
- **Kernel**: RBF (Radial Basis Function)
- **Rationale**: Effective for high-dimensional EEG features
- **Scientific Basis**: Cortes & Vapnik (1995) - Support-vector networks

#### Random Forest
- **Advantage**: Provides feature importance
- **Rationale**: Robust to overfitting, handles non-linear relationships

#### Class Balancing
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **Purpose**: Handles imbalanced emotion datasets

## 📖 Usage

### Loading Your Own EEG Data

```python
from src.utils.loader import load_eeg_file

# Supports .fif, .edf, .bdf, .set, .vhdr formats
raw = load_eeg_file('path/to/your/data.edf')
```

### Custom Preprocessing

```python
from src.preprocessing.filtering import apply_bandpass_filter, apply_notch_filter
from src.preprocessing.ica import run_ica_pipeline
from src.preprocessing.referencing import apply_common_average_reference

# Step-by-step preprocessing
raw = apply_bandpass_filter(raw, l_freq=0.5, h_freq=50.0)
raw = apply_notch_filter(raw, freqs=50.0)
raw_clean, ica, artifacts = run_ica_pipeline(raw, n_components=20)
raw_ref = apply_common_average_reference(raw_clean)
```

### Feature Extraction

```python
from src.features.psd_features import compute_psd_welch, extract_band_power

# Compute PSD
psds, freqs = compute_psd_welch(epochs, fmin=0.5, fmax=50.0)

# Extract band powers
band_powers = extract_band_power(psds, freqs)
```

### Visualization

```python
from src.visualization.plots import plot_raw_eeg, plot_psd
from src.visualization.topography import plot_band_topomaps

# Plot raw EEG
fig = plot_raw_eeg(raw, duration=10.0)

# Plot topographic maps
fig = plot_band_topomaps(epochs)
```

## 🧩 Modules

### Preprocessing
- `filtering.py`: Bandpass and notch filters with scientific justification
- `ica.py`: ICA-based artifact removal
- `asr.py`: Artifact Subspace Reconstruction
- `referencing.py`: Multiple re-referencing strategies

### Features
- `segmentation.py`: Epoch extraction and windowing
- `psd_features.py`: PSD computation, band powers, ratios, asymmetry

### Models
- `classifier.py`: SVM and Random Forest with cross-validation and SMOTE

### Visualization
- `plots.py`: Time-series, PSD, spectrograms, confusion matrices
- `topography.py`: Scalp topographic maps

### Utils
- `loader.py`: Data loading and synthetic data generation

## 📚 References

1. **Alarcao & Fonseca (2019)**. Emotions recognition using EEG signals: A survey. *IEEE Transactions on Affective Computing*, 10(3), 374-393.

2. **Davidson (2004)**. What does the prefrontal cortex "do" in affect: perspectives on frontal EEG asymmetry research. *Biological Psychology*, 67(1-2), 219-234.

3. **Jung et al. (2000)**. Removing electroencephalographic artifacts by blind source separation. *Psychophysiology*, 37(2), 163-178.

4. **Mullen et al. (2015)**. Real-time neuroimaging and cognitive monitoring using wearable dry EEG. *IEEE Transactions on Biomedical Engineering*, 62(11), 2553-2567.

5. **Widmann et al. (2015)**. Digital filter design for electrophysiological data. *Journal of Neuroscience Methods*, 250, 34-46.

6. **Cortes & Vapnik (1995)**. Support-vector networks. *Machine Learning*, 20(3), 273-297.

## 🤝 Contributing

This is a research-grade implementation. Contributions are welcome! Please ensure:
- All methods are scientifically justified
- Code follows the modular structure
- Documentation includes references
- Tests are provided for new features

## 🙏 Acknowledgments

Built with:
- [MNE-Python](https://mne.tools/) - EEG/MEG analysis
- [Scikit-Learn](https://scikit-learn.org/) - Machine learning
- [Streamlit](https://streamlit.io/) - Web interface
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Visualization

---

**Note**: This system uses synthetic demo data for demonstration. For real-world emotion recognition, use validated EEG datasets like DEAP, SEED, or DREAMER.
