# EEG Preprocessing Pipeline Demonstration

## Overview

This demonstration script (`preprocessing_pipeline_demo.py`) implements a complete, scientifically correct EEG preprocessing pipeline using MNE-Python. Every step includes scientific justification and generates visualizations.

## Pipeline Steps

### 1. Data Loading
- Loads raw EEG data (32 channels, 250 Hz sampling rate)
- For demonstration: creates synthetic data with realistic frequency components
- For real use: supports .fif, .edf, .bdf, .set, .vhdr formats

### 2. Notch Filtering (50 Hz)
**Scientific Justification**: Removes powerline interference without affecting nearby frequencies

**Implementation**:
```python
raw_notch = apply_notch_filter(raw, freqs=50.0)
```

**Output**: PSD plots showing 50 Hz peak removal

### 3. Bandpass Filtering (0.5-45 Hz)
**Scientific Justification**:
- **High-pass (0.5 Hz)**: Removes DC drift and slow baseline shifts
- **Low-pass (45 Hz)**: Removes high-frequency muscle artifacts
- Preserves all physiologically relevant EEG bands (Delta through Gamma)

**Implementation**:
```python
raw_filtered = apply_bandpass_filter(raw, l_freq=0.5, h_freq=45.0)
```

### 4. PSD Visualization
**Purpose**: Verify filtering effectiveness

**Outputs**:
- `psd_before_filtering.png`: Shows 50 Hz powerline peak
- `psd_after_filtering.png`: Shows 50 Hz peak removed
- `psd_comparison.png`: Side-by-side comparison

### 5. Independent Component Analysis (ICA)
**Scientific Justification**: 
- Separates EEG into statistically independent components
- Artifacts (eye blinks, muscle) are spatially/temporally distinct
- Allows selective removal without affecting brain signals

**Reference**: Jung et al. (2000) - Removing electroencephalographic artifacts by blind source separation

**Implementation**:
```python
# Fit ICA (20 components, FastICA algorithm)
ica = fit_ica(raw_filtered, n_components=20, method='fastica')

# Detect artifacts
artifacts = detect_artifact_components(ica, raw_filtered, threshold=0.8)

# Remove artifact components
raw_clean = apply_ica(ica, raw_filtered, exclude=artifacts)
```

**Component Rejection Criteria**:
1. High correlation (>0.8) with EOG/ECG channels
2. Characteristic spatial patterns (frontal for EOG, central for ECG)
3. Temporal characteristics matching artifacts (not brain activity)

**Output**: 
- `ica_components.png`: Topographic maps of all components
- `ica_before_after.png`: Time-series comparison

### 6. Artifact Subspace Reconstruction (ASR)
**Scientific Justification**:
- Removes high-amplitude burst artifacts (e.g., sudden movements)
- Uses sliding window PCA to identify artifact subspaces
- Reconstructs clean signals while preserving brain activity

**Reference**: Mullen et al. (2015) - Real-time neuroimaging using wearable dry EEG

**Cutoff Parameter Explanation**:
- **Cutoff = 5** (default): Standard deviations above baseline
- **Higher values (e.g., 10)**: More conservative, fewer artifacts removed
- **Lower values (e.g., 3)**: More aggressive, more artifacts removed
- **Recommended**: 5 for balanced approach

**Implementation**:
```python
raw_asr_clean = apply_asr(raw, cutoff=5.0)
```

**Output**: `asr_before_after.png` - Shows burst artifact removal

### 7. Common Average Referencing (CAR)
**Scientific Justification**:

CAR subtracts the average of all electrodes from each electrode. This is crucial because:

1. **Removes Reference Bias**: EEG measures potential differences (requires a reference). Original reference may introduce bias or be contaminated.

2. **Removes Common-Mode Noise**: Noise affecting all channels equally (e.g., environmental interference) is eliminated.

3. **Improves Spatial Resolution**: Enhances local activity patterns by removing global components.

4. **Standard for Emotion Recognition**: Most widely used reference in emotion recognition studies.

5. **Zero-Sum Assumption**: Valid for large electrode arrays where brain activity sums to zero.

**Why CAR is Important in Multi-Channel EEG**:
- EEG is a **relative measurement** (potential difference between electrode and reference)
- Choice of reference affects all measurements
- CAR provides a **reference-free** representation
- Enhances **spatial specificity** of signals
- Enables better **source localization**

**Reference**: Nunez & Srinivasan (2006) - Electric fields of the brain: The neurophysics of EEG

**Implementation**:
```python
raw_final = apply_common_average_reference(raw)
```

### 8. Save Preprocessed Data
Final preprocessed data is saved in MNE format for later use:
```python
raw_final.save('data/processed/preprocessed_eeg_raw.fif')
```

## Running the Demo

```bash
cd "D:\iit hackathon\eeg-emotion-recognition"
python preprocessing_pipeline_demo.py
```

## Output Files

All visualizations are saved to the project directory:

1. **psd_before_filtering.png** - Baseline PSD with 50 Hz peak
2. **psd_after_filtering.png** - PSD after filtering (50 Hz removed)
3. **psd_comparison.png** - Side-by-side before/after comparison
4. **ica_components.png** - Topographic maps of ICA components
5. **ica_before_after.png** - EEG time-series before/after ICA
6. **asr_before_after.png** - EEG time-series before/after ASR
7. **data/processed/preprocessed_eeg_raw.fif** - Final preprocessed data

## Scientific Validation

✅ **Filtering**: Standard neuroscience practice (0.5-45 Hz + 50 Hz notch)  
✅ **ICA**: FastICA algorithm with artifact detection (Jung et al., 2000)  
✅ **ASR**: Cutoff=5 for balanced artifact removal (Mullen et al., 2015)  
✅ **CAR**: Most common reference for emotion recognition (Nunez & Srinivasan, 2006)

## Next Steps

After preprocessing, the data is ready for:
1. **Segmentation**: Create 2-4 second epochs
2. **Feature Extraction**: Compute PSD features (band powers, asymmetry)
3. **Classification**: Train emotion recognition models

## Modular Design

Each preprocessing step is implemented as a separate, reusable function:
- `preprocessing/filtering.py` - Filtering functions
- `preprocessing/ica.py` - ICA functions
- `preprocessing/asr.py` - ASR functions
- `preprocessing/referencing.py` - Re-referencing functions

This allows you to:
- Use individual steps independently
- Customize parameters for your data
- Build custom preprocessing pipelines
- Reproduce results exactly

## References

1. **Jung et al. (2000)** - Removing electroencephalographic artifacts by blind source separation. *Psychophysiology*, 37(2), 163-178.

2. **Mullen et al. (2015)** - Real-time neuroimaging and cognitive monitoring using wearable dry EEG. *IEEE Trans. Biomed. Eng.*, 62(11), 2553-2567.

3. **Nunez & Srinivasan (2006)** - Electric fields of the brain: The neurophysics of EEG. Oxford University Press.

4. **Widmann et al. (2015)** - Digital filter design for electrophysiological data. *Journal of Neuroscience Methods*, 250, 34-46.
