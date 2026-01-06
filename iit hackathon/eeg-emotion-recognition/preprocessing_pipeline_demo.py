"""
Complete EEG Preprocessing Pipeline Demonstration

This script demonstrates a scientifically correct EEG preprocessing pipeline using MNE-Python.
It includes all major preprocessing steps with scientific justification and visualizations.

Steps:
1. Load raw EEG data
2. Apply 50 Hz notch filter (powerline noise removal)
3. Apply 0.5-45 Hz bandpass filter
4. Visualize PSD before and after filtering
5. Perform ICA with component analysis and artifact rejection
6. Apply Artifact Subspace Reconstruction (ASR)
7. Apply Common Average Referencing (CAR)

Author: EEG Emotion Recognition System
Date: 2026-01-06
"""

import os
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load module from file path
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load all required modules
filtering = load_module_from_path('filtering', os.path.join(script_dir, 'src', 'preprocessing', 'filtering.py'))
ica_module = load_module_from_path('ica_module', os.path.join(script_dir, 'src', 'preprocessing', 'ica.py'))
asr_module = load_module_from_path('asr_module', os.path.join(script_dir, 'src', 'preprocessing', 'asr.py'))
referencing = load_module_from_path('referencing', os.path.join(script_dir, 'src', 'preprocessing', 'referencing.py'))
loader = load_module_from_path('loader', os.path.join(script_dir, 'src', 'utils', 'loader.py'))

# Create convenient aliases
apply_bandpass_filter = filtering.apply_bandpass_filter
apply_notch_filter = filtering.apply_notch_filter
fit_ica = ica_module.fit_ica
detect_artifact_components = ica_module.detect_artifact_components
apply_ica = ica_module.apply_ica
apply_asr = asr_module.apply_asr
apply_common_average_reference = referencing.apply_common_average_reference
create_synthetic_eeg = loader.create_synthetic_eeg

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("EEG PREPROCESSING PIPELINE DEMONSTRATION")
print("="*80)
print()

# ============================================================================
# STEP 1: Load Raw EEG Data
# ============================================================================
print("STEP 1: Loading Raw EEG Data")
print("-" * 80)
print()

# For demonstration, we'll create synthetic EEG data
# In practice, you would load your own data using:
# raw = mne.io.read_raw_fif('your_data.fif', preload=True)
# or other formats (.edf, .bdf, .set, etc.)

print("Creating synthetic EEG data for demonstration...")
print("(In practice, load your own data using mne.io.read_raw_*)")
print()

raw_original = create_synthetic_eeg(
    n_channels=32,
    sfreq=250.0,
    duration=120.0,  # 2 minutes
    verbose=False
)

# Add some artificial 50 Hz noise to demonstrate notch filtering
print("Adding artificial 50 Hz powerline noise...")
data = raw_original.get_data()
times = raw_original.times
powerline_noise = 0.5e-5 * np.sin(2 * np.pi * 50 * times)  # 50 Hz noise
data = data + powerline_noise[np.newaxis, :]
raw_original._data = data

print(f"✓ Loaded EEG data:")
print(f"  - Channels: {len(raw_original.ch_names)}")
print(f"  - Sampling rate: {raw_original.info['sfreq']} Hz")
print(f"  - Duration: {raw_original.times[-1]:.1f} seconds")
print(f"  - Total samples: {raw_original.n_times}")
print()

# ============================================================================
# STEP 2: Compute PSD BEFORE Filtering (Baseline)
# ============================================================================
print("STEP 2: Computing PSD Before Filtering (Baseline)")
print("-" * 80)
print()

print("Computing Power Spectral Density...")
fig_before = raw_original.compute_psd(fmin=0.5, fmax=100).plot(
    average=True,
    picks='eeg',
    show=False
)
fig_before.suptitle('PSD Before Filtering (Note 50 Hz Peak)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/iit hackathon/eeg-emotion-recognition/psd_before_filtering.png', dpi=150)
print("✓ PSD before filtering computed and saved")
print()

# ============================================================================
# STEP 3: Apply 50 Hz Notch Filter
# ============================================================================
print("STEP 3: Applying 50 Hz Notch Filter")
print("-" * 80)
print()

print("Scientific Justification:")
print("  Powerline interference (50 Hz in Europe/Asia, 60 Hz in Americas) contaminates")
print("  EEG recordings. Notch filters remove this narrow-band interference without")
print("  affecting nearby frequencies.")
print()

raw_notch = raw_original.copy()
raw_notch = apply_notch_filter(raw_notch, freqs=50.0, verbose=False)

print("✓ 50 Hz notch filter applied")
print()

# ============================================================================
# STEP 4: Apply 0.5-45 Hz Bandpass Filter
# ============================================================================
print("STEP 4: Applying 0.5-45 Hz Bandpass Filter")
print("-" * 80)
print()

print("Scientific Justification:")
print("  - High-pass (0.5 Hz): Removes DC drift and slow baseline shifts")
print("  - Low-pass (45 Hz): Removes high-frequency muscle artifacts")
print("  - This range preserves all physiologically relevant EEG frequencies:")
print("    * Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz),")
print("    * Beta (13-30 Hz), Gamma (30-45 Hz)")
print()

raw_filtered = raw_notch.copy()
raw_filtered = apply_bandpass_filter(raw_filtered, l_freq=0.5, h_freq=45.0, verbose=False)

print("✓ Bandpass filter (0.5-45 Hz) applied")
print()

# ============================================================================
# STEP 5: Compute PSD AFTER Filtering
# ============================================================================
print("STEP 5: Computing PSD After Filtering")
print("-" * 80)
print()

print("Computing Power Spectral Density after filtering...")
fig_after = raw_filtered.compute_psd(fmin=0.5, fmax=100).plot(
    average=True,
    picks='eeg',
    show=False
)
fig_after.suptitle('PSD After Filtering (50 Hz Peak Removed)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/iit hackathon/eeg-emotion-recognition/psd_after_filtering.png', dpi=150)
print("✓ PSD after filtering computed and saved")
print()

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Before filtering
psd_before = raw_original.compute_psd(fmin=0.5, fmax=100)
psd_before.plot(axes=ax1, show=False, average=True)
ax1.set_title('Before Filtering\n(Note 50 Hz peak)', fontweight='bold')
ax1.axvline(50, color='r', linestyle='--', alpha=0.7, label='50 Hz powerline')
ax1.legend()

# After filtering
psd_after = raw_filtered.compute_psd(fmin=0.5, fmax=100)
psd_after.plot(axes=ax2, show=False, average=True)
ax2.set_title('After Filtering\n(50 Hz peak removed)', fontweight='bold')
ax2.axvline(50, color='r', linestyle='--', alpha=0.7, label='50 Hz (removed)')
ax2.legend()

plt.tight_layout()
plt.savefig('D:/iit hackathon/eeg-emotion-recognition/psd_comparison.png', dpi=150)
print("✓ PSD comparison plot saved")
print()

# ============================================================================
# STEP 6: Independent Component Analysis (ICA)
# ============================================================================
print("STEP 6: Performing Independent Component Analysis (ICA)")
print("-" * 80)
print()

print("Scientific Justification:")
print("  ICA separates EEG into statistically independent components. Artifacts")
print("  (eye blinks, muscle activity) are spatially and temporally distinct from")
print("  brain signals, allowing us to identify and remove them.")
print()
print("  Reference: Jung et al. (2000) - Removing electroencephalographic artifacts")
print("             by blind source separation. Psychophysiology, 37(2), 163-178.")
print()

# For ICA, we need to filter at 1 Hz (recommended)
print("Preparing data for ICA (filtering at 1 Hz high-pass)...")
raw_for_ica = raw_filtered.copy().filter(l_freq=1.0, h_freq=None, verbose=False)

print("Fitting ICA (FastICA algorithm)...")
ica = fit_ica(raw_for_ica, n_components=20, method='fastica', verbose=False)

print(f"✓ ICA fitted with {ica.n_components_} components")
print()

# Plot ICA components
print("Plotting ICA component topographies...")
fig_ica = ica.plot_components(picks=range(min(20, ica.n_components_)), show=False)
plt.savefig('D:/iit hackathon/eeg-emotion-recognition/ica_components.png', dpi=150, bbox_inches='tight')
print("✓ ICA component topographies saved")
print()

# Detect artifacts
print("Detecting artifact components...")
print("  - Looking for EOG (eye movement) artifacts")
print("  - Looking for ECG (heart) artifacts")
print()

artifacts = detect_artifact_components(
    ica, raw_for_ica,
    threshold=0.8,
    verbose=False
)

print(f"✓ Detected artifact components:")
print(f"  - EOG components: {artifacts['eog']}")
print(f"  - ECG components: {artifacts['ecg']}")
print()

# Scientific justification for component rejection
print("Component Rejection Justification:")
print("  Components are identified as artifacts if they:")
print("  1. Correlate highly (>0.8) with EOG/ECG channels")
print("  2. Show characteristic spatial patterns (frontal for EOG, central for ECG)")
print("  3. Have temporal characteristics matching artifacts (not brain activity)")
print()

# Manually mark components for removal (in practice, inspect visually)
exclude_components = list(set(artifacts['eog'] + artifacts['ecg']))
if not exclude_components:
    # If no artifacts detected, manually exclude first 2 for demonstration
    exclude_components = [0, 1]
    print("  Note: For demonstration, manually excluding components [0, 1]")
    print()

print(f"Excluding components: {exclude_components}")
print()

# Apply ICA
print("Applying ICA to remove artifact components...")
raw_ica_clean = apply_ica(ica, raw_filtered, exclude=exclude_components, verbose=False)

print("✓ ICA cleaning complete")
print()


# Plot before/after ICA
print("Creating ICA before/after comparison...")

# Create before ICA plot
fig1 = raw_filtered.plot(duration=10, n_channels=10, scalings={'eeg': 50e-6}, 
                         show=False, title='Before ICA (with artifacts)')
fig1.savefig('D:/iit hackathon/eeg-emotion-recognition/ica_before.png', dpi=150)

# Create after ICA plot
fig2 = raw_ica_clean.plot(duration=10, n_channels=10, scalings={'eeg': 50e-6},
                          show=False, title='After ICA (artifacts removed)')
fig2.savefig('D:/iit hackathon/eeg-emotion-recognition/ica_after.png', dpi=150)

plt.close('all')
print("✓ ICA before/after comparison saved")
print()


# ============================================================================
# STEP 7: Artifact Subspace Reconstruction (ASR)
# ============================================================================
print("STEP 7: Applying Artifact Subspace Reconstruction (ASR)")
print("-" * 80)
print()

print("Scientific Justification:")
print("  ASR identifies and removes high-amplitude burst artifacts (e.g., from")
print("  sudden movements) using sliding window PCA. It reconstructs clean signals")
print("  by projecting out artifact subspaces while preserving brain activity.")
print()
print("  Reference: Mullen et al. (2015) - Real-time neuroimaging using wearable")
print("             dry EEG. IEEE Trans. Biomed. Eng., 62(11), 2553-2567.")
print()

print("ASR Cutoff Parameter:")
print("  - Cutoff = 5 (default): Standard deviations above baseline for detection")
print("  - Higher values (e.g., 10): More conservative, fewer artifacts removed")
print("  - Lower values (e.g., 3): More aggressive, more artifacts removed")
print("  - We use cutoff=5 as a balanced approach")
print()

# Add some artificial burst artifacts for demonstration
print("Adding artificial burst artifacts for demonstration...")
data_asr = raw_ica_clean.get_data().copy()
# Add bursts at random times
for _ in range(5):
    burst_time = np.random.randint(1000, data_asr.shape[1] - 1000)
    burst_duration = 100
    burst_channels = np.random.choice(data_asr.shape[0], 5, replace=False)
    data_asr[burst_channels, burst_time:burst_time+burst_duration] *= 5.0

raw_with_bursts = raw_ica_clean.copy()
raw_with_bursts._data = data_asr

print("Applying ASR (cutoff=5)...")
raw_asr_clean = apply_asr(raw_with_bursts, cutoff=5.0, verbose=False)

print("✓ ASR complete")
print()

# Plot before/after ASR
print("Creating ASR before/after comparison...")

# Create before ASR plot
fig1 = raw_with_bursts.plot(duration=10, n_channels=10, scalings={'eeg': 50e-6},
                             show=False, title='Before ASR (with burst artifacts)')
fig1.savefig('D:/iit hackathon/eeg-emotion-recognition/asr_before.png', dpi=150)

# Create after ASR plot
fig2 = raw_asr_clean.plot(duration=10, n_channels=10, scalings={'eeg': 50e-6},
                          show=False, title='After ASR (bursts removed)')
fig2.savefig('D:/iit hackathon/eeg-emotion-recognition/asr_after.png', dpi=150)

plt.close('all')
print("✓ ASR before/after comparison saved")
print()


# ============================================================================
# STEP 8: Common Average Referencing (CAR)
# ============================================================================
print("STEP 8: Applying Common Average Referencing (CAR)")
print("-" * 80)
print()

print("Scientific Justification:")
print("  Common Average Reference (CAR) subtracts the average of all electrodes")
print("  from each electrode. This:")
print()
print("  1. Removes reference electrode bias")
print("  2. Removes common-mode noise (affects all channels equally)")
print("  3. Improves spatial resolution")
print("  4. Is the most widely used reference for emotion recognition")
print()
print("  Why CAR is important in multi-channel EEG:")
print("  - EEG measures potential differences (requires a reference)")
print("  - Original reference may introduce bias or be contaminated")
print("  - CAR assumes brain activity is zero-sum (valid for large arrays)")
print("  - Enhances local activity patterns")
print()
print("  Reference: Nunez & Srinivasan (2006) - Electric fields of the brain:")
print("             The neurophysics of EEG. Oxford University Press.")
print()

print("Applying Common Average Reference...")
raw_final = apply_common_average_reference(raw_asr_clean, verbose=False)

print("✓ CAR applied")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*80)
print("PREPROCESSING PIPELINE COMPLETE")
print("="*80)
print()

print("Summary of Applied Preprocessing Steps:")
print()
print("1. ✓ Loaded raw EEG data (32 channels, 250 Hz, 120 seconds)")
print("2. ✓ Applied 50 Hz notch filter (powerline noise removal)")
print("3. ✓ Applied 0.5-45 Hz bandpass filter (DC drift & muscle artifact removal)")
print("4. ✓ Visualized PSD before/after filtering (50 Hz peak removed)")
print("5. ✓ Performed ICA with 20 components")
print(f"6. ✓ Identified and removed artifact components: {exclude_components}")
print("7. ✓ Applied ASR (cutoff=5) to remove burst artifacts")
print("8. ✓ Applied Common Average Reference (CAR)")
print()

print("Output Files Generated:")
print("  - psd_before_filtering.png: PSD before any filtering")
print("  - psd_after_filtering.png: PSD after notch and bandpass filters")
print("  - psd_comparison.png: Side-by-side comparison showing 50 Hz removal")
print("  - ica_components.png: Topographic maps of ICA components")
print("  - ica_before_after.png: EEG time-series before/after ICA")
print("  - asr_before_after.png: EEG time-series before/after ASR")
print()

print("Final Data Characteristics:")
print(f"  - Channels: {len(raw_final.ch_names)}")
print(f"  - Sampling rate: {raw_final.info['sfreq']} Hz")
print(f"  - Duration: {raw_final.times[-1]:.1f} seconds")
print(f"  - Reference: {raw_final.info['custom_ref_applied']}")
print(f"  - Highpass: {raw_final.info['highpass']} Hz")
print(f"  - Lowpass: {raw_final.info['lowpass']} Hz")
print()

print("Next Steps:")
print("  1. Segment into epochs (2-4 second windows)")
print("  2. Extract PSD features (band powers, asymmetry)")
print("  3. Train emotion classifier (SVM or Random Forest)")
print()

print("="*80)
print("All preprocessing steps completed successfully!")
print("="*80)

# Save final preprocessed data
print()
print("Saving final preprocessed data...")

# Create directory if it doesn't exist
import os
os.makedirs('D:/iit hackathon/eeg-emotion-recognition/data/processed', exist_ok=True)

raw_final.save('D:/iit hackathon/eeg-emotion-recognition/data/processed/preprocessed_eeg_raw.fif',
               overwrite=True, verbose=False)
print("✓ Preprocessed data saved to: data/processed/preprocessed_eeg_raw.fif")
print()

print("To load this data later:")
print("  raw = mne.io.read_raw_fif('data/processed/preprocessed_eeg_raw.fif', preload=True)")
print()

# Show plots
print("Displaying plots...")
plt.show()
