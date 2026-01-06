"""
EEG Feature Extraction for Emotion Recognition

This script demonstrates scientifically correct feature extraction from EEG data
for emotion recognition tasks. It implements baseline correction and band power
extraction following neuroscience best practices.

Steps:
1. Segment EEG data into baseline (pre-stimulus) and trial (stimulus) periods
2. Apply baseline correction: Corrected_Power = Trial_Power - Baseline_Power
3. Compute Power Spectral Density (PSD) using Welch's method
4. Extract band power features for Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz)
5. Store features in structured Pandas DataFrame
6. Plot PSD curves showing frequency bands

Author: EEG Emotion Recognition System
Date: 2026-01-06
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy import signal

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load module from file path
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load required modules
loader = load_module_from_path('loader', os.path.join(script_dir, 'src', 'utils', 'loader.py'))
create_synthetic_eeg = loader.create_synthetic_eeg

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("EEG FEATURE EXTRACTION FOR EMOTION RECOGNITION")
print("="*80)
print()

# ============================================================================
# STEP 1: Load and Prepare EEG Data
# ============================================================================
print("STEP 1: Loading and Preparing EEG Data")
print("-" * 80)
print()

# Create synthetic EEG data with multiple trials
print("Creating synthetic EEG data with multiple emotion trials...")
print("(In practice, load your preprocessed data)")
print()

# Simulate 5 trials, each with baseline + stimulus period
n_trials = 5
baseline_duration = 3.0  # 3 seconds baseline
trial_duration = 5.0     # 5 seconds stimulus
sfreq = 250.0           # 250 Hz sampling rate
n_channels = 14         # 14 EEG channels

# Create emotion labels for trials
emotions = ['happy', 'sad', 'neutral', 'fear', 'happy']

print(f"Configuration:")
print(f"  - Number of trials: {n_trials}")
print(f"  - Baseline duration: {baseline_duration} seconds")
print(f"  - Trial duration: {trial_duration} seconds")
print(f"  - Sampling rate: {sfreq} Hz")
print(f"  - Channels: {n_channels}")
print(f"  - Emotions: {emotions}")
print()

# ============================================================================
# STEP 2: Segment EEG into Baseline and Trial Periods
# ============================================================================
print("STEP 2: Segmenting EEG into Baseline and Trial Periods")
print("-" * 80)
print()

print("Scientific Justification:")
print("  Baseline correction removes subject-specific baseline activity and")
print("  inter-trial variability, isolating stimulus-evoked responses.")
print("  This is critical for emotion recognition as it normalizes individual")
print("  differences in resting-state EEG.")
print()

# Storage for baseline and trial data
baseline_epochs = []
trial_epochs = []

for trial_idx in range(n_trials):
    # Create continuous data for this trial (baseline + stimulus)
    total_duration = baseline_duration + trial_duration
    raw = create_synthetic_eeg(
        n_channels=n_channels,
        sfreq=sfreq,
        duration=total_duration,
        verbose=False
    )
    
    # Add emotion-specific frequency modulation to trial period
    data = raw.get_data()
    trial_start_sample = int(baseline_duration * sfreq)
    
    # Modulate specific frequency bands based on emotion
    if emotions[trial_idx] == 'happy':
        # Happy: increased alpha power
        alpha_mod = 1.5 * np.sin(2 * np.pi * 10 * raw.times[trial_start_sample:])
        data[:, trial_start_sample:] += alpha_mod[np.newaxis, :] * 1e-6
    elif emotions[trial_idx] == 'sad':
        # Sad: increased theta power
        theta_mod = 1.5 * np.sin(2 * np.pi * 6 * raw.times[trial_start_sample:])
        data[:, trial_start_sample:] += theta_mod[np.newaxis, :] * 1e-6
    elif emotions[trial_idx] == 'fear':
        # Fear: increased beta power
        beta_mod = 1.5 * np.sin(2 * np.pi * 20 * raw.times[trial_start_sample:])
        data[:, trial_start_sample:] += beta_mod[np.newaxis, :] * 1e-6
    
    raw._data = data
    
    # Extract baseline period (pre-stimulus)
    baseline_raw = raw.copy().crop(tmin=0, tmax=baseline_duration)
    baseline_epochs.append(baseline_raw.get_data())
    
    # Extract trial period (stimulus)
    trial_raw = raw.copy().crop(tmin=baseline_duration, tmax=total_duration)
    trial_epochs.append(trial_raw.get_data())

print(f"✓ Segmented {n_trials} trials into baseline and stimulus periods")
print(f"  - Baseline shape per trial: {baseline_epochs[0].shape}")
print(f"  - Trial shape per trial: {trial_epochs[0].shape}")
print()

# ============================================================================
# STEP 3: Compute Power Spectral Density (PSD) using Welch's Method
# ============================================================================
print("STEP 3: Computing PSD using Welch's Method")
print("-" * 80)
print()

print("Scientific Justification:")
print("  Welch's method reduces variance in PSD estimation by:")
print("  1. Dividing signal into overlapping segments")
print("  2. Computing periodogram for each segment")
print("  3. Averaging periodograms")
print("  This provides more stable frequency domain features.")
print()

# Define frequency bands
freq_bands = {
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

print("Frequency Bands:")
for band_name, (low, high) in freq_bands.items():
    print(f"  - {band_name}: {low}-{high} Hz")
print()

# Welch parameters
nperseg = int(2 * sfreq)  # 2-second windows
noverlap = int(nperseg * 0.5)  # 50% overlap

print(f"Welch's Method Parameters:")
print(f"  - Window length: {nperseg/sfreq} seconds ({nperseg} samples)")
print(f"  - Overlap: {noverlap/sfreq} seconds ({noverlap} samples)")
print()

# Compute PSD for all trials
baseline_psds = []
trial_psds = []
freqs = None

for trial_idx in range(n_trials):
    # Baseline PSD
    baseline_psd_trial = []
    for ch_idx in range(n_channels):
        f, psd = signal.welch(
            baseline_epochs[trial_idx][ch_idx, :],
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap
        )
        baseline_psd_trial.append(psd)
        if freqs is None:
            freqs = f
    baseline_psds.append(np.array(baseline_psd_trial))
    
    # Trial PSD
    trial_psd_trial = []
    for ch_idx in range(n_channels):
        f, psd = signal.welch(
            trial_epochs[trial_idx][ch_idx, :],
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap
        )
        trial_psd_trial.append(psd)
    trial_psds.append(np.array(trial_psd_trial))

print(f"✓ Computed PSD for all trials")
print(f"  - Frequency resolution: {freqs[1] - freqs[0]:.3f} Hz")
print(f"  - Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
print()

# ============================================================================
# STEP 4: Apply Baseline Correction
# ============================================================================
print("STEP 4: Applying Baseline Correction")
print("-" * 80)
print()

print("Formula: Corrected_Power = Trial_Power - Baseline_Power")
print()
print("Physiological Interpretation:")
print("  Baseline correction isolates stimulus-evoked changes in brain activity")
print("  by removing pre-stimulus baseline. This accounts for:")
print("  - Individual differences in resting-state EEG")
print("  - Inter-trial variability")
print("  - Non-stimulus-related fluctuations")
print()

# Apply baseline correction
corrected_psds = []
for trial_idx in range(n_trials):
    corrected_psd = trial_psds[trial_idx] - baseline_psds[trial_idx]
    corrected_psds.append(corrected_psd)

print(f"✓ Applied baseline correction to all trials")
print()

# ============================================================================
# STEP 5: Extract Band Power Features
# ============================================================================
print("STEP 5: Extracting Band Power Features")
print("-" * 80)
print()

print("Band Power Calculation:")
print("  Band power is computed by integrating PSD over the frequency band:")
print("  Power = ∫[f_low to f_high] PSD(f) df")
print()

# Extract band powers
features_list = []

for trial_idx in range(n_trials):
    trial_features = {
        'trial': trial_idx + 1,
        'emotion': emotions[trial_idx]
    }
    
    # For each channel
    for ch_idx in range(n_channels):
        ch_name = f'Ch{ch_idx+1}'
        
        # Extract band powers
        for band_name, (low_freq, high_freq) in freq_bands.items():
            # Find frequency indices
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            
            # Integrate PSD over band (using trapezoidal rule)
            band_power = np.trapz(
                corrected_psds[trial_idx][ch_idx, freq_mask],
                freqs[freq_mask]
            )
            
            # Store feature
            feature_name = f'{ch_name}_{band_name}'
            trial_features[feature_name] = band_power
    
    features_list.append(trial_features)

# Create DataFrame
features_df = pd.DataFrame(features_list)

print(f"✓ Extracted band power features")
print(f"  - Total features per trial: {len(features_df.columns) - 2}")
print(f"  - Features per channel: {len(freq_bands)}")
print(f"  - Total channels: {n_channels}")
print()

print("Feature DataFrame Preview:")
print(features_df.head())
print()

# ============================================================================
# STEP 6: Visualize PSD with Frequency Bands
# ============================================================================
print("STEP 6: Visualizing PSD Curves with Frequency Bands")
print("-" * 80)
print()

# Select one channel and one trial for visualization
vis_channel = 0
vis_trial = 0

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Baseline PSD
ax1.plot(freqs, baseline_psds[vis_trial][vis_channel, :], 'b-', linewidth=2, label='Baseline PSD')
ax1.set_xlim(0, 35)
ax1.set_xlabel('Frequency (Hz)', fontsize=12)
ax1.set_ylabel('Power Spectral Density', fontsize=12)
ax1.set_title(f'Baseline PSD - Channel {vis_channel+1}', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Shade frequency bands
for band_name, (low, high) in freq_bands.items():
    ax1.axvspan(low, high, alpha=0.2, label=f'{band_name} ({low}-{high} Hz)')
ax1.legend(loc='upper right')

# Plot 2: Trial PSD
ax2.plot(freqs, trial_psds[vis_trial][vis_channel, :], 'g-', linewidth=2, label='Trial PSD')
ax2.set_xlim(0, 35)
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Power Spectral Density', fontsize=12)
ax2.set_title(f'Trial PSD - Channel {vis_channel+1} (Emotion: {emotions[vis_trial]})', 
              fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Shade frequency bands
for band_name, (low, high) in freq_bands.items():
    ax2.axvspan(low, high, alpha=0.2, label=f'{band_name} ({low}-{high} Hz)')
ax2.legend(loc='upper right')

# Plot 3: Baseline-Corrected PSD
ax3.plot(freqs, corrected_psds[vis_trial][vis_channel, :], 'r-', linewidth=2, 
         label='Baseline-Corrected PSD')
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.set_xlim(0, 35)
ax3.set_xlabel('Frequency (Hz)', fontsize=12)
ax3.set_ylabel('Power Change', fontsize=12)
ax3.set_title(f'Baseline-Corrected PSD - Channel {vis_channel+1}', 
              fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Shade frequency bands
colors = {'Theta': 'yellow', 'Alpha': 'cyan', 'Beta': 'magenta'}
for band_name, (low, high) in freq_bands.items():
    ax3.axvspan(low, high, alpha=0.2, color=colors[band_name], 
                label=f'{band_name} ({low}-{high} Hz)')
ax3.legend(loc='upper right')

plt.tight_layout()
plt.savefig('D:/iit hackathon/eeg-emotion-recognition/feature_extraction_psd.png', dpi=150)
print("✓ PSD visualization saved: feature_extraction_psd.png")
print()

# ============================================================================
# STEP 7: Visualize Band Powers Across Trials
# ============================================================================
print("STEP 7: Visualizing Band Powers Across Trials")
print("-" * 80)
print()

# Create bar plot of band powers for one channel across all trials
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (band_name, (low, high)) in enumerate(freq_bands.items()):
    ax = axes[idx]
    
    # Extract band powers for this band across all trials
    feature_col = f'Ch{vis_channel+1}_{band_name}'
    powers = features_df[feature_col].values
    
    # Create bar plot
    bars = ax.bar(range(1, n_trials+1), powers, color=colors[band_name], alpha=0.7, edgecolor='black')
    
    # Color bars by emotion
    emotion_colors = {'happy': 'gold', 'sad': 'blue', 'neutral': 'gray', 'fear': 'red'}
    for bar, emotion in zip(bars, emotions):
        bar.set_facecolor(emotion_colors.get(emotion, 'gray'))
    
    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel(f'{band_name} Power', fontsize=12)
    ax.set_title(f'{band_name} Band Power ({low}-{high} Hz)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, n_trials+1))
    ax.set_xticklabels([f'{i+1}\n({emotions[i]})' for i in range(n_trials)])
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/iit hackathon/eeg-emotion-recognition/feature_extraction_bands.png', dpi=150)
print("✓ Band power visualization saved: feature_extraction_bands.png")
print()

# ============================================================================
# STEP 8: Save Features to CSV
# ============================================================================
print("STEP 8: Saving Features to CSV")
print("-" * 80)
print()

# Save features
features_df.to_csv('D:/iit hackathon/eeg-emotion-recognition/eeg_features.csv', index=False)
print("✓ Features saved to: eeg_features.csv")
print()

print(f"Feature Statistics:")
print(features_df.describe())
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*80)
print("FEATURE EXTRACTION COMPLETE")
print("="*80)
print()

print("Summary of Extracted Features:")
print()
print(f"1. ✓ Segmented {n_trials} trials into baseline and stimulus periods")
print(f"2. ✓ Applied baseline correction (Trial - Baseline)")
print(f"3. ✓ Computed PSD using Welch's method")
print(f"4. ✓ Extracted {len(freq_bands)} band powers × {n_channels} channels = {len(freq_bands) * n_channels} features per trial")
print(f"5. ✓ Created structured DataFrame with {len(features_df)} trials")
print(f"6. ✓ Generated PSD visualizations")
print()

print("Output Files:")
print("  - feature_extraction_psd.png: PSD curves with frequency bands")
print("  - feature_extraction_bands.png: Band powers across trials")
print("  - eeg_features.csv: Extracted features in CSV format")
print()

print("Physiological Interpretation:")
print("  - Theta (4-8 Hz): Associated with emotional processing, memory")
print("  - Alpha (8-13 Hz): Relaxation, calmness (↑ = relaxed, ↓ = aroused)")
print("  - Beta (13-30 Hz): Active thinking, anxiety, arousal")
print()

print("Next Steps:")
print("  1. Use these features for emotion classification")
print("  2. Apply feature selection/dimensionality reduction")
print("  3. Train ML models (SVM, Random Forest)")
print()

print("="*80)
print("All feature extraction steps completed successfully!")
print("="*80)

# Show plots
print()
print("Displaying plots...")
plt.show()
