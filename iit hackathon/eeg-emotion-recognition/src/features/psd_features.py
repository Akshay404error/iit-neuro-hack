"""
Power Spectral Density (PSD) Feature Extraction

This module extracts frequency-domain features from EEG for emotion recognition.

Scientific Rationale:
- Frequency bands correlate with emotional states:
  * Delta (0.5-4 Hz): Deep sleep, unconscious processes
  * Theta (4-8 Hz): Drowsiness, meditation, memory
  * Alpha (8-13 Hz): Relaxation, calmness (inverse with arousal)
  * Beta (13-30 Hz): Active thinking, focus, anxiety
  * Gamma (30-50 Hz): High-level cognition, attention
- Frontal alpha asymmetry: Left > Right = positive valence
- PSD features are robust and interpretable

References:
- Davidson (2004). What does the prefrontal cortex "do" in affect: perspectives on frontal EEG asymmetry research.
  Biological Psychology, 67(1-2), 219-234.
- Welch (1967). The use of fast Fourier transform for the estimation of power spectra.
  IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.
"""

import mne
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from scipy import signal


# Standard EEG frequency bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}


def compute_psd_welch(
    epochs: mne.Epochs,
    fmin: float = 0.5,
    fmax: float = 50.0,
    n_fft: int = 256,
    n_overlap: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    Scientific Justification:
    - Welch's method: Reduces variance by averaging periodograms
    - n_fft: Frequency resolution (256 samples ≈ 0.5 Hz at 250 Hz sampling)
    - Overlap: Reduces variance (50% overlap is standard)
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    fmin : float, default=0.5
        Minimum frequency (Hz)
    fmax : float, default=50.0
        Maximum frequency (Hz)
    n_fft : int, default=256
        FFT length (determines frequency resolution)
    n_overlap : int, optional
        Number of overlapping samples. If None, uses n_fft // 2
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    tuple
        (psds, freqs) - Power spectral densities and frequencies
        psds shape: (n_epochs, n_channels, n_freqs)
        
    Examples
    --------
    >>> psds, freqs = compute_psd_welch(epochs, fmin=0.5, fmax=50.0)
    >>> print(f"PSD shape: {psds.shape}")
    """
    if verbose:
        print("="*60)
        print("COMPUTING POWER SPECTRAL DENSITY (Welch's method)")
        print("="*60)
        print(f"Frequency range: {fmin}-{fmax} Hz")
        print(f"FFT length: {n_fft}")
    
    # Compute PSD
    psds, freqs = mne.time_frequency.psd_welch(
        epochs,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_overlap,
        verbose=verbose
    )
    
    if verbose:
        print(f"✓ PSD computed")
        print(f"PSD shape: {psds.shape} (epochs, channels, frequencies)")
        print(f"Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz")
        print("="*60)
    
    return psds, freqs


def extract_band_power(
    psds: np.ndarray,
    freqs: np.ndarray,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract band power features from PSD.
    
    Scientific Justification:
    - Band power = integral of PSD over frequency band
    - Represents total energy in that frequency range
    - Correlates with cognitive and emotional states
    
    Parameters
    ----------
    psds : np.ndarray
        Power spectral densities (n_epochs, n_channels, n_freqs)
    freqs : np.ndarray
        Frequency values
    bands : dict, optional
        Frequency bands dictionary. If None, uses standard bands.
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    pd.DataFrame
        Band power features (n_epochs, n_channels * n_bands)
        Columns: 'ch0_delta', 'ch0_theta', ..., 'ch31_gamma'
        
    Examples
    --------
    >>> psds, freqs = compute_psd_welch(epochs)
    >>> band_powers = extract_band_power(psds, freqs)
    >>> print(band_powers.head())
    """
    if bands is None:
        bands = FREQ_BANDS
    
    if verbose:
        print("="*60)
        print("EXTRACTING BAND POWER FEATURES")
        print("="*60)
        print(f"Frequency bands: {bands}")
    
    n_epochs, n_channels, n_freqs = psds.shape
    
    # Initialize feature dictionary
    features = {}
    
    # Extract band power for each channel and band
    for ch_idx in range(n_channels):
        for band_name, (fmin, fmax) in bands.items():
            # Find frequency indices
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            
            # Compute band power (mean PSD in band)
            band_power = np.mean(psds[:, ch_idx, freq_mask], axis=1)
            
            # Store feature
            feature_name = f'ch{ch_idx}_{band_name}'
            features[feature_name] = band_power
    
    # Convert to DataFrame
    df = pd.DataFrame(features)
    
    if verbose:
        print(f"✓ Extracted {len(df.columns)} features")
        print(f"Feature shape: {df.shape}")
        print("="*60)
    
    return df


def extract_band_power_ratio(
    psds: np.ndarray,
    freqs: np.ndarray,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract band power ratio features.
    
    Scientific Justification:
    - Ratios normalize for individual differences in absolute power
    - Common ratios:
      * Theta/Beta: Attention, ADHD marker
      * Alpha/Beta: Relaxation vs. activation
      * (Alpha+Theta)/Beta: Engagement index
    
    Parameters
    ----------
    psds : np.ndarray
        Power spectral densities
    freqs : np.ndarray
        Frequency values
    bands : dict, optional
        Frequency bands
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    pd.DataFrame
        Band power ratio features
        
    Examples
    --------
    >>> ratios = extract_band_power_ratio(psds, freqs)
    """
    if bands is None:
        bands = FREQ_BANDS
    
    # First extract absolute band powers
    band_powers = extract_band_power(psds, freqs, bands, verbose=False)
    
    if verbose:
        print("="*60)
        print("EXTRACTING BAND POWER RATIOS")
        print("="*60)
    
    n_channels = psds.shape[1]
    ratio_features = {}
    
    # Compute ratios for each channel
    for ch_idx in range(n_channels):
        # Get band powers for this channel
        theta = band_powers[f'ch{ch_idx}_theta'].values
        alpha = band_powers[f'ch{ch_idx}_alpha'].values
        beta = band_powers[f'ch{ch_idx}_beta'].values
        
        # Compute ratios (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        ratio_features[f'ch{ch_idx}_theta_beta'] = theta / (beta + epsilon)
        ratio_features[f'ch{ch_idx}_alpha_beta'] = alpha / (beta + epsilon)
        ratio_features[f'ch{ch_idx}_theta_alpha_beta'] = (theta + alpha) / (beta + epsilon)
    
    df = pd.DataFrame(ratio_features)
    
    if verbose:
        print(f"✓ Extracted {len(df.columns)} ratio features")
        print("="*60)
    
    return df


def extract_asymmetry_features(
    psds: np.ndarray,
    freqs: np.ndarray,
    channel_pairs: List[Tuple[int, int]],
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract hemispheric asymmetry features.
    
    Scientific Justification:
    - Frontal alpha asymmetry: Emotional valence marker
      * Left > Right: Positive emotions (approach)
      * Right > Left: Negative emotions (withdrawal)
    - Computed as: log(Right) - log(Left)
    
    Parameters
    ----------
    psds : np.ndarray
        Power spectral densities
    freqs : np.ndarray
        Frequency values
    channel_pairs : list of tuple
        Pairs of (left_ch, right_ch) indices
        E.g., [(0, 1), (2, 3)] for F3-F4, F7-F8
    bands : dict, optional
        Frequency bands
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    pd.DataFrame
        Asymmetry features
        
    Examples
    --------
    >>> # F3-F4 asymmetry (channels 0 and 1)
    >>> asym = extract_asymmetry_features(psds, freqs, channel_pairs=[(0, 1)])
    """
    if bands is None:
        bands = FREQ_BANDS
    
    if verbose:
        print("="*60)
        print("EXTRACTING ASYMMETRY FEATURES")
        print("="*60)
        print(f"Channel pairs: {channel_pairs}")
    
    # Extract band powers
    band_powers = extract_band_power(psds, freqs, bands, verbose=False)
    
    asymmetry_features = {}
    
    # Compute asymmetry for each pair and band
    for pair_idx, (left_ch, right_ch) in enumerate(channel_pairs):
        for band_name in bands.keys():
            # Get left and right band powers
            left_power = band_powers[f'ch{left_ch}_{band_name}'].values
            right_power = band_powers[f'ch{right_ch}_{band_name}'].values
            
            # Compute asymmetry: log(Right) - log(Left)
            # Add epsilon to avoid log(0)
            epsilon = 1e-10
            asymmetry = np.log(right_power + epsilon) - np.log(left_power + epsilon)
            
            feature_name = f'asym_pair{pair_idx}_{band_name}'
            asymmetry_features[feature_name] = asymmetry
    
    df = pd.DataFrame(asymmetry_features)
    
    if verbose:
        print(f"✓ Extracted {len(df.columns)} asymmetry features")
        print("="*60)
    
    return df


def extract_all_psd_features(
    epochs: mne.Epochs,
    channel_pairs: Optional[List[Tuple[int, int]]] = None,
    include_ratios: bool = True,
    include_asymmetry: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract complete set of PSD-based features.
    
    This is the recommended function for feature extraction.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    channel_pairs : list of tuple, optional
        Channel pairs for asymmetry features
        If None and include_asymmetry=True, will skip asymmetry
    include_ratios : bool, default=True
        Include band power ratios
    include_asymmetry : bool, default=True
        Include asymmetry features
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    pd.DataFrame
        Complete feature matrix (n_epochs, n_features)
        
    Examples
    --------
    >>> # Extract all features
    >>> features = extract_all_psd_features(epochs, channel_pairs=[(0, 1)])
    >>> print(f"Feature matrix shape: {features.shape}")
    """
    if verbose:
        print("="*60)
        print("EXTRACTING ALL PSD FEATURES")
        print("="*60)
    
    # Step 1: Compute PSD
    psds, freqs = compute_psd_welch(epochs, verbose=verbose)
    
    # Step 2: Extract band powers
    band_powers = extract_band_power(psds, freqs, verbose=verbose)
    
    # Step 3: Extract ratios (optional)
    if include_ratios:
        ratios = extract_band_power_ratio(psds, freqs, verbose=verbose)
        features = pd.concat([band_powers, ratios], axis=1)
    else:
        features = band_powers
    
    # Step 4: Extract asymmetry (optional)
    if include_asymmetry and channel_pairs is not None:
        asymmetry = extract_asymmetry_features(psds, freqs, channel_pairs, verbose=verbose)
        features = pd.concat([features, asymmetry], axis=1)
    
    if verbose:
        print("="*60)
        print(f"✓ Total features extracted: {features.shape[1]}")
        print(f"Feature matrix shape: {features.shape}")
        print("="*60)
    
    return features
