"""
Artifact Subspace Reconstruction (ASR)

This module implements ASR for removing high-amplitude burst artifacts from EEG.

Scientific Rationale:
- ASR identifies and removes non-stationary high-amplitude artifacts
- Uses sliding window PCA to detect subspaces contaminated by artifacts
- Reconstructs clean signal by projecting out artifact subspace
- Preserves underlying brain activity better than simple rejection

References:
- Mullen et al. (2015). Real-time neuroimaging and cognitive monitoring using
  wearable dry EEG. IEEE Transactions on Biomedical Engineering, 62(11), 2553-2567.
- Chang et al. (2020). Evaluation of artifact subspace reconstruction for automatic
  artifact components removal in multi-channel EEG recordings. IEEE Transactions on
  Biomedical Engineering, 67(4), 1114-1121.
"""

import mne
import numpy as np
from typing import Optional
from scipy import linalg


def apply_asr(
    raw: mne.io.Raw,
    cutoff: float = 5.0,
    window_len: float = 0.5,
    verbose: bool = True
) -> mne.io.Raw:
    """
    Apply Artifact Subspace Reconstruction to remove burst artifacts.
    
    Scientific Justification:
    - Cutoff (default 5): Standard deviations above baseline for artifact detection
      Higher values = more conservative (fewer artifacts removed)
    - Window length (0.5s): Balance between temporal resolution and statistical power
    - ASR is particularly effective for motion artifacts in mobile EEG
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (should be filtered first)
    cutoff : float, default=5.0
        Standard deviation cutoff for artifact detection
        Typical range: 3-10 (lower = more aggressive)
    window_len : float, default=0.5
        Window length in seconds for computing statistics
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.io.Raw
        Cleaned raw data with burst artifacts removed
        
    Notes
    -----
    This is a simplified ASR implementation. For production use, consider
    using the ASRPY library or EEGLAB's clean_rawdata plugin.
    
    Examples
    --------
    >>> raw_clean = apply_asr(raw, cutoff=5.0, window_len=0.5)
    """
    if verbose:
        print("="*60)
        print("APPLYING ARTIFACT SUBSPACE RECONSTRUCTION (ASR)")
        print("="*60)
        print(f"Cutoff: {cutoff} SD")
        print(f"Window length: {window_len} s")
    
    # Get data
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    n_channels, n_samples = data.shape
    
    # Calculate window size in samples
    window_samples = int(window_len * sfreq)
    
    # Step 1: Compute baseline statistics from clean portion
    # Use first 30 seconds as baseline (assuming it's relatively clean)
    baseline_samples = min(int(30 * sfreq), n_samples // 2)
    baseline_data = data[:, :baseline_samples]
    
    # Compute covariance matrix of baseline
    baseline_cov = np.cov(baseline_data)
    
    # Step 2: Process data in sliding windows
    cleaned_data = data.copy()
    n_windows = (n_samples - window_samples) // (window_samples // 2) + 1
    
    for i in range(n_windows):
        start = i * (window_samples // 2)
        end = min(start + window_samples, n_samples)
        
        if end - start < window_samples // 2:
            break
        
        window_data = data[:, start:end]
        
        # Compute window covariance
        window_cov = np.cov(window_data)
        
        # Eigenvalue decomposition
        try:
            eigvals, eigvecs = linalg.eigh(window_cov, baseline_cov)
            
            # Identify artifact subspace (eigenvalues > cutoff^2)
            artifact_mask = eigvals > cutoff ** 2
            
            if np.any(artifact_mask):
                # Reconstruct clean signal by removing artifact subspace
                clean_eigvals = eigvals.copy()
                clean_eigvals[artifact_mask] = cutoff ** 2
                
                # Reconstruct covariance
                clean_cov = eigvecs @ np.diag(clean_eigvals) @ eigvecs.T
                
                # Apply Mahalanobis whitening
                M = linalg.sqrtm(linalg.inv(clean_cov))
                cleaned_window = M @ window_data
                
                # Reverse whitening with baseline
                M_baseline = linalg.sqrtm(baseline_cov)
                cleaned_window = M_baseline @ cleaned_window
                
                cleaned_data[:, start:end] = cleaned_window
        except Exception as e:
            if verbose:
                print(f"⚠ Warning: Could not process window {i}: {e}")
            continue
    
    # Create new Raw object with cleaned data
    raw_clean = raw.copy()
    raw_clean._data = cleaned_data
    
    if verbose:
        print("✓ ASR complete")
        print("="*60)
    
    return raw_clean


def apply_asr_pyprep(
    raw: mne.io.Raw,
    cutoff: float = 5.0,
    verbose: bool = True
) -> mne.io.Raw:
    """
    Apply ASR using PyPREP library (if available).
    
    PyPREP provides a more robust ASR implementation based on EEGLAB's clean_rawdata.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    cutoff : float, default=5.0
        Standard deviation cutoff
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.io.Raw
        Cleaned raw data
        
    Examples
    --------
    >>> raw_clean = apply_asr_pyprep(raw, cutoff=5.0)
    """
    try:
        from pyprep.removeTrend import removeTrend
        from pyprep.find_noisy_channels import NoisyChannels
        
        if verbose:
            print("="*60)
            print("APPLYING ASR (PyPREP)")
            print("="*60)
        
        # PyPREP's ASR implementation
        # Note: This is a placeholder - PyPREP's ASR is part of PrepPipeline
        # For full implementation, use PrepPipeline class
        
        if verbose:
            print("⚠ PyPREP ASR requires PrepPipeline class")
            print("Falling back to basic ASR implementation")
        
        return apply_asr(raw, cutoff=cutoff, verbose=verbose)
        
    except ImportError:
        if verbose:
            print("⚠ PyPREP not installed, using basic ASR implementation")
        return apply_asr(raw, cutoff=cutoff, verbose=verbose)


def detect_bad_channels_asr(
    raw: mne.io.Raw,
    cutoff: float = 5.0,
    verbose: bool = True
) -> list:
    """
    Detect bad channels using ASR criteria.
    
    Channels with persistent high-amplitude artifacts may need to be interpolated.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    cutoff : float, default=5.0
        Standard deviation cutoff
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    list
        List of bad channel names
        
    Examples
    --------
    >>> bad_channels = detect_bad_channels_asr(raw, cutoff=5.0)
    >>> raw.info['bads'] = bad_channels
    """
    data = raw.get_data()
    ch_names = raw.ch_names
    
    # Compute channel-wise statistics
    ch_std = np.std(data, axis=1)
    median_std = np.median(ch_std)
    
    # Identify channels with excessive variance
    bad_mask = ch_std > cutoff * median_std
    bad_channels = [ch_names[i] for i in np.where(bad_mask)[0]]
    
    if verbose:
        print(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
    
    return bad_channels
