"""
Independent Component Analysis (ICA) for Artifact Removal

This module implements ICA-based artifact removal for EEG preprocessing.

Scientific Rationale:
- ICA separates EEG into statistically independent components
- Artifacts (eye blinks, muscle activity) are spatially and temporally distinct
- Removing artifact components preserves brain signals while eliminating noise

References:
- Jung et al. (2000). Removing electroencephalographic artifacts by blind source separation.
  Psychophysiology, 37(2), 163-178.
- Delorme & Makeig (2004). EEGLAB: an open source toolbox for analysis of single-trial
  EEG dynamics. Journal of Neuroscience Methods, 134(1), 9-21.
"""

import mne
from mne.preprocessing import ICA
import numpy as np
from typing import Optional, List


def fit_ica(
    raw: mne.io.Raw,
    n_components: Optional[int] = None,
    method: str = 'fastica',
    random_state: int = 42,
    max_iter: int = 800,
    verbose: bool = True
) -> ICA:
    """
    Fit ICA to decompose EEG into independent components.
    
    Scientific Justification:
    - FastICA: Fast convergence, robust to outliers
    - n_components: Typically 15-25 for emotion recognition (balance between decomposition and overfitting)
    - High-pass filter (1 Hz) recommended before ICA to improve stationarity
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (should be filtered first)
    n_components : int, optional
        Number of ICA components. If None, uses min(n_channels, n_samples//3)
    method : str, default='fastica'
        ICA algorithm ('fastica', 'infomax', 'picard')
    random_state : int, default=42
        Random seed for reproducibility
    max_iter : int, default=800
        Maximum iterations for convergence
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    ICA
        Fitted ICA object
        
    Examples
    --------
    >>> raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=None)
    >>> ica = fit_ica(raw_filtered, n_components=20)
    """
    if verbose:
        print("="*60)
        print("FITTING ICA")
        print("="*60)
        print(f"Method: {method}")
        print(f"Number of components: {n_components if n_components else 'auto'}")
    
    # Create ICA object
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter=max_iter,
        verbose=verbose
    )
    
    # Fit ICA
    # Note: It's recommended to use a copy of raw filtered at 1 Hz for ICA fitting
    ica.fit(raw, verbose=verbose)
    
    if verbose:
        print(f"✓ ICA fitted with {ica.n_components_} components")
        print("="*60)
    
    return ica


def detect_artifact_components(
    ica: ICA,
    raw: mne.io.Raw,
    eog_channels: Optional[List[str]] = None,
    ecg_channels: Optional[List[str]] = None,
    threshold: float = 0.8,
    verbose: bool = True
) -> dict:
    """
    Automatically detect artifact components (EOG, ECG).
    
    Scientific Justification:
    - EOG artifacts: Correlate with eye movement channels
    - ECG artifacts: Correlate with heart activity
    - Threshold 0.8: Conservative to avoid removing brain components
    
    Parameters
    ----------
    ica : ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw EEG data
    eog_channels : list of str, optional
        EOG channel names. If None, attempts auto-detection
    ecg_channels : list of str, optional
        ECG channel names. If None, attempts auto-detection
    threshold : float, default=0.8
        Correlation threshold for artifact detection
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    dict
        Dictionary with 'eog' and 'ecg' component indices
        
    Examples
    --------
    >>> artifacts = detect_artifact_components(ica, raw, threshold=0.8)
    >>> print(f"EOG components: {artifacts['eog']}")
    """
    artifacts = {'eog': [], 'ecg': []}
    
    if verbose:
        print("="*60)
        print("DETECTING ARTIFACT COMPONENTS")
        print("="*60)
    
    # Detect EOG artifacts
    try:
        if eog_channels is None:
            # Try to find EOG channels automatically
            eog_inds, eog_scores = ica.find_bads_eog(raw, threshold=threshold, verbose=verbose)
        else:
            eog_inds, eog_scores = ica.find_bads_eog(
                raw, ch_name=eog_channels, threshold=threshold, verbose=verbose
            )
        artifacts['eog'] = eog_inds
        
        if verbose:
            print(f"✓ Found {len(eog_inds)} EOG components: {eog_inds}")
    except Exception as e:
        if verbose:
            print(f"⚠ Could not detect EOG components: {e}")
    
    # Detect ECG artifacts
    try:
        if ecg_channels is None:
            ecg_inds, ecg_scores = ica.find_bads_ecg(raw, threshold=threshold, verbose=verbose)
        else:
            ecg_inds, ecg_scores = ica.find_bads_ecg(
                raw, ch_name=ecg_channels, threshold=threshold, verbose=verbose
            )
        artifacts['ecg'] = ecg_inds
        
        if verbose:
            print(f"✓ Found {len(ecg_inds)} ECG components: {ecg_inds}")
    except Exception as e:
        if verbose:
            print(f"⚠ Could not detect ECG components: {e}")
    
    if verbose:
        print("="*60)
    
    return artifacts


def apply_ica(
    ica: ICA,
    raw: mne.io.Raw,
    exclude: Optional[List[int]] = None,
    verbose: bool = True
) -> mne.io.Raw:
    """
    Apply ICA to remove artifact components from raw data.
    
    Parameters
    ----------
    ica : ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw EEG data
    exclude : list of int, optional
        Component indices to exclude. If None, uses ica.exclude
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.io.Raw
        Cleaned raw data with artifacts removed
        
    Examples
    --------
    >>> raw_clean = apply_ica(ica, raw, exclude=[0, 1, 5])
    """
    if exclude is not None:
        ica.exclude = exclude
    
    if verbose:
        print("="*60)
        print("APPLYING ICA")
        print("="*60)
        print(f"Excluding components: {ica.exclude}")
    
    # Apply ICA (creates a copy by default)
    raw_clean = ica.apply(raw.copy(), verbose=verbose)
    
    if verbose:
        print("✓ ICA applied, artifacts removed")
        print("="*60)
    
    return raw_clean


def run_ica_pipeline(
    raw: mne.io.Raw,
    n_components: Optional[int] = None,
    eog_channels: Optional[List[str]] = None,
    ecg_channels: Optional[List[str]] = None,
    threshold: float = 0.8,
    manual_exclude: Optional[List[int]] = None,
    verbose: bool = True
) -> tuple:
    """
    Complete ICA pipeline: fit, detect artifacts, and apply.
    
    This is the recommended workflow for ICA-based artifact removal.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (should be filtered at 1 Hz high-pass)
    n_components : int, optional
        Number of ICA components
    eog_channels : list of str, optional
        EOG channel names
    ecg_channels : list of str, optional
        ECG channel names
    threshold : float, default=0.8
        Artifact detection threshold
    manual_exclude : list of int, optional
        Additional components to exclude manually
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    tuple
        (raw_clean, ica, artifacts) - Cleaned data, ICA object, and artifact dict
        
    Examples
    --------
    >>> # Prepare data: filter at 1 Hz for ICA
    >>> raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None)
    >>> raw_clean, ica, artifacts = run_ica_pipeline(raw_for_ica, n_components=20)
    """
    # Step 1: Fit ICA
    ica = fit_ica(raw, n_components=n_components, verbose=verbose)
    
    # Step 2: Detect artifacts
    artifacts = detect_artifact_components(
        ica, raw, eog_channels=eog_channels, ecg_channels=ecg_channels,
        threshold=threshold, verbose=verbose
    )
    
    # Step 3: Combine automatic and manual exclusions
    exclude = list(set(artifacts['eog'] + artifacts['ecg']))
    if manual_exclude:
        exclude = list(set(exclude + manual_exclude))
    
    # Step 4: Apply ICA
    raw_clean = apply_ica(ica, raw, exclude=exclude, verbose=verbose)
    
    return raw_clean, ica, artifacts
