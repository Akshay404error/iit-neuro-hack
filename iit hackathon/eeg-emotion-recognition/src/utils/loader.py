"""
Data Loading Utilities

This module provides utilities for loading EEG data and generating synthetic demo data.
"""

import mne
import numpy as np
from typing import Optional, List


def load_eeg_file(
    filepath: str,
    preload: bool = True,
    verbose: bool = True
) -> mne.io.Raw:
    """
    Load EEG file from various formats.
    
    Supports: .fif, .edf, .bdf, .set (EEGLAB), .vhdr (BrainVision)
    
    Parameters
    ----------
    filepath : str
        Path to EEG file
    preload : bool, default=True
        Load data into memory
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.io.Raw
        Raw EEG data
        
    Examples
    --------
    >>> raw = load_eeg_file('data.fif')
    >>> raw = load_eeg_file('data.edf')
    """
    if verbose:
        print(f"Loading EEG file: {filepath}")
    
    # Determine file type and load
    if filepath.endswith('.fif'):
        raw = mne.io.read_raw_fif(filepath, preload=preload, verbose=verbose)
    elif filepath.endswith('.edf'):
        raw = mne.io.read_raw_edf(filepath, preload=preload, verbose=verbose)
    elif filepath.endswith('.bdf'):
        raw = mne.io.read_raw_bdf(filepath, preload=preload, verbose=verbose)
    elif filepath.endswith('.set'):
        raw = mne.io.read_raw_eeglab(filepath, preload=preload, verbose=verbose)
    elif filepath.endswith('.vhdr'):
        raw = mne.io.read_raw_brainvision(filepath, preload=preload, verbose=verbose)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    if verbose:
        print(f"✓ Loaded {len(raw.ch_names)} channels, {raw.n_times} samples")
        print(f"Sampling rate: {raw.info['sfreq']} Hz")
        print(f"Duration: {raw.times[-1]:.2f} seconds")
    
    return raw


def create_synthetic_eeg(
    n_channels: int = 32,
    sfreq: float = 250.0,
    duration: float = 60.0,
    ch_types: str = 'eeg',
    montage: str = 'standard_1020',
    verbose: bool = True
) -> mne.io.Raw:
    """
    Create synthetic EEG data for demonstration.
    
    Scientific Justification:
    - Simulates realistic EEG with multiple frequency components
    - Adds noise to mimic real recordings
    - Useful for testing pipeline without real data
    
    Parameters
    ----------
    n_channels : int, default=32
        Number of EEG channels
    sfreq : float, default=250.0
        Sampling frequency (Hz)
    duration : float, default=60.0
        Duration in seconds
    ch_types : str, default='eeg'
        Channel type
    montage : str, default='standard_1020'
        Electrode montage
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.io.Raw
        Synthetic raw EEG data
        
    Examples
    --------
    >>> raw = create_synthetic_eeg(n_channels=32, duration=60.0)
    """
    if verbose:
        print("="*60)
        print("CREATING SYNTHETIC EEG DATA")
        print("="*60)
        print(f"Channels: {n_channels}")
        print(f"Sampling rate: {sfreq} Hz")
        print(f"Duration: {duration} s")
    
    # Create time vector
    n_samples = int(sfreq * duration)
    times = np.arange(n_samples) / sfreq
    
    # Create synthetic data with multiple frequency components
    data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Add different frequency components
        # Delta (0.5-4 Hz)
        delta = 0.5 * np.sin(2 * np.pi * 2 * times + np.random.rand() * 2 * np.pi)
        
        # Theta (4-8 Hz)
        theta = 0.3 * np.sin(2 * np.pi * 6 * times + np.random.rand() * 2 * np.pi)
        
        # Alpha (8-13 Hz)
        alpha = 0.8 * np.sin(2 * np.pi * 10 * times + np.random.rand() * 2 * np.pi)
        
        # Beta (13-30 Hz)
        beta = 0.2 * np.sin(2 * np.pi * 20 * times + np.random.rand() * 2 * np.pi)
        
        # Gamma (30-50 Hz)
        gamma = 0.1 * np.sin(2 * np.pi * 40 * times + np.random.rand() * 2 * np.pi)
        
        # Add noise
        noise = 0.1 * np.random.randn(n_samples)
        
        # Combine components
        data[ch] = delta + theta + alpha + beta + gamma + noise
    
    # Convert to microvolts (typical EEG scale)
    data = data * 1e-5  # Scale to ~10 µV
    
    # Create channel names
    if n_channels <= 32:
        # Standard 10-20 system channel names
        ch_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2',
            'F9', 'F10', 'FT9', 'FT10', 'TP9', 'TP10',
            'P9', 'P10', 'PO9', 'PO10', 'Oz', 'Iz', 'FCz'
        ][:n_channels]
    else:
        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    
    # Create info structure
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )
    
    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=verbose)
    
    # Set montage
    try:
        montage_obj = mne.channels.make_standard_montage(montage)
        raw.set_montage(montage_obj, on_missing='ignore')
    except Exception as e:
        if verbose:
            print(f"⚠ Could not set montage: {e}")
    
    if verbose:
        print("✓ Synthetic EEG created")
        print("="*60)
    
    return raw


def create_synthetic_emotion_dataset(
    n_trials_per_emotion: int = 30,
    trial_duration: float = 4.0,
    emotions: Optional[List[str]] = None,
    n_channels: int = 32,
    sfreq: float = 250.0,
    verbose: bool = True
) -> tuple:
    """
    Create synthetic emotion dataset for demonstration.
    
    Parameters
    ----------
    n_trials_per_emotion : int, default=30
        Number of trials per emotion
    trial_duration : float, default=4.0
        Duration of each trial (seconds)
    emotions : list of str, optional
        Emotion labels. If None, uses ['happy', 'sad', 'neutral']
    n_channels : int, default=32
        Number of EEG channels
    sfreq : float, default=250.0
        Sampling frequency
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    tuple
        (raw, labels) - Raw EEG data and emotion labels
        
    Examples
    --------
    >>> raw, labels = create_synthetic_emotion_dataset(n_trials_per_emotion=30)
    """
    if emotions is None:
        emotions = ['happy', 'sad', 'neutral']
    
    if verbose:
        print("="*60)
        print("CREATING SYNTHETIC EMOTION DATASET")
        print("="*60)
        print(f"Emotions: {emotions}")
        print(f"Trials per emotion: {n_trials_per_emotion}")
        print(f"Trial duration: {trial_duration} s")
    
    # Calculate total duration
    n_trials = len(emotions) * n_trials_per_emotion
    total_duration = n_trials * trial_duration
    
    # Create synthetic EEG
    raw = create_synthetic_eeg(
        n_channels=n_channels,
        sfreq=sfreq,
        duration=total_duration,
        verbose=False
    )
    
    # Create labels
    labels = []
    for emotion in emotions:
        labels.extend([emotion] * n_trials_per_emotion)
    
    if verbose:
        print(f"✓ Created dataset with {n_trials} trials")
        print(f"Total duration: {total_duration} s")
        print("="*60)
    
    return raw, labels


def get_channel_pairs_for_asymmetry(
    ch_names: List[str],
    verbose: bool = True
) -> List[tuple]:
    """
    Get channel pairs for asymmetry analysis.
    
    Returns pairs of (left, right) hemisphere channels.
    
    Parameters
    ----------
    ch_names : list of str
        Channel names
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    list of tuple
        Channel pairs (left_idx, right_idx)
        
    Examples
    --------
    >>> pairs = get_channel_pairs_for_asymmetry(raw.ch_names)
    """
    # Common left-right pairs in 10-20 system
    pair_names = [
        ('Fp1', 'Fp2'),
        ('F3', 'F4'),
        ('F7', 'F8'),
        ('C3', 'C4'),
        ('T3', 'T4'),
        ('P3', 'P4'),
        ('T5', 'T6'),
        ('O1', 'O2')
    ]
    
    pairs = []
    for left_name, right_name in pair_names:
        if left_name in ch_names and right_name in ch_names:
            left_idx = ch_names.index(left_name)
            right_idx = ch_names.index(right_name)
            pairs.append((left_idx, right_idx))
    
    if verbose:
        print(f"Found {len(pairs)} channel pairs for asymmetry analysis")
        for left_idx, right_idx in pairs:
            print(f"  - {ch_names[left_idx]} <-> {ch_names[right_idx]}")
    
    return pairs
