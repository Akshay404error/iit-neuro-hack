"""
EEG Segmentation Module

This module implements epoch extraction and windowing for EEG analysis.

Scientific Rationale:
- Segmentation divides continuous EEG into fixed-length epochs
- Epoch length (2-4s) balances temporal resolution and statistical power
- Overlapping windows increase sample size for machine learning
- Baseline correction removes DC offset and slow drifts

References:
- Cohen (2014). Analyzing neural time series data: theory and practice. MIT Press.
- Makeig et al. (2004). Mining event-related brain dynamics. Trends in Cognitive Sciences, 8(5), 204-210.
"""

import mne
import numpy as np
from typing import Optional, Union, List


def create_fixed_length_epochs(
    raw: mne.io.Raw,
    duration: float = 2.0,
    overlap: float = 0.0,
    preload: bool = True,
    verbose: bool = True
) -> mne.Epochs:
    """
    Create fixed-length epochs from continuous EEG data.
    
    Scientific Justification:
    - Duration 2-4s: Optimal for capturing emotional states
      - Too short: Insufficient data for frequency analysis
      - Too long: Emotion may change within epoch
    - Overlap: Increases sample size without collecting more data
      - 50% overlap is common (doubles sample size)
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    duration : float, default=2.0
        Epoch duration in seconds
    overlap : float, default=0.0
        Overlap between consecutive epochs in seconds
        E.g., overlap=1.0 with duration=2.0 gives 50% overlap
    preload : bool, default=True
        Load data into memory
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.Epochs
        Fixed-length epochs
        
    Examples
    --------
    >>> # Non-overlapping 2-second epochs
    >>> epochs = create_fixed_length_epochs(raw, duration=2.0, overlap=0.0)
    >>> 
    >>> # 50% overlapping 4-second epochs
    >>> epochs = create_fixed_length_epochs(raw, duration=4.0, overlap=2.0)
    """
    if verbose:
        print("="*60)
        print("CREATING FIXED-LENGTH EPOCHS")
        print("="*60)
        print(f"Duration: {duration} s")
        print(f"Overlap: {overlap} s")
    
    # Create fixed-length epochs
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=overlap,
        preload=preload,
        verbose=verbose
    )
    
    if verbose:
        print(f"✓ Created {len(epochs)} epochs")
        print(f"Epoch shape: {epochs.get_data().shape}")
        print("="*60)
    
    return epochs


def create_event_based_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    tmin: float = -0.2,
    tmax: float = 2.0,
    baseline: Optional[tuple] = (-0.2, 0.0),
    preload: bool = True,
    verbose: bool = True
) -> mne.Epochs:
    """
    Create event-based epochs (e.g., stimulus-locked).
    
    Scientific Justification:
    - Event-locked analysis: Aligns epochs to stimulus/response onset
    - Baseline correction: Removes pre-stimulus activity
    - tmin/tmax: Captures pre-stimulus baseline and post-stimulus response
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    events : np.ndarray
        Event array (n_events, 3) from mne.find_events()
    event_id : dict
        Event ID dictionary (e.g., {'happy': 1, 'sad': 2})
    tmin : float, default=-0.2
        Start time before event (seconds)
    tmax : float, default=2.0
        End time after event (seconds)
    baseline : tuple, optional
        Baseline period (start, end) in seconds
        None = no baseline correction
    preload : bool, default=True
        Load data into memory
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.Epochs
        Event-based epochs
        
    Examples
    --------
    >>> events = mne.find_events(raw, stim_channel='STI')
    >>> event_id = {'happy': 1, 'sad': 2, 'neutral': 3}
    >>> epochs = create_event_based_epochs(raw, events, event_id)
    """
    if verbose:
        print("="*60)
        print("CREATING EVENT-BASED EPOCHS")
        print("="*60)
        print(f"Event types: {event_id}")
        print(f"Time window: [{tmin}, {tmax}] s")
        print(f"Baseline: {baseline}")
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=preload,
        verbose=verbose
    )
    
    if verbose:
        print(f"✓ Created {len(epochs)} epochs")
        for event_name, event_code in event_id.items():
            n_events = len(epochs[event_name])
            print(f"  - {event_name}: {n_events} epochs")
        print("="*60)
    
    return epochs


def apply_baseline_correction(
    epochs: mne.Epochs,
    baseline: tuple = (None, 0),
    verbose: bool = True
) -> mne.Epochs:
    """
    Apply baseline correction to epochs.
    
    Scientific Justification:
    - Removes pre-stimulus activity and DC offset
    - Baseline period should be before stimulus onset
    - Common baselines: (-0.2, 0), (None, 0), or entire epoch mean
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    baseline : tuple, default=(None, 0)
        Baseline period (start, end) in seconds
        (None, 0) = from epoch start to time 0
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.Epochs
        Baseline-corrected epochs
        
    Examples
    --------
    >>> epochs_corrected = apply_baseline_correction(epochs, baseline=(-0.2, 0))
    """
    if verbose:
        print(f"Applying baseline correction: {baseline}")
    
    epochs_corrected = epochs.copy().apply_baseline(baseline, verbose=verbose)
    
    if verbose:
        print("✓ Baseline correction applied")
    
    return epochs_corrected


def reject_bad_epochs(
    epochs: mne.Epochs,
    reject: Optional[dict] = None,
    flat: Optional[dict] = None,
    verbose: bool = True
) -> mne.Epochs:
    """
    Reject epochs with artifacts based on amplitude criteria.
    
    Scientific Justification:
    - Removes epochs with extreme amplitudes (artifacts)
    - Reject criteria depend on electrode type and preprocessing
    - Typical EEG reject: 100-150 µV (after filtering/ICA)
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    reject : dict, optional
        Rejection criteria (e.g., {'eeg': 100e-6})
        If None, uses default: {'eeg': 100e-6}
    flat : dict, optional
        Flat channel criteria (e.g., {'eeg': 1e-6})
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.Epochs
        Epochs with bad epochs dropped
        
    Examples
    --------
    >>> # Reject epochs with amplitude > 100 µV
    >>> epochs_clean = reject_bad_epochs(epochs, reject={'eeg': 100e-6})
    """
    if reject is None:
        reject = {'eeg': 100e-6}  # 100 µV
    
    if verbose:
        print("="*60)
        print("REJECTING BAD EPOCHS")
        print("="*60)
        print(f"Rejection criteria: {reject}")
        if flat:
            print(f"Flat criteria: {flat}")
        print(f"Epochs before rejection: {len(epochs)}")
    
    # Drop bad epochs
    epochs_clean = epochs.copy().drop_bad(reject=reject, flat=flat, verbose=verbose)
    
    if verbose:
        n_rejected = len(epochs) - len(epochs_clean)
        rejection_rate = n_rejected / len(epochs) * 100
        print(f"✓ Rejected {n_rejected} epochs ({rejection_rate:.1f}%)")
        print(f"Epochs after rejection: {len(epochs_clean)}")
        print("="*60)
    
    return epochs_clean


def create_epochs_with_labels(
    raw: mne.io.Raw,
    labels: List[str],
    duration: float = 2.0,
    overlap: float = 0.0,
    verbose: bool = True
) -> tuple:
    """
    Create epochs with corresponding emotion labels.
    
    This is useful when you have continuous data with known emotion labels.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    labels : list of str
        Emotion labels for each epoch (e.g., ['happy', 'sad', 'neutral'])
    duration : float, default=2.0
        Epoch duration in seconds
    overlap : float, default=0.0
        Overlap between epochs in seconds
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    tuple
        (epochs, labels_array) - Epochs and corresponding labels
        
    Examples
    --------
    >>> # Assume 60 seconds of data with known emotions
    >>> labels = ['happy'] * 15 + ['sad'] * 15  # 30 epochs of 2s each
    >>> epochs, y = create_epochs_with_labels(raw, labels, duration=2.0)
    """
    # Create epochs
    epochs = create_fixed_length_epochs(
        raw, duration=duration, overlap=overlap, verbose=verbose
    )
    
    # Ensure labels match number of epochs
    if len(labels) != len(epochs):
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match "
            f"number of epochs ({len(epochs)})"
        )
    
    labels_array = np.array(labels)
    
    if verbose:
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  - {label}: {count} epochs")
    
    return epochs, labels_array
