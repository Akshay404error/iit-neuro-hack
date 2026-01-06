"""
EEG Re-referencing Module

This module implements different re-referencing strategies for EEG preprocessing.

Scientific Rationale:
- Re-referencing removes reference electrode bias and improves spatial resolution
- Common Average Reference (CAR): Most common for emotion recognition
- Linked mastoids: Traditional reference for clinical EEG
- Laplacian: Enhances local activity, reduces volume conduction

References:
- Yao (2001). A method to standardize a reference of scalp EEG recordings to a point at infinity.
  Physiological Measurement, 22(4), 693.
- Nunez & Srinivasan (2006). Electric fields of the brain: The neurophysics of EEG.
  Oxford University Press.
"""

import mne
import numpy as np
from typing import Optional, List


def apply_common_average_reference(
    raw: mne.io.Raw,
    verbose: bool = True
) -> mne.io.Raw:
    """
    Apply Common Average Reference (CAR).
    
    Scientific Justification:
    - CAR subtracts the average of all electrodes from each electrode
    - Removes common-mode noise and reference bias
    - Most widely used reference for emotion recognition studies
    - Assumes brain activity is zero-sum (valid for large electrode arrays)
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.io.Raw
        Re-referenced raw data
        
    Examples
    --------
    >>> raw_car = apply_common_average_reference(raw)
    """
    if verbose:
        print("="*60)
        print("APPLYING COMMON AVERAGE REFERENCE (CAR)")
        print("="*60)
    
    # Apply CAR
    raw_car = raw.copy().set_eeg_reference(ref_channels='average', verbose=verbose)
    
    if verbose:
        print("✓ CAR applied")
        print("="*60)
    
    return raw_car


def apply_mastoid_reference(
    raw: mne.io.Raw,
    mastoid_channels: Optional[List[str]] = None,
    verbose: bool = True
) -> mne.io.Raw:
    """
    Apply linked mastoid reference.
    
    Scientific Justification:
    - Mastoids (M1, M2) are electrically neutral locations
    - Linked mastoids = average of left and right mastoid electrodes
    - Traditional reference in clinical EEG
    - Good for frontal/central electrode analysis
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    mastoid_channels : list of str, optional
        Names of mastoid channels (e.g., ['M1', 'M2'], ['TP9', 'TP10'])
        If None, attempts to auto-detect
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.io.Raw
        Re-referenced raw data
        
    Examples
    --------
    >>> raw_mastoid = apply_mastoid_reference(raw, mastoid_channels=['M1', 'M2'])
    """
    if verbose:
        print("="*60)
        print("APPLYING LINKED MASTOID REFERENCE")
        print("="*60)
    
    # Auto-detect mastoid channels if not provided
    if mastoid_channels is None:
        ch_names = raw.ch_names
        # Common mastoid channel names
        possible_names = ['M1', 'M2', 'TP9', 'TP10', 'A1', 'A2']
        mastoid_channels = [ch for ch in possible_names if ch in ch_names]
        
        if not mastoid_channels:
            raise ValueError(
                "Could not auto-detect mastoid channels. "
                "Please specify mastoid_channels parameter."
            )
    
    if verbose:
        print(f"Using mastoid channels: {mastoid_channels}")
    
    # Apply mastoid reference
    raw_mastoid = raw.copy().set_eeg_reference(
        ref_channels=mastoid_channels, verbose=verbose
    )
    
    if verbose:
        print("✓ Mastoid reference applied")
        print("="*60)
    
    return raw_mastoid


def apply_laplacian_reference(
    raw: mne.io.Raw,
    verbose: bool = True
) -> mne.io.Raw:
    """
    Apply surface Laplacian (current source density) reference.
    
    Scientific Justification:
    - Laplacian enhances local cortical activity
    - Reduces volume conduction from distant sources
    - Improves spatial resolution
    - Useful for detecting focal emotional responses
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with standard 10-20 montage
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.io.Raw
        Laplacian-referenced raw data
        
    Notes
    -----
    Requires electrode positions (montage). This is a simplified implementation
    using nearest-neighbor Laplacian. For full CSD, use MNE's compute_current_source_density.
    
    Examples
    --------
    >>> # Ensure montage is set
    >>> montage = mne.channels.make_standard_montage('standard_1020')
    >>> raw.set_montage(montage)
    >>> raw_laplacian = apply_laplacian_reference(raw)
    """
    if verbose:
        print("="*60)
        print("APPLYING LAPLACIAN REFERENCE")
        print("="*60)
    
    # Check if montage is set
    if raw.get_montage() is None:
        if verbose:
            print("⚠ No montage found. Setting standard 10-20 montage...")
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
    
    # Compute Current Source Density (CSD) - this is the Laplacian
    try:
        raw_laplacian = mne.preprocessing.compute_current_source_density(
            raw.copy(), verbose=verbose
        )
        
        if verbose:
            print("✓ Laplacian reference applied")
    except Exception as e:
        if verbose:
            print(f"⚠ Could not apply Laplacian: {e}")
            print("Falling back to CAR")
        raw_laplacian = apply_common_average_reference(raw, verbose=False)
    
    if verbose:
        print("="*60)
    
    return raw_laplacian


def apply_rest_reference(
    raw: mne.io.Raw,
    verbose: bool = True
) -> mne.io.Raw:
    """
    Apply Reference Electrode Standardization Technique (REST).
    
    Scientific Justification:
    - REST approximates an infinity reference
    - Uses forward modeling to estimate reference-free potentials
    - Theoretically optimal reference
    - Requires electrode positions and head model
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with montage
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.io.Raw
        REST-referenced raw data
        
    Notes
    -----
    This is a placeholder. Full REST implementation requires:
    1. Electrode positions
    2. Head model (spherical or realistic)
    3. Forward solution computation
    
    For production use, consider using EEGLAB's REST plugin.
    
    Examples
    --------
    >>> raw_rest = apply_rest_reference(raw)
    """
    if verbose:
        print("="*60)
        print("APPLYING REST REFERENCE")
        print("="*60)
        print("⚠ REST not fully implemented. Using CAR instead.")
        print("="*60)
    
    # Fallback to CAR
    return apply_common_average_reference(raw, verbose=False)


def choose_reference(
    raw: mne.io.Raw,
    reference_type: str = 'average',
    mastoid_channels: Optional[List[str]] = None,
    verbose: bool = True
) -> mne.io.Raw:
    """
    Apply specified re-referencing strategy.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    reference_type : str, default='average'
        Type of reference: 'average', 'mastoid', 'laplacian', 'rest'
    mastoid_channels : list of str, optional
        Mastoid channel names (for 'mastoid' reference)
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    mne.io.Raw
        Re-referenced raw data
        
    Examples
    --------
    >>> # Common average reference (recommended for emotion recognition)
    >>> raw_ref = choose_reference(raw, reference_type='average')
    >>> 
    >>> # Linked mastoids
    >>> raw_ref = choose_reference(raw, reference_type='mastoid', 
    ...                            mastoid_channels=['M1', 'M2'])
    """
    if reference_type == 'average':
        return apply_common_average_reference(raw, verbose=verbose)
    elif reference_type == 'mastoid':
        return apply_mastoid_reference(raw, mastoid_channels=mastoid_channels, verbose=verbose)
    elif reference_type == 'laplacian':
        return apply_laplacian_reference(raw, verbose=verbose)
    elif reference_type == 'rest':
        return apply_rest_reference(raw, verbose=verbose)
    else:
        raise ValueError(
            f"Unknown reference type: {reference_type}. "
            f"Choose from: 'average', 'mastoid', 'laplacian', 'rest'"
        )
