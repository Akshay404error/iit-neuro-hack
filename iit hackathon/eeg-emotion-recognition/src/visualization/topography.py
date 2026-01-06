"""
Topographic Visualization

This module provides topographic mapping functions for EEG data.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List


def plot_topomap(
    data: np.ndarray,
    info: mne.Info,
    title: str = "Topographic Map",
    cmap: str = 'RdBu_r',
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot topographic map.
    
    Parameters
    ----------
    data : np.ndarray
        Data values for each channel (n_channels,)
    info : mne.Info
        MNE info object with channel positions
    title : str
        Plot title
    cmap : str
        Colormap
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im, _ = mne.viz.plot_topomap(
        data,
        info,
        axes=ax,
        show=False,
        cmap=cmap,
        contours=6
    )
    
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    
    return fig


def plot_band_topomaps(
    epochs: mne.Epochs,
    bands: Optional[Dict[str, tuple]] = None,
    figsize: tuple = (15, 10)
) -> plt.Figure:
    """
    Plot topographic maps for each frequency band.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    bands : dict, optional
        Frequency bands
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if bands is None:
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
    
    n_bands = len(bands)
    fig, axes = plt.subplots(1, n_bands, figsize=figsize)
    
    if n_bands == 1:
        axes = [axes]
    
    for ax, (band_name, (fmin, fmax)) in zip(axes, bands.items()):
        # Compute PSD
        psds, freqs = mne.time_frequency.psd_welch(
            epochs, fmin=fmin, fmax=fmax, verbose=False
        )
        
        # Average across epochs and frequencies
        band_power = psds.mean(axis=(0, 2))
        
        # Plot topomap
        im, _ = mne.viz.plot_topomap(
            band_power,
            epochs.info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            contours=6
        )
        
        ax.set_title(f'{band_name}\n({fmin}-{fmax} Hz)')
    
    plt.tight_layout()
    return fig


def plot_asymmetry_topomap(
    epochs: mne.Epochs,
    band: tuple = (8, 13),
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot hemispheric asymmetry topographic map.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    band : tuple
        Frequency band (fmin, fmax)
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fmin, fmax = band
    
    # Compute PSD
    psds, freqs = mne.time_frequency.psd_welch(
        epochs, fmin=fmin, fmax=fmax, verbose=False
    )
    
    # Average across epochs and frequencies
    band_power = psds.mean(axis=(0, 2))
    
    # Compute log power
    log_power = np.log(band_power + 1e-10)
    
    # Create asymmetry map (this is simplified - proper asymmetry requires paired channels)
    fig, ax = plt.subplots(figsize=figsize)
    
    im, _ = mne.viz.plot_topomap(
        log_power,
        epochs.info,
        axes=ax,
        show=False,
        cmap='RdBu_r',
        contours=6
    )
    
    ax.set_title(f'Alpha Asymmetry ({fmin}-{fmax} Hz)')
    plt.colorbar(im, ax=ax, label='Log Power')
    
    return fig


def plot_evoked_topomap(
    evoked: mne.Evoked,
    times: List[float],
    figsize: tuple = (12, 4)
) -> plt.Figure:
    """
    Plot topographic maps at specific time points.
    
    Parameters
    ----------
    evoked : mne.Evoked
        Evoked response
    times : list of float
        Time points to plot (seconds)
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig = evoked.plot_topomap(
        times=times,
        ch_type='eeg',
        show=False,
        colorbar=True,
        size=3
    )
    
    return fig


def plot_sensor_positions(
    info: mne.Info,
    figsize: tuple = (8, 8)
) -> plt.Figure:
    """
    Plot sensor positions.
    
    Parameters
    ----------
    info : mne.Info
        MNE info object
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    mne.viz.plot_sensors(
        info,
        kind='topomap',
        show_names=True,
        show=False
    )
    
    plt.title('Sensor Positions')
    
    return fig
