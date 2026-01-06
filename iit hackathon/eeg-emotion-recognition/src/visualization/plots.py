"""
EEG Visualization - Time-series and Spectral Plots

This module provides visualization functions for EEG data analysis.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


def plot_raw_eeg(
    raw: mne.io.Raw,
    duration: float = 10.0,
    n_channels: int = 10,
    scalings: Optional[dict] = None,
    title: str = "Raw EEG Signal"
) -> plt.Figure:
    """
    Plot raw EEG time-series.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    duration : float, default=10.0
        Duration to plot (seconds)
    n_channels : int, default=10
        Number of channels to plot
    scalings : dict, optional
        Scaling factors for different channel types
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if scalings is None:
        scalings = {'eeg': 20e-6}  # 20 µV
    
    fig = raw.plot(
        duration=duration,
        n_channels=n_channels,
        scalings=scalings,
        title=title,
        show=False
    )
    
    return fig


def plot_psd(
    raw: mne.io.Raw,
    fmin: float = 0.5,
    fmax: float = 50.0,
    picks: Optional[List[str]] = None,
    title: str = "Power Spectral Density"
) -> plt.Figure:
    """
    Plot Power Spectral Density.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    fmin : float, default=0.5
        Minimum frequency
    fmax : float, default=50.0
        Maximum frequency
    picks : list of str, optional
        Channels to plot
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig = raw.plot_psd(
        fmin=fmin,
        fmax=fmax,
        picks=picks,
        average=False,
        spatial_colors=True,
        show=False
    )
    fig.suptitle(title)
    
    return fig


def plot_epochs(
    epochs: mne.Epochs,
    picks: Optional[List[str]] = None,
    scalings: Optional[dict] = None,
    title: str = "Epochs"
) -> plt.Figure:
    """
    Plot epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    picks : list of str, optional
        Channels to plot
    scalings : dict, optional
        Scaling factors
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if scalings is None:
        scalings = {'eeg': 50e-6}
    
    fig = epochs.plot(
        picks=picks,
        scalings=scalings,
        title=title,
        show=False
    )
    
    return fig


def plot_band_powers(
    band_powers: dict,
    ch_names: List[str],
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot band powers across channels.
    
    Parameters
    ----------
    band_powers : dict
        Dictionary of band powers {band_name: powers_array}
    ch_names : list of str
        Channel names
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(ch_names))
    width = 0.15
    
    for i, (band_name, powers) in enumerate(band_powers.items()):
        offset = (i - len(band_powers) / 2) * width
        ax.bar(x + offset, powers, width, label=band_name)
    
    ax.set_xlabel('Channel')
    ax.set_ylabel('Power (µV²)')
    ax.set_title('Band Powers Across Channels')
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : list of str
        Class names
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance_df,
    top_n: int = 20,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int, default=20
        Number of top features to plot
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    top_features = importance_df.head(top_n)
    
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_spectrogram(
    raw: mne.io.Raw,
    channel: str,
    fmin: float = 0.5,
    fmax: float = 50.0,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot spectrogram for a single channel.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    channel : str
        Channel name
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Get data for the channel
    ch_idx = raw.ch_names.index(channel)
    data = raw.get_data(picks=[ch_idx])[0]
    sfreq = raw.info['sfreq']
    
    # Compute spectrogram
    from scipy import signal as sp_signal
    f, t, Sxx = sp_signal.spectrogram(data, sfreq, nperseg=int(sfreq))
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Limit frequency range
    freq_mask = (f >= fmin) & (f <= fmax)
    
    im = ax.pcolormesh(
        t, f[freq_mask], 10 * np.log10(Sxx[freq_mask, :]),
        shading='gouraud', cmap='viridis'
    )
    
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Spectrogram - {channel}')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)')
    
    plt.tight_layout()
    return fig
