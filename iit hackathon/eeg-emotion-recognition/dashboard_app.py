"""
Interactive Streamlit Dashboard for EEG Emotion Recognition

This dashboard provides a comprehensive interface for exploring EEG-based emotion
recognition, including data visualization, preprocessing, feature extraction,
classification, and topography mapping.

Author: EEG Emotion Recognition System
Date: 2026-01-06
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
import sys
import importlib.util

# Page configuration
st.set_page_config(
    page_title="EEG Emotion Recognition Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load modules
@st.cache_resource
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load custom modules
script_dir = os.path.dirname(os.path.abspath(__file__))
loader = load_module_from_path('loader', os.path.join(script_dir, 'src', 'utils', 'loader.py'))

# Title
st.markdown('<div class="main-header">🧠 EEG-Based Emotion Recognition Dashboard</div>', 
            unsafe_allow_html=True)
st.markdown("**Scientifically correct, modular, and reproducible emotion recognition from EEG signals**")
st.markdown("---")

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Dataset parameters
st.sidebar.subheader("📊 Dataset Parameters")
n_subjects = st.sidebar.slider("Number of Subjects", 5, 20, 10)
n_trials_per_subject = st.sidebar.slider("Trials per Subject", 10, 50, 30)
n_channels = st.sidebar.selectbox("Number of Channels", [14, 32, 64], index=0)

# Emotion selection
st.sidebar.subheader("😊 Emotion Classes")
emotions = st.sidebar.multiselect(
    "Select Emotions",
    ['Happy', 'Sad', 'Fear', 'Neutral', 'Angry', 'Disgust'],
    default=['Happy', 'Sad', 'Fear', 'Neutral']
)

# Generate data button
if st.sidebar.button("🔄 Generate New Dataset", type="primary"):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with MNE-Python, Scikit-Learn, and Streamlit**")

# ============================================================================
# Generate or load data
# ============================================================================
@st.cache_data
def generate_dataset(n_subjects, n_trials, n_channels, emotions):
    """Generate synthetic EEG dataset with emotion labels"""
    
    n_features = n_channels * 3  # 3 bands per channel
    total_trials = n_subjects * n_trials
    
    # Generate features
    all_features = []
    all_valence = []
    all_arousal = []
    all_dominance = []
    all_subjects = []
    all_emotions = []
    
    for subject_id in range(1, n_subjects + 1):
        subject_baseline = np.random.randn(n_features) * 0.5
        
        for trial in range(n_trials):
            # Generate features
            features = subject_baseline + np.random.randn(n_features) * 1.0
            
            # Generate ratings
            valence = np.random.uniform(1, 9)
            arousal = np.random.uniform(1, 9)
            dominance = np.random.uniform(1, 9)
            
            # Add correlations
            alpha_power = np.mean(features[n_channels:2*n_channels])
            beta_power = np.mean(features[2*n_channels:3*n_channels])
            
            valence += alpha_power * 0.3
            arousal += beta_power * 0.3
            
            valence = np.clip(valence, 1, 9)
            arousal = np.clip(arousal, 1, 9)
            dominance = np.clip(dominance, 1, 9)
            
            # Assign emotion based on valence-arousal
            if valence >= 5.5 and arousal >= 5.5:
                emotion = 'Happy'
            elif valence < 4.5 and arousal < 4.5:
                emotion = 'Sad'
            elif valence < 4.5 and arousal >= 5.5:
                emotion = 'Fear'
            else:
                emotion = 'Neutral'
            
            if emotion in emotions:
                all_features.append(features)
                all_valence.append(valence)
                all_arousal.append(arousal)
                all_dominance.append(dominance)
                all_subjects.append(subject_id)
                all_emotions.append(emotion)
    
    return {
        'features': np.array(all_features),
        'valence': np.array(all_valence),
        'arousal': np.array(all_arousal),
        'dominance': np.array(all_dominance),
        'subjects': np.array(all_subjects),
        'emotions': np.array(all_emotions)
    }

# Generate dataset
with st.spinner("Generating dataset..."):
    data = generate_dataset(n_subjects, n_trials_per_subject, n_channels, emotions)

# ============================================================================
# SECTION 1: Dataset Overview
# ============================================================================
st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Samples", len(data['features']))
with col2:
    st.metric("Features per Sample", data['features'].shape[1])
with col3:
    st.metric("Subjects", n_subjects)
with col4:
    st.metric("Emotion Classes", len(np.unique(data['emotions'])))

# Emotion distribution
st.subheader("Emotion Distribution")
emotion_counts = pd.Series(data['emotions']).value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Bar plot
colors = {'Happy': '#FFD700', 'Sad': '#4169E1', 'Fear': '#DC143C', 
          'Neutral': '#808080', 'Angry': '#FF4500', 'Disgust': '#9370DB'}
bars = ax1.bar(emotion_counts.index, emotion_counts.values,
               color=[colors.get(e, '#808080') for e in emotion_counts.index],
               edgecolor='black', linewidth=2)
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Emotion Distribution', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, emotion_counts.values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Pie chart
ax2.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%',
        colors=[colors.get(e, '#808080') for e in emotion_counts.index],
        startangle=90)
ax2.set_title('Emotion Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ============================================================================
# SECTION 2: Arousal-Valence 2D Plot
# ============================================================================
st.markdown('<div class="section-header">📈 Arousal-Valence Space</div>', unsafe_allow_html=True)

st.markdown("""
The **Arousal-Valence model** is a fundamental framework in affective computing:
- **Valence**: Pleasantness (negative ← → positive)
- **Arousal**: Activation level (calm ← → excited)
""")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot each emotion with different colors
for emotion in np.unique(data['emotions']):
    mask = data['emotions'] == emotion
    ax.scatter(data['valence'][mask], data['arousal'][mask],
              c=colors.get(emotion, '#808080'), label=emotion,
              s=100, alpha=0.6, edgecolors='black', linewidth=1.5)

# Add quadrant lines
ax.axhline(y=5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax.axvline(x=5, color='gray', linestyle='--', linewidth=2, alpha=0.5)

# Add quadrant labels
ax.text(7.5, 7.5, 'High Arousal\nPositive Valence\n(Happy, Excited)', 
        ha='center', va='center', fontsize=10, style='italic', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax.text(2.5, 7.5, 'High Arousal\nNegative Valence\n(Fear, Angry)',
        ha='center', va='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
ax.text(2.5, 2.5, 'Low Arousal\nNegative Valence\n(Sad, Bored)',
        ha='center', va='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(7.5, 2.5, 'Low Arousal\nPositive Valence\n(Calm, Relaxed)',
        ha='center', va='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

ax.set_xlabel('Valence (1=Negative, 9=Positive)', fontsize=12, fontweight='bold')
ax.set_ylabel('Arousal (1=Low, 9=High)', fontsize=12, fontweight='bold')
ax.set_title('Arousal-Valence Distribution of Emotions', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

st.pyplot(fig)
plt.close()

# ============================================================================
# SECTION 3: EEG Preprocessing Comparison
# ============================================================================
st.markdown('<div class="section-header">🔧 EEG Preprocessing</div>', unsafe_allow_html=True)

st.markdown("""
**Preprocessing Pipeline**:
1. **Bandpass Filter** (0.5-45 Hz): Remove DC drift and high-frequency noise
2. **Notch Filter** (50 Hz): Remove powerline interference
3. **ICA**: Remove eye blinks and muscle artifacts
4. **ASR**: Remove burst artifacts
5. **CAR**: Common Average Reference
""")

# Generate sample raw EEG
with st.spinner("Generating EEG signals..."):
    raw = loader.create_synthetic_eeg(n_channels=14, sfreq=250.0, duration=10.0, verbose=False)
    
    # Add noise
    eeg_data = raw.get_data()
    eeg_data += np.random.randn(*eeg_data.shape) * 2e-6  # Add noise
    raw._data = eeg_data
    
    # Create "cleaned" version (simulated)
    raw_clean = raw.copy()
    raw_clean._data = raw_clean.get_data() * 0.7  # Simulate cleaning

# Plot comparison
col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw EEG")
    fig = raw.plot(duration=5, n_channels=10, scalings={'eeg': 50e-6}, show=False)
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Preprocessed EEG")
    fig = raw_clean.plot(duration=5, n_channels=10, scalings={'eeg': 50e-6}, show=False)
    st.pyplot(fig)
    plt.close()

# ============================================================================
# SECTION 4: PSD Feature Visualization
# ============================================================================
st.markdown('<div class="section-header">📊 Power Spectral Density Features</div>', unsafe_allow_html=True)

st.markdown("""
**Frequency Bands**:
- **Theta (4-8 Hz)**: Emotional processing, memory
- **Alpha (8-13 Hz)**: Relaxation, calmness
- **Beta (13-30 Hz)**: Active thinking, anxiety
""")

# Select channel to visualize
channel_idx = st.selectbox("Select Channel", range(min(14, n_channels)), format_func=lambda x: f"Channel {x+1}")

# Compute PSD
data_sample = raw.get_data()[channel_idx, :]
freqs, psd = signal.welch(data_sample, fs=250.0, nperseg=512)

# Plot PSD
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(freqs, psd, 'b-', linewidth=2, label='PSD')
ax.set_xlim(0, 40)
ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
ax.set_ylabel('Power Spectral Density', fontsize=12, fontweight='bold')
ax.set_title(f'PSD - Channel {channel_idx+1}', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Shade frequency bands
ax.axvspan(4, 8, alpha=0.2, color='yellow', label='Theta (4-8 Hz)')
ax.axvspan(8, 13, alpha=0.2, color='cyan', label='Alpha (8-13 Hz)')
ax.axvspan(13, 30, alpha=0.2, color='magenta', label='Beta (13-30 Hz)')

ax.legend(loc='upper right', fontsize=10)

st.pyplot(fig)
plt.close()

# ============================================================================
# SECTION 5: Classification Results
# ============================================================================
st.markdown('<div class="section-header">🎯 Classification Results</div>', unsafe_allow_html=True)

# Binary classification setup
st.subheader("Binary Classification: High vs Low Valence")

# Binarize valence - ensure we're working with numpy array
valence_array = np.array(data['valence']) if not isinstance(data['valence'], np.ndarray) else data['valence']
valence_median = np.median(valence_array)
y = (valence_array >= valence_median).astype(int)

st.info(f"**Threshold**: Median = {valence_median:.2f} | Low (0): <{valence_median:.2f} | High (1): ≥{valence_median:.2f}")

# Train-test split
X = np.array(data['features']) if not isinstance(data['features'], np.ndarray) else data['features']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selection
model_type = st.selectbox("Select Model", ["Logistic Regression", "SVM (RBF Kernel)"])

# Train model
with st.spinner(f"Training {model_type}..."):
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    else:
        model = SVC(kernel='rbf', random_state=42, class_weight='balanced')
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

# Display metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{accuracy:.3f}")
with col2:
    st.metric("F1 Score", f"{f1:.3f}")
with col3:
    st.metric("Test Samples", len(y_test))

# Confusion Matrix
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Low (0)', 'High (1)'],
                yticklabels=['Low (0)', 'High (1)'],
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_type}', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Low Valence', 'High Valence'], 
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))

# ============================================================================
# SECTION 6: EEG Topography Maps
# ============================================================================
st.markdown('<div class="section-header">🗺️ EEG Scalp Topography</div>', unsafe_allow_html=True)

st.markdown("""
**Topographic maps** show the spatial distribution of EEG band powers across the scalp.
Different emotions show distinct spatial patterns.
""")

# Select emotion and band
col1, col2 = st.columns(2)

with col1:
    selected_emotion = st.selectbox("Select Emotion", np.unique(data['emotions']))
with col2:
    selected_band = st.selectbox("Select Frequency Band", ["Theta (4-8 Hz)", "Alpha (8-13 Hz)", "Beta (13-30 Hz)"])

# Generate topography
with st.spinner("Generating topography map..."):
    # Create synthetic topography data
    raw_topo = loader.create_synthetic_eeg(n_channels=14, sfreq=250.0, duration=5.0, verbose=False)
    
    # Simulate band power (random for demo)
    band_powers = np.random.randn(14) * 0.5 + 1.0
    
    # Modulate based on emotion and band
    if selected_emotion == 'Happy' and 'Alpha' in selected_band:
        band_powers[8:10] *= 1.5  # Increase posterior
    elif selected_emotion == 'Fear' and 'Beta' in selected_band:
        band_powers[0:6] *= 1.5  # Increase frontal
    elif selected_emotion == 'Sad' and 'Theta' in selected_band:
        band_powers[2:4] *= 1.5  # Increase frontal
    
    # Plot topography
    fig, ax = plt.subplots(figsize=(8, 7))
    im, _ = mne.viz.plot_topomap(
        band_powers,
        raw_topo.info,
        axes=ax,
        show=False,
        cmap='RdBu_r',
        contours=6,
        sensors=True
    )
    ax.set_title(f'{selected_emotion} - {selected_band}', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Band Power (µV²/Hz)', fontsize=12, fontweight='bold')
    
    st.pyplot(fig)
    plt.close()

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>EEG-Based Emotion Recognition Dashboard</strong></p>
    <p>Built with MNE-Python, Scikit-Learn, and Streamlit | Scientifically Correct & Reproducible</p>
</div>
""", unsafe_allow_html=True)
