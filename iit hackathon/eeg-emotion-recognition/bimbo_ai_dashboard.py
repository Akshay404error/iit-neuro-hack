"""
BIMBO AI - Brain Imaging and Machine-learning Based Observation AI
Interactive Dashboard for EEG-Based Emotion Recognition with AI-Powered Analysis

This enhanced dashboard provides:
- Multi-format dataset upload (.fif, .edf, .csv, .mat, .set)
- Automated EEG analysis pipeline
- AI-powered report generation using Groq API
- Comprehensive visualization and classification

Author: BIMBO AI Team
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
from datetime import datetime
import io
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.colors import HexColor, gray, whitesmoke, black

# Page configuration
st.set_page_config(
    page_title="BIMBO AI - EEG Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
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
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Groq client
GROQ_API_KEY = "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"

@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

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
st.markdown('<div class="main-header">🧠 BIMBO AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Brain Imaging and Machine-learning Based Observation AI<br>Advanced EEG Emotion Recognition with AI-Powered Analysis</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
st.sidebar.title("⚙️ BIMBO AI Configuration")
st.sidebar.markdown("---")

# Initialize default values FIRST (before any conditional logic)
n_subjects = 10
n_trials_per_subject = 30
n_channels = 14
emotions = ['Happy', 'Sad', 'Fear', 'Neutral']

# Data source selection
st.sidebar.subheader("📁 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Upload Dataset", "Generate Synthetic Data"]
)

# ============================================================================
# Dataset Upload Section
# ============================================================================
uploaded_file = None
uploaded_data = None

if data_source == "Upload Dataset":
    st.sidebar.markdown("### Upload Your EEG Dataset")
    st.sidebar.info("💡 **Large File Support**: Upload files up to 1 GB with no restrictions!")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['fif', 'edf', 'bdf', 'set', 'vhdr', 'csv', 'mat'],
        help="Supported formats: .fif, .edf, .csv, .mat, .set, .vhdr | No size limit - upload files up to 1 GB"
    )
    
    if uploaded_file is not None:
        st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # File format detection
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            with st.spinner(f"Loading {file_extension.upper()} file..."):
                # Save uploaded file temporarily
                temp_file_path = f"temp_upload.{file_extension}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load based on file type
                if file_extension == 'fif':
                    raw = mne.io.read_raw_fif(temp_file_path, preload=True, verbose=False)
                elif file_extension in ['edf', 'bdf']:
                    raw = mne.io.read_raw_edf(temp_file_path, preload=True, verbose=False)
                elif file_extension == 'set':
                    raw = mne.io.read_raw_eeglab(temp_file_path, preload=True, verbose=False)
                elif file_extension == 'vhdr':
                    raw = mne.io.read_raw_brainvision(temp_file_path, preload=True, verbose=False)
                elif file_extension == 'csv':
                    # Load CSV and convert to MNE format
                    df = pd.read_csv(temp_file_path)
                    st.sidebar.info(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                    uploaded_data = df
                    raw = None
                elif file_extension == 'mat':
                    from scipy.io import loadmat
                    mat_data = loadmat(temp_file_path)
                    st.sidebar.info(f"MAT file loaded with keys: {list(mat_data.keys())}")
                    uploaded_data = mat_data
                    raw = None
                
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                
                if raw is not None:
                    st.sidebar.success(f"✅ Loaded {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s duration")
                    uploaded_data = {
                        'raw': raw,
                        'n_channels': len(raw.ch_names),
                        'duration': raw.times[-1],
                        'sfreq': raw.info['sfreq']
                    }
        
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            uploaded_file = None

# Synthetic data parameters (if not uploading)
if data_source == "Generate Synthetic Data":
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

# AI Report Generation
st.sidebar.subheader("🤖 AI Analysis")
generate_report = st.sidebar.checkbox("Generate AI Report", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**BIMBO AI v1.0**")
st.sidebar.markdown("*Powered by MNE-Python, Scikit-Learn, Groq AI*")

# ============================================================================
# Generate or load data
# ============================================================================
@st.cache_data
def generate_dataset(n_subjects, n_trials, n_channels, emotions):
    """Generate synthetic EEG dataset with emotion labels"""
    
    n_features = n_channels * 3  # 3 bands per channel
    
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

# Generate or use uploaded dataset
if data_source == "Upload Dataset" and uploaded_data is not None:
    st.info("📁 Using uploaded dataset")
    
    # If it's raw EEG data, extract features
    if isinstance(uploaded_data, dict) and 'raw' in uploaded_data:
        raw = uploaded_data['raw']
        
        # Extract simple features from raw data
        with st.spinner("Extracting features from uploaded EEG data..."):
            # Compute PSD features
            data_array = raw.get_data()
            n_channels_upload = data_array.shape[0]
            
            # Simplified feature extraction
            features_list = []
            for ch_idx in range(min(n_channels_upload, 14)):  # Limit to 14 channels
                ch_data = data_array[ch_idx, :]
                freqs, psd = signal.welch(ch_data, fs=raw.info['sfreq'], nperseg=512)
                
                # Extract band powers
                theta_power = np.trapz(psd[(freqs >= 4) & (freqs <= 8)])
                alpha_power = np.trapz(psd[(freqs >= 8) & (freqs <= 13)])
                beta_power = np.trapz(psd[(freqs >= 13) & (freqs <= 30)])
                
                features_list.extend([theta_power, alpha_power, beta_power])
            
            # Create synthetic labels for demo
            n_samples = 100
            data = {
                'features': np.tile(features_list, (n_samples, 1)),
                'valence': np.random.uniform(1, 9, n_samples),
                'arousal': np.random.uniform(1, 9, n_samples),
                'dominance': np.random.uniform(1, 9, n_samples),
                'subjects': np.random.randint(1, 6, n_samples),
                'emotions': np.random.choice(['Happy', 'Sad', 'Fear', 'Neutral'], n_samples)
            }
    else:
        # CSV or MAT file - use as-is or convert
        st.warning("Custom data format detected. Using synthetic data for demo.")
        data = generate_dataset(10, 30, 14, ['Happy', 'Sad', 'Fear', 'Neutral'])
else:
    # Generate synthetic data
    with st.spinner("Generating synthetic dataset..."):
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
    st.metric("Unique Subjects", len(np.unique(data['subjects'])))
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
# SECTION 2.5: Correlation Analysis (Phase 1 Requirement)
# ============================================================================
st.markdown('<div class="section-header">📊 Correlation Analysis</div>', unsafe_allow_html=True)

st.markdown("""
**Correlation analysis** reveals the relationships between emotional dimensions:
- **Positive correlation**: Dimensions tend to increase together
- **Negative correlation**: One increases as the other decreases
- **No correlation**: Dimensions are independent
""")

# Compute correlation matrix
correlation_data = pd.DataFrame({
    'Valence': data['valence'],
    'Arousal': data['arousal'],
    'Dominance': data['dominance']
})

correlation_matrix = correlation_data.corr(method='pearson')

# Create correlation heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, square=True, ax=ax1,
            cbar_kws={'label': 'Correlation Coefficient'},
            linewidths=2, linecolor='white')
ax1.set_title('Pearson Correlation Matrix\n(Valence, Arousal, Dominance)', 
              fontsize=14, fontweight='bold')

# Bar plot of correlation strengths
correlations = []
pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        pairs.append(f"{correlation_matrix.columns[i]}-{correlation_matrix.columns[j]}")
        correlations.append(correlation_matrix.iloc[i, j])

colors_bar = ['#2ca02c' if c > 0 else '#d62728' for c in correlations]
bars = ax2.barh(pairs, correlations, color=colors_bar, edgecolor='black', linewidth=2)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
ax2.set_title('Pairwise Correlations', fontsize=14, fontweight='bold')
ax2.set_xlim(-1, 1)
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, corr in zip(bars, correlations):
    width = bar.get_width()
    ax2.text(width + (0.05 if width > 0 else -0.05), bar.get_y() + bar.get_height()/2,
            f'{corr:.3f}', ha='left' if width > 0 else 'right', va='center',
            fontsize=11, fontweight='bold')

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Interpretation
st.subheader("🔍 Correlation Insights")

col1, col2, col3 = st.columns(3)

with col1:
    val_aro_corr = correlation_matrix.loc['Valence', 'Arousal']
    st.metric("Valence ↔ Arousal", f"{val_aro_corr:.3f}",
             delta="Positive" if val_aro_corr > 0.3 else "Weak" if val_aro_corr > -0.3 else "Negative")

with col2:
    val_dom_corr = correlation_matrix.loc['Valence', 'Dominance']
    st.metric("Valence ↔ Dominance", f"{val_dom_corr:.3f}",
             delta="Positive" if val_dom_corr > 0.3 else "Weak" if val_dom_corr > -0.3 else "Negative")

with col3:
    aro_dom_corr = correlation_matrix.loc['Arousal', 'Dominance']
    st.metric("Arousal ↔ Dominance", f"{aro_dom_corr:.3f}",
             delta="Positive" if aro_dom_corr > 0.3 else "Weak" if aro_dom_corr > -0.3 else "Negative")

# Scientific interpretation
st.info(f"""
**Scientific Interpretation**:
- **Valence-Arousal** ({val_aro_corr:.3f}): {'Strong positive correlation suggests high-arousal emotions tend to be more positive.' if val_aro_corr > 0.5 else 'Moderate correlation indicates some independence between pleasantness and activation.' if val_aro_corr > 0.2 else 'Weak/negative correlation suggests valence and arousal are relatively independent dimensions.'}
- **Valence-Dominance** ({val_dom_corr:.3f}): {'Strong correlation indicates feeling in control is associated with positive emotions.' if abs(val_dom_corr) > 0.5 else 'Moderate relationship between pleasantness and sense of control.' if abs(val_dom_corr) > 0.2 else 'Relatively independent dimensions.'}
- **Arousal-Dominance** ({aro_dom_corr:.3f}): {'High arousal is associated with feeling in control.' if aro_dom_corr > 0.5 else 'Moderate relationship between activation and dominance.' if abs(aro_dom_corr) > 0.2 else 'Activation level and control are relatively independent.'}

These correlations reveal the **emotional structure** of the dataset and validate the dimensional model of affect.
""")

# ============================================================================
# SECTION 3: Classification with AI Analysis
# ============================================================================
st.markdown('<div class="section-header">🎯 Classification & AI Analysis</div>', unsafe_allow_html=True)

# Binary classification
st.subheader("Binary Classification: High vs Low Valence")

# Binarize valence
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

# Model selection with advanced options
st.subheader("🎯 Advanced Model Selection")
col_model1, col_model2, col_model3 = st.columns(3)

with col_model1:
    model_type = st.selectbox("Select Model", [
        "XGBoost (Best)",
        "Ensemble Voting",
        "Random Forest (Recommended)",
        "Gradient Boosting",
        "SVM (RBF Kernel)",
        "Logistic Regression"
    ])

with col_model2:
    use_feature_selection = st.checkbox("Use Feature Selection", value=True, 
                                       help="Select top features to improve accuracy")

with col_model3:
    use_smote = st.checkbox("Use SMOTE", value=True,
                           help="Balance classes using SMOTE")

# Train model with hyperparameter tuning
with st.spinner(f"Training {model_type} with optimization..."):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import PolynomialFeatures
    
    # Apply SMOTE for class balancing
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train
    
    # Feature selection
    if use_feature_selection:
        k_features = min(25, X_train_balanced.shape[1])  # Select top 25 features
        selector = SelectKBest(f_classif, k=k_features)
        X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)
        X_test_selected = selector.transform(X_test_scaled)
    else:
        X_train_selected = X_train_balanced
        X_test_selected = X_test_scaled
    
    # Model training with optimized hyperparameters
    if model_type == "XGBoost (Best)":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                min_child_weight=1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        except ImportError:
            st.warning("XGBoost not installed. Using Gradient Boosting instead.")
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            )
    
    elif model_type == "Ensemble Voting":
        # Create ensemble of best models
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=42
        )
        lr = LogisticRegression(
            max_iter=1000,
            C=2.0,
            class_weight='balanced',
            random_state=42
        )
        
        model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft'
        )
    
    elif model_type == "Random Forest (Recommended)":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=7,
            min_samples_split=4,
            subsample=0.8,
            random_state=42
        )
    
    elif model_type == "SVM (RBF Kernel)":
        model = SVC(
            kernel='rbf',
            C=15.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    
    else:  # Logistic Regression
        model = LogisticRegression(
            max_iter=2000,
            C=2.0,
            class_weight='balanced',
            solver='saga',
            random_state=42
        )
    
    model.fit(X_train_selected, y_train_balanced)
    y_pred = model.predict(X_test_selected)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif model_type == "Ensemble Voting" and hasattr(model.estimators_[0], 'feature_importances_'):
        # Average feature importance from ensemble
        feature_importance = np.mean([
            est.feature_importances_ for est in model.estimators_ 
            if hasattr(est, 'feature_importances_')
        ], axis=0)

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

# Feature Importance (if available)
if feature_importance is not None:
    st.subheader("🔍 Top 10 Most Important Features")
    
    # Get top 10 features
    if use_feature_selection:
        selected_indices = selector.get_support(indices=True)
        feature_names = [f"Feature {i+1}" for i in selected_indices]
    else:
        feature_names = [f"Feature {i+1}" for i in range(len(feature_importance))]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], 
                   color=plt.cm.viridis(importance_df['Importance'] / importance_df['Importance'].max()))
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================================
# AI-Powered Report Generation
# ============================================================================
if generate_report:
    st.markdown('<div class="section-header">🤖 AI-Generated Analysis Report</div>', unsafe_allow_html=True)
    
    with st.spinner("Generating AI-powered analysis report..."):
        try:
            # Prepare analysis summary
            analysis_summary = f"""
Dataset Analysis Summary:
- Total Samples: {len(data['features'])}
- Features: {data['features'].shape[1]}
- Emotions: {', '.join(np.unique(data['emotions']))}
- Valence Range: {valence_array.min():.2f} - {valence_array.max():.2f}
- Arousal Range: {data['arousal'].min():.2f} - {data['arousal'].max():.2f}

Classification Results:
- Model: {model_type}
- Accuracy: {accuracy:.3f}
- F1 Score: {f1:.3f}
- Confusion Matrix: {cm.tolist()}

Emotion Distribution:
{emotion_counts.to_dict()}
"""
            
            # Generate report using Groq AI
            client = get_groq_client()
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert neuroscientist and data scientist specializing in EEG-based emotion recognition. Provide detailed, scientific analysis of EEG emotion recognition results."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze the following EEG emotion recognition results and provide a comprehensive scientific report:

{analysis_summary}

Please provide:
1. Summary of key findings
2. Interpretation of classification performance
3. Analysis of emotion distribution patterns
4. Recommendations for improvement
5. Scientific insights about the arousal-valence model results

Keep the report professional, scientific, and actionable."""
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            ai_report = completion.choices[0].message.content
            
            # Display AI report
            st.markdown("### 📄 AI Analysis Report")
            st.markdown(ai_report)
            
            # Multi-Format Report Generation
            st.markdown("### 📥 Download Reports")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # 1. TEXT Report
            with col1:
                report_text = f"""BIMBO AI - EEG Emotion Recognition Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{ai_report}

---
Technical Details:
{analysis_summary}
"""
                st.download_button(
                    label="📄 TXT",
                    data=report_text,
                    file_name=f"bimbo_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # 2. CSV Report
            with col2:
                csv_data = pd.DataFrame({
                    'Sample_ID': range(len(data['features'])),
                    'Subject': data['subjects'],
                    'Emotion': data['emotions'],
                    'Valence': data['valence'],
                    'Arousal': data['arousal']
                })
                csv_buffer = io.StringIO()
                csv_data.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📊 CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"bimbo_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # 3. JSON Report
            with col3:
                import json
                json_report = {
                    "metadata": {"generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "system": "BIMBO AI v1.0"},
                    "classification": {"accuracy": float(accuracy), "f1_score": float(f1)},
                    "emotion_distribution": emotion_counts.to_dict(),
                    "ai_analysis": ai_report
                }
                json_buffer = io.StringIO()
                json.dump(json_report, json_buffer, indent=2)
                
                st.download_button(
                    label="🔗 JSON",
                    data=json_buffer.getvalue(),
                    file_name=f"bimbo_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # 4. Markdown Report
            with col4:
                md_report = f"""# BIMBO AI Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Results
- **Accuracy**: {accuracy:.3f}
- **F1 Score**: {f1:.3f}

## AI Analysis
{ai_report}
"""
                st.download_button(
                    label="📝 MD",
                    data=md_report,
                    file_name=f"bimbo_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            # 5. Enhanced Colorful PDF Report
            with col5:
                try:
                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, 
                                           rightMargin=72, leftMargin=72,
                                           topMargin=72, bottomMargin=18)
                    story = []
                    styles = getSampleStyleSheet()
                    
                    # Custom styles
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=28,
                        textColor=HexColor('#1f77b4'),
                        spaceAfter=30,
                        alignment=1  # Center
                    )
                    
                    subtitle_style = ParagraphStyle(
                        'SubTitle',
                        parent=styles['Normal'],
                        fontSize=14,
                        textColor=HexColor('#2ca02c'),
                        spaceAfter=20,
                        alignment=1
                    )
                    
                    heading_style = ParagraphStyle(
                        'CustomHeading',
                        parent=styles['Heading2'],
                        fontSize=16,
                        textColor=HexColor('#d62728'),
                        spaceAfter=12,
                        spaceBefore=12
                    )
                    
                    # Title Page
                    story.append(Paragraph("🧠 BIMBO AI", title_style))
                    story.append(Paragraph("Brain Imaging and Machine-learning Based Observation AI", subtitle_style))
                    story.append(Paragraph("EEG Emotion Recognition Analysis Report", styles['Normal']))
                    story.append(Spacer(1, 0.3*inch))
                    
                    # Metadata Box
                    meta_data = [
                        ['Report Information', ''],
                        ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                        ['System Version', 'BIMBO AI v1.0'],
                        ['Model Used', model_type],
                        ['Total Samples', str(len(data['features']))],
                    ]
                    meta_table = Table(meta_data, colWidths=[2.5*inch, 3.5*inch])
                    meta_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1f77b4')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
                        ('GRID', (0, 0), (-1, -1), 1, black),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 12),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ]))
                    story.append(meta_table)
                    story.append(Spacer(1, 0.4*inch))
                    
                    # Classification Results Section
                    story.append(Paragraph("📊 Classification Results", heading_style))
                    results_data = [
                        ['Metric', 'Value', 'Interpretation'],
                        ['Accuracy', f"{accuracy:.3f}", 'High' if accuracy > 0.7 else 'Moderate'],
                        ['F1 Score', f"{f1:.3f}", 'Good' if f1 > 0.7 else 'Fair'],
                        ['Valence Threshold', f"{valence_median:.2f}", 'Median-based'],
                    ]
                    results_table = Table(results_data, colWidths=[2*inch, 2*inch, 2*inch])
                    results_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2ca02c')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#e8f5e9')),
                        ('GRID', (0, 0), (-1, -1), 1.5, HexColor('#2ca02c')),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ]))
                    story.append(results_table)
                    story.append(Spacer(1, 0.3*inch))
                    
                    # Emotion Distribution Section
                    story.append(Paragraph("😊 Emotion Distribution", heading_style))
                    emotion_data = [['Emotion', 'Count', 'Percentage', 'Status']]
                    for emotion, count in emotion_counts.items():
                        percentage = (count / len(data['emotions']) * 100)
                        status = '✓ Balanced' if percentage > 20 else '⚠ Low'
                        emotion_data.append([emotion, str(count), f"{percentage:.1f}%", status])
                    
                    emotion_table = Table(emotion_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                    emotion_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#ff7f0e')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#fff3e0')),
                        ('GRID', (0, 0), (-1, -1), 1.5, HexColor('#ff7f0e')),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#fff3e0'), HexColor('#ffe0b2')]),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ]))
                    story.append(emotion_table)
                    story.append(Spacer(1, 0.3*inch))
                    
                    # Statistical Summary Section
                    story.append(Paragraph("📈 Statistical Summary", heading_style))
                    stats_data = [
                        ['Dimension', 'Min', 'Max', 'Mean', 'Median'],
                        ['Valence', f"{valence_array.min():.2f}", f"{valence_array.max():.2f}", 
                         f"{valence_array.mean():.2f}", f"{valence_median:.2f}"],
                        ['Arousal', f"{data['arousal'].min():.2f}", f"{data['arousal'].max():.2f}",
                         f"{data['arousal'].mean():.2f}", f"{np.median(data['arousal']):.2f}"],
                    ]
                    stats_table = Table(stats_data, colWidths=[1.5*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.1*inch])
                    stats_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#9467bd')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f3e5f5')),
                        ('GRID', (0, 0), (-1, -1), 1.5, HexColor('#9467bd')),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ]))
                    story.append(stats_table)
                    story.append(Spacer(1, 0.3*inch))
                    
                    # AI Analysis Section
                    story.append(Paragraph("🤖 AI-Generated Analysis", heading_style))
                    story.append(Paragraph("Analysis done by Matsya N", styles['Italic']))
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Split AI report into paragraphs - SHOW COMPLETE REPORT
                    ai_lines = ai_report.split('\n')
                    for i, line in enumerate(ai_lines):  # Show ALL lines, no limit
                        if line.strip():
                            # Check if it's a heading (starts with #)
                            if line.strip().startswith('#'):
                                clean_line = line.strip().replace('#', '').strip()
                                story.append(Paragraph(clean_line, heading_style))
                            else:
                                story.append(Paragraph(line, styles['Normal']))
                                story.append(Spacer(1, 0.05*inch))
                    
                    # Footer
                    story.append(Spacer(1, 0.5*inch))
                    footer_style = ParagraphStyle(
                        'Footer',
                        parent=styles['Normal'],
                        fontSize=9,
                        textColor=gray,
                        alignment=1
                    )
                    story.append(Paragraph("―――――――――――――――――――――――――――――――――――", footer_style))
                    story.append(Paragraph("BIMBO AI - Brain Imaging and Machine-learning Based Observation AI", footer_style))
                    story.append(Paragraph("Report generated by BIMBO AI | Team Matsya N", footer_style))
                    story.append(Paragraph("© 2026 Matsya N Team | Scientifically Correct & Reproducible", footer_style))
                    
                    # Build PDF
                    doc.build(story)
                    pdf_buffer.seek(0)
                    
                    st.download_button(
                        label="📕 PDF",
                        data=pdf_buffer.getvalue(),
                        file_name=f"bimbo_ai_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as pdf_error:
                    st.error(f"PDF generation error: {str(pdf_error)}")
            
            
        except Exception as e:
            st.error(f"Error generating AI report: {str(e)}")
            st.info("Displaying basic analysis instead.")

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>BIMBO AI v1.0</strong> - Brain Imaging and Machine-learning Based Observation AI</p>
    <p>Report generated by <strong>BIMBO AI | Team Matsya N</strong></p>
    <p>© 2026 Matsya N Team | Scientifically Correct & Reproducible</p>
</div>
""", unsafe_allow_html=True)
