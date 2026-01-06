"""
EEG-Based Emotion Recognition - Streamlit Application

Interactive web interface for the complete EEG emotion recognition pipeline.
"""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.loader import create_synthetic_emotion_dataset, get_channel_pairs_for_asymmetry
from preprocessing.filtering import preprocess_with_filters
from preprocessing.ica import run_ica_pipeline
from preprocessing.asr import apply_asr
from preprocessing.referencing import choose_reference
from features.segmentation import create_fixed_length_epochs, reject_bad_epochs
from features.psd_features import extract_all_psd_features
from models.classifier import EmotionClassifier
from visualization.plots import plot_raw_eeg, plot_psd, plot_confusion_matrix, plot_feature_importance
from visualization.topography import plot_band_topomaps

# Page config
st.set_page_config(
    page_title="EEG Emotion Recognition",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 EEG-Based Emotion Recognition System")
st.markdown("**Scientifically correct, modular, and reproducible emotion recognition from EEG signals**")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Initialize session state
if 'raw' not in st.session_state:
    st.session_state.raw = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'epochs' not in st.session_state:
    st.session_state.epochs = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Loading",
    "🔧 Preprocessing",
    "📈 Feature Extraction",
    "🤖 Model Training",
    "📋 Results"
])

# ============================================================================
# TAB 1: Data Loading
# ============================================================================
with tab1:
    st.header("Data Loading")
    
    st.markdown("""
    ### Scientific Background
    
    This system analyzes EEG signals to recognize emotional states. The preprocessing pipeline follows
    neuroscience best practices:
    
    - **Filtering**: Remove DC drift and high-frequency noise
    - **ICA**: Remove ocular and muscle artifacts
    - **ASR**: Remove burst artifacts
    - **Re-referencing**: Common Average Reference for spatial filtering
    
    **Frequency Bands**:
    - Delta (0.5-4 Hz): Deep sleep, unconscious processes
    - Theta (4-8 Hz): Drowsiness, meditation, memory
    - Alpha (8-13 Hz): Relaxation, calmness
    - Beta (13-30 Hz): Active thinking, focus
    - Gamma (30-50 Hz): High-level cognition
    """)
    
    st.subheader("Generate Demo Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_trials = st.number_input("Trials per emotion", min_value=10, max_value=100, value=30)
        trial_duration = st.number_input("Trial duration (s)", min_value=2.0, max_value=10.0, value=4.0)
    
    with col2:
        n_channels = st.number_input("Number of channels", min_value=8, max_value=64, value=32)
        sfreq = st.number_input("Sampling rate (Hz)", min_value=100.0, max_value=500.0, value=250.0)
    
    emotions = st.multiselect(
        "Emotion labels",
        ['happy', 'sad', 'neutral', 'angry', 'fear'],
        default=['happy', 'sad', 'neutral']
    )
    
    if st.button("🎲 Generate Synthetic Data"):
        with st.spinner("Generating synthetic EEG data..."):
            raw, labels = create_synthetic_emotion_dataset(
                n_trials_per_emotion=n_trials,
                trial_duration=trial_duration,
                emotions=emotions,
                n_channels=n_channels,
                sfreq=sfreq,
                verbose=False
            )
            
            st.session_state.raw = raw
            st.session_state.labels = labels
            
            st.success(f"✅ Generated {len(labels)} trials ({raw.times[-1]:.1f} seconds)")
    
    if st.session_state.raw is not None:
        st.subheader("Data Overview")
        
        raw = st.session_state.raw
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Channels", len(raw.ch_names))
        col2.metric("Sampling Rate", f"{raw.info['sfreq']} Hz")
        col3.metric("Duration", f"{raw.times[-1]:.1f} s")
        
        # Plot raw EEG
        st.subheader("Raw EEG Signal")
        fig = plot_raw_eeg(raw, duration=10.0, n_channels=10)
        st.pyplot(fig)
        plt.close()

# ============================================================================
# TAB 2: Preprocessing
# ============================================================================
with tab2:
    st.header("Preprocessing")
    
    if st.session_state.raw is None:
        st.warning("⚠️ Please load data first (Tab 1)")
    else:
        st.subheader("Filtering")
        
        col1, col2 = st.columns(2)
        with col1:
            l_freq = st.slider("High-pass cutoff (Hz)", 0.1, 2.0, 0.5, 0.1)
            h_freq = st.slider("Low-pass cutoff (Hz)", 30.0, 100.0, 50.0, 5.0)
        
        with col2:
            apply_notch = st.checkbox("Apply notch filter", value=True)
            notch_freq = st.selectbox("Notch frequency (Hz)", [50, 60], index=0)
        
        if st.button("🔧 Apply Filtering"):
            with st.spinner("Applying filters..."):
                raw_filtered = st.session_state.raw.copy()
                raw_filtered = preprocess_with_filters(
                    raw_filtered,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    notch_freq=notch_freq,
                    apply_notch=apply_notch,
                    verbose=False
                )
                st.session_state.raw = raw_filtered
                st.success("✅ Filtering complete")
        
        # PSD plot
        if st.checkbox("Show Power Spectral Density"):
            fig = plot_psd(st.session_state.raw, fmin=0.5, fmax=50.0)
            st.pyplot(fig)
            plt.close()
        
        st.subheader("Re-referencing")
        
        reference_type = st.selectbox(
            "Reference type",
            ['average', 'mastoid', 'laplacian'],
            index=0
        )
        
        if st.button("🔧 Apply Re-referencing"):
            with st.spinner("Applying re-referencing..."):
                raw_ref = st.session_state.raw.copy()
                raw_ref = choose_reference(raw_ref, reference_type=reference_type, verbose=False)
                st.session_state.raw = raw_ref
                st.success(f"✅ {reference_type.capitalize()} reference applied")

# ============================================================================
# TAB 3: Feature Extraction
# ============================================================================
with tab3:
    st.header("Feature Extraction")
    
    if st.session_state.raw is None:
        st.warning("⚠️ Please load data first (Tab 1)")
    else:
        st.subheader("Segmentation")
        
        col1, col2 = st.columns(2)
        with col1:
            epoch_duration = st.slider("Epoch duration (s)", 1.0, 5.0, 2.0, 0.5)
            overlap = st.slider("Overlap (s)", 0.0, 2.0, 0.0, 0.5)
        
        with col2:
            reject_threshold = st.slider("Rejection threshold (µV)", 50, 200, 100, 10)
        
        if st.button("📊 Create Epochs"):
            with st.spinner("Creating epochs..."):
                epochs = create_fixed_length_epochs(
                    st.session_state.raw,
                    duration=epoch_duration,
                    overlap=overlap,
                    verbose=False
                )
                
                # Reject bad epochs
                epochs = reject_bad_epochs(
                    epochs,
                    reject={'eeg': reject_threshold * 1e-6},
                    verbose=False
                )
                
                st.session_state.epochs = epochs
                st.success(f"✅ Created {len(epochs)} epochs")
        
        if st.session_state.epochs is not None:
            st.subheader("PSD Feature Extraction")
            
            include_ratios = st.checkbox("Include band power ratios", value=True)
            include_asymmetry = st.checkbox("Include asymmetry features", value=True)
            
            if st.button("📈 Extract Features"):
                with st.spinner("Extracting PSD features..."):
                    epochs = st.session_state.epochs
                    
                    # Get channel pairs for asymmetry
                    channel_pairs = None
                    if include_asymmetry:
                        channel_pairs = get_channel_pairs_for_asymmetry(
                            epochs.ch_names, verbose=False
                        )
                    
                    # Extract features
                    features = extract_all_psd_features(
                        epochs,
                        channel_pairs=channel_pairs,
                        include_ratios=include_ratios,
                        include_asymmetry=include_asymmetry,
                        verbose=False
                    )
                    
                    st.session_state.features = features
                    st.success(f"✅ Extracted {features.shape[1]} features")
                    
                    # Show feature preview
                    st.dataframe(features.head())
            
            # Topographic maps
            if st.checkbox("Show Band Power Topomaps"):
                with st.spinner("Generating topomaps..."):
                    fig = plot_band_topomaps(st.session_state.epochs, figsize=(15, 3))
                    st.pyplot(fig)
                    plt.close()

# ============================================================================
# TAB 4: Model Training
# ============================================================================
with tab4:
    st.header("Model Training")
    
    if st.session_state.features is None:
        st.warning("⚠️ Please extract features first (Tab 3)")
    else:
        st.subheader("Classifier Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox("Model type", ['svm', 'random_forest'], index=0)
            test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        
        with col2:
            apply_smote = st.checkbox("Apply SMOTE (class balancing)", value=True)
            cv_folds = st.slider("Cross-validation folds", 3, 10, 5, 1)
        
        if st.button("🤖 Train Model"):
            with st.spinner("Training model..."):
                from sklearn.model_selection import train_test_split
                
                # Prepare data
                X = st.session_state.features
                y = np.array(st.session_state.labels[:len(X)])
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Train classifier
                clf = EmotionClassifier(model_type=model_type, verbose=False)
                clf.fit(X_train, y_train, apply_smote=apply_smote)
                
                # Evaluate
                results = clf.evaluate(X_test, y_test)
                
                # Cross-validation
                cv_results = clf.cross_validate(X, y, cv=cv_folds)
                
                # Store in session
                st.session_state.model = clf
                st.session_state.test_data = (X_test, y_test)
                st.session_state.results = results
                st.session_state.cv_results = cv_results
                
                st.success("✅ Model trained successfully!")
        
        if st.session_state.model is not None:
            st.subheader("Training Results")
            
            results = st.session_state.results
            cv_results = st.session_state.cv_results
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{results['accuracy']:.3f}")
            col2.metric("Precision", f"{results['precision']:.3f}")
            col3.metric("Recall", f"{results['recall']:.3f}")
            col4.metric("F1-Score", f"{results['f1']:.3f}")
            
            st.metric("CV Accuracy", f"{cv_results['mean']:.3f} ± {cv_results['std']:.3f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            X_test, y_test = st.session_state.test_data
            y_pred = st.session_state.model.predict(X_test)
            
            unique_labels = sorted(set(y_test))
            fig = plot_confusion_matrix(results['confusion_matrix'], unique_labels)
            st.pyplot(fig)
            plt.close()
            
            # Feature importance (Random Forest only)
            if model_type == 'random_forest':
                st.subheader("Feature Importance")
                importance_df = st.session_state.model.get_feature_importance()
                
                top_n = st.slider("Number of top features", 10, 50, 20, 5)
                fig = plot_feature_importance(importance_df, top_n=top_n)
                st.pyplot(fig)
                plt.close()

# ============================================================================
# TAB 5: Results
# ============================================================================
with tab5:
    st.header("Results & Export")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first (Tab 4)")
    else:
        st.subheader("Model Summary")
        
        st.markdown(f"""
        **Model Type**: {st.session_state.model.model_type.upper()}
        
        **Performance**:
        - Test Accuracy: {st.session_state.results['accuracy']:.3f}
        - Cross-Validation: {st.session_state.cv_results['mean']:.3f} ± {st.session_state.cv_results['std']:.3f}
        
        **Features**: {st.session_state.features.shape[1]}
        
        **Classes**: {list(st.session_state.model.label_encoder.classes_)}
        """)
        
        st.subheader("Make Predictions")
        
        st.markdown("The trained model can now predict emotions from new EEG data.")
        
        if st.button("🔮 Predict on Test Set"):
            X_test, y_test = st.session_state.test_data
            y_pred = st.session_state.model.predict(X_test)
            y_proba = st.session_state.model.predict_proba(X_test)
            
            # Show predictions
            results_df = pd.DataFrame({
                'True Label': y_test,
                'Predicted Label': y_pred,
                'Confidence': y_proba.max(axis=1)
            })
            
            st.dataframe(results_df.head(20))
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv,
                file_name="emotion_predictions.csv",
                mime="text/csv"
            )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About

This system implements a scientifically correct EEG emotion recognition pipeline following neuroscience best practices.

**Modules**:
- Preprocessing (filtering, ICA, ASR, re-referencing)
- Feature extraction (PSD, band powers, asymmetry)
- Classification (SVM, Random Forest)
- Visualization (topomaps, spectrograms)

**References**:
- Alarcao & Fonseca (2019). Emotions recognition using EEG signals: A survey.
- Davidson (2004). Frontal EEG asymmetry research.
""")
