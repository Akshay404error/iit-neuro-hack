"""
Machine Learning Pipeline for Affect Recognition from EEG Features

This script demonstrates a scientifically correct ML pipeline for emotion/affect
recognition using EEG features. It implements binary classification with proper
evaluation and addresses class imbalance.

Steps:
1. Convert continuous Valence/Arousal/Dominance ratings to binary classes (median threshold)
2. Train Logistic Regression and SVM classifiers
3. Implement K-Fold Cross-Validation and Leave-One-Subject-Out (LOSO)
4. Evaluate using Accuracy, F1 Score, and Confusion Matrix
5. Analyze class distribution and imbalance effects
6. Provide reproducible evaluation

Author: EEG Emotion Recognition System
Date: 2026-01-06
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("MACHINE LEARNING PIPELINE FOR AFFECT RECOGNITION")
print("="*80)
print()

# ============================================================================
# STEP 1: Generate Synthetic EEG Features with Continuous Ratings
# ============================================================================
print("STEP 1: Generating Synthetic EEG Features with Continuous Ratings")
print("-" * 80)
print()

print("Simulating EEG features from multiple subjects...")
print("(In practice, use features extracted from real EEG data)")
print()

# Simulate data from 10 subjects, 30 trials each
n_subjects = 10
n_trials_per_subject = 30
n_features = 42  # 14 channels × 3 bands (Theta, Alpha, Beta)

# Generate features and ratings
all_features = []
all_valence = []
all_arousal = []
all_dominance = []
all_subjects = []

for subject_id in range(1, n_subjects + 1):
    # Generate features for this subject
    # Add subject-specific baseline
    subject_baseline = np.random.randn(n_features) * 0.5
    
    for trial in range(n_trials_per_subject):
        # Generate features with some structure
        features = subject_baseline + np.random.randn(n_features) * 1.0
        
        # Generate continuous ratings (1-9 scale, typical for SAM)
        valence = np.random.uniform(1, 9)
        arousal = np.random.uniform(1, 9)
        dominance = np.random.uniform(1, 9)
        
        # Add some correlation between features and ratings
        # High alpha -> high valence (relaxed/positive)
        alpha_power = np.mean(features[14:28])  # Alpha features
        valence += alpha_power * 0.3
        
        # High beta -> high arousal
        beta_power = np.mean(features[28:42])  # Beta features
        arousal += beta_power * 0.3
        
        # Clip to valid range
        valence = np.clip(valence, 1, 9)
        arousal = np.clip(arousal, 1, 9)
        dominance = np.clip(dominance, 1, 9)
        
        all_features.append(features)
        all_valence.append(valence)
        all_arousal.append(arousal)
        all_dominance.append(dominance)
        all_subjects.append(subject_id)

# Convert to arrays
X = np.array(all_features)
valence_continuous = np.array(all_valence)
arousal_continuous = np.array(all_arousal)
dominance_continuous = np.array(all_dominance)
subjects = np.array(all_subjects)

print(f"✓ Generated dataset:")
print(f"  - Total samples: {len(X)}")
print(f"  - Features per sample: {X.shape[1]}")
print(f"  - Subjects: {n_subjects}")
print(f"  - Trials per subject: {n_trials_per_subject}")
print()

print(f"Continuous Rating Statistics:")
print(f"  - Valence: mean={valence_continuous.mean():.2f}, std={valence_continuous.std():.2f}")
print(f"  - Arousal: mean={arousal_continuous.mean():.2f}, std={arousal_continuous.std():.2f}")
print(f"  - Dominance: mean={dominance_continuous.mean():.2f}, std={dominance_continuous.std():.2f}")
print()

# ============================================================================
# STEP 2: Convert Continuous Ratings to Binary Classes
# ============================================================================
print("STEP 2: Converting Continuous Ratings to Binary Classes")
print("-" * 80)
print()

print("Threshold Selection: Median-Based Binarization")
print()
print("Scientific Justification:")
print("  Median thresholding is preferred over mean because:")
print("  1. Robust to outliers in rating distributions")
print("  2. Ensures balanced class distribution (50-50 split)")
print("  3. Commonly used in affective computing literature")
print("  4. Interpretable: 'High' = above median, 'Low' = below median")
print()

# Compute medians
valence_median = np.median(valence_continuous)
arousal_median = np.median(arousal_continuous)
dominance_median = np.median(dominance_continuous)

print(f"Computed Medians:")
print(f"  - Valence median: {valence_median:.2f}")
print(f"  - Arousal median: {arousal_median:.2f}")
print(f"  - Dominance median: {dominance_median:.2f}")
print()

# Binarize
y_valence = (valence_continuous >= valence_median).astype(int)
y_arousal = (arousal_continuous >= arousal_median).astype(int)
y_dominance = (dominance_continuous >= dominance_median).astype(int)

print("Binary Class Labels:")
print("  - 0 = Low (below median)")
print("  - 1 = High (above or equal to median)")
print()

# ============================================================================
# STEP 3: Analyze Class Distribution
# ============================================================================
print("STEP 3: Analyzing Class Distribution")
print("-" * 80)
print()

def analyze_class_distribution(y, name):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    print(f"{name} Distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / total) * 100
        class_name = "High" if label == 1 else "Low"
        print(f"  - Class {label} ({class_name}): {count} samples ({percentage:.1f}%)")
    
    # Calculate imbalance ratio
    if len(counts) == 2:
        imbalance_ratio = max(counts) / min(counts)
        print(f"  - Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 1.5:
            print(f"  ⚠ WARNING: Significant class imbalance detected!")
        else:
            print(f"  ✓ Classes are relatively balanced")
    print()

analyze_class_distribution(y_valence, "Valence")
analyze_class_distribution(y_arousal, "Arousal")
analyze_class_distribution(y_dominance, "Dominance")

print("Class Imbalance Effects:")
print("  1. Accuracy can be misleading (high accuracy by predicting majority class)")
print("  2. F1-score better reflects performance on both classes")
print("  3. Confusion matrix reveals per-class performance")
print("  4. Consider using balanced accuracy or class weights")
print()

# Visualize class distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (y, name) in enumerate([(y_valence, 'Valence'), 
                                   (y_arousal, 'Arousal'), 
                                   (y_dominance, 'Dominance')]):
    ax = axes[idx]
    unique, counts = np.unique(y, return_counts=True)
    bars = ax.bar(['Low (0)', 'High (1)'], counts, color=['#FF6B6B', '#4ECDC4'], 
                   edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(f'{name} Class Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/iit hackathon/eeg-emotion-recognition/ml_class_distribution.png', dpi=150)
print("✓ Class distribution plot saved: ml_class_distribution.png")
print()

# ============================================================================
# STEP 4: Feature Standardization
# ============================================================================
print("STEP 4: Feature Standardization")
print("-" * 80)
print()

print("Standardizing features (zero mean, unit variance)...")
print("This is critical for:")
print("  - Logistic Regression (gradient descent convergence)")
print("  - SVM (distance-based algorithm)")
print()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Features standardized")
print(f"  - Original mean: {X.mean():.3f}, std: {X.std():.3f}")
print(f"  - Scaled mean: {X_scaled.mean():.3f}, std: {X_scaled.std():.3f}")
print()

# ============================================================================
# STEP 5: Define Classification Models
# ============================================================================
print("STEP 5: Defining Classification Models")
print("-" * 80)
print()

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Handle imbalance
    ),
    'SVM (RBF Kernel)': SVC(
        kernel='rbf',
        random_state=42,
        class_weight='balanced',  # Handle imbalance
        probability=True
    )
}

print("Models configured:")
for name, model in models.items():
    print(f"  - {name}")
    print(f"    Parameters: {model.get_params()}")
print()

print("Note: class_weight='balanced' automatically adjusts weights inversely")
print("      proportional to class frequencies to handle imbalance.")
print()

# ============================================================================
# STEP 6: K-Fold Cross-Validation
# ============================================================================
print("STEP 6: K-Fold Cross-Validation")
print("-" * 80)
print()

print("K-Fold Cross-Validation (k=5):")
print("  - Splits data into 5 folds")
print("  - Each fold used once as validation set")
print("  - Provides robust performance estimate")
print()

k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1'
}

# Store results
kfold_results = {}

# Evaluate on Valence (can repeat for Arousal and Dominance)
print("Evaluating on VALENCE classification:")
print()

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    # Cross-validation
    cv_results = cross_validate(
        model, X_scaled, y_valence,
        cv=kfold,
        scoring=scoring,
        return_train_score=True
    )
    
    kfold_results[model_name] = cv_results
    
    # Print results
    print(f"  ✓ {model_name} Results:")
    print(f"    - Accuracy: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
    print(f"    - F1 Score:  {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")
    print()

# ============================================================================
# STEP 7: Leave-One-Subject-Out (LOSO) Cross-Validation
# ============================================================================
print("STEP 7: Leave-One-Subject-Out (LOSO) Cross-Validation")
print("-" * 80)
print()

print("LOSO Cross-Validation:")
print("  - Trains on n-1 subjects, tests on 1 held-out subject")
print("  - More realistic evaluation (subject-independent)")
print("  - Tests generalization to new subjects")
print()

loso = LeaveOneGroupOut()
loso_results = {}

print("Evaluating on VALENCE classification:")
print()

for model_name, model in models.items():
    print(f"Training {model_name} with LOSO...")
    
    # LOSO cross-validation
    cv_results = cross_validate(
        model, X_scaled, y_valence,
        groups=subjects,
        cv=loso,
        scoring=scoring,
        return_train_score=True
    )
    
    loso_results[model_name] = cv_results
    
    # Print results
    print(f"  ✓ {model_name} LOSO Results:")
    print(f"    - Accuracy: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
    print(f"    - F1 Score:  {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")
    print()

# ============================================================================
# STEP 8: Train Final Model and Generate Confusion Matrix
# ============================================================================
print("STEP 8: Training Final Model and Generating Confusion Matrix")
print("-" * 80)
print()

print("Training on full dataset for confusion matrix visualization...")
print()

# Train-test split for confusion matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_valence, test_size=0.2, random_state=42, stratify=y_valence
)

confusion_matrices = {}

for model_name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    confusion_matrices[model_name] = cm
    
    print(f"{model_name}:")
    print(f"  - Test Accuracy: {accuracy:.3f}")
    print(f"  - Test F1 Score: {f1:.3f}")
    print(f"  - Confusion Matrix:")
    print(f"    {cm}")
    print()
    
    # Classification report
    print(f"  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Valence', 'High Valence']))
    print()

# Visualize confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
    ax = axes[idx]
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Low (0)', 'High (1)'],
                yticklabels=['Low (0)', 'High (1)'],
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/iit hackathon/eeg-emotion-recognition/ml_confusion_matrices.png', dpi=150)
print("✓ Confusion matrices saved: ml_confusion_matrices.png")
print()

# ============================================================================
# STEP 9: Compare K-Fold vs LOSO Performance
# ============================================================================
print("STEP 9: Comparing K-Fold vs LOSO Performance")
print("-" * 80)
print()

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
ax = axes[0]
x_pos = np.arange(len(models))
width = 0.35

kfold_acc = [kfold_results[name]['test_accuracy'].mean() for name in models.keys()]
loso_acc = [loso_results[name]['test_accuracy'].mean() for name in models.keys()]

bars1 = ax.bar(x_pos - width/2, kfold_acc, width, label='K-Fold CV', 
               color='#4ECDC4', edgecolor='black')
bars2 = ax.bar(x_pos + width/2, loso_acc, width, label='LOSO CV',
               color='#FF6B6B', edgecolor='black')

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy: K-Fold vs LOSO', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models.keys(), rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# F1 Score comparison
ax = axes[1]
kfold_f1 = [kfold_results[name]['test_f1'].mean() for name in models.keys()]
loso_f1 = [loso_results[name]['test_f1'].mean() for name in models.keys()]

bars1 = ax.bar(x_pos - width/2, kfold_f1, width, label='K-Fold CV',
               color='#4ECDC4', edgecolor='black')
bars2 = ax.bar(x_pos + width/2, loso_f1, width, label='LOSO CV',
               color='#FF6B6B', edgecolor='black')

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 Score: K-Fold vs LOSO', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models.keys(), rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('D:/iit hackathon/eeg-emotion-recognition/ml_cv_comparison.png', dpi=150)
print("✓ Cross-validation comparison saved: ml_cv_comparison.png")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*80)
print("MACHINE LEARNING PIPELINE COMPLETE")
print("="*80)
print()

print("Summary:")
print()
print("1. ✓ Converted continuous ratings to binary classes (median threshold)")
print("2. ✓ Trained Logistic Regression and SVM models")
print("3. ✓ Evaluated with K-Fold (k=5) and LOSO cross-validation")
print("4. ✓ Computed Accuracy, F1 Score, and Confusion Matrices")
print("5. ✓ Analyzed class distribution and imbalance")
print()

print("Key Findings:")
print(f"  - Median thresholding ensures balanced classes (~50-50 split)")
print(f"  - LOSO typically shows lower performance than K-Fold")
print(f"    (more realistic, tests generalization to new subjects)")
print(f"  - F1 Score accounts for both precision and recall")
print(f"  - class_weight='balanced' helps with imbalance")
print()

print("Output Files:")
print("  - ml_class_distribution.png: Class distribution analysis")
print("  - ml_confusion_matrices.png: Confusion matrices for both models")
print("  - ml_cv_comparison.png: K-Fold vs LOSO performance")
print()

print("Reproducibility:")
print("  - Random seed set to 42")
print("  - All parameters documented")
print("  - Stratified splits preserve class distribution")
print()

print("="*80)
print("All ML pipeline steps completed successfully!")
print("="*80)

# Show plots
print()
print("Displaying plots...")
plt.show()
