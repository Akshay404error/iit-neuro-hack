"""
Machine Learning Classifier for EEG Emotion Recognition

This module implements ML classifiers for emotion recognition from EEG features.

Scientific Rationale:
- SVM with RBF kernel: Effective for high-dimensional EEG features
- Random Forest: Provides feature importance, robust to overfitting
- Cross-validation: Ensures generalization to unseen data
- SMOTE: Handles class imbalance in emotion datasets

References:
- Alarcao & Fonseca (2019). Emotions recognition using EEG signals: A survey.
  IEEE Transactions on Affective Computing, 10(3), 374-393.
- Cortes & Vapnik (1995). Support-vector networks. Machine Learning, 20(3), 273-297.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from typing import Optional, Dict, Tuple
import joblib


class EmotionClassifier:
    """
    Emotion classifier with preprocessing and model training.
    
    Examples
    --------
    >>> clf = EmotionClassifier(model_type='svm')
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> clf.evaluate(X_test, y_test)
    """
    
    def __init__(
        self,
        model_type: str = 'svm',
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize emotion classifier.
        
        Parameters
        ----------
        model_type : str, default='svm'
            Type of classifier: 'svm', 'random_forest'
        random_state : int, default=42
            Random seed for reproducibility
        verbose : bool, default=True
            Print information
        """
        self.model_type = model_type
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the ML model."""
        if self.model_type == 'svm':
            # SVM with RBF kernel
            # Scientific justification: RBF kernel handles non-linear relationships
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=self.random_state,
                probability=True,  # Enable probability estimates
                verbose=self.verbose
            )
        elif self.model_type == 'random_forest':
            # Random Forest
            # Scientific justification: Ensemble method, provides feature importance
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=self.random_state,
                verbose=self.verbose
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        apply_smote: bool = True
    ):
        """
        Train the classifier.
        
        Scientific Justification:
        - Standardization: Zero mean, unit variance (important for SVM)
        - SMOTE: Synthetic Minority Over-sampling (handles class imbalance)
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Labels (n_samples,)
        apply_smote : bool, default=True
            Apply SMOTE for class balancing
        """
        if self.verbose:
            print("="*60)
            print(f"TRAINING {self.model_type.upper()} CLASSIFIER")
            print("="*60)
            print(f"Training samples: {X.shape[0]}")
            print(f"Features: {X.shape[1]}")
        
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        if self.verbose:
            unique, counts = np.unique(y, return_counts=True)
            print("\nClass distribution:")
            for label, count in zip(unique, counts):
                print(f"  - {label}: {count} samples")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply SMOTE if requested
        if apply_smote:
            try:
                smote = SMOTE(random_state=self.random_state)
                X_scaled, y_encoded = smote.fit_resample(X_scaled, y_encoded)
                
                if self.verbose:
                    print("\n✓ SMOTE applied")
                    unique, counts = np.unique(y_encoded, return_counts=True)
                    print("Class distribution after SMOTE:")
                    for label_idx, count in zip(unique, counts):
                        label = self.label_encoder.inverse_transform([label_idx])[0]
                        print(f"  - {label}: {count} samples")
            except Exception as e:
                if self.verbose:
                    print(f"\n⚠ Could not apply SMOTE: {e}")
        
        # Train model
        if self.verbose:
            print("\nTraining model...")
        
        self.model.fit(X_scaled, y_encoded)
        
        if self.verbose:
            print("✓ Model trained")
            print("="*60)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict emotion labels.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict emotion probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted probabilities (n_samples, n_classes)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        y_proba = self.model.predict_proba(X_scaled)
        
        return y_proba
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict:
        """
        Evaluate classifier performance.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            True labels
            
        Returns
        -------
        dict
            Evaluation metrics
        """
        y_pred = self.predict(X)
        
        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted'
        )
        cm = confusion_matrix(y, y_pred)
        
        if self.verbose:
            print("="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
            print("\nClassification Report:")
            print(classification_report(y, y_pred))
            print("="*60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict:
        """
        Perform cross-validation.
        
        Scientific Justification:
        - Stratified K-Fold: Preserves class distribution in each fold
        - 5-fold CV: Standard for small-medium datasets
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Labels
        cv : int, default=5
            Number of folds
            
        Returns
        -------
        dict
            Cross-validation scores
        """
        if self.verbose:
            print("="*60)
            print(f"CROSS-VALIDATION ({cv}-fold)")
            print("="*60)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Compute scores
        scores = cross_val_score(
            self.model, X_scaled, y_encoded, cv=skf, scoring='accuracy'
        )
        
        if self.verbose:
            print(f"Accuracy scores: {scores}")
            print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            print("="*60)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (for Random Forest only).
        
        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        if self.model_type != 'random_forest':
            raise ValueError("Feature importance only available for Random Forest")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        df = pd.DataFrame({
            'feature': [self.feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        
        return df
    
    def save(self, filepath: str):
        """Save the trained model."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)
        
        if self.verbose:
            print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        
        if self.verbose:
            print(f"✓ Model loaded from {filepath}")


def hyperparameter_tuning(
    X: pd.DataFrame,
    y: np.ndarray,
    model_type: str = 'svm',
    cv: int = 5,
    verbose: bool = True
) -> Tuple[Dict, float]:
    """
    Perform hyperparameter tuning using Grid Search.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Labels
    model_type : str, default='svm'
        Model type
    cv : int, default=5
        Cross-validation folds
    verbose : bool, default=True
        Print information
        
    Returns
    -------
    tuple
        (best_params, best_score)
    """
    if verbose:
        print("="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
    
    # Prepare data
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    X_scaled = scaler.fit_transform(X.values if isinstance(X, pd.DataFrame) else X)
    y_encoded = label_encoder.fit_transform(y)
    
    # Define parameter grids
    if model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        model = SVC(probability=True)
    elif model_type == 'random_forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy', verbose=1 if verbose else 0
    )
    grid_search.fit(X_scaled, y_encoded)
    
    if verbose:
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best score: {grid_search.best_score_:.3f}")
        print("="*60)
    
    return grid_search.best_params_, grid_search.best_score_
