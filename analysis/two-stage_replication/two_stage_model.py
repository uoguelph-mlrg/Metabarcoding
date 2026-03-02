"""
Two-Stage Random Forest Model for Zero-Inflated Relative Abundance Prediction.

Stage 1: Random Forest Classifier - Predicts zero vs non-zero abundance
Stage 2: Random Forest Regressor - Predicts actual abundance for non-zero samples

This approach handles the zero-inflation in the data by first determining if a 
sample has any abundance, then predicting the actual value.
"""
import os
import pickle
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
)
import argparse

from config import set_seed

# Features from utils.py
OBSERVATION_FEATURES = [
    "Excess",
    "trackingsl_specimen_count",
    "Bulk_Sample_wet_weight",
    "SumExcessSpecimens",
    "ExcessNumberTaxa",
    "total_reads_per_sample",
    "length_min_mm",
    "month",
    "avg_length_mm",
    "reads_per_specimen",
    "specimens_per_gram",
    "taxonomic_completeness",
    "read_consistency",
    "relative_read_abundance"
]

TAXONOMY_FEATURES = ["phylum", "class", "order", "family"]


@dataclass
class TwoStageConfig:
    """Configuration for two-stage model."""
    # Classifier hyperparameters (tuned via hyperparameter search)
    clf_n_estimators: int = 200
    clf_max_depth: Optional[int] = 20
    clf_min_samples_split: int = 10
    clf_min_samples_leaf: int = 1
    clf_max_features: str = "sqrt"
    clf_class_weight: str = "balanced_subsample"  # Handle class imbalance
    
    # Regressor hyperparameters (tuned via hyperparameter search)
    reg_n_estimators: int = 100
    reg_max_depth: Optional[int] = 10
    reg_min_samples_split: int = 10
    reg_min_samples_leaf: int = 1
    reg_max_features: str = "sqrt"
    
    # General settings
    n_jobs: int = -1
    random_state: int = 42
    
    # Data split fractions
    train_frac: float = 0.8
    val_frac: float = 0.1
    
    # Data paths
    data_dir: str = "data"
    model_dir: str = "models"
    results_dir: str = "results"
    
    # Features to exclude
    exclude_columns: tuple = ("sample_id", "bin_uri")


class TwoStageRandomForest:
    """
    Two-stage model for zero-inflated abundance prediction.
    
    Stage 1: Classifier predicts P(abundance > 0)
    Stage 2: Regressor predicts abundance | abundance > 0
    """
    
    def __init__(self, config: TwoStageConfig):
        self.config = config
        self.classifier = None
        self.regressor = None
        self.feature_names = None
        self.is_fitted = False
        
    def _init_classifier(self) -> RandomForestClassifier:
        """Initialize the classifier for zero vs non-zero prediction."""
        return RandomForestClassifier(
            n_estimators=self.config.clf_n_estimators,
            max_depth=self.config.clf_max_depth,
            min_samples_split=self.config.clf_min_samples_split,
            min_samples_leaf=self.config.clf_min_samples_leaf,
            max_features=self.config.clf_max_features,
            class_weight=self.config.clf_class_weight,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbose=0
        )
    
    def _init_regressor(self) -> RandomForestRegressor:
        """Initialize the regressor for abundance prediction."""
        return RandomForestRegressor(
            n_estimators=self.config.reg_n_estimators,
            max_depth=self.config.reg_max_depth,
            min_samples_split=self.config.reg_min_samples_split,
            min_samples_leaf=self.config.reg_min_samples_leaf,
            max_features=self.config.reg_max_features,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbose=0
        )
    
    def preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Extract numeric features, excluding ID columns."""
        cols_to_drop = [col for col in self.config.exclude_columns if col in X.columns]
        X_numeric = X.drop(columns=cols_to_drop, errors='ignore')
        X_numeric = X_numeric.select_dtypes(include=[np.number])
        
        if self.feature_names is None:
            self.feature_names = X_numeric.columns.tolist()
        
        return X_numeric[self.feature_names].values
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, verbose: bool = True):
        """
        Fit both stages of the model.
        
        Args:
            X_train: Training features
            y_train: Training targets (rel_abundance)
            verbose: Print progress
        """
        # Preprocess features
        X_train_proc = self.preprocess_features(X_train)
        y_train_arr = y_train.values.ravel()
        
        # Create binary labels for classifier (0 = zero, 1 = non-zero)
        y_binary = (y_train_arr > 0).astype(int)
        
        if verbose:
            n_zeros = (y_binary == 0).sum()
            n_nonzeros = (y_binary == 1).sum()
            print(f"\n{'='*60}")
            print("STAGE 1: CLASSIFIER (Zero vs Non-Zero)")
            print(f"{'='*60}")
            print(f"Training samples: {len(y_binary)}")
            print(f"  Zero abundance: {n_zeros} ({n_zeros/len(y_binary)*100:.1f}%)")
            print(f"  Non-zero abundance: {n_nonzeros} ({n_nonzeros/len(y_binary)*100:.1f}%)")
        
        # Train classifier
        self.classifier = self._init_classifier()
        if verbose:
            print("\nTraining classifier...")
        self.classifier.fit(X_train_proc, y_binary)
        
        # Train regressor on non-zero samples only
        nonzero_mask = y_train_arr > 0
        X_train_nonzero = X_train_proc[nonzero_mask]
        y_train_nonzero = y_train_arr[nonzero_mask]
        
        if verbose:
            print(f"\n{'='*60}")
            print("STAGE 2: REGRESSOR (Abundance for Non-Zero Samples)")
            print(f"{'='*60}")
            print(f"Training samples (non-zero only): {len(y_train_nonzero)}")
            print(f"  Min abundance: {y_train_nonzero.min():.6f}")
            print(f"  Max abundance: {y_train_nonzero.max():.6f}")
            print(f"  Mean abundance: {y_train_nonzero.mean():.6f}")
        
        self.regressor = self._init_regressor()
        if verbose:
            print("\nTraining regressor...")
        self.regressor.fit(X_train_nonzero, y_train_nonzero)
        
        self.is_fitted = True
        
        if verbose:
            print("\nModel training complete!")
        
        return self
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict abundance using the two-stage approach.
        
        Args:
            X: Features
            threshold: Classification threshold for non-zero prediction
            
        Returns:
            Predicted abundance values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_proc = self.preprocess_features(X)
        
        # Stage 1: Predict zero vs non-zero
        nonzero_proba = self.classifier.predict_proba(X_proc)[:, 1]
        is_nonzero = nonzero_proba >= threshold
        
        # Stage 2: Predict abundance for samples predicted as non-zero
        predictions = np.zeros(len(X_proc))
        if is_nonzero.sum() > 0:
            predictions[is_nonzero] = self.regressor.predict(X_proc[is_nonzero])
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_proba_nonzero(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of non-zero abundance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_proc = self.preprocess_features(X)
        return self.classifier.predict_proba(X_proc)[:, 1]
    
    def evaluate_classifier(self, X: pd.DataFrame, y_true: pd.DataFrame, 
                           threshold: float = 0.5, verbose: bool = True) -> Dict:
        """
        Evaluate the classifier stage.
        
        Returns metrics: accuracy, precision, recall, F1-score
        """
        X_proc = self.preprocess_features(X)
        y_true_arr = y_true.values.ravel()
        y_binary_true = (y_true_arr > 0).astype(int)
        
        # Predict
        proba = self.classifier.predict_proba(X_proc)[:, 1]
        y_binary_pred = (proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_binary_true, y_binary_pred),
            'precision': precision_score(y_binary_true, y_binary_pred),
            'recall': recall_score(y_binary_true, y_binary_pred),
            'f1': f1_score(y_binary_true, y_binary_pred),
            'confusion_matrix': confusion_matrix(y_binary_true, y_binary_pred)
        }
        
        if verbose:
            print("\nClassifier Evaluation:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print("\nConfusion Matrix:")
            print(f"  TN: {metrics['confusion_matrix'][0,0]:6d}  FP: {metrics['confusion_matrix'][0,1]:6d}")
            print(f"  FN: {metrics['confusion_matrix'][1,0]:6d}  TP: {metrics['confusion_matrix'][1,1]:6d}")
        
        return metrics
    
    def evaluate_regressor(self, X: pd.DataFrame, y_true: pd.DataFrame,
                          verbose: bool = True) -> Dict:
        """
        Evaluate the regressor stage on truly non-zero samples.
        
        Returns metrics: RMSE, MAE, R², MAPE, Median AE
        """
        X_proc = self.preprocess_features(X)
        y_true_arr = y_true.values.ravel()
        
        # Filter to non-zero ground truth samples
        nonzero_mask = y_true_arr > 0
        X_nonzero = X_proc[nonzero_mask]
        y_nonzero_true = y_true_arr[nonzero_mask]
        
        # Predict
        y_nonzero_pred = self.regressor.predict(X_nonzero)
        y_nonzero_pred = np.maximum(y_nonzero_pred, 0)  # Ensure non-negative
        
        # Calculate metrics
        mse = mean_squared_error(y_nonzero_true, y_nonzero_pred)
        metrics = {
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_nonzero_true, y_nonzero_pred),
            'r2': r2_score(y_nonzero_true, y_nonzero_pred),
            'median_ae': median_absolute_error(y_nonzero_true, y_nonzero_pred),
            'n_samples': len(y_nonzero_true)
        }
        
        # MAPE (Mean Absolute Percentage Error) - avoiding division by zero
        mape = np.mean(np.abs((y_nonzero_true - y_nonzero_pred) / y_nonzero_true)) * 100
        metrics['mape'] = mape
        
        if verbose:
            print(f"\nRegressor Evaluation (on {metrics['n_samples']} non-zero samples):")
            print(f"  RMSE:      {metrics['rmse']:.6f}")
            print(f"  MAE:       {metrics['mae']:.6f}")
            print(f"  R²:        {metrics['r2']:.4f}")
            print(f"  Median AE: {metrics['median_ae']:.6f}")
            print(f"  MAPE:      {metrics['mape']:.2f}%")
        
        return metrics
    
    def evaluate_combined(self, X: pd.DataFrame, y_true: pd.DataFrame,
                         threshold: float = 0.5, verbose: bool = True) -> Dict:
        """
        Evaluate the full two-stage pipeline end-to-end.
        """
        y_true_arr = y_true.values.ravel()
        
        # Get full pipeline predictions
        y_pred = self.predict(X, threshold=threshold)
        
        # Overall metrics
        mse = mean_squared_error(y_true_arr, y_pred)
        metrics = {
            'overall_rmse': np.sqrt(mse),
            'overall_mae': mean_absolute_error(y_true_arr, y_pred),
            'overall_r2': r2_score(y_true_arr, y_pred),
            'overall_median_ae': median_absolute_error(y_true_arr, y_pred)
        }
        
        # Evaluate on different subsets
        # 1. True zeros
        zero_mask = y_true_arr == 0
        if zero_mask.sum() > 0:
            metrics['zero_mae'] = mean_absolute_error(y_true_arr[zero_mask], y_pred[zero_mask])
            metrics['zero_pred_zero_ratio'] = (y_pred[zero_mask] == 0).mean()
        
        # 2. True non-zeros
        nonzero_mask = y_true_arr > 0
        if nonzero_mask.sum() > 0:
            metrics['nonzero_rmse'] = np.sqrt(mean_squared_error(
                y_true_arr[nonzero_mask], y_pred[nonzero_mask]))
            metrics['nonzero_mae'] = mean_absolute_error(
                y_true_arr[nonzero_mask], y_pred[nonzero_mask])
        
        if verbose:
            print("\nCombined Two-Stage Evaluation:")
            print(f"  Overall RMSE:      {metrics['overall_rmse']:.6f}")
            print(f"  Overall MAE:       {metrics['overall_mae']:.6f}")
            print(f"  Overall R²:        {metrics['overall_r2']:.4f}")
            print(f"  Overall Median AE: {metrics['overall_median_ae']:.6f}")
            if 'zero_mae' in metrics:
                print(f"\n  On true zeros (n={zero_mask.sum()}):")
                print(f"    MAE: {metrics['zero_mae']:.6f}")
                print(f"    Correctly predicted as zero: {metrics['zero_pred_zero_ratio']*100:.1f}%")
            if 'nonzero_mae' in metrics:
                print(f"\n  On true non-zeros (n={nonzero_mask.sum()}):")
                print(f"    RMSE: {metrics['nonzero_rmse']:.6f}")
                print(f"    MAE: {metrics['nonzero_mae']:.6f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from both models."""
        clf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'classifier_importance': self.classifier.feature_importances_
        })
        
        reg_importance = pd.DataFrame({
            'feature': self.feature_names,
            'regressor_importance': self.regressor.feature_importances_
        })
        
        importance = clf_importance.merge(reg_importance, on='feature')
        importance['avg_importance'] = (
            importance['classifier_importance'] + importance['regressor_importance']
        ) / 2
        
        return importance.sort_values('avg_importance', ascending=False)
    
    def save(self, path: str = None):
        """Save the model to disk."""
        if path is None:
            os.makedirs(self.config.model_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            path = os.path.join(self.config.model_dir, f"two_stage_rf_{timestamp}.pkl")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'regressor': self.regressor,
                'feature_names': self.feature_names,
                'config': self.config
            }, f)
        
        print(f"Model saved to: {path}")
        return path
    
    @classmethod
    def load(cls, path: str) -> 'TwoStageRandomForest':
        """Load a saved model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['config'])
        model.classifier = data['classifier']
        model.regressor = data['regressor']
        model.feature_names = data['feature_names']
        model.is_fitted = True
        
        return model


def load_data(data_dir: str) -> Tuple:
    """
    Load preprocessed train, validation, and test data from CSV files.
    
    Use preprocess_data.py to create these files from ecuador_training_data.csv.
    """
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    y_val = pd.read_csv(os.path.join(data_dir, "y_val.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))
    
    print(f"Loaded data from: {data_dir}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_data_from_raw(config: TwoStageConfig) -> Tuple:
    """
    Load and preprocess data from ecuador_training_data.csv.
    Uses same preprocessing as utils.py but includes taxonomy features.
    """
    data_path = os.path.join(config.data_dir, "ecuador_training_data.csv")
    df = pd.read_csv(data_path)
    
    # Rename columns to match expected format
    df = df.rename(columns={"sample-eventid": "sample_id"})
    
    # Parse date and extract day of year as numeric feature
    if "collection_start_date" in df.columns:
        df["collection_day"] = pd.to_datetime(
            df["collection_start_date"], format="%m/%d/%Y", errors="coerce"
        ).dt.dayofyear
        df["collection_day"] = df["collection_day"].fillna(0)
    else:
        df["collection_day"] = 0
    
    # Build index mappings
    unique_samples = df["sample_id"].unique()
    n_samples = len(unique_samples)
    unique_bins = df["bin_uri"].unique()
    
    # Compute relative abundance (target)
    sample_totals = df.groupby("sample_id")["occurrences"].transform("sum")
    df["rel_abundance"] = df["occurrences"] / (sample_totals + 1e-10)
    
    # Normalize reads per sample (log-transformed)
    df["total_reads_norm"] = np.log1p(df["total_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
    df["avg_reads_norm"] = np.log1p(df["avg_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
    df["max_reads_norm"] = np.log1p(df["max_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
    df["min_reads_norm"] = np.log1p(df["min_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
    
    # Get available feature columns
    feature_cols_present = [c for c in OBSERVATION_FEATURES if c in df.columns]
    taxonomy_cols_present = [c for c in TAXONOMY_FEATURES if c in df.columns]
    # Hierarchical and path encoding (match preprocess_data.py)
    TAXONOMIC_HIERARCHY = ['kingdom', 'phylum', 'class', 'order', 'family', 'subfamily', 'genus', 'species']
    available_hierarchy = [feat for feat in TAXONOMIC_HIERARCHY if feat in df.columns]
    taxonomy_encoded_cols = [f'{level}_hierarchical' for level in available_hierarchy]
    if available_hierarchy:
        taxonomy_encoded_cols.append('taxonomic_path')
    # One-hot, binary, target encoding for other categoricals (if present)
    onehot_cols = [c for c in df.columns if any(c.startswith(f'{col}_') for col in df.columns if df[col].nunique() <= 10 and col not in taxonomy_cols_present)]
    binary_cols = [f'{col}_encoded' for col in df.columns if df[col].nunique() == 2 and col not in taxonomy_cols_present]
    target_cols = [f'{col}_target' for col in df.columns if df[col].nunique() > 10 and col not in taxonomy_cols_present]
    all_feature_cols = feature_cols_present + taxonomy_encoded_cols + onehot_cols + binary_cols + target_cols
    
    # Build df_long with required columns + features
    base_cols = ["sample_id", "bin_uri", "occurrences", "rel_abundance"]
    df_long = df[base_cols + all_feature_cols].copy()
    
    # Create train/val/test splits at sample level
    np.random.seed(config.random_state)
    sample_indices = np.arange(n_samples)
    np.random.shuffle(sample_indices)
    
    n_train = int(n_samples * config.train_frac)
    n_val = int(n_samples * config.val_frac)
    
    train_sample_idx = sample_indices[:n_train]
    val_sample_idx = sample_indices[n_train:n_train + n_val]
    test_sample_idx = sample_indices[n_train + n_val:]
    
    # Fill missing numeric features with median from training set per BIN
    X_train_subset = df_long.loc[
        df_long["sample_id"].isin(set(unique_samples[train_sample_idx])), 
        feature_cols_present + ["bin_uri"]
    ]
    bin_medians = X_train_subset.groupby("bin_uri").median()
    
    for col in feature_cols_present:
        if col not in bin_medians.columns:
            continue
        median_map = bin_medians[col].to_dict()
        df_long[col] = df_long.apply(
            lambda row: median_map.get(row["bin_uri"], np.nan) if pd.isna(row[col]) else row[col],
            axis=1
        )
        df_long[col] = df_long[col].fillna(df_long[col].median())
    
    # Normalize numeric features based on training set statistics
    train_mask = df_long["sample_id"].isin(set(unique_samples[train_sample_idx]))
    for col in feature_cols_present:
        train_mean = df_long.loc[train_mask, col].mean()
        train_std = df_long.loc[train_mask, col].std(ddof=0) + 1e-10
        df_long[col] = (df_long[col] - train_mean) / train_std
    
    # Get train, val, test data
    def compute_data_split(df_long, sample_idx):
        sample_set = set(unique_samples[sample_idx])
        mask = df_long["sample_id"].isin(sample_set)
        X = df_long.loc[mask, ["sample_id", "bin_uri"] + all_feature_cols].copy()
        y = df_long.loc[mask, "rel_abundance"]
        return X, pd.DataFrame(y)
    
    X_train, y_train = compute_data_split(df_long, train_sample_idx)
    X_val, y_val = compute_data_split(df_long, val_sample_idx)
    X_test, y_test = compute_data_split(df_long, test_sample_idx)
    
    print(f"Loaded {len(df_long)} observations")
    print(f"  {n_samples} samples, {len(unique_bins)} bins")
    print(f"  Observation features: {len(feature_cols_present)}")
    print(f"  Taxonomy features (encoded): {len(taxonomy_encoded_cols)}")
    print(f"  Total features: {len(all_feature_cols)}")
    print(f"  Train: {len(train_sample_idx)} samples")
    print(f"  Val: {len(val_sample_idx)} samples")
    print(f"  Test: {len(test_sample_idx)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser(description="Train Two-Stage Random Forest Model")
    
    # Classifier args
    parser.add_argument("--clf_n_estimators", type=int, default=100)
    parser.add_argument("--clf_max_depth", type=int, default=None)
    parser.add_argument("--clf_class_weight", type=str, default="balanced")
    
    # Regressor args
    parser.add_argument("--reg_n_estimators", type=int, default=100)
    parser.add_argument("--reg_max_depth", type=int, default=None)
    
    # General args
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save", action="store_true")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create config
    config = TwoStageConfig(
        clf_n_estimators=args.clf_n_estimators,
        clf_max_depth=args.clf_max_depth,
        clf_class_weight=args.clf_class_weight,
        reg_n_estimators=args.reg_n_estimators,
        reg_max_depth=args.reg_max_depth,
        data_dir=args.data_dir,
        random_state=args.seed
    )
    
    # Load data
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.data_dir)
    
    # Create and train model
    model = TwoStageRandomForest(config)
    model.fit(X_train, y_train)
    
    # Evaluate on all splits
    for name, X, y in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        print(f"\n{'='*60}")
        print(f"{name.upper()} SET EVALUATION")
        print(f"{'='*60}")
        
        clf_metrics = model.evaluate_classifier(X, y, threshold=args.threshold)
        reg_metrics = model.evaluate_regressor(X, y)
        combined_metrics = model.evaluate_combined(X, y, threshold=args.threshold)
    
    # Feature importance
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE")
    print(f"{'='*60}")
    importance = model.get_feature_importance()
    print(importance.to_string(index=False))
    
    # Save model
    if args.save:
        model.save()
    
    return model


if __name__ == "__main__":
    main()
