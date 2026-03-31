"""
Train a Random Forest Regressor for metabarcoding relative abundance prediction.
"""
import os
import sys
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse

from config import RandomForestConfig, set_seed

# Features from utils.py
OBSERVATION_FEATURES = [
    # Observed features (already in dataset) 
    "total_reads_per_sample",
    "repl_w_reads_fractn",
    "latitude", 
    "longitude",
    "Excess",
    "Bulk_Sample_wet_weight",
    "SumExcessSpecimens",
    "ExcessNumberTaxa",
    "length_min_mm", 
    "length_max_mm",
    # Computed bin-level features
    "collection_day",   # derived from collection_start_date
    "total_reads_norm", # total_reads normalized by sample total
    "avg_reads_norm",   # avg_reads normalized by sample total
    "max_reads_norm",   # max_reads normalized by sample total
    "min_reads_norm",   # min_reads normalized by sample total
]

TAXONOMY_FEATURES = ["phylum", "class", "order", "family", "subfamily", "genus", "species"]


def load_data(config: RandomForestConfig):
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
    
    # Label encode taxonomy features
    label_encoders = {}
    for col in taxonomy_cols_present:
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    taxonomy_encoded_cols = [c + "_encoded" for c in taxonomy_cols_present]
    all_feature_cols = feature_cols_present + taxonomy_encoded_cols
    
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


def preprocess_features(X: pd.DataFrame, config: RandomForestConfig) -> np.ndarray:
    """
    Preprocess features by removing non-numeric columns.
    
    Args:
        X: DataFrame with features
        config: Configuration object
        
    Returns:
        Numpy array of numeric features
    """
    # Drop excluded columns
    cols_to_drop = [col for col in config.exclude_columns if col in X.columns]
    X_numeric = X.drop(columns=cols_to_drop, errors='ignore')
    
    # Handle any remaining non-numeric columns
    X_numeric = X_numeric.select_dtypes(include=[np.number])
    
    return X_numeric.values, X_numeric.columns.tolist()


def train_random_forest(config: RandomForestConfig, verbose: bool = True):
    """
    Train a Random Forest Regressor.
    
    Args:
        config: Configuration object
        verbose: Whether to print progress
        
    Returns:
        Trained model and feature names
    """
    set_seed(config.random_state)
    
    # Load data
    if verbose:
        print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(config)
    
    if verbose:
        print(f"  Train size: {len(X_train)}")
        print(f"  Val size: {len(X_val)}")
        print(f"  Test size: {len(X_test)}")
    
    # Preprocess features
    if verbose:
        print("\nPreprocessing features...")
    X_train_proc, feature_names = preprocess_features(X_train, config)
    X_val_proc, _ = preprocess_features(X_val, config)
    X_test_proc, _ = preprocess_features(X_test, config)
    
    y_train_arr = y_train.values.ravel()
    y_val_arr = y_val.values.ravel()
    y_test_arr = y_test.values.ravel()
    
    if verbose:
        print(f"  Number of features: {X_train_proc.shape[1]}")
        print(f"  Feature names: {feature_names}")
    
    # Initialize model
    if verbose:
        print("\nInitializing Random Forest Regressor...")
        print(f"  n_estimators: {config.n_estimators}")
        print(f"  max_depth: {config.max_depth}")
        print(f"  min_samples_split: {config.min_samples_split}")
        print(f"  min_samples_leaf: {config.min_samples_leaf}")
        print(f"  max_features: {config.max_features}")
    
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        verbose=1 if verbose else 0
    )
    
    # Train model
    if verbose:
        print("\nTraining Random Forest...")
    model.fit(X_train_proc, y_train_arr)
    
    # Evaluate on all splits
    if verbose:
        print("\nEvaluating model...")
    
    results = {}
    for name, X_proc, y_arr in [
        ("train", X_train_proc, y_train_arr),
        ("val", X_val_proc, y_val_arr),
        ("test", X_test_proc, y_test_arr)
    ]:
        y_pred = model.predict(X_proc)
        
        mse = mean_squared_error(y_arr, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_arr, y_pred)
        r2 = r2_score(y_arr, y_pred)
        
        results[name] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        if verbose:
            print(f"\n  {name.upper()} Results:")
            print(f"    MSE:  {mse:.6f}")
            print(f"    RMSE: {rmse:.6f}")
            print(f"    MAE:  {mae:.6f}")
            print(f"    R²:   {r2:.6f}")
    
    # Feature importance
    if verbose:
        print("\nFeature Importance (top 10):")
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(importance.head(10).to_string(index=False))
    
    return model, feature_names, results


def save_model(model, feature_names: list, results: dict, config: RandomForestConfig):
    """Save the trained model and results."""
    # Create directories if they don't exist
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    model_name = results.get("model", "two_stage_replication")
    # Save model
    model_path = os.path.join(config.model_dir, f"rf_model_{model_name}_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'config': config
        }, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save results
    results_path = os.path.join(config.results_dir, f"training_results_{model_name}_{timestamp}.csv")
    results_df = pd.DataFrame(results).T
    results_df.to_csv(results_path)
    print(f"Results saved to: {results_path}")
    
    return model_path, results_path


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest Regressor")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=None, help="Max tree depth")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Min samples to split")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="Min samples in leaf")
    parser.add_argument("--max_features", type=str, default="sqrt", help="Max features per split")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="two_stage_replication", help="Model variant name stored in output artifacts")
    parser.add_argument("--save", action="store_false", help="Save the model")
    
    args = parser.parse_args()
    
    # Create config
    config = RandomForestConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        data_dir=args.data_dir,
        random_state=args.seed
    )
    
    # Train model
    model, feature_names, results = train_random_forest(config)
    results["model"] = args.model
    
    # Save if requested
    if args.save:
        save_model(model, feature_names, results, config)
    
    return model, results


if __name__ == "__main__":
    main()
