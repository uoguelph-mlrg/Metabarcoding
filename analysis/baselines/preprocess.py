"""
Preprocessing module for baseline models.
Adapted from random_forest/preprocess_data.py
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, List, Optional


OBSERVATION_FEATURES = [
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
    "collection_day",
    "total_reads_norm",
    "avg_reads_norm",
    "max_reads_norm",
    "min_reads_norm",
]

TAXONOMY_FEATURES = ["phylum", "class", "order", "family", "subfamily", "genus", "species"]


class DataPreprocessor:
    """Preprocessor for metabarcoding baseline data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.taxonomy_encoded_cols: List[str] = []
        self.fitted = False
        
    def load_and_preprocess(
        self,
        data_path: str,
        train_frac: float = 0.8,
        val_frac: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
               pd.Series, pd.Series, pd.Series, Dict]:
        """
        Load raw data, preprocess, and create train/val/test splits.
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, metadata
        """
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Rename columns
        df = df.rename(columns={"sample-eventid": "sample_id"})
        
        # Parse date and extract day of year
        if "collection_start_date" in df.columns:
            df["collection_day"] = pd.to_datetime(
                df["collection_start_date"], format="%m/%d/%Y", errors="coerce"
            ).dt.dayofyear
            df["collection_day"] = df["collection_day"].fillna(0)
        else:
            df["collection_day"] = 0
        
        # Get unique samples and bins
        unique_samples = df["sample_id"].unique()
        n_samples = len(unique_samples)
        unique_bins = df["bin_uri"].unique()
        
        print(f"Found {n_samples} samples, {len(unique_bins)} bins, {len(df)} observations")
        
        # Compute relative abundance (target variable)
        sample_totals = df.groupby("sample_id")["occurrences"].transform("sum")
        df["rel_abundance"] = df["occurrences"] / (sample_totals + 1e-10)
        
        # Create binary presence/absence indicator (for zero-inflated models)
        df["presence"] = (df["occurrences"] > 0).astype(int)
        
        # Normalize reads per sample (log-transformed)
        df["total_reads_norm"] = np.log1p(df["total_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
        df["avg_reads_norm"] = np.log1p(df["avg_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
        df["max_reads_norm"] = np.log1p(df["max_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
        df["min_reads_norm"] = np.log1p(df["min_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2)
        
        # Get available features
        feature_cols_present = [c for c in OBSERVATION_FEATURES if c in df.columns]
        taxonomy_cols_present = [c for c in TAXONOMY_FEATURES if c in df.columns]
        
        print(f"Observation features: {len(feature_cols_present)}")
        print(f"Taxonomy features: {len(taxonomy_cols_present)}")
        
        # Label encode taxonomy features
        for col in taxonomy_cols_present:
            df[col] = df[col].fillna("Unknown")
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        self.taxonomy_encoded_cols = [c + "_encoded" for c in taxonomy_cols_present]
        self.feature_cols = feature_cols_present + self.taxonomy_encoded_cols
        
        print(f"Total features: {len(self.feature_cols)}")
        
        # Build working dataframe
        base_cols = ["sample_id", "bin_uri", "occurrences", "rel_abundance", "presence"]
        df_long = df[base_cols + self.feature_cols].copy()
        
        # Create train/val/test splits at sample level
        np.random.seed(self.random_state)
        sample_indices = np.arange(n_samples)
        np.random.shuffle(sample_indices)
        
        n_train = int(n_samples * train_frac)
        n_val = int(n_samples * val_frac)
        
        train_sample_idx = sample_indices[:n_train]
        val_sample_idx = sample_indices[n_train:n_train + n_val]
        test_sample_idx = sample_indices[n_train + n_val:]
        
        train_samples = set(unique_samples[train_sample_idx])
        val_samples = set(unique_samples[val_sample_idx])
        test_samples = set(unique_samples[test_sample_idx])
        
        print(f"\nSplit sizes:")
        print(f"  Train samples: {len(train_sample_idx)}")
        print(f"  Val samples: {len(val_sample_idx)}")
        print(f"  Test samples: {len(test_sample_idx)}")
        
        # Fill missing numeric features with median from training set
        train_mask = df_long["sample_id"].isin(train_samples)
        for col in feature_cols_present:
            train_median = df_long.loc[train_mask, col].median()
            df_long[col] = df_long[col].fillna(train_median)
        
        # Normalize numeric features using training statistics
        X_train_numeric = df_long.loc[train_mask, feature_cols_present]
        self.scaler.fit(X_train_numeric)
        
        df_long[feature_cols_present] = self.scaler.transform(df_long[feature_cols_present])
        self.fitted = True
        
        # Create splits
        def get_split(sample_set):
            mask = df_long["sample_id"].isin(sample_set)
            X = df_long.loc[mask, self.feature_cols].copy()
            y = df_long.loc[mask, "rel_abundance"].copy()
            presence = df_long.loc[mask, "presence"].copy()
            meta = df_long.loc[mask, ["sample_id", "bin_uri"]].copy()
            return X, y, presence, meta
        
        X_train, y_train, presence_train, meta_train = get_split(train_samples)
        X_val, y_val, presence_val, meta_val = get_split(val_samples)
        X_test, y_test, presence_test, meta_test = get_split(test_samples)
        
        print(f"\nObservation counts:")
        print(f"  Train: {len(X_train)}")
        print(f"  Val: {len(X_val)}")
        print(f"  Test: {len(X_test)}")
        
        # Check zero inflation
        zero_train = (y_train == 0).sum()
        zero_val = (y_val == 0).sum()
        zero_test = (y_test == 0).sum()
        
        print(f"\nZero inflation:")
        print(f"  Train: {zero_train}/{len(y_train)} ({zero_train/len(y_train)*100:.1f}%)")
        print(f"  Val: {zero_val}/{len(y_val)} ({zero_val/len(y_val)*100:.1f}%)")
        print(f"  Test: {zero_test}/{len(y_test)} ({zero_test/len(y_test)*100:.1f}%)")
        
        metadata = {
            "train_meta": meta_train,
            "val_meta": meta_val,
            "test_meta": meta_test,
            "presence_train": presence_train,
            "presence_val": presence_val,
            "presence_test": presence_test,
            "feature_cols": self.feature_cols,
            "n_samples": n_samples,
            "n_bins": len(unique_bins),
            "zero_fraction_train": zero_train / len(y_train),
        }
        
        return X_train, X_val, X_test, y_train, y_val, y_test, metadata


def load_data(data_path: str, random_state: int = 42) -> Tuple:
    """Convenience function to load and preprocess data."""
    preprocessor = DataPreprocessor(random_state=random_state)
    return preprocessor.load_and_preprocess(data_path)
