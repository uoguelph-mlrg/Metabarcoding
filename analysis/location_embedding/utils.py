# Features to use in MLP (observation-level + computed bin-level)
from typing import Tuple, Dict, Any, Literal, Optional
import os
import pandas as pd
import numpy as np
from config import Config
import logging as log

from embed_location import add_location_embeddings, build_location_embedder, EmbedderSpec

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

LOCATION_RAW_FEATURES = ["latitude", "longitude"]

TAXONOMY_FEATURES = [
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "subfamily",
    "genus",
    "species"
]

def load(
    config: Config, 
    save_data: bool = True,
    fixed_split_indices: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, Dict[Any, int], Dict[Any, int], Dict[str, np.ndarray]]:
    """
    Load and preprocess the CSV data.

    Args:
        config: Configuration object with train_frac, val_frac
        save_data: Whether to save split CSVs to disk
        fixed_split_indices: Optional dict with 'train', 'val', 'test' keys containing
                            sample indices for reproducible splits across different calls

    Returns:
    Tuple containing:
        - splits: dict with 'train', 'val', 'test' keys mapping to dicts with 'X', 'y', 'sample_ids'
        - bins_df: DataFrame with bin features and taxonomy
        - bin_index: mapping bin_uri -> col index
        - sample_index: mapping sample_id -> row index
        - split_indices: dict with 'train', 'val', 'test' sample indices (for reuse)
    """
    df = pd.read_csv(config.data_path)

    # Rename columns to match expected format
    df = df.rename(columns={
        "sample-eventid": "sample_id"
    })

    # Parse date and extract day of year as numeric feature
    if "collection_start_date" in df.columns:
        df["collection_day"] = pd.to_datetime(df["collection_start_date"], format="%m/%d/%Y", errors="coerce").dt.dayofyear
        df["collection_day"] = df["collection_day"].fillna(0)
    else:
        df["collection_day"] = 0

    # Build index mappings
    unique_samples = df["sample_id"].unique()
    n_samples = len(unique_samples)
    sample_index = {s: i for i, s in enumerate(unique_samples)}
    unique_bins = df["bin_uri"].unique()
    bin_index = {b: i for i, b in enumerate(unique_bins)}

    # Normalize + log-transform occurences to get target y
    sample_totals = df.groupby("sample_id")["occurrences"].transform("sum")
    # Also store actual proportions for cross-entropy loss
    df["rel_abundance"] = df["occurrences"] / (sample_totals + 1e-10)

    # Apply logarithmic preprocessing (only log transform, no sample normalization)
    df["total_reads_per_sample"] = np.log1p(df["total_reads_per_sample"])
    df["total_reads_norm"] = np.log1p(df["total_reads"])
    df["avg_reads_norm"] = np.log1p(df["avg_reads"])
    df["max_reads_norm"] = np.log1p(df["max_reads"])
    df["min_reads_norm"] = np.log1p(df["min_reads"])

    # Resolve location embedding params: explicit kwargs take priority, then config attributes, then defaults
    def _cfg(attr, default):
        return getattr(config, attr, default)

    _use_loc_emb     = _cfg('use_location_embedding',    False)
    _emb_model       = _cfg('location_embedder_model',   'satclip')
    _keep_raw_gps    = _cfg('keep_raw_gps_features',     False)
    _emb_prefix      = _cfg('location_embedding_prefix', 'loc_emb')
    _emb_device      = _cfg('location_embedder_device',  'cpu')
    _emb_batch       = _cfg('location_embedder_batch_size', 2048)
    _satclip_ckpt    = _cfg('satclip_ckpt_path',         None)
    _range_db        = _cfg('range_db_path',             None)
    _range_model     = _cfg('range_model_name',          'RANGE+')
    _range_beta      = _cfg('range_beta',                0.5)
    _ae_year         = _cfg('alphaearth_year',           2024)
    _ae_scale        = _cfg('alphaearth_scale_meters',   10)
    _ae_project      = _cfg('alphaearth_project',        'metabarcoding-491221')

    feature_list = list(OBSERVATION_FEATURES)
    location_embedding_cols = []
    if _use_loc_emb:
        spec = EmbedderSpec(
            model_name=_emb_model,
            device=_emb_device,
            batch_size=_emb_batch,
            satclip_ckpt_path=_satclip_ckpt,
            range_db_path=_range_db,
            range_model_name=_range_model,
            range_beta=_range_beta,
            alphaearth_year=_ae_year,
            alphaearth_scale_meters=_ae_scale,
            alphaearth_project=_ae_project,
        )
        embedder = build_location_embedder(spec)
        df, location_embedding_cols = add_location_embeddings(
            df,
            embedder,
            lat_col="latitude",
            lon_col="longitude",
            prefix=_emb_prefix,
        )

        feature_list = [f for f in feature_list if f not in LOCATION_RAW_FEATURES]
        feature_list.extend(location_embedding_cols)
        if _keep_raw_gps:
            feature_list.extend(LOCATION_RAW_FEATURES)

        log.info(
            "Using location embeddings via %s (%d dims)",
            _emb_model,
            len(location_embedding_cols),
        )
    
    # Log a warning if any expected features are missing
    missing_features = [c for c in feature_list + TAXONOMY_FEATURES if c not in df.columns]
    if missing_features:
        log.warning(f"Missing features in dataset: {', '.join(missing_features)}.")

    # Build df_long with required columns + features
    base_cols = ["sample_id", "bin_uri", "occurrences", "rel_abundance"]
    feature_cols_present = [c for c in feature_list if c in df.columns]
    df_long = df[base_cols + feature_cols_present].copy()

    # Build taxonomic_data with taxonomy and features
    taxonomic_data = df.groupby("bin_uri").first()[[c for c in TAXONOMY_FEATURES if c in df.columns]].reset_index()

    # Ensure taxonomic_data is ordered by bin_index
    taxonomic_data["_idx"] = taxonomic_data["bin_uri"].map(bin_index)
    taxonomic_data = taxonomic_data.sort_values("_idx").drop(columns=["_idx"]).reset_index(drop=True)

    # Create train/val/test splits at sample level
    # Use fixed indices if provided for reproducibility across calls
    if fixed_split_indices is not None:
        train_sample_idx = fixed_split_indices["train"]
        val_sample_idx = fixed_split_indices["val"]
        test_sample_idx = fixed_split_indices["test"]
    else:
        sample_indices = np.arange(n_samples)
        np.random.shuffle(sample_indices)

        n_train = int(n_samples * config.train_frac)
        n_val = int(n_samples * config.val_frac)

        train_sample_idx = sample_indices[:n_train]
        val_sample_idx = sample_indices[n_train:n_train + n_val]
        test_sample_idx = sample_indices[n_train + n_val:]
    
    # Store split indices for reuse
    split_indices = {
        "train": train_sample_idx,
        "val": val_sample_idx,
        "test": test_sample_idx,
    }

    # Fill missing numeric features with bin-level medians from the training split.
    # Use vectorized operations to avoid extremely slow row-wise apply on wide feature sets
    # (e.g., RANGE embeddings add 1280 columns).
    X = df_long.loc[
        df_long["sample_id"].isin(set(unique_samples[train_sample_idx])), feature_cols_present + ["bin_uri"]
    ]
    bin_medians = X.groupby("bin_uri").median()
    for col in feature_cols_present:
        # Skip columns that are already complete to avoid unnecessary work.
        if not df_long[col].isna().any():
            continue
        if col not in bin_medians.columns:
            continue

        # Fill NaNs from bin-specific medians, then fall back to global median.
        df_long[col] = df_long[col].fillna(df_long["bin_uri"].map(bin_medians[col]))
        df_long[col] = df_long[col].fillna(df_long[col].median())
    
    # Normalize based on training set statistics
    for col in feature_cols_present:
        # Avoid division by zero by stabilizing the denominator (not by shifting the standardized feature)
        std = float(X[col].std(ddof=0))
        df_long[col] = (df_long[col] - float(X[col].mean())) / (std + 1e-10)

    # Get train, val, test data
    def compute_data_split(df_long, sample_idx):
        sample_set = set(unique_samples[sample_idx])
        mask = df_long["sample_id"].isin(sample_set)
        X = df_long.loc[mask, ["sample_id", "bin_uri"] + feature_cols_present]
        X = X.set_index(["sample_id", "bin_uri"])
        y_prob = df_long.loc[mask, "rel_abundance"]
        return X, y_prob
    X_train, y_train = compute_data_split(df_long, train_sample_idx)
    X_val, y_val = compute_data_split(df_long, val_sample_idx)
    X_test, y_test = compute_data_split(df_long, test_sample_idx)
    log.info(f"Loaded {len(df_long)} observations")
    log.info(f"  {len(unique_samples)} samples, {len(unique_bins)} bins")
    log.info(f"  Features: {len(feature_cols_present)} ({', '.join(feature_cols_present)})")
    log.info(f"  Train: {len(train_sample_idx)} samples ({100 * config.train_frac:.0f}%)")
    log.info(f"  Val: {len(val_sample_idx)} samples ({100 * config.val_frac:.0f}%)")
    log.info(f"  Test: {len(test_sample_idx)} samples ({100 * (1 - config.train_frac - config.val_frac):.0f}%)")

    return {
        "train": {"X": X_train, "y": y_train, "y_prob": y_train},
        "val": {"X": X_val, "y": y_val, "y_prob": y_val},
        "test": {"X": X_test, "y": y_test, "y_prob": y_test},
    }, taxonomic_data, bin_index, sample_index, split_indices


def load_processed(
    data_dir: str
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, Dict[Any, int], Dict[Any, int], Dict[str, np.ndarray]]:
    """
    Load preprocessed splits saved by `load()` from a directory.

    Expected files in `data_dir`:
      - X_train.csv, X_val.csv, X_test.csv (MultiIndex: sample_id, bin_uri)
      - y_train.csv, y_val.csv, y_test.csv (single column, aligned to X_*.csv row order)
      - taxonomic_data.csv (must include bin_uri and taxonomy columns)

    Returns the same objects as `load()`, except `split_indices` is empty
    (since the original sample-index permutation is not recoverable from files alone).
    """
    def _read_X(split: str) -> pd.DataFrame:
        path = os.path.join(data_dir, f"X_{split}.csv")
        X = pd.read_csv(path)
        if "sample_id" not in X.columns or "bin_uri" not in X.columns:
            raise ValueError(f"{path} must contain columns 'sample_id' and 'bin_uri'")
        X = X.set_index(["sample_id", "bin_uri"])
        return X

    def _read_y(split: str, n_rows: int) -> pd.Series:
        path = os.path.join(data_dir, f"y_{split}.csv")
        y_df = pd.read_csv(path)
        if y_df.shape[1] == 1:
            y = y_df.iloc[:, 0]
        else:
            # fallback: try a known column name
            y = y_df["rel_abundance"] if "rel_abundance" in y_df.columns else y_df.iloc[:, 0]
        if len(y) != n_rows:
            raise ValueError(f"{path} has {len(y)} rows but X_{split}.csv has {n_rows}")
        return y.astype(np.float32)

    X_train = _read_X("train")
    X_val = _read_X("val")
    X_test = _read_X("test")

    y_train = _read_y("train", len(X_train))
    y_val = _read_y("val", len(X_val))
    y_test = _read_y("test", len(X_test))

    tax_path = os.path.join(data_dir, "taxonomic_data.csv")
    bins_df = pd.read_csv(tax_path)
    if "bin_uri" not in bins_df.columns:
        raise ValueError(f"{tax_path} must contain 'bin_uri'")

    # Build index mappings from the processed splits (ensures consistency)
    unique_samples = pd.Index(
        pd.concat([
            X_train.reset_index()["sample_id"],
            X_val.reset_index()["sample_id"],
            X_test.reset_index()["sample_id"],
        ]).unique()
    )
    sample_index = {s: i for i, s in enumerate(unique_samples)}

    unique_bins = pd.Index(
        pd.concat([
            X_train.reset_index()["bin_uri"],
            X_val.reset_index()["bin_uri"],
            X_test.reset_index()["bin_uri"],
        ]).unique()
    )
    bin_index = {b: i for i, b in enumerate(unique_bins)}

    # Reorder bins_df to match bin_index where possible
    bins_df["_idx"] = bins_df["bin_uri"].map(bin_index)
    bins_df = bins_df.sort_values("_idx").drop(columns=["_idx"]).reset_index(drop=True)

    # No split_indices available when loading from disk
    split_indices: Dict[str, np.ndarray] = {"train": np.array([]), "val": np.array([]), "test": np.array([])}

    return {
        "train": {"X": X_train, "y": y_train, "y_prob": y_train},
        "val": {"X": X_val, "y": y_val, "y_prob": y_val},
        "test": {"X": X_test, "y": y_test, "y_prob": y_test},
    }, bins_df, bin_index, sample_index, split_indices