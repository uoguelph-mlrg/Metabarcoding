# Features to use in MLP (observation-level + computed bin-level)
from re import L
from typing import Tuple, Dict, Any, Literal, Optional, List
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
import logging as log

OBSERVATION_FEATURES = [
    # Observed features (already in dataset) 
    "total_reads_per_sample",
    "repl_w_reads_fractn",
    "latitude", 
    "longitude",
    #"Excess",
    #"Bulk_Sample_wet_weight",
    #"SumExcessSpecimens",
    #"ExcessNumberTaxa",
    #"length_min_mm", 
    #"length_max_mm",
    # Computed bin-level features
    "collection_day",   # derived from collection_start_date
    "total_reads_norm", # total_reads normalized by sample total
    "avg_reads_norm",   # avg_reads normalized by sample total
    "max_reads_norm",   # max_reads normalized by sample total
    "min_reads_norm",   # min_reads normalized by sample total
]

TAXONOMY_FEATURES = [
    "species",
    "genus",
    #"subfamily",
    "family",
    "order",
    "class",
    "phylum",
    #"kingdom",
]

def _compute_barcodebert_embeddings(
    config: Config,
    barcode_data_path: str,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Run BarcodeBERT inference on sequences in barcode_data_path and return a
    dict mapping bin_uri -> mean-pooled embedding vector (numpy float32).

    The function uses mean-pooling of the last hidden state across all token
    positions (recommended by the BarcodeBERT authors).

    Args:
        config: Configuration object with device settings.
        barcode_data_path: Path to TSV/CSV with 'bin_uri' and 'seq' columns.
        batch_size: Number of sequences per inference batch.

    Returns:
        Dict[str, np.ndarray]: {bin_uri: np.ndarray of shape [hidden_dim]}
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    MODEL_NAME = "emmabhl/BarcodeBERT_finetuned"
    log.info(f"Loading BarcodeBERT from HuggingFace ({MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    device = torch.device(config.device)
    model = model.to(device).eval()

    # Read data: one consensus sequence per BIN
    sep = "\t" if barcode_data_path.endswith(".tsv") else ","
    df = pd.read_csv(barcode_data_path, sep=sep)
    if "bin_uri" not in df.columns or "seq" not in df.columns:
        raise ValueError(
            f"{barcode_data_path} must contain 'bin_uri' and 'seq' columns. "
            f"Found: {list(df.columns)}"
        )

    # Aggregate: take the first (consensus) sequence per BIN
    bin_seqs = df.groupby("bin_uri")["seq"].first().to_dict()
    uris = list(bin_seqs.keys())
    sequences = [bin_seqs[u] for u in uris]

    log.info(f"Running BarcodeBERT inference on {len(sequences)} BINs (batch_size={batch_size}) ...")

    emb_dict: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start : start + batch_size]
            batch_uris = uris[start : start + batch_size]

            # KmerTokenizer is single-sequence only: encode each one individually
            # and stack — safe because padding makes all outputs the same length
            batch_input_ids = []
            batch_attention_mask = []
            for seq in batch_seqs:
                encoded = tokenizer(seq, padding=True)
                batch_input_ids.append(encoded["input_ids"])
                batch_attention_mask.append(encoded["attention_mask"])

            input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(device)
            attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state: [B, seq_len, hidden_dim]
            last_hidden = outputs.last_hidden_state
            # Mean-pool over non-padding token positions
            mask_exp = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            sum_hidden = (last_hidden * mask_exp).sum(dim=1)  # [B, hidden_dim]
            count = mask_exp.sum(dim=1).clamp(min=1e-9)  # [B, 1]
            mean_pooled = (sum_hidden / count).cpu().numpy()  # [B, hidden_dim]

            for uri, emb in zip(batch_uris, mean_pooled):
                emb_dict[str(uri)] = emb.astype(np.float32)

            if (start // batch_size) % 10 == 0:
                log.debug(f"  Processed {start + len(batch_seqs)}/{len(sequences)} sequences")

    log.info("BarcodeBERT inference complete.")
    return emb_dict


def _load_or_compute_embeddings(
    config: Config,
    bin_uris_ordered: List[str],
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Load or compute DNA sequence embeddings for BINs.

    Priority:
        1. Load from config.embedding_path if the file exists.
        2. Otherwise compute via BarcodeBERT using config.barcode_data_path and save to 
           config.embedding_path (if provided) so future runs skip inference.
        3. If neither path is usable, raise a descriptive error.

    Args:
        config: Configuration object with embedding_path, barcode_data_path, and device settings.
        bin_uris_ordered: List of bin URIs in the order they appear in bins_df.

    Returns:
        Tuple containing:
        - embeddings: np.ndarray of shape [n_bins, emb_dim] or None if no embeddings loaded
        - bins_with_embedding: np.ndarray of shape [n_bins] (bool) indicating which bins have valid embeddings
    """
    embedding_path = config.embedding_path
    barcode_data_path = config.barcode_data_path
    n_bins = len(bin_uris_ordered)

    emb_dict: Dict[str, np.ndarray] = {}  # bin_uri -> np.ndarray

    if embedding_path is not None and os.path.exists(embedding_path):
        log.info(f"Loading precomputed embeddings from {embedding_path}")
        emb_dict = np.load(embedding_path, allow_pickle=True).item()
    elif barcode_data_path is not None:
        log.info(
            f"Precomputed embeddings not found; running BarcodeBERT inference "
            f"on {barcode_data_path}"
        )
        emb_dict = _compute_barcodebert_embeddings(config, barcode_data_path)
        # Cache to disk for future runs
        if embedding_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(embedding_path)), exist_ok=True)
            np.save(embedding_path, np.array(emb_dict, dtype=object), allow_pickle=True)
            log.info(f"Saved computed embeddings to {embedding_path}")
    else:
        raise ValueError(
            "use_embedding=True requires at least one of:\n"
            "  config.embedding_path  — path to a precomputed .npy embedding file\n"
            "  config.barcode_data_path — path to a TSV with 'bin_uri' and 'seq' columns"
        )

    # Determine embedding dimension from first available vector
    emb_dim = next(iter(emb_dict.values())).shape[0]
    embeddings = np.zeros((n_bins, emb_dim), dtype=np.float32)
    bins_with_embedding = np.zeros(n_bins, dtype=bool)

    for idx, uri in enumerate(bin_uris_ordered):
        if uri in emb_dict:
            embeddings[idx] = emb_dict[uri].astype(np.float32)
            bins_with_embedding[idx] = True

    n_missing = int((~bins_with_embedding).sum())
    n_present = int(bins_with_embedding.sum())
    log.info(
        f"Embeddings loaded: {n_present}/{n_bins} bins have sequences; "
        f"{n_missing} will use taxonomy fallback."
    )

    return embeddings, bins_with_embedding


def load(
    config: Config, 
    save_data: bool = True,
    fixed_split_indices: Optional[Dict[str, np.ndarray]] = None,
    read_count_preprocessing: str = "original"
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, Dict[Any, int], Dict[Any, int], Dict[str, np.ndarray]]:
    """
    Load and preprocess the CSV data.

    Args:
        config: Configuration object with train_frac, val_frac
        save_data: Whether to save split CSVs to disk
        fixed_split_indices: Optional dict with 'train', 'val', 'test' keys containing
                            sample indices for reproducible splits across different calls
        read_count_preprocessing: One of "original" (no preprocessing), "normalized" (normalize per
                                sample), or "logarithm" (only apply log transform)

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

    # Apply read count preprocessing based on the specified method
    if read_count_preprocessing == "original":
        # Normalize by sample + log transform (original method)
        df["total_reads_norm"] = df["total_reads"]
        df["avg_reads_norm"] = df["avg_reads"]
        df["max_reads_norm"] = df["max_reads"]
        df["min_reads_norm"] = df["min_reads"]
    elif read_count_preprocessing == "normalized":
        # Only normalize by sample (no log)
        df["total_reads_norm"] = df["total_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2
        df["avg_reads_norm"] = df["avg_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2
        df["max_reads_norm"] = df["max_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2
        df["min_reads_norm"] = df["min_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2
    elif read_count_preprocessing == "logarithm":
        # Only log transform (no sample normalization)
        df["total_reads_per_sample"] = np.log1p(df["total_reads_per_sample"])
        df["total_reads_norm"] = np.log1p(df["total_reads"])
        df["avg_reads_norm"] = np.log1p(df["avg_reads"])
        df["max_reads_norm"] = np.log1p(df["max_reads"])
        df["min_reads_norm"] = np.log1p(df["min_reads"])
    else:
        raise ValueError(f"Unknown read_count_preprocessing: {read_count_preprocessing}. "
                        f"Must be one of: 'original', 'normalized', 'logarithm'")
    # Log a warning if any expected features are missing
    missing_features = [c for c in OBSERVATION_FEATURES + TAXONOMY_FEATURES if c not in df.columns]
    if missing_features:
        log.warning(f"Missing features in dataset: {', '.join(missing_features)}.")

    # Build df_long with required columns + features
    base_cols = ["sample_id", "bin_uri", "occurrences", "rel_abundance"]
    feature_cols_present = [c for c in OBSERVATION_FEATURES if c in df.columns]
    df_long = df[base_cols + feature_cols_present].copy()

    # Build bins_df with taxonomy columns (if use_taxonomy) and/or embedding (if use_embedding)
    if config.use_taxonomy:
        bins_df = df.groupby("bin_uri").first()[[c for c in TAXONOMY_FEATURES if c in df.columns]].reset_index()
    else:
        bins_df = pd.DataFrame({"bin_uri": df["bin_uri"].unique()})

    # Ensure bins_df is ordered by bin_index
    bins_df["_idx"] = bins_df["bin_uri"].map(bin_index)
    bins_df = bins_df.sort_values("_idx").drop(columns=["_idx"]).reset_index(drop=True)

    # Load or compute embeddings if needed
    embeddings_array: Optional[np.ndarray] = None
    bins_with_embedding_arr: Optional[np.ndarray] = None
    if config.use_embedding:
        embeddings_array, bins_with_embedding_arr = _load_or_compute_embeddings(config, bins_df["bin_uri"].tolist())
        # Add embedding columns to bins_df
        bins_df["embedding"] = [embeddings_array[i] for i in range(len(bins_df))]
        bins_df["has_embedding"] = bins_with_embedding_arr

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

    # Fill missing numeric features with their median values given the BIN in the training set
    X = df_long.loc[
        df_long["sample_id"].isin(set(unique_samples[train_sample_idx])), feature_cols_present + ["bin_uri"]
    ]
    bin_medians = X.groupby("bin_uri").median()
    for col in feature_cols_present:
        if col not in bin_medians.columns:
            continue
        median_map = bin_medians[col].to_dict()
        df_long[col] = df_long.apply(
            lambda row: median_map.get(row["bin_uri"], np.nan) if pd.isna(row[col]) else row[col],
            axis=1
        )
        # Now fill any remaining missing values with overall median
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
    
    # Save the data splits in the `data` folder
    if save_data:
        data_dir = os.path.dirname(config.data_path)
        # Create subdirectory for this preprocessing method
        save_dir = os.path.join(data_dir, config.read_count_preprocessing)
        os.makedirs(save_dir, exist_ok=True)

        for X, y, split in [(X_train,y_train,"train"), (X_val,y_val,"val"), (X_test,y_test,"test")]:
            X.to_csv(f"{save_dir}/X_{split}.csv")
            pd.Series(y).to_csv(f"{save_dir}/y_{split}.csv", index=False)
        bins_df.to_csv(f"{save_dir}/bins_data.csv", index=False)

    return {
        "train": {"X": X_train, "y": y_train, "y_prob": y_train},
        "val": {"X": X_val, "y": y_val, "y_prob": y_val},
        "test": {"X": X_test, "y": y_test, "y_prob": y_test},
    }, bins_df, bin_index, sample_index, split_indices


def load_processed(
    data_dir: str
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, Dict[Any, int], Dict[Any, int], Dict[str, np.ndarray]]:
    """
    Load preprocessed splits saved by `load()` from a directory.

    Expected files in `data_dir`:
      - X_train.csv, X_val.csv, X_test.csv (MultiIndex: sample_id, bin_uri)
      - y_train.csv, y_val.csv, y_test.csv (single column, aligned to X_*.csv row order)
      - bins_data.csv (must include bin_uri and taxonomy columns)

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

    bin_data_path = os.path.join(data_dir, "bins_data.csv")
    bins_df = pd.read_csv(bin_data_path)
    if "bin_uri" not in bins_df.columns:
        raise ValueError(f"{bin_data_path} must contain 'bin_uri'")

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


if __name__ == "__main__":
    """Generate datasets with different read count preprocessing methods."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate datasets with different read count preprocessing")
    parser.add_argument("--seed", type=int, default=14, help="Random seed")
    args = parser.parse_args()
    
    # Setup logging
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    # Create config
    cfg = Config()
    
    # Generate datasets with all three preprocessing methods
    preprocessing_methods = ["original", "normalized", "logarithm"]
    
    log.info("="*70)
    log.info("GENERATING DATASETS WITH DIFFERENT PREPROCESSING METHODS")
    log.info("="*70)
    
    # First pass: generate the split indices
    log.info("\nGenerating split indices...")
    _, _, _, _, split_indices = load(
        cfg, 
        save_data=False,
        read_count_preprocessing="original"
    )
    
    # Second pass: generate datasets using the same split indices
    for method in preprocessing_methods:
        log.info(f"\n{'='*70}")
        log.info(f"Preprocessing method: {method.upper()}")
        log.info(f"{'='*70}")
        
        _ = load(
            cfg,
            save_data=True,
            fixed_split_indices=split_indices,
            read_count_preprocessing=method
        )
        
        log.info(f"✓ Saved {method} dataset to data/{method}/")
    
    log.info("\n" + "="*70)
    log.info("DATASET GENERATION COMPLETE")
    log.info("="*70)
    log.info("\nGenerated datasets:")
    for method in preprocessing_methods:
        log.info(f"  - data/{method}/")
