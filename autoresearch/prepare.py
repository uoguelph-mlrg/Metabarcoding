import logging as log
import os
import pickle
import tempfile
from typing import Any, Dict, List, Literal, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train import Config

TIME_BUDGET = 300  # wall-clock training seconds (excluding eval/preprocessing overhead)

# Features to use in MLP (observation-level + computed bin-level)
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

PREPROCESSING_STATE_FILENAME = "preprocessing_state.pkl"


def _default_preprocessing_state_path(config: Config, filename: str = PREPROCESSING_STATE_FILENAME) -> str:
    data_dir = os.path.abspath(config.results_dir)
    return os.path.join(data_dir, filename)

def save_preprocessing_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(path)) as tmp:
        pickle.dump(state, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path = tmp.name
    os.replace(tmp_path, path)

def load_preprocessing_state(path: str) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        state = pickle.load(fh)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid preprocessing state in {path}: expected dict")
    return state


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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load or compute DNA sequence embeddings for BINs.

    Priority:
        1. Load from config.embedding_path if the file exists.
        2. Otherwise compute via BarcodeBERT using config.barcode_data_path and save to 
            config.embedding_path (if provided) so future runs skip inference.
        3. If neither path is usable, raise a descriptive error.

    Args:
        config: Configuration object with embedding_path, barcode_data_path, and device settings.
        bin_uris_ordered: List of bin URIs in the order they appear in taxonomy_df.

    Returns:
        Tuple containing:
        - embeddings: np.ndarray of shape [n_bins, emb_dim]
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
    preprocessing_state_path: Optional[str] = None,
    preprocessing_state_filename: str = PREPROCESSING_STATE_FILENAME,
) -> Tuple[
    Dict[str, Dict[str, Any]], pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray], 
    Dict[Any, int], Dict[Any, int], Dict[str, np.ndarray], str
]:
    """
    Load and preprocess the CSV data.

    Args:
        config: Configuration object with train_frac, val_frac
        save_data: Whether to save split CSVs to disk
        fixed_split_indices: Optional dict with 'train', 'val', 'test' keys containing sample 
            indices for reproducible splits across different calls
        preprocessing_state_path: Optional path to preprocessing state file; used for replay mode 
            when the file already exists. If missing, preprocessing is computed and the artifact is 
            written there.
        preprocessing_state_filename: Artifact filename used when writing default state

    Returns:
    Tuple containing:
        - splits: dict with 'train', 'val', 'test' keys mapping to dicts with 'X', 'y', 'sample_ids'
        - taxonomy_df: DataFrame with bin features and taxonomy or embedding info
        - embeddings_array: np.ndarray of shape [n_bins, emb_dim] if use_embedding else None
        - bins_with_embedding_arr: np.ndarray of shape [n_bins] bool indicating which bins have embeddings (if use_embedding) else None
        - bin_index: mapping bin_uri -> col index
        - sample_index: mapping sample_id -> row index
        - split_indices: dict with 'train', 'val', 'test' sample indices (for reuse)
        - preprocessing_state_path: path to the saved preprocessing state artifact 
    """
    replay_state: Optional[Dict[str, Any]] = None
    if preprocessing_state_path is not None and os.path.exists(preprocessing_state_path):
        replay_state = load_preprocessing_state(preprocessing_state_path)

    default_state_path = _default_preprocessing_state_path(config, preprocessing_state_filename)
    resolved_state_path = os.path.abspath(preprocessing_state_path) if preprocessing_state_path else default_state_path

    df = pd.read_csv(config.data_path)

    df = df.rename(columns={
        "sample-eventid": "sample_id"
    })

    ################################################################################################
    # Data curation
    ################################################################################################
    # Keep samples with enough total reads. Threshold = min(5th percentile, 50k) on raw read counts.
    if "total_reads_per_sample" in df.columns:
        sample_reads = pd.to_numeric(
            df.groupby("sample_id")["total_reads_per_sample"].first(),
            errors="coerce",
        )
    elif "total_reads" in df.columns:
        sample_reads = pd.to_numeric(
            df.groupby("sample_id")["total_reads"].sum(),
            errors="coerce",
        )
        log.warning(
            "Column 'total_reads_per_sample' not found; using summed 'total_reads' per sample for filtering."
        )
    else:
        sample_reads = pd.Series(dtype=float)
        log.warning(
            "No read-count column found for sample filtering ('total_reads_per_sample' or 'total_reads')."
        )

    if len(sample_reads) > 0:
        q05 = float(sample_reads.quantile(0.05))
        # Cap the read filter at 50k to avoid dropping a large tail of reasonably well-sequenced 
        # samples (in the cases where the dataset is well curated)
        reads_threshold = min(q05, 50000.0)
        kept_sample_ids = sample_reads[sample_reads >= reads_threshold].index
        dropped_samples = int((sample_reads < reads_threshold).sum())

        if len(kept_sample_ids) == 0:
            raise ValueError(
                f"Sample filtering removed all samples at threshold {reads_threshold:.2f}."
            )

        df = df[df["sample_id"].isin(kept_sample_ids)].copy()
        log.info(
            "Applied sample read-count filter: threshold=min(q05=%.2f, 50000)=%.2f; kept %d/%d samples; dropped %d.",
            q05,
            reads_threshold,
            len(kept_sample_ids),
            len(sample_reads),
            dropped_samples,
        )

    ################################################################################################
    # Feature engineering and preprocessing
    ################################################################################################
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
    
    ################################################################################################
    # Train/val/test split
    ################################################################################################
    if fixed_split_indices is not None:
        # Use fixed indices if provided for reproducibility across calls.
        train_sample_idx = fixed_split_indices["train"]
        val_sample_idx = fixed_split_indices["val"]
        test_sample_idx = fixed_split_indices["test"]
        if (
            train_sample_idx.max(initial=-1) >= n_samples
            or val_sample_idx.max(initial=-1) >= n_samples
            or test_sample_idx.max(initial=-1) >= n_samples
        ):
            log.warning(
                "Provided fixed_split_indices are incompatible with current filtered samples; falling back to random split."
            )
            fixed_split_indices = None
    elif config.remove_excess:
        # If removing excess samples from train/val, build test set only from excess samples and split the rest randomly.
        excess_samples = df[df["Excess"] > 0]["sample_id"].unique()
        non_excess_samples = df[df["Excess"] <= 0]["sample_id"].unique()
        
        n_non_excess = len(non_excess_samples)
        n_val = int(n_non_excess * config.val_frac)
        n_train = n_non_excess - n_val
        
        np.random.shuffle(non_excess_samples)
        train_sample_idx = np.array([sample_index[s] for s in non_excess_samples[:n_train]])
        val_sample_idx = np.array([sample_index[s] for s in non_excess_samples[n_train:n_train + n_val]])
        test_sample_idx = np.array([sample_index[s] for s in excess_samples])
        
    else:
        # WARNING: to be updated once we corrected the excess
        df = df[df["Excess"] <= 0]
        
        # Randomly split samples into train/val/test according to config fractions.
        sample_indices = np.arange(n_samples)
        np.random.shuffle(sample_indices)

        n_train = int(n_samples * config.train_frac)
        n_val = int(n_samples * config.val_frac)

        train_sample_idx = sample_indices[:n_train]
        val_sample_idx = sample_indices[n_train:n_train + n_val]
        test_sample_idx = sample_indices[n_train + n_val:]
    
    split_indices = {
        "train": train_sample_idx,
        "val": val_sample_idx,
        "test": test_sample_idx,
    }

    missing_features = [c for c in OBSERVATION_FEATURES if c not in df.columns]
    if missing_features:
        log.warning(f"Missing features in dataset: {', '.join(missing_features)}.")
    missing_features = [c for c in TAXONOMY_FEATURES if c not in df.columns]
    if config.use_taxonomy and missing_features:
        log.warning(f"Missing taxonomy features in dataset: {', '.join(missing_features)}.")

    # Build df_long with required columns + features
    base_cols = ["sample_id", "bin_uri", "occurrences", "rel_abundance"]
    if replay_state is not None:
        feature_cols_present = list(replay_state.get("feature_cols_present", []))
        if not feature_cols_present:
            raise ValueError("Preprocessing replay failed: missing 'feature_cols_present' in artifact")
        replay_missing = [c for c in feature_cols_present if c not in df.columns]
        if replay_missing:
            raise ValueError(
                "Preprocessing replay failed: expected feature columns are missing from input data: "
                + ", ".join(replay_missing)
            )
    else:
        feature_cols_present = [c for c in OBSERVATION_FEATURES if c in df.columns]
    df_long = df[base_cols + feature_cols_present].copy()

    # Ensure feature columns are numeric; non-numeric values become NaN and are imputed later.
    for col in feature_cols_present:
        if not pd.api.types.is_numeric_dtype(df_long[col]):
            log.warning(f"Feature column '{col}' is not numeric; attempting to coerce to numeric with NaN for invalid values.")
            df_long[col] = pd.to_numeric(df_long[col], errors="coerce")

    # Build taxonomy_df with taxonomy columns (if use_taxonomy) and/or embedding (if use_embedding)
    if config.use_taxonomy:
        taxonomy_df = df.groupby("bin_uri").first()[[c for c in TAXONOMY_FEATURES if c in df.columns]].reset_index()
    else:
        taxonomy_df = pd.DataFrame({"bin_uri": df["bin_uri"].unique()})

    # Ensure taxonomy_df is ordered by bin_index
    taxonomy_df["_idx"] = taxonomy_df["bin_uri"].map(bin_index)
    taxonomy_df = taxonomy_df.sort_values("_idx").drop(columns=["_idx"]).reset_index(drop=True)

    # Load or compute embeddings if needed
    embeddings_array: Optional[np.ndarray] = None
    bins_with_embedding_arr: Optional[np.ndarray] = None
    if config.use_embedding:
        embeddings_array, bins_with_embedding_arr = _load_or_compute_embeddings(config, taxonomy_df["bin_uri"].tolist())

    if replay_state is not None:
        train_feature_means = dict(replay_state.get("train_feature_means", {}))
        train_feature_stds = dict(replay_state.get("train_feature_stds", {}))
        bin_medians = pd.DataFrame(replay_state.get("bin_medians", {}))
        feature_medians = pd.Series(replay_state.get("feature_medians", {}))
        # Restore embeddings from replay state if available
        if config.use_embedding and "embeddings_dict" in replay_state:
            cached_emb_dict = replay_state.get("embeddings_dict", {})
            embeddings_array = np.zeros((len(taxonomy_df), len(next(iter(cached_emb_dict.values())))), dtype=np.float32)
            bins_with_embedding_arr = np.zeros(len(taxonomy_df), dtype=bool)
            for idx, uri in enumerate(taxonomy_df["bin_uri"]):
                if uri in cached_emb_dict:
                    embeddings_array[idx] = np.array(cached_emb_dict[uri], dtype=np.float32)
                    bins_with_embedding_arr[idx] = True
    else:
        # Fill missing numeric features with their median values given the BIN in the training set.
        X = df_long.loc[
            df_long["sample_id"].isin(set(unique_samples[train_sample_idx])), feature_cols_present + ["bin_uri"]
        ]
        X_features = X[feature_cols_present]
        train_feature_means = X_features.mean().to_dict()
        train_feature_stds = X_features.std(ddof=0).to_dict()
        bin_medians = X.groupby("bin_uri")[feature_cols_present].median()
        feature_medians = X_features.median()

    for col in feature_cols_present:
        median_map = dict(bin_medians.get(col, {}))
        # First pass: fill NaNs with the BIN-specific train-set median (vectorized).
        missing = df_long[col].isna()
        df_long.loc[missing, col] = df_long.loc[missing, "bin_uri"].map(median_map)
        # Second pass: global fallback for BINs with no usable train median.
        if col not in feature_medians:
            raise ValueError(f"Preprocessing replay failed: missing global median for feature '{col}'")
        df_long[col] = df_long[col].fillna(float(feature_medians[col]))

        if col not in train_feature_means or col not in train_feature_stds:
            raise ValueError(f"Preprocessing replay failed: missing mean/std for feature '{col}'")
        std = float(train_feature_stds[col])
        mean = float(train_feature_means[col])
        df_long[col] = (df_long[col] - mean) / (std + 1e-10)
        

    if replay_state is None:
        # Store the sample IDs corresponding to each split for reproducibility and downstream use.
        split_sample_ids = {
            "train": unique_samples[train_sample_idx].tolist(),
            "val": unique_samples[val_sample_idx].tolist(),
            "test": unique_samples[test_sample_idx].tolist(),
        }
        state = {
            "source_data_path": os.path.abspath(config.data_path),
            "feature_cols_present": feature_cols_present,
            "log_transform_columns": [
                c for c in ["total_reads_per_sample", "total_reads_norm", "avg_reads_norm", "max_reads_norm", "min_reads_norm"]
                if c in feature_cols_present
            ],
            "train_feature_means": train_feature_means,
            "train_feature_stds": train_feature_stds,
            "feature_medians": feature_medians.to_dict(),
            "bin_medians": bin_medians.to_dict(),
            "split_indices": {
                "train": train_sample_idx.astype(np.int64).tolist(),
                "val": val_sample_idx.astype(np.int64).tolist(),
                "test": test_sample_idx.astype(np.int64).tolist(),
            },
            "split_sample_ids": split_sample_ids,
            "sample_filter": {
                "enabled": len(sample_reads) > 0,
                "threshold": float(reads_threshold) if len(sample_reads) > 0 else None,
            },
        }
        # Add embeddings to state if computed
        if config.use_embedding and embeddings_array is not None:
            # Store embeddings dict for reproducibility on resume
            embeddings_dict = {
                uri: embeddings_array[idx].tolist()
                for idx, uri in enumerate(taxonomy_df["bin_uri"])
            }
            state["embeddings_dict"] = embeddings_dict
            state["embedding_path"] = config.embedding_path
            state["use_embedding"] = True
        save_preprocessing_state(resolved_state_path, state)

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
        data_path = os.path.dirname(config.data_path)

        for X, y, split in [(X_train,y_train,"train"), (X_val,y_val,"val"), (X_test,y_test,"test")]:
            X.to_csv(f"{data_path}/X_{split}.csv")
            pd.Series(y).to_csv(f"{data_path}/y_{split}.csv", index=False)
        taxonomy_df.to_csv(f"{data_path}/bins_data.csv", index=False)

    result = (
        {
            "train": {"X": X_train, "y": y_train, "y_prob": y_train},
            "val": {"X": X_val, "y": y_val, "y_prob": y_val},
            "test": {"X": X_test, "y": y_test, "y_prob": y_test},
        }, 
        taxonomy_df,                # flat DataFrame: bin_uri + taxonomy only
        embeddings_array,           # np.ndarray [n_bins, emb_dim] or None
        bins_with_embedding_arr,    # np.ndarray [n_bins] bool or None
        bin_index, 
        sample_index, 
        split_indices, 
        resolved_state_path
    )

    return result


def load_processed(
    data_dir: str
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, Dict[Any, int], Dict[Any, int], Dict[str, np.ndarray]]:
    """
    Load preprocessed splits saved by `load()` from a directory.

    Expected files in `data_dir`:
    - X_train.csv, X_val.csv, X_test.csv (MultiIndex: sample_id, bin_uri)
    - y_train.csv, y_val.csv, y_test.csv (single column, aligned to X_*.csv row order)
    - bins_data.csv (must include bin_uri and taxonomy or embedding columns)

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
            # Contract: if rel_abundance is absent, first column is treated as target.
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

    bins_path = os.path.join(data_dir, "bins_data.csv")
    taxonomy_df = pd.read_csv(bins_path)
    if "bin_uri" not in taxonomy_df.columns:
        raise ValueError(f"{bins_path} must contain 'bin_uri'")

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

    # Reorder taxonomy_df to match bin_index where possible
    taxonomy_df["_idx"] = taxonomy_df["bin_uri"].map(bin_index)
    taxonomy_df = taxonomy_df.sort_values("_idx").drop(columns=["_idx"]).reset_index(drop=True)

    # No split_indices available when loading from disk
    split_indices: Dict[str, np.ndarray] = {"train": np.array([]), "val": np.array([]), "test": np.array([])}

    return {
        "train": {"X": X_train, "y": y_train, "y_prob": y_train},
        "val": {"X": X_val, "y": y_val, "y_prob": y_val},
        "test": {"X": X_test, "y": y_test, "y_prob": y_test},
    }, taxonomy_df, bin_index, sample_index, split_indices


def load_or_preprocess(
    config: Config,
    fixed_split_indices: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, Dict[Any, int], Dict[Any, int], Dict[str, np.ndarray]]:
    """
    Fast path: if preprocessed splits already exist in the data directory, load them
    via load_processed() — skips all CSV parsing and feature engineering.
    Slow path: runs load() with save_data=True so subsequent calls take the fast path.

    The seven sentinel files checked are: X_train.csv, X_val.csv, X_test.csv,
    y_train.csv, y_val.csv, y_test.csv, bins_data.csv.

    NOTE: when loading from cache, fixed_split_indices is ignored — the split is
    already baked into the saved files. This is intentional: autoresearch runs must
    compare models on a fixed, reproducible split.
    """
    data_dir = os.path.dirname(config.data_path)
    sentinel_files = [
        os.path.join(data_dir, f"{prefix}.csv")
        for prefix in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "bins_data")
    ]
    if all(os.path.exists(p) for p in sentinel_files):
        log.info("load_or_preprocess: cached splits found — loading from disk (skipping preprocessing).")
        return load_processed(data_dir)
    else:
        log.info("load_or_preprocess: no cached splits found — running full preprocessing and saving.")
        return load(config, save_data=True, fixed_split_indices=fixed_split_indices)


class Loss:
    """
    Fixed loss functions for the metabarcoding autoresearch harness.

    This class lives in prepare.py (read-only to agents) to protect the evaluation
    harness from accidental modification. The loss *type* is still agent-tunable via
    Config.loss_type; only the implementation is locked here.

    Two modes:
    - "cross_entropy": Sample-level distributional loss.
        Input: logits [batch_size, n_bins], target: relative abundances (sums to 1).
        Computes -sum(target * log_softmax(logits)) per sample, averaged over batch.
        This equals KL divergence up to a constant (entropy of target).
    - "logistic": Bin-level BCE with logits (continuous targets in [0,1]).
    """

    def __init__(self, task: Literal["cross_entropy", "logistic"] = "cross_entropy"):
        self.task = task
        if task == "cross_entropy":
            pass
        elif task == "logistic":
            # BCEWithLogitsLoss accepts continuous targets in [0,1]: the loss
            # −[y·log σ(z) + (1−y)·log(1−σ(z))] is a valid cross-entropy over
            # a Bernoulli whose probability is y, so it is appropriate even when
            # targets are fractional relative abundances rather than hard 0/1 labels.
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown task {task}")

    def cross_entropy_soft_targets(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.dim() == 3 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 3 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        if mask is None:
            mask = (logits > float("-inf")).float()
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_safe = torch.where(mask.bool(), log_probs, torch.zeros_like(log_probs))
        return (-torch.sum(targets * log_probs_safe, dim=-1)).mean()

    def __call__(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.task == "cross_entropy":
            return self.cross_entropy_soft_targets(outputs, targets, mask)
        return self.criterion(outputs, targets)


def collate_samples(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for sample-level batching.
    
    Since each sample can have a different number of bins, we need to:
    1. Pad all samples to the same length (max bins in batch)
    2. Create a mask to ignore padded positions in loss computation
    
    Args:
        batch: List of dicts from MBDataset._get_sample(), each with:
            - input: [n_bins, n_features]
            - target: [n_bins] (relative abundances summing to 1)
            - bin_idx: [n_bins]
            - sample_idx: scalar
    
    Returns:
        Dict with padded tensors:
            - input: [batch_size, max_bins, n_features]
            - target: [batch_size, max_bins]
            - bin_idx: [batch_size, max_bins]
            - sample_idx: [batch_size]
            - mask: [batch_size, max_bins] (1 for valid, 0 for padded)
    """
    # Find max number of bins in this batch
    max_bins = max(item["input"].shape[0] for item in batch)
    n_features = batch[0]["input"].shape[1]
    batch_size = len(batch)
    
    # Initialize padded arrays
    inputs = np.zeros((batch_size, max_bins, n_features), dtype=np.float32)
    targets = np.zeros((batch_size, max_bins), dtype=np.float32)
    bin_indices = np.zeros((batch_size, max_bins), dtype=np.int64)
    sample_indices = np.zeros(batch_size, dtype=np.int64)
    masks = np.zeros((batch_size, max_bins), dtype=np.float32)
    
    for i, item in enumerate(batch):
        n_bins = item["input"].shape[0]
        inputs[i, :n_bins, :] = item["input"]
        targets[i, :n_bins] = item["target"]
        bin_indices[i, :n_bins] = item["bin_idx"]
        sample_indices[i] = item["sample_idx"]
        masks[i, :n_bins] = 1.0
    
    return {
        "input": torch.from_numpy(inputs),
        "target": torch.from_numpy(targets),
        "bin_idx": torch.from_numpy(bin_indices),
        "sample_idx": torch.from_numpy(sample_indices),
        "mask": torch.from_numpy(masks),
    }


class MBDataset(Dataset):
    """
    Lightweight dataset representing (sample, bin) pairs with normalized targets.
    Expects preprocessed wide matrix or long table.
    
    Two modes:
    - "sample": Returns all bins for a single sample. Use with collate_samples.
    Good for cross-entropy loss where bins within a sample form a distribution.
    - "bin": Returns individual (sample, bin) observations.
    Good for logistic/MSE loss on individual bins.
    """

    def __init__(
        self, 
        data: Dict[str, pd.DataFrame],
        bin_index: Dict[Any, int], 
        sample_index: Dict[Any, int],
        loss_mode: Literal["sample", "bin"] = "sample"
    ):
        """
        Args:
            data: Dict with 'X' (features DataFrame with MultiIndex) and 'y' (targets)
            bin_index: mapping bin_uri -> col index
            sample_index: mapping sample_id -> row index
            loss_mode: "sample" for cross-entropy, "bin" for logistic loss
        """
        self.bin_index = bin_index
        self.sample_index = sample_index
        self.loss_mode = loss_mode
        
        # Extract data arrays
        self.bin_uris = data["X"].index.get_level_values("bin_uri").map(bin_index).to_numpy(dtype=np.int64)
        self.sample_ids = data["X"].index.get_level_values("sample_id").map(sample_index).to_numpy(dtype=np.int64)
        self.X = data["X"].to_numpy(dtype=np.float32)
        self.y = data["y"].to_numpy(dtype=np.float32)
        
        # Build mapping from sample_idx to list of row indices in this split
        # Only include samples that actually appear in this data split
        self._sample_to_indices: Dict[int, np.ndarray] = {}
        unique_sample_ids = np.unique(self.sample_ids)
        for sample_idx in unique_sample_ids:
            self._sample_to_indices[sample_idx] = np.where(self.sample_ids == sample_idx)[0]
        
        # For sample mode, we iterate over samples present in this split
        self._sample_list = list(self._sample_to_indices.keys())
        
        if loss_mode == "sample":
            self._len, self._get = self._len_sample, self._get_sample
        elif loss_mode == "bin":
            self._len, self._get = self._len_bin, self._get_bin
        else:
            raise ValueError(f"Unknown loss_mode {loss_mode}")

    def __len__(self):
        return self._len()

    def __getitem__(self, idx):
        return self._get(idx)

    # -------------------- Sample mode --------------------
    # Returns all bins for one sample. Use with collate_samples for batching.
    
    def _len_sample(self):
        return len(self._sample_list)

    def _get_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get all bins for sample at position idx in the sample list.
        
        Returns:
            Dict with:
                - input: [n_bins, n_features] features for each bin
                - target: [n_bins] relative abundances (sum to 1)
                - bin_idx: [n_bins] bin indices
                - sample_idx: scalar sample index
        """
        sample_idx = self._sample_list[idx]
        indices = self._sample_to_indices[sample_idx]
        
        return {
            "input": self.X[indices],                           # [n_bins, n_features]
            "target": self.y[indices],                          # [n_bins] - relative abundances
            "bin_idx": self.bin_uris[indices],                  # [n_bins]
            "sample_idx": np.array(sample_idx, dtype=np.int64), # scalar
        }

    # -------------------- Bin mode --------------------
    # Returns individual (sample, bin) observations.
    
    def _len_bin(self):
        return len(self.X)

    def _get_bin(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get single observation at row idx.
        
        Returns:
            Dict with:
                - input: [n_features] features
                - target: scalar target value
                - bin_idx: scalar bin index
                - sample_idx: scalar sample index
        """
        return {
            "input": self.X[idx],
            "target": self.y[idx],
            "bin_idx": self.bin_uris[idx],
            "sample_idx": np.array(self.sample_ids[idx], dtype=np.int64),
        }

