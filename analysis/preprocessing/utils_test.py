# Features to use in MLP (observation-level + computed bin-level)
from typing import Tuple, Dict, Any, Literal, Optional, List
import os
import sys
import pickle
import shutil
import tempfile
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
    "total_reads",
    "avg_reads",
    "max_reads",
    "min_reads",
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
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Run BarcodeBERT inference on sequences in config.data_path and return a
    dict mapping bin_uri -> mean-pooled embedding vector (numpy float32).

    The function uses mean-pooling of the last hidden state across all token
    positions (recommended by the BarcodeBERT authors).

    Args:
        config: Configuration object with data_path and device settings.
        batch_size: Number of sequences per inference batch.

    Returns:
        Dict[str, np.ndarray]: {bin_uri: np.ndarray of shape [hidden_dim]}
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    MODEL_NAME = getattr(config, "barcode_hf_model", "emmabhl/BarcodeBERT_finetuned")
    log.info(f"Loading BarcodeBERT from HuggingFace ({MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    device = torch.device(config.device)
    model = model.to(device).eval()

    # Read data: one consensus sequence per BIN
    data_path = config.data_path
    sep = "\t" if data_path.endswith(".tsv") else ","
    df = pd.read_csv(data_path, sep=sep)
    if "bin_uri" not in df.columns or "seq" not in df.columns:
        raise ValueError(
            f"{data_path} must contain 'bin_uri' and 'seq' columns. "
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
    Compute DNA sequence embeddings for BINs via BarcodeBERT.

    Caching to disk is handled by the preprocessed_dir mechanism in load().

    Args:
        config: Configuration object with data_path and device settings.
        bin_uris_ordered: List of bin URIs in the order they appear in taxonomy_df.

    Returns:
        Tuple containing:
        - embeddings: np.ndarray of shape [n_bins, emb_dim]
        - bins_with_embedding: np.ndarray of shape [n_bins] (bool) indicating which bins have valid embeddings
    """
    n_bins = len(bin_uris_ordered)
    log.info(f"Running BarcodeBERT inference on {config.data_path}")
    emb_dict = _compute_barcodebert_embeddings(config)

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


def _save_to_preprocessed_dir(
    out_dir: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    taxonomy_df: pd.DataFrame,
    embeddings_array: Optional[np.ndarray],
    bins_with_embedding_arr: Optional[np.ndarray],
    bin_index: Dict[Any, int],
    sample_index: Dict[Any, int],
    split_indices: Dict[str, np.ndarray],
    state_path: str,
) -> None:
    """Save all preprocessed artifacts to out_dir and write a sentinel file last."""
    os.makedirs(out_dir, exist_ok=True)
    sentinel = os.path.join(out_dir, "_complete")
    if os.path.exists(sentinel):
        os.remove(sentinel)

    for X, y, split in [(X_train, y_train, "train"), (X_val, y_val, "val"), (X_test, y_test, "test")]:
        X.to_csv(os.path.join(out_dir, f"X_{split}.csv"))
        pd.Series(y).to_csv(os.path.join(out_dir, f"y_{split}.csv"), index=False)
    taxonomy_df.to_csv(os.path.join(out_dir, "bins_data.csv"), index=False)

    emb = embeddings_array if embeddings_array is not None else np.array([], dtype=np.float32)
    np.save(os.path.join(out_dir, "embeddings_array.npy"), emb)
    bwe = bins_with_embedding_arr if bins_with_embedding_arr is not None else np.array([], dtype=bool)
    np.save(os.path.join(out_dir, "bins_with_embedding_arr.npy"), bwe)

    with open(os.path.join(out_dir, "bin_index.pkl"), "wb") as f:
        pickle.dump(bin_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out_dir, "sample_index.pkl"), "wb") as f:
        pickle.dump(sample_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out_dir, "split_indices.pkl"), "wb") as f:
        pickle.dump(split_indices, f, protocol=pickle.HIGHEST_PROTOCOL)

    shutil.copy2(state_path, os.path.join(out_dir, PREPROCESSING_STATE_FILENAME))
    open(sentinel, "w").close()
    log.info(f"Saved complete preprocessed cache to {out_dir}")


def _load_from_preprocessed_dir(
    cache_dir: str,
) -> Tuple[
    Dict[str, Dict[str, Any]], pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray],
    Dict[Any, int], Dict[Any, int], Dict[str, np.ndarray], str
]:
    """Load all preprocessing artifacts from a directory written by _save_to_preprocessed_dir."""
    def _read_X(split: str) -> pd.DataFrame:
        X = pd.read_csv(os.path.join(cache_dir, f"X_{split}.csv"))
        return X.set_index(["sample_id", "bin_uri"])

    def _read_y(split: str) -> pd.Series:
        y_df = pd.read_csv(os.path.join(cache_dir, f"y_{split}.csv"))
        return y_df.iloc[:, 0].astype(np.float32)

    X_train = _read_X("train")
    X_val   = _read_X("val")
    X_test  = _read_X("test")
    y_train = _read_y("train")
    y_val   = _read_y("val")
    y_test  = _read_y("test")

    taxonomy_df = pd.read_csv(os.path.join(cache_dir, "bins_data.csv"))

    emb_arr = np.load(os.path.join(cache_dir, "embeddings_array.npy"))
    embeddings_array = emb_arr if emb_arr.size > 0 else None

    bwe_arr = np.load(os.path.join(cache_dir, "bins_with_embedding_arr.npy"))
    bins_with_embedding_arr = bwe_arr if bwe_arr.size > 0 else None

    with open(os.path.join(cache_dir, "bin_index.pkl"), "rb") as f:
        bin_index = pickle.load(f)
    with open(os.path.join(cache_dir, "sample_index.pkl"), "rb") as f:
        sample_index = pickle.load(f)
    with open(os.path.join(cache_dir, "split_indices.pkl"), "rb") as f:
        split_indices = pickle.load(f)

    state_path = os.path.join(cache_dir, PREPROCESSING_STATE_FILENAME)

    return (
        {
            "train": {"X": X_train, "y": y_train, "y_prob": y_train},
            "val":   {"X": X_val,   "y": y_val,   "y_prob": y_val},
            "test":  {"X": X_test,  "y": y_test,  "y_prob": y_test},
        },
        taxonomy_df,
        embeddings_array,
        bins_with_embedding_arr,
        bin_index,
        sample_index,
        split_indices,
        state_path,
    )


def load(
    config: Config,
    fixed_split_indices: Optional[Dict[str, np.ndarray]] = None,
    read_count_preprocessing: Literal["original", "normalized", "logarithm"] = "original",
    preprocessing_state_path: Optional[str] = None,
    preprocessing_state_filename: str = PREPROCESSING_STATE_FILENAME,
    preprocessed_dir: Optional[str] = None,
) -> Tuple[
    Dict[str, Dict[str, Any]], pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray],
    Dict[Any, int], Dict[Any, int], Dict[str, np.ndarray], str
]:
    """
    Load and preprocess the CSV data.

    Args:
        config: Configuration object with train_frac, val_frac
        fixed_split_indices: Optional dict with 'train', 'val', 'test' keys containing sample
            indices for reproducible splits across different calls
        read_count_preprocessing: One of "original" (no preprocessing), "normalized" (normalize per
                                sample), or "logarithm" (only apply log transform)
        preprocessing_state_path: Optional path to preprocessing state file; used for replay mode
            when the file already exists. If missing, preprocessing is computed and the artifact is
            written there.
        preprocessing_state_filename: Artifact filename used when writing default state
        preprocessed_dir: If set, save all artifacts here on first run and fast-path load on
            subsequent runs (skips CSV parsing and BarcodeBERT inference).

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
    if preprocessed_dir is not None:
        sentinel = os.path.join(preprocessed_dir, "_complete")
        if os.path.exists(sentinel):
            log.info(f"load(): cache hit at {preprocessed_dir} — loading from disk, skipping preprocessing.")
            return _load_from_preprocessed_dir(preprocessed_dir)

    replay_state: Optional[Dict[str, Any]] = None
    if preprocessing_state_path is not None and os.path.exists(preprocessing_state_path):
        replay_state = load_preprocessing_state(preprocessing_state_path)

    default_state_path = _default_preprocessing_state_path(config, preprocessing_state_filename)
    resolved_state_path = os.path.abspath(preprocessing_state_path) if preprocessing_state_path else default_state_path

    df = pd.read_csv(config.data_path)

    df = df.rename(columns={
        "sample": "sample_id"
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

    # Normalize + log-transform occurences to get target y
    sample_totals = df.groupby("sample_id")["occurrences"].transform("sum")
    # Also store actual proportions for cross-entropy loss
    df["rel_abundance"] = df["occurrences"] / (sample_totals + 1e-10)

    # Apply read count preprocessing based on the specified method
    if read_count_preprocessing == "original":
        # No preprocessing; keep raw read counts as-is
        pass
    elif read_count_preprocessing == "normalized":
        # Only normalize by sample (no log)
        df["total_reads"] = df["total_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2
        df["avg_reads"] = df["avg_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2
        df["max_reads"] = df["max_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2
        df["min_reads"] = df["min_reads"] / (df["total_reads_per_sample"] + 1e-10) * 1e2
    elif read_count_preprocessing == "logarithm":
        # Only log transform (no sample normalization)
        df["total_reads_per_sample"] = np.log1p(df["total_reads_per_sample"])
        df["total_reads"] = np.log1p(df["total_reads"])
        df["avg_reads"] = np.log1p(df["avg_reads"])
        df["max_reads"] = np.log1p(df["max_reads"])
        df["min_reads"] = np.log1p(df["min_reads"])
    else:
        raise ValueError(f"Unknown read_count_preprocessing: {read_count_preprocessing}. "
                        f"Must be one of: 'original', 'normalized', 'logarithm'")

    # Build sample index mappings before splitting so split sizes are computed correctly.
    unique_samples = df["sample_id"].unique()
    n_samples = len(unique_samples)
    sample_index = {s: i for i, s in enumerate(unique_samples)}
    
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

    unique_bins = df["bin_uri"].unique()
    bin_index = {b: i for i, b in enumerate(unique_bins)}

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

    ################################################################################################
    # Location embedding (optional)
    ################################################################################################
    location_embedding_cols: List[str] = []
    if replay_state is None and getattr(config, "location_embedder", None) is not None:
        from embed_location import build_location_embedder, add_location_embeddings, EmbedderSpec
        _emb_model = config.location_embedder
        _cache_path: Optional[str] = None
        if getattr(config, "preprocessed_dir", None) is not None:
            _cache_path = os.path.join(
                os.path.abspath(config.preprocessed_dir),
                f"loc_emb_{_emb_model}.npy",
            )
            _cache_coords_path = _cache_path.replace(".npy", "_coords.npy")

        if _cache_path is not None and os.path.exists(_cache_path) and os.path.exists(_cache_coords_path):
            cached_embs = np.load(_cache_path)
            cached_coords = np.load(_cache_coords_path)
            log.info("Loaded location embeddings from cache: %s", _cache_path)
            # Re-join cached embeddings to df rows by matching lat/lon
            n_emb_dims = cached_embs.shape[1]
            location_embedding_cols = [f"loc_emb_{i:03d}" for i in range(n_emb_dims)]
            coord_to_idx = {tuple(row): i for i, row in enumerate(cached_coords)}
            lat = df["latitude"].to_numpy()
            lon = df["longitude"].to_numpy()
            row_embs = np.array([
                cached_embs[coord_to_idx.get((lat[i], lon[i]), -1)]
                if (lat[i], lon[i]) in coord_to_idx else np.zeros(n_emb_dims, dtype=np.float32)
                for i in range(len(df))
            ], dtype=np.float32)
            for j, col in enumerate(location_embedding_cols):
                df[col] = row_embs[:, j]
        else:
            spec = EmbedderSpec(
                model_name=_emb_model,
                device=config.device,
                satclip_ckpt_path=getattr(config, "satclip_ckpt_path", None),
                range_db_path=getattr(config, "range_db_path", None),
            )
            embedder = build_location_embedder(spec)
            df, location_embedding_cols = add_location_embeddings(
                df, embedder, lat_col="latitude", lon_col="longitude"
            )
            log.info("Computed location embeddings via %s (%d dims)", _emb_model, len(location_embedding_cols))
            if _cache_path is not None:
                unique_coords = df[["latitude", "longitude"]].drop_duplicates().to_numpy(dtype=np.float64)
                coord_to_idx = {tuple(row): i for i, row in enumerate(unique_coords)}
                n_emb_dims = len(location_embedding_cols)
                uniq_embs = np.zeros((len(unique_coords), n_emb_dims), dtype=np.float32)
                for j, col in enumerate(location_embedding_cols):
                    for k, coord in enumerate(unique_coords):
                        rows_with_coord = (df["latitude"] == coord[0]) & (df["longitude"] == coord[1])
                        uniq_embs[k, j] = df.loc[rows_with_coord, col].iloc[0]
                os.makedirs(os.path.dirname(_cache_path), exist_ok=True)
                np.save(_cache_path, uniq_embs)
                np.save(_cache_coords_path, unique_coords)
                log.info("Saved location embedding cache to %s", _cache_path)

    # Build feature list: when using location embedder, replace raw GPS with embedding cols
    LOCATION_RAW_FEATURES = ["latitude", "longitude"]
    base_feature_list = list(OBSERVATION_FEATURES)
    if location_embedding_cols:
        base_feature_list = [f for f in OBSERVATION_FEATURES if f not in LOCATION_RAW_FEATURES]
        base_feature_list.extend(location_embedding_cols)
        if getattr(config, "keep_raw_gps_features", False):
            base_feature_list.extend(LOCATION_RAW_FEATURES)
    elif replay_state is not None and getattr(config, "location_embedder", None) is not None:
        # Replay with location embedder: re-run embedding to recreate columns, then let
        # feature_cols_present from state take over (which already includes loc_emb_* cols).
        from embed_location import build_location_embedder, add_location_embeddings, EmbedderSpec
        spec = EmbedderSpec(
            model_name=config.location_embedder,
            device=config.device,
            satclip_ckpt_path=getattr(config, "satclip_ckpt_path", None),
            range_db_path=getattr(config, "range_db_path", None),
        )
        embedder = build_location_embedder(spec)
        df, location_embedding_cols = add_location_embeddings(
            df, embedder, lat_col="latitude", lon_col="longitude"
        )

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
        feature_cols_present = [c for c in base_feature_list if c in df.columns]
    df_long = df[base_cols + feature_cols_present].copy()

    # Ensure feature columns are numeric; non-numeric values become NaN and are imputed later.
    for col in feature_cols_present:
        if not pd.api.types.is_numeric_dtype(df_long[col]):
            log.warning(f"Feature column '{col}' is not numeric; attempting to coerce to numeric with NaN for invalid values.")
            df_long[col] = pd.to_numeric(df_long[col], errors="coerce")

    # Build taxonomy_df with taxonomy columns
    taxonomy_df = df.groupby("bin_uri").first()[[c for c in TAXONOMY_FEATURES if c in df.columns]].reset_index()

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

    if preprocessed_dir is not None:
        _save_to_preprocessed_dir(
            preprocessed_dir,
            X_train, y_train, X_val, y_val, X_test, y_test,
            taxonomy_df, embeddings_array, bins_with_embedding_arr,
            bin_index, sample_index, split_indices, resolved_state_path,
        )

    return (
        {
            "train": {"X": X_train, "y": y_train, "y_prob": y_train},
            "val": {"X": X_val, "y": y_val, "y_prob": y_val},
            "test": {"X": X_test, "y": y_test, "y_prob": y_test},
        },
        taxonomy_df,
        embeddings_array,
        bins_with_embedding_arr,
        bin_index,
        sample_index,
        split_indices,
        resolved_state_path,
    )


if __name__ == "__main__":
    """Generate datasets with different read count preprocessing methods."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate datasets with different read count preprocessing")
    args = parser.parse_args()
    
    # Setup logging
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Set seed for reproducibility
    np.random.seed()
    
    # Create config
    cfg = Config()
    
    # Generate datasets with all three preprocessing methods
    preprocessing_methods = ["original", "normalized", "logarithm"]
    
    log.info("="*70)
    log.info("GENERATING DATASETS WITH DIFFERENT PREPROCESSING METHODS")
    log.info("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # First pass: generate the split indices (no caching needed here)
    log.info("\nGenerating split indices...")
    _, _, _, _, _, _, split_indices, _ = load(
        cfg,
        read_count_preprocessing="original",
    )

    # Second pass: generate and cache each variant
    for method in preprocessing_methods:
        log.info(f"\n{'='*70}")
        log.info(f"Preprocessing method: {method.upper()}")
        log.info(f"{'='*70}")

        out_dir = os.path.join(script_dir, "data", method)
        _ = load(
            cfg,
            fixed_split_indices=split_indices,
            read_count_preprocessing=method,
            preprocessed_dir=out_dir,
        )

        log.info(f"✓ Saved {method} dataset to {out_dir}")

    log.info("\n" + "="*70)
    log.info("DATASET GENERATION COMPLETE")
    log.info("="*70)
    log.info("\nGenerated datasets:")
    for method in preprocessing_methods:
        log.info(f"  - {os.path.join(script_dir, 'data', method)}/")
