import logging as log
import os
import pickle
import shutil
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
    # Fast-path: load everything from a previously written cache directory.
    if config.preprocessed_dir is not None:
        sentinel = os.path.join(config.preprocessed_dir, "_complete")
        if os.path.exists(sentinel):
            cached = _load_from_preprocessed_dir(config.preprocessed_dir)
            # cached[2] is embeddings_array; skip cache if config needs embeddings but cache has none
            if config.use_embedding and cached[2] is None:
                log.info(f"load(): cache at {config.preprocessed_dir} has no embeddings but use_embedding=True — recomputing.")
            else:
                # Check that the cached location embedder matches the current config.
                cached_state_path = os.path.join(config.preprocessed_dir, PREPROCESSING_STATE_FILENAME)
                cached_loc_embedder = None
                if os.path.exists(cached_state_path):
                    try:
                        cached_state = load_preprocessing_state(cached_state_path)
                        cached_loc_embedder = cached_state.get("location_embedder", None)
                    except Exception:
                        pass
                current_loc_embedder = getattr(config, "location_embedder", None)
                if cached_loc_embedder != current_loc_embedder:
                    log.info(
                        f"load(): cache at {config.preprocessed_dir} was built with location_embedder={cached_loc_embedder!r} "
                        f"but current config has location_embedder={current_loc_embedder!r} — recomputing."
                    )
                else:
                    log.info(f"load(): cache hit at {config.preprocessed_dir} — loading from disk, skipping preprocessing.")
                    return cached

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

    # Normalize + log-transform occurences to get target y
    sample_totals = df.groupby("sample_id")["occurrences"].transform("sum")
    # Also store actual proportions for cross-entropy loss
    df["rel_abundance"] = df["occurrences"] / (sample_totals + 1e-10)

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
        # Always record which location embedder was used so cache invalidation works correctly.
        state["location_embedder"] = getattr(config, "location_embedder", None)
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

    if config.preprocessed_dir is not None and config.use_embedding:
        # Only write the cache from a run with use_embedding=True so embeddings are always present.
        # Runs with use_embedding=False skip the write to avoid poisoning the cache for other runs.
        _save_to_preprocessed_dir(
            config.preprocessed_dir,
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


MODEL_ALIASES: Dict[str, str] = {
	"satclip": "satclip",
	"range": "range",
	"range+": "range",
	"geoclip": "geoclip",
	"alphaearth": "alphaearth",
	"alpha_earth": "alphaearth",
}


_REPO_DIR = Path(__file__).resolve().parent.parent
_THIRD_PARTY_DIR = _REPO_DIR / "third_party"


def _ensure_third_party_on_syspath(*relative_parts: str) -> Path:
	"""Ensure a local third_party source path exists on sys.path and return it."""
	path = (_THIRD_PARTY_DIR / Path(*relative_parts)).resolve()
	if path.exists():
		path_str = str(path)
		if path_str not in sys.path:
			sys.path.insert(0, path_str)
	return path


def _normalize_model_name(name: str) -> str:
	key = name.strip().lower()
	if key not in MODEL_ALIASES:
		supported = ", ".join(sorted(set(MODEL_ALIASES.values())))
		raise ValueError(f"Unknown location embedder '{name}'. Supported: {supported}")
	return MODEL_ALIASES[key]


def _iter_batches(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
	for start in range(0, n, batch_size):
		end = min(start + batch_size, n)
		yield start, end


def _sanitize_latlon(latlon: np.ndarray) -> np.ndarray:
	# Always materialize a writable copy. Some upstream arrays (views/readonly buffers)
	# can be non-writeable and fail during in-place sanitization.
	arr = np.array(latlon, dtype=np.float64, copy=True, order="C")
	if arr.ndim != 2 or arr.shape[1] != 2:
		raise ValueError("Coordinates must be an array with shape (N, 2) in [lat, lon] order")

	arr[:, 0] = np.nan_to_num(arr[:, 0], nan=0.0, posinf=90.0, neginf=-90.0)
	arr[:, 1] = np.nan_to_num(arr[:, 1], nan=0.0, posinf=180.0, neginf=-180.0)
	arr[:, 0] = np.clip(arr[:, 0], -90.0, 90.0)
	arr[:, 1] = np.clip(arr[:, 1], -180.0, 180.0)
	return arr


def _latlon_to_unit_xyz(latlon: np.ndarray) -> np.ndarray:
	arr = _sanitize_latlon(latlon).astype(np.float64)
	lat_rad = np.deg2rad(arr[:, 0])
	lon_rad = np.deg2rad(arr[:, 1])
	x = np.cos(lat_rad) * np.cos(lon_rad)
	y = np.cos(lat_rad) * np.sin(lon_rad)
	z = np.sin(lat_rad)
	return np.column_stack([x, y, z]).astype(np.float32)


@dataclass
class EmbedderSpec:
	model_name: str
	device: str = "cpu"
	batch_size: int = 2048
	satclip_ckpt_path: Optional[str] = None
	range_db_path: Optional[str] = None


class BaseLocationEmbedder:
	def __init__(self, device: str = "cpu", batch_size: int = 2048) -> None:
		self.device = device
		self.batch_size = max(1, int(batch_size))

	@property
	def embedding_dim(self) -> int:
		raise NotImplementedError

	def encode(self, latlon: np.ndarray) -> np.ndarray:
		raise NotImplementedError


class GeoCLIPEmbedder(BaseLocationEmbedder):
	def __init__(self, device: str = "cpu", batch_size: int = 2048) -> None:
		super().__init__(device=device, batch_size=batch_size)
		try:
			import torch
			geoclip_mod = importlib.import_module("geoclip")
		except ImportError as exc:
			raise ImportError(
				"GeoCLIP backend requires 'geoclip' and 'torch'. Install with: pip install geoclip"
			) from exc

		self._torch = torch
		LocationEncoder = getattr(geoclip_mod, "LocationEncoder")
		self._model = LocationEncoder().double().to(self.device)
		self._model.eval()

	@property
	def embedding_dim(self) -> int:
		return 512

	def encode(self, latlon: np.ndarray) -> np.ndarray:
		coords = _sanitize_latlon(latlon)
		outs: List[np.ndarray] = []
		with self._torch.no_grad():
			for start, end in _iter_batches(len(coords), self.batch_size):
				batch = self._torch.tensor(coords[start:end], dtype=self._torch.float64, device=self.device)
				emb = self._model(batch).detach().cpu().numpy().astype(np.float32)
				outs.append(emb)
		return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.embedding_dim), dtype=np.float32)


class _RangeBackedEmbedder(BaseLocationEmbedder):
	def __init__(
		self,
		model_name: str,
		device: str,
		batch_size: int,
		satclip_ckpt_path: Optional[str],
		range_db_path: Optional[str],
		range_beta: float,
	) -> None:
		super().__init__(device=device, batch_size=batch_size)
		try:
			import torch
		except ImportError as exc:
			raise ImportError(
				"RANGE backend requires 'torch'. Install with: pip install torch"
			) from exc

		self._torch = torch
		self._model_name = model_name.upper()
		if self._model_name not in ("RANGE", "RANGE+"):
			raise ValueError(f"Unsupported RANGE model '{model_name}'. Use RANGE or RANGE+.")

		if range_db_path is None:
			raise ValueError("RANGE requires a precomputed database file path via 'range_db_path'.")
		range_db = Path(range_db_path).expanduser()
		if not range_db.is_absolute():
			range_db = (_REPO_DIR / range_db).resolve()
		if not range_db.is_file():
			raise FileNotFoundError(
				"RANGE database not found. "
				f"Expected file at: {range_db}. "
				"Pass --range_db_path explicitly if needed."
			)

		db = np.load(range_db, allow_pickle=True)
		if "satclip_embeddings" not in db or "image_embeddings" not in db:
			raise ValueError(
				"Invalid RANGE database: expected keys 'satclip_embeddings' and 'image_embeddings'."
			)

		db_satclip = db["satclip_embeddings"].astype(np.float32)
		db_satclip_norm = np.linalg.norm(db_satclip, ord=2, axis=1, keepdims=True)
		db_satclip = db_satclip / np.clip(db_satclip_norm, 1e-8, None)
		self._db_satclip = self._torch.tensor(db_satclip, dtype=self._torch.float32, device=self.device)
		self._db_image = self._torch.tensor(db["image_embeddings"].astype(np.float32), dtype=self._torch.float32, device=self.device)

		self._db_locs_xyz = None
		if self._model_name == "RANGE+":
			if "locs" not in db:
				raise ValueError("Invalid RANGE+ database: expected key 'locs'.")
			db_locs_latlon = db["locs"].astype(np.float32)
			self._db_locs_xyz = self._torch.tensor(_latlon_to_unit_xyz(db_locs_latlon), dtype=self._torch.float32, device=self.device)

		self._satclip = SatCLIPEmbedder(
			device=device,
			batch_size=batch_size,
			satclip_ckpt_path=satclip_ckpt_path,
		)
		self._temp = 15.0
		self._geo_temp = 40.0
		self._beta = float(range_beta)
		self._embedding_dim = 1280

	@property
	def embedding_dim(self) -> int:
		return self._embedding_dim

	def encode(self, latlon: np.ndarray) -> np.ndarray:
		coords_latlon = _sanitize_latlon(latlon).astype(np.float32)
		outs: List[np.ndarray] = []
		with self._torch.no_grad():
			for start, end in _iter_batches(len(coords_latlon), self.batch_size):
				batch_latlon_np = coords_latlon[start:end]
				curr_loc_np = self._satclip.encode(batch_latlon_np).astype(np.float32)
				curr_loc = self._torch.tensor(curr_loc_np, dtype=self._torch.float32, device=self.device)
				curr_loc = curr_loc / self._torch.clamp(curr_loc.norm(p=2, dim=-1, keepdim=True), min=1e-8)

				high_res_similarity = self._torch.softmax((curr_loc @ self._db_satclip.T) * self._temp, dim=-1)
				high_res_embeddings = high_res_similarity @ self._db_image

				if self._model_name == "RANGE+":
					query_xyz = self._torch.tensor(_latlon_to_unit_xyz(batch_latlon_np), dtype=self._torch.float32, device=self.device)
					angular_similarity = self._torch.softmax((query_xyz @ self._db_locs_xyz.T) * self._geo_temp, dim=-1)
					angular_high_res = angular_similarity @ self._db_image
					combined_high_res = (1.0 - self._beta) * angular_high_res + self._beta * high_res_embeddings
				else:
					combined_high_res = high_res_embeddings

				emb = self._torch.cat([combined_high_res, curr_loc], dim=1)
				outs.append(emb.detach().cpu().numpy().astype(np.float32))
		return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.embedding_dim), dtype=np.float32)


class SatCLIPEmbedder(BaseLocationEmbedder):
	def __init__(
		self,
		device: str = "cpu",
		batch_size: int = 2048,
		satclip_ckpt_path: Optional[str] = None,
	) -> None:
		super().__init__(device=device, batch_size=batch_size)
		try:
			import torch
			from huggingface_hub import hf_hub_download
		except ImportError as exc:
			raise ImportError(
				"SatCLIP backend requires 'torch' and 'huggingface-hub'. "
				"Install with: pip install torch huggingface-hub"
			) from exc

		# Ensure local vendored SatCLIP paths are discoverable first.
		vendored_satclip_pkg = _ensure_third_party_on_syspath("satclip", "satclip")
		vendored_satclip_root = _ensure_third_party_on_syspath("satclip")
		importlib.invalidate_caches()

		# SatCLIP loaders exposed by the official repo:
		# - load_lightweight.py: get_satclip_loc_encoder (no lightning dependency)
		# - load.py: get_satclip
		get_satclip = None
		import_errors: List[str] = []
		for mod_name in (
			"load_lightweight",
			"load",
			"satclip.load_lightweight",
			"satclip.load",
			"satclip.load_model",
		):
			try:
				mod = importlib.import_module(mod_name)
			except Exception as exc:  # pragma: no cover - import failures are env-specific
				import_errors.append(f"{mod_name}: {exc}")
				continue
			candidate = getattr(mod, "get_satclip", None) or getattr(mod, "get_satclip_loc_encoder", None)
			if candidate is not None:
				get_satclip = candidate
				break

		if get_satclip is None:
			raise ImportError(
				"SatCLIP backend could not find a loader function. Expected one of "
				"'get_satclip' or 'get_satclip_loc_encoder' from SatCLIP source. "
				"If using local clone, ensure third_party/satclip is present and SatCLIP deps are installed. "
				f"Checked paths: {vendored_satclip_pkg}, {vendored_satclip_root}. "
				f"Import errors: {' | '.join(import_errors)}"
			)

		ckpt_path = satclip_ckpt_path
		if ckpt_path is None:
			ckpt_path = hf_hub_download(
				repo_id="microsoft/SatCLIP-ResNet50-L10",
				filename="satclip-resnet50-l10.ckpt",
				repo_type="model",
			)

		self._torch = torch
		self._model = get_satclip(ckpt_path, device=device)
		self._model.to(device)
		self._model.eval()
		self._embedding_dim = int(getattr(self._model, "location_feature_dim", 256))

	@property
	def embedding_dim(self) -> int:
		return self._embedding_dim

	def encode(self, latlon: np.ndarray) -> np.ndarray:
		coords_latlon = _sanitize_latlon(latlon)
		# SatCLIP expects [lon, lat] coordinates (as in HF card examples)
		coords_lonlat = np.column_stack([coords_latlon[:, 1], coords_latlon[:, 0]])

		outs: List[np.ndarray] = []
		with self._torch.no_grad():
			for start, end in _iter_batches(len(coords_lonlat), self.batch_size):
				batch = self._torch.tensor(
					coords_lonlat[start:end], dtype=self._torch.float64, device=self.device
				)
				emb = self._model(batch)
				if hasattr(emb, "detach"):
					emb_np = emb.detach().cpu().numpy()
				else:
					emb_np = np.asarray(emb)
				outs.append(np.asarray(emb_np, dtype=np.float32))
		return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.embedding_dim), dtype=np.float32)


class RANGEEmbedder(_RangeBackedEmbedder):
	def __init__(
		self,
		device: str = "cpu",
		batch_size: int = 4096,
		satclip_ckpt_path: Optional[str] = None,
		range_db_path: Optional[str] = None,
		range_model_name: str = "RANGE+",  # "RANGE" or "RANGE+"
		range_beta: float = 0.5,           # interpolation weight for RANGE+ (0=RANGE, 1=RANGE+)
	) -> None:
		super().__init__(
			model_name=range_model_name,
			device=device,
			batch_size=batch_size,
			satclip_ckpt_path=satclip_ckpt_path,
			range_db_path=range_db_path,
			range_beta=range_beta,
		)


class AlphaEarthEmbedder(BaseLocationEmbedder):
	DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

	def __init__(
		self,
		device: str = "cpu",
		batch_size: int = 256,
		year: int = 2024,           # satellite imagery year
		scale_meters: int = 10,     # sampling resolution in metres
		project: Optional[str] = "metabarcoding-491221",  # GCP project for Earth Engine
	) -> None:
		super().__init__(device=device, batch_size=batch_size)
		try:
			ee = importlib.import_module("ee")
		except ImportError as exc:
			raise ImportError(
				"AlphaEarth backend requires Earth Engine API. Install with: pip install earthengine-api"
			) from exc

		self._ee = ee
		self.year = int(year)
		self.scale_meters = int(scale_meters)

		# Allow project override from env for cluster jobs.
		project = (
			project
			or os.environ.get("ALPHAEARTH_EE_PROJECT")
			or os.environ.get("EE_PROJECT")
		)

		# Optional service-account credentials path for non-interactive auth.
		# Preferred on clusters where browser-based auth is unavailable.
		credentials_path = (
			os.environ.get("ALPHAEARTH_EE_CREDENTIALS_JSON")
			or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
		)

		last_exc: Optional[Exception] = None

		if credentials_path:
			if not os.path.exists(credentials_path):
				raise RuntimeError(
					"Earth Engine credentials file not found. "
					"Set ALPHAEARTH_EE_CREDENTIALS_JSON (or GOOGLE_APPLICATION_CREDENTIALS) "
					f"to a valid JSON key file path. Got: {credentials_path}"
				)

			try:
				with open(credentials_path, "r", encoding="utf-8") as f:
					service_account_email = json.load(f).get("client_email")
			except Exception as exc:
				raise RuntimeError(
					"Could not read service-account JSON key for Earth Engine initialization. "
					f"Path: {credentials_path}"
				) from exc

			if not service_account_email:
				raise RuntimeError(
					"Service-account JSON is missing 'client_email'. "
					f"Path: {credentials_path}"
				)

			try:
				credentials = ee.ServiceAccountCredentials(service_account_email, credentials_path)
				ee.Initialize(credentials=credentials, project=project) if project else ee.Initialize(credentials=credentials)
				last_exc = None
			except Exception as exc:
				last_exc = exc

		if last_exc is not None or not credentials_path:
			try:
				ee.Initialize(project=project) if project else ee.Initialize()
			except Exception as exc:
				root = last_exc if last_exc is not None else exc
				raise RuntimeError(
					"Earth Engine is not initialized. "
					"For local interactive auth, run `earthengine authenticate`. "
					"For clusters, set ALPHAEARTH_EE_CREDENTIALS_JSON (or GOOGLE_APPLICATION_CREDENTIALS) "
					"to a service-account JSON key and set ALPHAEARTH_EE_PROJECT to your GCP project ID. "
					f"Initialization error: {root}"
				) from exc

	@property
	def embedding_dim(self) -> int:
		return 64

	def _encode_batch(self, latlon_batch: np.ndarray) -> np.ndarray:
		ee = self._ee
		features = []
		for i, (lat, lon) in enumerate(latlon_batch):
			geom = ee.Geometry.Point([float(lon), float(lat)])
			features.append(ee.Feature(geom, {"idx": int(i)}))

		start = f"{self.year}-01-01"
		end = f"{self.year + 1}-01-01"
		collection = ee.ImageCollection(self.DATASET_ID).filterDate(start, end)
		image = collection.mosaic()

		sampled = image.sampleRegions(
			collection=ee.FeatureCollection(features),
			properties=["idx"],
			scale=self.scale_meters,
			geometries=False,
		).getInfo()

		out = np.zeros((len(latlon_batch), self.embedding_dim), dtype=np.float32)
		for feature in sampled.get("features", []):
			props = feature.get("properties", {})
			idx = int(props.get("idx", -1))
			if idx < 0 or idx >= len(latlon_batch):
				continue
			out[idx] = np.array([props.get(f"A{i:02d}", 0.0) for i in range(64)], dtype=np.float32)
		return out

	def encode(self, latlon: np.ndarray) -> np.ndarray:
		coords = _sanitize_latlon(latlon)
		outs: List[np.ndarray] = []
		for start, end in _iter_batches(len(coords), self.batch_size):
			outs.append(self._encode_batch(coords[start:end]))
		return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.embedding_dim), dtype=np.float32)


def build_location_embedder(spec: EmbedderSpec) -> BaseLocationEmbedder:
	model_name = _normalize_model_name(spec.model_name)
	if model_name == "satclip":
		return SatCLIPEmbedder(
			device=spec.device,
			batch_size=spec.batch_size,
			satclip_ckpt_path=spec.satclip_ckpt_path,
		)
	if model_name == "range":
		return RANGEEmbedder(
			device=spec.device,
			batch_size=spec.batch_size,
			satclip_ckpt_path=spec.satclip_ckpt_path,
			range_db_path=spec.range_db_path,
		)
	if model_name == "geoclip":
		return GeoCLIPEmbedder(device=spec.device, batch_size=spec.batch_size)
	if model_name == "alphaearth":
		return AlphaEarthEmbedder(device=spec.device, batch_size=spec.batch_size)
	raise ValueError(f"Unsupported model_name '{spec.model_name}'")


def add_location_embeddings(
	df: pd.DataFrame,
	embedder: BaseLocationEmbedder,
	*,
	lat_col: str = "latitude",
	lon_col: str = "longitude",
	prefix: str = "loc_emb",
) -> Tuple[pd.DataFrame, List[str]]:
	if lat_col not in df.columns or lon_col not in df.columns:
		raise ValueError(f"DataFrame must contain '{lat_col}' and '{lon_col}' columns")

	coord_df = df[[lat_col, lon_col]].copy()
	valid_mask = coord_df[lat_col].notna() & coord_df[lon_col].notna()

	unique_coords = coord_df.loc[valid_mask, [lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
	emb_cols = [f"{prefix}_{i:03d}" for i in range(embedder.embedding_dim)]

	if len(unique_coords) == 0:
		out_df = df.copy()
		empty_emb_df = pd.DataFrame(
			np.zeros((len(out_df), len(emb_cols)), dtype=np.float32),
			columns=emb_cols,
			index=out_df.index,
		)
		out_df = pd.concat([out_df, empty_emb_df], axis=1)
		return out_df, emb_cols

	unique_latlon = unique_coords[[lat_col, lon_col]].to_numpy(dtype=np.float64)
	unique_embeddings = embedder.encode(unique_latlon)
	if unique_embeddings.shape[1] != embedder.embedding_dim:
		raise ValueError(
			"Embedding dimension mismatch: "
			f"expected {embedder.embedding_dim}, got {unique_embeddings.shape[1]}"
		)

	emb_values_df = pd.DataFrame(unique_embeddings, columns=emb_cols, index=unique_coords.index)
	emb_df = pd.concat([unique_coords, emb_values_df], axis=1)

	merged = df.merge(emb_df, on=[lat_col, lon_col], how="left")
	merged[emb_cols] = merged[emb_cols].fillna(0.0)
	return merged, emb_cols


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
        self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-entropy loss for soft targets (probability distributions).
        
        Args:
            logits: Raw model outputs (before softmax) of shape [batch_size, n_bins_in_sample]
                Padded positions should have value -inf (will become 0 after softmax)
            targets: Target probability distributions of shape [batch_size, n_bins_in_sample]
                Must sum to 1 along last dim. Padded positions should be 0.
            mask: Optional mask of shape [batch_size, n_bins_in_sample]. 
                1 for valid positions, 0 for padded. If None, inferred from logits.
        
        Returns:
            Scalar loss averaged over the batch.
        """

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
        self, outputs: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None
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


def _shannon_diversity(values: np.ndarray, eps: float = 1e-10) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.nan
    probs = (arr + eps) / np.sum(arr + eps)
    return float(-np.sum(probs * np.log(probs + eps)))


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    rank_x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    rank_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        return None
    rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return rho if np.isfinite(rho) else None


def _r2_and_intercept(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10):
    if len(y_true) < 2:
        return np.nan, np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + eps))
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    intercept = float(intercept) if np.isfinite(intercept) else np.nan
    return r2, intercept


def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sample_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive prediction metrics (micro and macro-averaged).

    Micro metrics pool all observations; macro metrics average per sample to
    avoid domination by samples with many observed BINs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = np.clip(y_pred[valid], 0, 1)
    eps = 1e-10

    rmse_macro = np.nan
    mae_macro = np.nan
    kl_divergence = np.nan
    shannon_r2 = np.nan
    shannon_intercept = np.nan
    spearman_macro = np.nan

    if sample_labels is not None:
        sample_labels_v = np.asarray(sample_labels)[valid]
        rmse_per, mae_per, kl_per = [], [], []
        shannon_true_per, shannon_pred_per = [], []
        spearman_per = []
        for sample in np.unique(sample_labels_v):
            mask = sample_labels_v == sample
            true_s = y_true[mask]
            pred_s = y_pred[mask]
            if len(true_s) == 0:
                continue
            rmse_per.append(float(np.sqrt(np.mean((true_s - pred_s) ** 2))))
            mae_per.append(float(np.mean(np.abs(true_s - pred_s))))
            true_s_norm = (true_s + eps) / (true_s + eps).sum()
            pred_s_norm = (pred_s + eps) / (pred_s + eps).sum()
            kl_per.append(float(np.sum(true_s_norm * np.log(true_s_norm / pred_s_norm))))

            s_true = _shannon_diversity(true_s, eps)
            s_pred = _shannon_diversity(pred_s, eps)
            if np.isfinite(s_true) and np.isfinite(s_pred):
                shannon_true_per.append(s_true)
                shannon_pred_per.append(s_pred)

            if len(true_s) > 1:
                rho = _spearman_rho(true_s, pred_s)
                if rho is not None:
                    spearman_per.append(rho)

        if rmse_per:
            rmse_macro = float(np.mean(rmse_per))
            mae_macro = float(np.mean(mae_per))
            kl_divergence = float(np.mean(kl_per))

        if len(shannon_true_per) > 1:
            shannon_r2, shannon_intercept = _r2_and_intercept(
                np.array(shannon_true_per), np.array(shannon_pred_per), eps
            )

        if spearman_per:
            spearman_macro = float(np.mean(spearman_per))

    rmse_micro = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae_micro = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + eps))

    y_true_log = np.log(y_true + 1)
    y_pred_log = np.log(y_pred + 1)
    ss_res_log = np.sum((y_true_log - y_pred_log) ** 2)
    ss_tot_log = np.sum((y_true_log - np.mean(y_true_log)) ** 2)
    r2_log = float(1 - ss_res_log / (ss_tot_log + eps))

    zero_mask = y_true == 0
    nonzero_mask = y_true > 0
    rmse_zeros = float(np.sqrt(np.mean((y_true[zero_mask] - y_pred[zero_mask]) ** 2))) if zero_mask.sum() > 0 else np.nan
    mae_zeros = float(np.mean(np.abs(y_true[zero_mask] - y_pred[zero_mask]))) if zero_mask.sum() > 0 else np.nan
    rmse_nonzeros = float(np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask]) ** 2))) if nonzero_mask.sum() > 0 else np.nan
    mae_nonzeros = float(np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]))) if nonzero_mask.sum() > 0 else np.nan

    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    correlation = 0.0 if np.isnan(corr) else float(corr)

    nz = y_true != 0
    rel_error = np.zeros_like(y_true, dtype=float)
    rel_error[nz] = np.abs(y_pred[nz] - y_true[nz]) / np.abs(y_true[nz])
    absolute_relative_error = float(np.mean(rel_error[nz])) if nz.sum() > 0 else np.nan

    return {
        "RMSE (micro)": rmse_micro,
        "RMSE (macro)": rmse_macro,
        "MAE (micro)": mae_micro,
        "MAE (macro)": mae_macro,
        "Absolute Relative Error": absolute_relative_error,
        "R² (Shannon diversity)": shannon_r2,
        "Shannon intercept": shannon_intercept,
        "Spearman Rho (macro)": spearman_macro,
        "R²": r2,
        "R² (log + 1)": r2_log,
        "RMSE (zeros)": rmse_zeros,
        "MAE (zeros)": mae_zeros,
        "RMSE (non-zeros)": rmse_nonzeros,
        "MAE (non-zeros)": mae_nonzeros,
        "KL Divergence": kl_divergence,
        "Correlation": correlation,
        "n_zeros": float(zero_mask.sum()),
        "n_nonzeros": float(nonzero_mask.sum()),
    }


def metric_key(metric_name: str) -> str:
    """Normalize a metric name to a valid wandb/logging key."""
    return (
        metric_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("²", "2")
        .replace("+", "plus")
        .replace("-", "_")
        .replace("/", "_")
    )

