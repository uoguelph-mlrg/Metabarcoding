import logging as log
import os
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

    # Sample-level curation: keep samples with enough total reads.
    # Threshold is defined as min(5th percentile, 50000) on raw (non-log) read counts.
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
    
    # Log a warning if any expected features are missing
    missing_features = [c for c in OBSERVATION_FEATURES + TAXONOMY_FEATURES if c not in df.columns]
    if missing_features:
        log.warning(f"Missing features in dataset: {', '.join(missing_features)}.")

    # Build df_long with required columns + features
    base_cols = ["sample_id", "bin_uri", "occurrences", "rel_abundance"]
    feature_cols_present = [c for c in OBSERVATION_FEATURES if c in df.columns]
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
        if (
            train_sample_idx.max(initial=-1) >= n_samples
            or val_sample_idx.max(initial=-1) >= n_samples
            or test_sample_idx.max(initial=-1) >= n_samples
        ):
            log.warning(
                "Provided fixed_split_indices are incompatible with current filtered samples; falling back to random split."
            )
            fixed_split_indices = None

    if fixed_split_indices is None:
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
        # First pass: fill NaNs with the BIN-specific train-set median (vectorized).
        missing = df_long[col].isna()
        df_long.loc[missing, col] = df_long.loc[missing, "bin_uri"].map(median_map)
        # Second pass: global fallback for BINs with no usable train median.
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
        data_path = os.path.dirname(config.data_path)

        for X, y, split in [(X_train,y_train,"train"), (X_val,y_val,"val"), (X_test,y_test,"test")]:
            X.to_csv(f"{data_path}/X_{split}.csv")
            pd.Series(y).to_csv(f"{data_path}/y_{split}.csv", index=False)
        taxonomic_data.to_csv(f"{data_path}/taxonomic_data.csv", index=False)

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


def load_or_preprocess(
    config: Config,
    fixed_split_indices: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, Dict[Any, int], Dict[Any, int], Dict[str, np.ndarray]]:
    """
    Fast path: if preprocessed splits already exist in the data directory, load them
    via load_processed() — skips all CSV parsing and feature engineering.
    Slow path: runs load() with save_data=True so subsequent calls take the fast path.

    The seven sentinel files checked are: X_train.csv, X_val.csv, X_test.csv,
    y_train.csv, y_val.csv, y_test.csv, taxonomic_data.csv.

    NOTE: when loading from cache, fixed_split_indices is ignored — the split is
    already baked into the saved files. This is intentional: autoresearch runs must
    compare models on a fixed, reproducible split.
    """
    data_dir = os.path.dirname(config.data_path)
    sentinel_files = [
        os.path.join(data_dir, f"{prefix}.csv")
        for prefix in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "taxonomic_data")
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

