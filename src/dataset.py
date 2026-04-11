import os
from pathlib import Path
from random import sample
from typing import Any, Dict, List, Literal, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config


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