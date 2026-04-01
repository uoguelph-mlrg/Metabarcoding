"""
Ablation Study: Compare MLP architectures vs MLP + Latent

This script trains and compares:
1. MLP + Latent: Original model with MLP and per-bin latent factors (from train.py)
2. MLP without taxonomy: MLP-only on the exact same numeric features as the latent model MLP
   (NO taxonomy, NO BIN embedding)
3. MLP with taxonomy: MLP-only using hierarchical entity embeddings of taxonomy levels
   (NO BIN embedding)

All models use cross-entropy loss for fair comparison.
Results are saved to pickle for visualization.

Usage:
    python ablation_study.py --data_path ../../data/data_merged.csv
    python ablation_study.py --data_path ../../data/data_merged.csv --no_wandb
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Dict, Any, Tuple, Literal, Optional
import logging as log
import sys

# Add src folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import Config, set_seed
from train import Trainer  # Reuse existing Trainer for MLP + Latent
from utils import load, OBSERVATION_FEATURES
from mlp import MLPModel
from loss import Loss
from dataset import MBDataset, collate_samples

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# Data Loading with Taxonomy Features
# ============================================================================

TAXONOMY_COLS = ["phylum", "class", "order", "family", "subfamily", "genus", "species"]


def load_data_with_taxonomy(
    data_path: str, 
    config: Config,
    include_taxonomy: bool = False,
    fixed_split_indices: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, Dict[Any, int], Dict[Any, int], Optional[Dict[str, LabelEncoder]], Dict[str, np.ndarray]]:
    """
    Load data with optional taxonomic features encoded as integers.
    
    Args:
        data_path: Path to the CSV data file
        config: Configuration object
        include_taxonomy: Whether to include taxonomy features
        fixed_split_indices: If provided, use these sample indices for splits
                            (ensures same splits across different data loading calls)
    
    Returns:
        Tuple of (splits, taxonomy_data, bin_index, sample_index, taxonomy_encoders, split_indices)
    """
    df = pd.read_csv(data_path)
    df = df.rename(columns={"sample-eventid": "sample_id"})
    
    # Parse date feature
    if "collection_start_date" in df.columns:
        df["collection_day"] = pd.to_datetime(
            df["collection_start_date"], format="%m/%d/%Y", errors="coerce"
        ).dt.dayofyear.fillna(0)
    else:
        df["collection_day"] = 0
    
    # Build indices
    unique_samples = df["sample_id"].unique()
    n_samples = len(unique_samples)
    sample_index = {s: i for i, s in enumerate(unique_samples)}
    unique_bins = df["bin_uri"].unique()
    bin_index = {b: i for i, b in enumerate(unique_bins)}
    
    # Compute relative abundance
    sample_totals = df.groupby("sample_id")["occurrences"].transform("sum")
    df["rel_abundance"] = df["occurrences"] / (sample_totals + 1e-10)
    
    # Normalize reads per sample
    df["total_reads_per_sample"] = np.log1p(df["total_reads_per_sample"])
    df["total_reads_norm"] = np.log1p(df["total_reads"])
    df["avg_reads_norm"] = np.log1p(df["avg_reads"])
    df["max_reads_norm"] = np.log1p(df["max_reads"])
    df["min_reads_norm"] = np.log1p(df["min_reads"])
    
    # Determine feature columns
    feature_cols = [c for c in OBSERVATION_FEATURES if c in df.columns]
    
    # Encode taxonomy if requested
    taxonomy_encoders = None
    if include_taxonomy:
        taxonomy_encoders = {}
        for col in TAXONOMY_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("unknown").astype(str)
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col])
                taxonomy_encoders[col] = le
                feature_cols.append(f"{col}_encoded")
    
    # Build long-format dataframe
    base_cols = ["sample_id", "bin_uri", "occurrences", "rel_abundance"]
    df_long = df[base_cols + feature_cols].copy()
    
    # Extract taxonomy data
    taxonomy_data = df.groupby("bin_uri").first()[
        [c for c in TAXONOMY_COLS if c in df.columns]
    ].reset_index()
    taxonomy_data["_idx"] = taxonomy_data["bin_uri"].map(bin_index)
    taxonomy_data = taxonomy_data.sort_values("_idx").drop(columns=["_idx"]).reset_index(drop=True)
    
    # Create splits - use fixed indices if provided for reproducibility
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
    
    # Normalize numeric features based on training set
    numeric_features = [c for c in OBSERVATION_FEATURES if c in df_long.columns]
    X_train_subset = df_long.loc[
        df_long["sample_id"].isin(set(unique_samples[train_sample_idx])), 
        numeric_features + ["bin_uri"]
    ]
    
    # Fill missing values with bin medians from training set
    bin_medians = X_train_subset.groupby("bin_uri").median()
    for col in numeric_features:
        if col not in bin_medians.columns:
            continue
        median_map = bin_medians[col].to_dict()
        df_long[col] = df_long.apply(
            lambda row: median_map.get(row["bin_uri"], np.nan) if pd.isna(row[col]) else row[col],
            axis=1
        )
        df_long[col] = df_long[col].fillna(df_long[col].median())
    
    # Standardize features
    for col in numeric_features:
        mean_val = X_train_subset[col].mean()
        std_val = X_train_subset[col].std(ddof=0) + 1e-10
        df_long[col] = (df_long[col] - mean_val) / std_val
    
    # Also normalize taxonomy features if included
    if include_taxonomy:
        for col in TAXONOMY_COLS:
            encoded_col = f"{col}_encoded"
            if encoded_col in df_long.columns:
                mean_val = df_long[encoded_col].mean()
                std_val = df_long[encoded_col].std(ddof=0) + 1e-10
                df_long[encoded_col] = (df_long[encoded_col] - mean_val) / std_val
    
    def compute_split(df_long, sample_idx, feature_cols):
        sample_set = set(unique_samples[sample_idx])
        mask = df_long["sample_id"].isin(sample_set)
        X = df_long.loc[mask, ["sample_id", "bin_uri"] + feature_cols].set_index(["sample_id", "bin_uri"])
        y = df_long.loc[mask, "rel_abundance"]
        return X, y
    
    X_train, y_train = compute_split(df_long, train_sample_idx, feature_cols)
    X_val, y_val = compute_split(df_long, val_sample_idx, feature_cols)
    X_test, y_test = compute_split(df_long, test_sample_idx, feature_cols)
    
    log.info(f"Loaded {len(df_long)} observations")
    log.info(f"  {len(unique_samples)} samples, {len(unique_bins)} bins")
    log.info(f"  Features: {len(feature_cols)}")
    if include_taxonomy:
        log.info(f"  Taxonomy levels encoded: {list(taxonomy_encoders.keys())}")
    log.info(f"  Train: {len(train_sample_idx)} samples, Val: {len(val_sample_idx)} samples, Test: {len(test_sample_idx)} samples")
    
    splits = {
        "train": {"X": X_train, "y": y_train},
        "val": {"X": X_val, "y": y_val},
        "test": {"X": X_test, "y": y_test},
    }
    
    # Store split indices for reuse
    split_indices = {
        "train": train_sample_idx,
        "val": val_sample_idx,
        "test": test_sample_idx,
    }
    
    return splits, taxonomy_data, bin_index, sample_index, taxonomy_encoders, split_indices


# ============================================================================
# MLP-only models (NO BIN embedding)
# ============================================================================

def _build_taxonomy_id_matrix(
    bins_df: pd.DataFrame,
    n_bins: int,
    taxonomy_cols: list[str],
) -> tuple[np.ndarray, Dict[str, int]]:
    """
    Build a [n_bins, n_levels] integer matrix of taxonomy ids, with 0 reserved for "unknown".
    Returns:
      - tax_ids_per_bin: np.ndarray[int64] of shape [n_bins, n_levels]
      - cardinalities: dict level -> n_categories_including_unknown
    """
    tax_ids = np.zeros((n_bins, len(taxonomy_cols)), dtype=np.int64)
    cardinalities: Dict[str, int] = {}

    for j, col in enumerate(taxonomy_cols):
        if col not in bins_df.columns:
            # All unknown for this level
            cardinalities[col] = 1
            continue

        vals = bins_df[col].fillna("unknown").astype(str)
        # Ensure "unknown" is 0
        uniques = list(pd.Index(vals.unique()))
        cats = ["unknown"] + [v for v in uniques if v != "unknown"]
        mapping = {c: i for i, c in enumerate(cats)}
        ids = vals.map(mapping).astype(np.int64).to_numpy()
        if len(ids) != n_bins:
            raise ValueError(f"bins_df has {len(ids)} rows but expected n_bins={n_bins}")

        tax_ids[:, j] = ids
        cardinalities[col] = len(cats)

    return tax_ids, cardinalities


class MLPNoTaxonomy(nn.Module):
    """MLP-only baseline using the exact same numeric features as the latent model MLP."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        self.mlp = MLPModel(input_dim, hidden_dims=hidden_dims, dropout=dropout)

    def forward(self, features: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        # bin_idx is ignored by design (NO BIN embedding, NO taxonomy)
        return self.mlp(features)


class MLPWithHierarchicalTaxonomyEmbeddings(nn.Module):
    """
    MLP that augments numeric features with hierarchical entity embeddings:
      phylum/class/order/family/subfamily/genus/species (no BIN embedding).
    """

    def __init__(
        self,
        n_num_features: int,
        tax_ids_per_bin: np.ndarray,
        taxonomy_cardinalities: Dict[str, int],
        taxonomy_cols: list[str],
        hidden_dims: list[int],
        dropout: float,
        embedding_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        self.taxonomy_cols = taxonomy_cols
        # Register a lookup table: bin_idx -> [level_ids...]
        self.register_buffer(
            "tax_ids_per_bin",
            torch.tensor(tax_ids_per_bin, dtype=torch.long),
            persistent=False,
        )

        # Default embedding sizes (can be overridden)
        if embedding_dims is None:
            embedding_dims = {
                "phylum": 8,
                "class": 8,
                "order": 16,
                "family": 16,
                "subfamily": 24,
                "genus": 24,
                "species": 32,
            }

        self.embeddings = nn.ModuleDict()
        emb_out_dim = 0
        for col in taxonomy_cols:
            n_cat = int(taxonomy_cardinalities.get(col, 1))
            d = int(embedding_dims.get(col, 8))
            # We reserve 0 for "unknown", but we still learn an embedding for it
            self.embeddings[col] = nn.Embedding(n_cat, d)
            emb_out_dim += d

        self.mlp = MLPModel(n_num_features + emb_out_dim, hidden_dims=hidden_dims, dropout=dropout)

    def forward(self, features: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        tax_ids = self.tax_ids_per_bin[bin_idx]  # [N, n_levels]
        embs = []
        for j, col in enumerate(self.taxonomy_cols):
            embs.append(self.embeddings[col](tax_ids[:, j]))
        tax_vec = torch.cat(embs, dim=-1) if embs else torch.zeros((features.size(0), 0), device=features.device)
        x = torch.cat([features, tax_vec], dim=-1)
        return self.mlp(x)


# ============================================================================
# MLP-Only Trainer (uses cross-entropy loss like original)
# ============================================================================

class MLPOnlyTrainer:
    """
    Trainer for MLP-only model with bin embeddings.
    
    Uses cross-entropy loss (sample mode) for fair comparison with MLP + Latent.
    """
    
    def __init__(
        self, 
        cfg: Config, 
        data_splits: Dict[str, Dict[str, Any]],
        bin_index: Dict[Any, int],
        sample_index: Dict[Any, int],
        model_name: str = "mlp_only",
        taxonomy: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.model_name = model_name
        self.device = torch.device(cfg.device)
        self.bin_index = bin_index
        self.sample_index = sample_index
        self.idx_to_bin = {idx: str(bin_uri) for bin_uri, idx in bin_index.items()}
        self.idx_to_sample = {idx: str(sample_id) for sample_id, idx in sample_index.items()}
        
        n_features = data_splits["train"]["X"].shape[1]
        n_bins = len(bin_index)
        
        log.info(f"\nBuilding {model_name} model:")
        log.info(f"  Features: {n_features}")
        log.info(f"  Number of bins: {n_bins}")
        log.info(f"  Using cross-entropy loss (sample mode)")
        log.info(f"  BIN embedding: disabled")
        log.info(f"  Taxonomy embeddings: {'enabled' if taxonomy is not None else 'disabled'}")
        
        hidden_dims = [128, 64]
        if taxonomy is None:
            # NO taxonomy, NO bin embedding: exact same inputs as latent model MLP
            self.model = MLPNoTaxonomy(
                input_dim=n_features,
                hidden_dims=hidden_dims,
                dropout=cfg.dropout,
            ).to(self.device)
        else:
            # Hierarchical entity embeddings for taxonomy levels (still NO bin embedding)
            self.model = MLPWithHierarchicalTaxonomyEmbeddings(
                n_num_features=n_features,
                tax_ids_per_bin=taxonomy["tax_ids_per_bin"],
                taxonomy_cardinalities=taxonomy["cardinalities"],
                taxonomy_cols=taxonomy.get("taxonomy_cols", TAXONOMY_COLS),
                hidden_dims=hidden_dims,
                dropout=cfg.dropout,
            ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Use cross-entropy loss (sample mode) like the original model
        self.criterion = Loss(task="cross_entropy")
        
        # Create datasets in sample mode for cross-entropy
        train_ds = MBDataset(data_splits["train"], bin_index, sample_index, loss_mode="sample")
        val_ds = MBDataset(data_splits["val"], bin_index, sample_index, loss_mode="sample")
        test_ds = MBDataset(data_splits["test"], bin_index, sample_index, loss_mode="sample")
        
        batch_size = cfg.batch_size_sample
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_samples)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_samples)
        self.test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_samples)
        
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0
        
        root = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(root, "models", f"ablation_{model_name}_{time.strftime('%Y-%m-%d_%H:%M')}.pt")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
    
    def _to_device(self, batch):
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)
        bin_idx = batch["bin_idx"].to(self.device)
        sample_idx = batch["sample_idx"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)
        return inputs, targets, bin_idx, sample_idx, mask
    
    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        n_samples = 0
        
        for batch in self.train_loader:
            inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
            
            # Sample mode: inputs [B, max_bins, features]
            B, max_bins, n_feat = inputs.shape
            inputs_flat = inputs.view(B * max_bins, n_feat)
            bin_idx_flat = bin_idx.view(B * max_bins)
            
            # Forward pass
            outputs_flat = self.model(inputs_flat, bin_idx_flat)
            outputs = outputs_flat.view(B, max_bins)
            
            # Mask padded positions
            outputs = outputs.masked_fill(mask == 0, float('-inf'))
            
            # Cross-entropy loss
            loss = self.criterion(outputs, targets, mask)
            
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size
        
        return running_loss / max(1, n_samples)
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0
        n_samples = 0
        
        for batch in loader:
            inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
            
            B, max_bins, n_feat = inputs.shape
            inputs_flat = inputs.view(B * max_bins, n_feat)
            bin_idx_flat = bin_idx.view(B * max_bins)
            
            outputs_flat = self.model(inputs_flat, bin_idx_flat)
            outputs = outputs_flat.view(B, max_bins)
            outputs = outputs.masked_fill(mask == 0, float('-inf'))
            
            loss = self.criterion(outputs, targets, mask)
            
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size
        
        return running_loss / max(1, n_samples)
    
    @torch.no_grad()
    def get_predictions(self, loader: DataLoader = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get flattened predictions, targets, and their sample/BIN labels for evaluation."""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_sample_labels = []
        all_bin_labels = []
        
        eval_loader = loader if loader is not None else self.test_loader
        
        for batch in eval_loader:
            inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
            
            B, max_bins, n_feat = inputs.shape
            inputs_flat = inputs.view(B * max_bins, n_feat)
            bin_idx_flat = bin_idx.view(B * max_bins)
            
            outputs_flat = self.model(inputs_flat, bin_idx_flat)
            outputs = outputs_flat.view(B, max_bins)
            outputs = outputs.masked_fill(mask == 0, float('-inf'))
            
            # Apply softmax to get probabilities
            probs = F.softmax(outputs, dim=-1)
            
            # Extract valid (non-padded) predictions
            for b in range(B):
                valid_mask = mask[b].bool()
                y_pred = probs[b, valid_mask].cpu().numpy()
                y_true = targets[b, valid_mask].cpu().numpy()
                valid_bin_idx = bin_idx[b, valid_mask].cpu().numpy()
                sample_idx_scalar = int(sample_idx[b].item())
                sample_label = self.idx_to_sample.get(sample_idx_scalar, str(sample_idx_scalar))
                all_preds.extend(y_pred)
                all_targets.extend(y_true)
                all_sample_labels.extend([sample_label] * len(y_pred))
                all_bin_labels.extend([
                    self.idx_to_bin.get(int(bin_id), str(int(bin_id)))
                    for bin_id in valid_bin_idx
                ])
        
        return (
            np.array(all_preds),
            np.array(all_targets),
            np.array(all_sample_labels),
            np.array(all_bin_labels),
        )
    
    def save_model(self):
        torch.save({"model_state_dict": self.model.state_dict()}, self.save_path)
    
    def run(self, use_wandb: bool = True, max_epochs: int = 100) -> Dict[str, Any]:
        log.info(f"\nTraining {self.model_name}...")
        
        best_val = float('inf')
        no_improve = 0
        
        pbar = tqdm(range(max_epochs), desc=f"{self.model_name}")
        for epoch in pbar:
            train_loss = self.train_epoch()
            val_loss = self.validate(self.val_loader)
            self.scheduler.step(val_loss)
            
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f"{self.model_name}/epoch": epoch, 
                    f"{self.model_name}/train_loss": train_loss, 
                    f"{self.model_name}/val_loss": val_loss
                })
            
            pbar.set_postfix({"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"})
            
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                self.save_model()
                no_improve = 0
            else:
                no_improve += 1
                if self.cfg.patience and no_improve >= self.cfg.patience:
                    log.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Final evaluation
        predictions, targets, sample_labels, bin_labels = self.get_predictions(self.test_loader)
        
        log.info(f"\n{self.model_name} Results:")
        log.info(f"  Best val loss: {best_val:.6f}")
        
        return {
            "model_name": self.model_name,
            "best_val_loss": best_val,
            "predictions": predictions,
            "targets": targets,
            "sample_labels": sample_labels,
            "bin_labels": bin_labels,
        }


# ============================================================================
# Regression Metrics
# ============================================================================

def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics with zero/non-zero split."""
    metrics = {}
    
    # Overall metrics
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # R² and Pearson correlation
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    if len(preds) > 1 and np.std(preds) > 0 and np.std(targets) > 0:
        correlation = np.corrcoef(preds, targets)[0, 1]
    else:
        correlation = 0.0
    
    metrics["mae_all"] = mae
    metrics["mse_all"] = mse
    metrics["rmse_all"] = rmse
    metrics["r2"] = r2
    metrics["correlation"] = correlation
    
    # Zero values (target == 0)
    zero_mask = targets == 0
    if zero_mask.sum() > 0:
        metrics["mae_zero"] = np.mean(np.abs(preds[zero_mask] - targets[zero_mask]))
        metrics["mse_zero"] = np.mean((preds[zero_mask] - targets[zero_mask]) ** 2)
        metrics["n_zero"] = int(zero_mask.sum())
        metrics["mean_pred_when_zero"] = float(np.mean(preds[zero_mask]))
    else:
        metrics["mae_zero"] = 0.0
        metrics["mse_zero"] = 0.0
        metrics["n_zero"] = 0
        metrics["mean_pred_when_zero"] = 0.0
    
    # Non-zero values (target > 0)
    nonzero_mask = targets > 0
    if nonzero_mask.sum() > 0:
        metrics["mae_nonzero"] = np.mean(np.abs(preds[nonzero_mask] - targets[nonzero_mask]))
        metrics["mse_nonzero"] = np.mean((preds[nonzero_mask] - targets[nonzero_mask]) ** 2)
        metrics["n_nonzero"] = int(nonzero_mask.sum())
        metrics["mean_pred_when_nonzero"] = float(np.mean(preds[nonzero_mask]))
        metrics["mean_target_when_nonzero"] = float(np.mean(targets[nonzero_mask]))
    else:
        metrics["mae_nonzero"] = 0.0
        metrics["mse_nonzero"] = 0.0
        metrics["n_nonzero"] = 0
        metrics["mean_pred_when_nonzero"] = 0.0
        metrics["mean_target_when_nonzero"] = 0.0
    
    # KL divergence (treating as distributions within each sample)
    eps = 1e-10
    preds_safe = np.clip(preds, eps, 1.0)
    targets_safe = np.clip(targets, eps, 1.0)
    kl_div = np.mean(targets_safe * np.log(targets_safe / preds_safe))
    metrics["kl_divergence"] = kl_div
    
    return metrics


# ============================================================================
# Main Ablation Study
# ============================================================================

def run_ablation_study(
    data_path: str, 
    cfg: Config, 
    use_wandb: bool = True,
    max_epochs: int = 100,
    run_group: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run ablation variants (no baseline retraining).

    1. MLP without taxonomy: MLP-only with observation features
    2. MLP with taxonomy: MLP-only with hierarchical taxonomy embeddings

    All variants use the same train/val/test splits.
    """
    results = {}
    
    # ========================================================================
    # Step 1: Load data and get split indices (will be reused)
    # ========================================================================
    log.info("\n" + "="*70)
    log.info("PREPARING DATA SPLITS")
    log.info("="*70)
    
    set_seed(14)  # Fixed seed for reproducibility
    
    # Create splits using the SAME preprocessing as the latent model (utils.load)
    # We only use this call to determine split indices deterministically.
    _, _, _, _, split_indices = load(data_path, cfg, save_data=False)
    
    # ========================================================================
    # Step 2: Build non-taxonomy data once for all ablation variants
    # ========================================================================
    data_no_tax, bins_df, bin_index, sample_index, _, _ = load_data_with_taxonomy(
        data_path,
        cfg,
        include_taxonomy=False,
        fixed_split_indices=split_indices,
    )
    
    # ========================================================================
    # Step 3: Train MLP without taxonomy (same features as MLP + Latent)
    # ========================================================================
    log.info("\n" + "="*70)
    log.info("TRAINING MLP WITHOUT TAXONOMY")
    log.info("="*70)
    
    set_seed(14)  # Reset seed

    if use_wandb:
        wandb.init(
            project="metabarcoding",
            name=f"ablation_study_mlp_no_taxonomy_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
            group=run_group,
            tags=["ablation_study", "mlp_no_taxonomy", "variant_only"],
            config=cfg.__dict__,
            reinit=True,
        )
    
    try:
        mlp_no_tax_trainer = MLPOnlyTrainer(
            cfg, data_no_tax, bin_index, sample_index, 
            model_name="MLP (no taxonomy)",
            taxonomy=None,
        )
        mlp_no_tax_results = mlp_no_tax_trainer.run(use_wandb=use_wandb, max_epochs=max_epochs)
    finally:
        if use_wandb:
            wandb.finish()
    
    mlp_no_tax_metrics = compute_regression_metrics(
        mlp_no_tax_results["predictions"],
        mlp_no_tax_results["targets"]
    )
    
    results["mlp_no_taxonomy"] = {
        "model_name": "MLP (no taxonomy)",
        "best_val_loss": mlp_no_tax_results["best_val_loss"],
        "predictions": mlp_no_tax_results["predictions"],
        "targets": mlp_no_tax_results["targets"],
        "sample_labels": mlp_no_tax_results["sample_labels"],
        "bin_labels": mlp_no_tax_results["bin_labels"],
        "metrics": mlp_no_tax_metrics,
        "n_features": data_no_tax["train"]["X"].shape[1],
    }
    
    log.info(f"MLP (no taxonomy): MAE={mlp_no_tax_metrics['mae_all']:.6f}, MSE={mlp_no_tax_metrics['mse_all']:.6f}")
    
    # ========================================================================
    # Step 4: Train MLP with taxonomy
    # ========================================================================
    log.info("\n" + "="*70)
    log.info("TRAINING MLP WITH TAXONOMY")
    log.info("="*70)
    
    set_seed(14)  # Reset seed

    if use_wandb:
        wandb.init(
            project="metabarcoding",
            name=f"ablation_study_mlp_with_taxonomy_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
            group=run_group,
            tags=["ablation_study", "mlp_with_taxonomy", "variant_only"],
            config=cfg.__dict__,
            reinit=True,
        )
    
    # Build hierarchical taxonomy ids per BIN (0 reserved for "unknown")
    tax_ids_per_bin, card = _build_taxonomy_id_matrix(
        bins_df=bins_df,
        n_bins=len(bin_index),
        taxonomy_cols=TAXONOMY_COLS,
    )
    taxonomy_spec = {"tax_ids_per_bin": tax_ids_per_bin, "cardinalities": card, "taxonomy_cols": TAXONOMY_COLS}
    
    try:
        mlp_with_tax_trainer = MLPOnlyTrainer(
            cfg, data_no_tax, bin_index, sample_index,
            model_name="MLP (with taxonomy)",
            taxonomy=taxonomy_spec,
        )
        mlp_with_tax_results = mlp_with_tax_trainer.run(use_wandb=use_wandb, max_epochs=max_epochs)
    finally:
        if use_wandb:
            wandb.finish()
    
    mlp_with_tax_metrics = compute_regression_metrics(
        mlp_with_tax_results["predictions"],
        mlp_with_tax_results["targets"]
    )
    
    results["mlp_with_taxonomy"] = {
        "model_name": "MLP (with taxonomy)",
        "best_val_loss": mlp_with_tax_results["best_val_loss"],
        "predictions": mlp_with_tax_results["predictions"],
        "targets": mlp_with_tax_results["targets"],
        "sample_labels": mlp_with_tax_results["sample_labels"],
        "bin_labels": mlp_with_tax_results["bin_labels"],
        "metrics": mlp_with_tax_metrics,
        "n_features": data_no_tax["train"]["X"].shape[1],
    }
    
    log.info(f"MLP (with taxonomy): MAE={mlp_with_tax_metrics['mae_all']:.6f}, MSE={mlp_with_tax_metrics['mse_all']:.6f}")
    
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation Study: MLP variants vs MLP + Latent"
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data CSV file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum epochs for MLP-only training")
    args = parser.parse_args()
    
    # Setup
    set_seed(14)
    cfg = Config()
    
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    run_group = f"ablation_study_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Run ablation study
    results = run_ablation_study(
        args.data_path,
        cfg,
        use_wandb=use_wandb,
        max_epochs=args.max_epochs,
        run_group=run_group,
    )
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    for variant, variant_results in results.items():
        results_path = os.path.join(output_dir, f"ablation_study_{variant}.pkl")
        save_results({variant: variant_results}, results_path)
    
    # Print summary
    log.info(f"\n{'='*70}")
    log.info("ABLATION VARIANT TRAINING COMPLETE")
    log.info(f"{'='*70}")
    
    log.info("\nSummary:")
    for key, result in results.items():
        metrics = result["metrics"]
        log.info(f"\n{result['model_name']} ({result['n_features']} features):")
        log.info(f"  MAE: {metrics['mae_all']:.6f}")
        log.info(f"  MSE: {metrics['mse_all']:.6f}")
        log.info(f"  Correlation: {metrics['correlation']:.4f}")
        log.info(f"  MAE (zero): {metrics['mae_zero']:.6f}")
        log.info(f"  MAE (non-zero): {metrics['mae_nonzero']:.6f}")
    
    log.info(f"\nResults saved to: {output_dir}")
    log.info(f"Run visualization with one or more files from: {output_dir}")
