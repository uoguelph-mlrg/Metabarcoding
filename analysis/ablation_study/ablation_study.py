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
    python ablation_study.py
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
from utils import load, OBSERVATION_FEATURES, TAXONOMY_FEATURES
from mlp import MLPModel
from loss import Loss
from dataset import MBDataset, collate_samples

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Derived from TAXONOMY_FEATURES (utils) so taxonomy_df columns stay in sync.
TAXONOMY_COLS = list(TAXONOMY_FEATURES)


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
        
        hidden_dims = self.cfg.mlp_hidden_dims  # Can tune this if desired, but keeping it fixed for fair comparison
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
            self.model.parameters(), lr=cfg.mlp_lr, weight_decay=cfg.weight_decay
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
    cfg: Config,
    use_wandb: bool = True,
    max_epochs: int = 100,
    run_group: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run ablation variants (no baseline retraining).

    1. MLP without taxonomy: MLP-only with observation features
    2. MLP with taxonomy: MLP-only with hierarchical taxonomy embeddings

    All variants use the same train/val/test splits and identical preprocessing
    as the latent model (via utils.load).
    """
    results = {}

    # ========================================================================
    # Step 1: Load data using the canonical utils.load pipeline.
    # This ensures identical sample filtering, feature engineering,
    # imputation, standardization, and train/val/test splits as the
    # MLP + Latent baseline.
    # ========================================================================
    log.info("\n" + "="*70)
    log.info("PREPARING DATA SPLITS")
    log.info("="*70)

    set_seed()  # Fixed seed for reproducibility

    splits, taxonomy_df, _, _, bin_index, sample_index, _, _ = load(
        cfg, save_data=False
    )

    # ========================================================================
    # Step 2: Train MLP without taxonomy (same features as MLP + Latent)
    # ========================================================================
    log.info("\n" + "="*70)
    log.info("TRAINING MLP WITHOUT TAXONOMY")
    log.info("="*70)
    
    set_seed()  # Reset seed

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
            cfg, splits, bin_index, sample_index,
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
        "n_features": splits["train"]["X"].shape[1],
    }
    
    log.info(f"MLP (no taxonomy): MAE={mlp_no_tax_metrics['mae_all']:.6f}, MSE={mlp_no_tax_metrics['mse_all']:.6f}")
    
    # ========================================================================
    # Step 4: Train MLP with taxonomy
    # ========================================================================
    log.info("\n" + "="*70)
    log.info("TRAINING MLP WITH TAXONOMY")
    log.info("="*70)
    
    set_seed()  # Reset seed

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
        bins_df=taxonomy_df,
        n_bins=len(bin_index),
        taxonomy_cols=TAXONOMY_COLS,
    )
    taxonomy_spec = {"tax_ids_per_bin": tax_ids_per_bin, "cardinalities": card, "taxonomy_cols": TAXONOMY_COLS}
    
    try:
        mlp_with_tax_trainer = MLPOnlyTrainer(
            cfg, splits, bin_index, sample_index,
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
        "n_features": splits["train"]["X"].shape[1],
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
    set_seed()
    cfg = Config()
    
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    run_group = f"ablation_study_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Run ablation study
    results = run_ablation_study(
        cfg,
        use_wandb=use_wandb,
        max_epochs=args.max_epochs,
        run_group=run_group,
    )
    
    # Save results in a timestamped subdirectory so successive runs don't collide.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir, time.strftime("%Y%m%d_%H%M%S"))
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
