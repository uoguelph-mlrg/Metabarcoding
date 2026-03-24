from __future__ import annotations
import os
# Prevent BLAS/OMP threading conflict with PyTorch MPS on macOS
#os.environ.setdefault("OMP_NUM_THREADS", "1")
#os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import re
import time
from typing import Literal, Optional, List, Tuple, Dict, Any
import logging as log
from dataclasses import asdict

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import from src folder (reusing existing infrastructure)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from neighbor_graph import NeighbourGraph
from latent_solver import LatentSolver
from dataset import MBDataset, collate_samples
from loss import Loss
from mlp import MLPModel
from model import Model
from config import Config, set_seed
from utils import load, load_processed

class Trainer:
    """
    Trains MLP and latent embedding jointly in a single forward-backward pass per epoch.
    """
    def __init__(
        self,
        cfg: Config,
        data_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        loss_type: Literal["cross_entropy", "logistic"] = "cross_entropy",
        fixed_split_indices: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.cfg = cfg
        
        # Load data
        if data_dir is not None:
            data, bins_df, bin_index, sample_index, split_indices = load_processed(data_dir)
        else:
            if data_path is None:
                raise ValueError("Either data_path or data_dir must be provided")
            data, bins_df, bin_index, sample_index, split_indices = load(
                data_path, self.cfg, fixed_split_indices=fixed_split_indices
            )
        self.data = data
        self.bin_index = bin_index
        self.sample_index = sample_index
        self.split_indices = split_indices
        
        # Build neighbours graph and latent solver
        self.neighbour_graph = NeighbourGraph(self.cfg, bins_df)
        self.neighbour_graph.build()
        
        latent_solver = LatentSolver(self.cfg, self.neighbour_graph)
        latent_solver.build_V_and_H(data["train"]["X"], bin_index, method="nw")
        
        # Build model
        self.device = torch.device(self.cfg.device)
        input_dim = data["train"]["X"].shape[1]
        self.input_dim = input_dim
        
        # In two-phase training, Phase 1 uses input_dim only (no latent).
        # _expand_mlp_for_latent() will expand the first layer after Phase 1.
        # Otherwise, build the full input_dim + latent_dim MLP from the start.
        mlp_input_dim = input_dim + self.cfg.latent_dim
        mlp_model = MLPModel(mlp_input_dim, hidden_dims=[64, 128, 64, 32], dropout=self.cfg.dropout).to(self.device)
        
        self.model = Model(
            mlp_model, 
            latent_solver, 
            n_bins=len(bin_index),
            latent_dim=self.cfg.latent_dim,
            latent_init_std=self.cfg.latent_init_std,
            device=self.device
        )
        
        # Pre-compute H as a torch sparse tensor for smoothness regularization in train_epoch
        H_coo = latent_solver.H.tocoo()
        H_indices = torch.LongTensor(np.vstack([H_coo.row, H_coo.col]))
        H_values = torch.FloatTensor(H_coo.data)
        self.H_torch = torch.sparse_coo_tensor(
            H_indices, H_values, size=H_coo.shape, device=self.device
        )

        # Joint optimizer: MLP and latent trained simultaneously with separate learning rates
        self.optimizer = torch.optim.AdamW([
            {"params": self.model.mlp.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": [self.model.latent_embedding.weight], "lr": cfg.latent_lr, "weight_decay": 0.0},
        ])
        # LR scheduler — total steps = epochs (one scheduler step per epoch)
        total_steps = self.cfg.epochs
        warmup_steps = max(1, int(0.1 * total_steps))
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )
        
        # Loss configuration
        self.loss_type = loss_type
        self.loss_mode = "sample" if loss_type == "cross_entropy" else "bin"
        self.criterion = Loss(task=loss_type)
        
        # Build datasets
        log.debug(f"\n   Building datasets...")
        train = MBDataset(data["train"], bin_index, sample_index, loss_mode=self.loss_mode)
        val = MBDataset(data["val"], bin_index, sample_index, loss_mode=self.loss_mode)
        test = MBDataset(data["test"], bin_index, sample_index, loss_mode=self.loss_mode)

        # Always build an ordered bin-mode dataset for latent solving and aligned diagnostics.
        # This guarantees that (inputs, targets, bin_idx, sample_idx) share a single, consistent row order.
        train_bin_ordered = MBDataset(data["train"], bin_index, sample_index, loss_mode="bin")
        
        # Initialize data loaders
        # For sample mode, use custom collate function to handle variable-length samples
        batch_size = cfg.batch_size_sample if self.loss_mode == "sample" else cfg.batch_size_bin
        collate_fn = collate_samples if self.loss_mode == "sample" else None
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val, batch_size=1, shuffle=False, collate_fn=collate_fn)
        self.test_loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # Non-shuffled loader for stable evaluation/tracking (sample-mode ordering may not match row ordering)
        self.train_loader_ordered = DataLoader(train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Canonical ordered loader for latent solving (row-aligned with its own targets)
        self.train_loader_bin_ordered = DataLoader(
            train_bin_ordered,
            batch_size=self.cfg.batch_size_bin,
            shuffle=False,
            collate_fn=None,
        )
        
        # Loss tracking for visualization
        self.train_losses = []  # List of (0, epoch, loss) tuples
        self.val_losses = []    # List of (0, epoch, loss) tuples

        # Latent importance diagnostics (populated during run())
        # Each entry: dict with keys epoch, weight_norm_ratio, embedding_std,
        #             ablation_delta (only every diag_ablation_interval epochs)
        self.latent_diagnostics: List[Dict[str, Any]] = []

        # Prepare save path
        root = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(root, "models", f"{time.strftime('%Y-%m-%d_%H:%M')}.pt")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_latent_weight_ratio(self) -> float:
        """
        Compute the per-column weight activity ratio of latent vs feature inputs
        in the MLP first layer.

        Returns latent_col_mean_norm / feat_col_mean_norm, where:
            feat_col_mean_norm  = ||W[:, :input_dim]||_F  / input_dim
            latent_col_mean_norm= ||W[:, input_dim:]||_F  / latent_dim

        A ratio < 1 means latent columns are weaker than feature columns on
        average; approaching 0 signals the MLP is ignoring the latent input.
        """
        W = self.model.mlp.net[0].weight.data  # [hidden0, input_dim + latent_dim]
        W_feat   = W[:, :self.input_dim]
        W_latent = W[:, self.input_dim:]
        feat_col_mean   = W_feat.norm(p='fro').item()   / self.input_dim
        latent_col_mean = W_latent.norm(p='fro').item() / self.cfg.latent_dim
        return latent_col_mean / (feat_col_mean + 1e-12)

    @torch.no_grad()
    def _compute_ablation_delta(self) -> float:
        """
        Compute the increase in validation loss when the latent embedding is zeroed.
        A small delta (≈ 0) means the MLP is not using the latent at all.
        """
        # Save embeddings
        saved = self.model.latent_embedding.weight.data.clone()
        # Zero out
        self.model.latent_embedding.weight.data.zero_()
        loss_no_latent = self.validate(split="val")
        # Restore
        self.model.latent_embedding.weight.data.copy_(saved)
        loss_with_latent = self.validate(split="val")
        return loss_no_latent - loss_with_latent

    def _collect_diagnostics(self, epoch: int, run_ablation: bool = False) -> Dict[str, Any]:
        """Collect one snapshot of latent importance diagnostics."""
        diag: Dict[str, Any] = {
            "epoch": epoch,
            "weight_norm_ratio": self._compute_latent_weight_ratio(),
            "embedding_std": float(self.model.latent_embedding.weight.data.std()),
            "ablation_delta": None,
        }
        if run_ablation:
            diag["ablation_delta"] = self._compute_ablation_delta()
        return diag

    # ------------------------------------------------------------------
    def train_epoch(self) -> float:
        """Train for one epoch. Returns average CE loss (regularization excluded)."""
        self.model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in self.train_loader:
            inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)

            if self.loss_mode == "sample":
                # Sample mode: inputs [B, max_bins, features], bin_idx [B, max_bins]
                # Forward pass on all bins at once
                B, max_bins, n_feat = inputs.shape
                inputs_flat = inputs.view(B * max_bins, n_feat)  # [B*max_bins, features]
                bin_idx_flat = bin_idx.view(B * max_bins)        # [B*max_bins]
                
                outputs_flat = self.model(inputs_flat, bin_idx_flat)  # [B*max_bins]
                outputs = outputs_flat.view(B, max_bins)              # [B, max_bins]
                
                # Apply mask: set padded positions to large negative (will become ~0 after softmax)
                outputs = outputs.masked_fill(mask == 0, float('-inf'))
                
                # Loss expects [B, max_bins] logits, [B, max_bins] target probs, and mask
                loss = self.criterion(outputs, targets, mask)
            else:
                # Bin mode: standard forward pass
                outputs = self.model(inputs, bin_idx)
                loss = self.criterion(outputs, targets)

            # Smoothness regularization: λ * ||Z - HZ||^2
            Z = self.model.latent_embedding.weight  # [n_bins, latent_dim]
            HZ = torch.sparse.mm(self.H_torch, Z)
            smooth_loss = self.cfg.latent_smooth_reg * torch.sum((Z - HZ) ** 2)
            # Norm regularization: λ_norm * ||Z||^2
            norm_loss = self.cfg.latent_norm_reg * torch.sum(Z ** 2)
            total_loss = loss + smooth_loss + norm_loss

            self.optimizer.zero_grad()
            total_loss.backward()

            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.mlp.parameters()) + [self.model.latent_embedding.weight],
                    self.cfg.grad_clip,
                )

            self.optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size  # track CE loss only
            n_samples += batch_size

        epoch_loss = running_loss / max(1, n_samples)
        return epoch_loss


    @torch.no_grad()
    def validate(self, split: Literal["train", "val", "test"]) -> float:
        data_loader = (
            self.val_loader if split == "val" else 
            self.test_loader if split == "test" else 
            self.train_loader
        )
        if data_loader is None:
            raise ValueError(f"Unknown split {split}")
        
        self.model.eval()
        running_loss = 0.0
        n_samples = 0

        for batch in data_loader:
            inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
            
            if self.loss_mode == "sample":
                B, max_bins, n_feat = inputs.shape
                inputs_flat = inputs.view(B * max_bins, n_feat)
                bin_idx_flat = bin_idx.view(B * max_bins)
                
                outputs_flat = self.model(inputs_flat, bin_idx_flat)
                outputs = outputs_flat.view(B, max_bins)
                outputs = outputs.masked_fill(mask == 0, float('-inf'))
                
                loss = self.criterion(outputs, targets, mask)
            else:
                outputs = self.model(inputs, bin_idx)
                loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        val_loss = running_loss / max(1, n_samples)
        return val_loss

    @torch.no_grad()
    def compute_metrics(self, split: str) -> Dict[str, float]:
        """Compute interpretable metrics for a given split."""
        data_loader = (
            self.train_loader if split == "train" else
            self.val_loader if split == "val" else
            self.test_loader if split == "test" else
            None
        )
        if data_loader is None:
            return {}
        self.model.eval()
        
        # Metrics to compute
        classification_accuracy = []
        classification_precision = []
        classification_recall = []
        classification_specificity = []
        classification_f1 = []
        mae_when_present = []
        mae_when_absent = []
        
        for batch in data_loader:
            inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
            
            if self.loss_mode == "sample":
                # Sample mode: inputs [B, max_bins, features]
                B, max_bins, n_feat = inputs.shape
                inputs_flat = inputs.view(B * max_bins, n_feat)
                bin_idx_flat = bin_idx.view(B * max_bins)
                
                outputs_flat = self.model(inputs_flat, bin_idx_flat)
                outputs = outputs_flat.view(B, max_bins)
                
                # Apply mask if present, else create default mask
                if mask is not None:
                    outputs = outputs.masked_fill(mask == 0, float('-inf'))
                    mask_use = mask
                else:
                    mask_use = torch.ones((B, max_bins), device=outputs.device)
                
                # Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=-1)
                
                # Flatten but only keep valid (non-padded) entries
                mask_flat = mask_use.view(-1).bool()
                probs_np = probs.view(-1)[mask_flat].cpu().numpy()
                targets_np = targets.view(-1)[mask_flat].cpu().numpy()
            else:
                outputs = self.model(inputs, bin_idx)
                probs = torch.sigmoid(outputs)
                probs_np = probs.cpu().numpy().flatten()
                targets_np = targets.cpu().numpy().flatten()
            
            # Classification metrics (predicted absent when prob < epsilon)
            epsilon = 1e-2
            tp = sum((p >= epsilon) and (t > 0) for p, t in zip(probs_np, targets_np))
            tn = sum((p < epsilon) and (t == 0) for p, t in zip(probs_np, targets_np))
            fp = sum((p >= epsilon) and (t == 0) for p, t in zip(probs_np, targets_np))
            fn = sum((p < epsilon) and (t > 0) for p, t in zip(probs_np, targets_np))
            accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
            precision = tp / max(1, (tp + fp))
            recall = tp / max(1, (tp + fn))
            specificity = tn / max(1, (tn + fp))
            f1 = 2 * (precision * recall) / max(1e-12, (precision + recall))
            classification_accuracy.append(accuracy)
            classification_precision.append(precision)
            classification_recall.append(recall)
            classification_specificity.append(specificity)
            classification_f1.append(f1)
            
            # MAE when present / absent
            for p, t in zip(probs_np, targets_np):
                if t > 0:
                    mae_when_present.append(abs(p - t))
                else:
                    mae_when_absent.append(abs(p - t))
        
        return {
            "classification_accuracy": float(np.mean(classification_accuracy)),
            "classification_precision": float(np.mean(classification_precision)),
            "classification_recall": float(np.mean(classification_recall)),
            "classification_specificity": float(np.mean(classification_specificity)),
            "classification_f1": float(np.mean(classification_f1)),
            "mae_when_present": float(np.mean(mae_when_present)) if mae_when_present else 0.0,
            "mae_when_absent": float(np.mean(mae_when_absent)) if mae_when_absent else 0.0,
        }

    @torch.no_grad()
    def get_predictions(self, split: Literal["train", "val", "test"] = "test") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions and targets for a given split.
        Returns:
            Tuple of (predictions, targets, sample_labels, bin_labels) as flat 1D arrays.
            - predictions:   float32 (N,) — predicted probabilities/values
            - targets:       float32 (N,) — true relative abundances
            - sample_labels: (N,) str    — sample ID for each entry (from self.sample_index)
            - bin_labels:    (N,) str    — BIN URI for each entry (from self.bin_index)
            where N = total valid (sample, BIN) pairs across the requested split.
        """
        data_loader = (
            self.train_loader if split == "train" else
            self.val_loader if split == "val" else
            self.test_loader
        )
        self.model.eval()
        # Accumulate per-sample lists
        sample_pred: Dict[int, list] = {}
        sample_true: Dict[int, list] = {}
        sample_bins: Dict[int, list] = {}  # integer bin indices (from self.bin_index)
        for batch in data_loader:
            if self.loss_mode == "sample":
                inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
                B = inputs.shape[0]
                for b in range(B):
                    s_idx = sample_idx[b].item() if hasattr(sample_idx[b], 'item') else int(sample_idx[b])
                    
                    # Handle mask - if None, all are valid
                    if mask is not None:
                        valid_mask = mask[b].bool()
                    else:
                        valid_mask = torch.ones(inputs.shape[1], dtype=torch.bool, device=inputs.device)
                    
                    inputs_flat = inputs[b][valid_mask]
                    bin_idx_flat = bin_idx[b][valid_mask]
                    outputs = self.model(inputs_flat, bin_idx_flat)
                    outputs = outputs.unsqueeze(0)  # [1, n_bins]
                    probs = F.softmax(outputs, dim=-1).squeeze(0).cpu().numpy()
                    y_true = targets[b][valid_mask].cpu().numpy()
                    sample_pred.setdefault(s_idx, []).extend(probs.tolist())
                    sample_true.setdefault(s_idx, []).extend(y_true.tolist())
                    sample_bins.setdefault(s_idx, []).extend(bin_idx_flat.cpu().numpy().tolist())
            else:
                # Bin mode: batch is a dict (from DataLoader, no collate_fn)
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                bin_idx = batch["bin_idx"].to(self.device)
                sample_idx = batch["sample_idx"].to(self.device)
                # If batch_size=1, make sure dims are right
                if len(inputs.shape) == 1:
                    inputs = inputs.unsqueeze(0)
                    targets = targets.unsqueeze(0)
                    bin_idx = bin_idx.unsqueeze(0)
                    sample_idx = sample_idx.unsqueeze(0)
                for i in range(inputs.shape[0]):
                    s_idx = sample_idx[i].item() if hasattr(sample_idx[i], 'item') else int(sample_idx[i])
                    b_idx = int(bin_idx[i].item() if hasattr(bin_idx[i], 'item') else bin_idx[i])
                    input_i = inputs[i].unsqueeze(0)
                    bin_idx_i = bin_idx[i].unsqueeze(0)
                    output = self.model(input_i, bin_idx_i)
                    prob = torch.sigmoid(output).cpu().numpy().item()
                    y_true = targets[i].cpu().numpy().item()
                    sample_pred.setdefault(s_idx, []).append(prob)
                    sample_true.setdefault(s_idx, []).append(y_true)
                    sample_bins.setdefault(s_idx, []).append(b_idx)
        
        # Normalize predictions per sample if using logistic loss
        if self.loss_type == "logistic":
            for s_idx in sample_pred:
                preds = np.array(sample_pred[s_idx])
                pred_sum = preds.sum()
                if pred_sum > 0:
                    sample_pred[s_idx] = (preds / pred_sum).tolist()
        
        # Invert index dicts to recover string labels
        idx_to_sample = {v: k for k, v in self.sample_index.items()}
        idx_to_bin = {v: k for k, v in self.bin_index.items()}
        # Flatten into 1D arrays, ordered by sample_idx
        preds_flat, trues_flat, s_labels, b_labels = [], [], [], []
        for s_idx in sorted(sample_pred.keys()):
            n = len(sample_pred[s_idx])
            preds_flat.extend(sample_pred[s_idx])
            trues_flat.extend(sample_true[s_idx])
            s_labels.extend([idx_to_sample[s_idx]] * n)
            b_labels.extend([idx_to_bin[int(b)] for b in sample_bins[s_idx]])
        return (
            np.array(preds_flat, dtype=np.float32),
            np.array(trues_flat, dtype=np.float32),
            np.array(s_labels),
            np.array(b_labels),
        )

    def run(self, use_wandb: bool = True) -> Dict[str, Any]:
        """Run joint MLP+latent training."""

        best_val = float('inf')
        pbar = tqdm(range(self.cfg.epochs), desc="Training", leave=True)
        for epoch in pbar:
            self.train_epoch()
            train_loss = self.validate(split="train")
            val_loss = self.validate(split="val")

            self.train_losses.append((0, epoch, train_loss))
            self.val_losses.append((0, epoch, val_loss))

            # Collect latent importance diagnostics every epoch (ablation periodically)
            run_ablation = (
                self.cfg.diag_ablation_interval > 0
                and epoch % self.cfg.diag_ablation_interval == 0
            )
            diag = self._collect_diagnostics(epoch, run_ablation=run_ablation)
            self.latent_diagnostics.append(diag)

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "latent_weight_ratio": diag["weight_norm_ratio"],
                    "latent_embedding_std": diag["embedding_std"],
                    **({"ablation_delta": diag["ablation_delta"]}
                       if diag["ablation_delta"] is not None else {}),
                })

            self.scheduler.step()

            if epoch % 10 == 0:
                pbar.set_postfix({
                    "train": f"{train_loss:.6f}",
                    "val": f"{val_loss:.6f}",
                    "latent_ratio": f"{diag['weight_norm_ratio']:.3f}",
                    "emb_std": f"{diag['embedding_std']:.4f}",
                })
                log.info(
                    f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}, "
                    f"latent_ratio={diag['weight_norm_ratio']:.4f}, "
                    f"emb_std={diag['embedding_std']:.6f}"
                    + (f", ablation_delta={diag['ablation_delta']:.6f}"
                       if diag["ablation_delta"] is not None else "")
                )

            if val_loss < best_val:
                best_val = val_loss
                self.model.save_model(self.save_path)

        self._plot_training_progress()
        self._plot_latent_importance()

        # Load best checkpoint before final evaluation
        if os.path.exists(self.save_path):
            self.model.load_model(self.save_path)
            log.debug(f"Loaded best checkpoint from {self.save_path} for final evaluation")

        predictions, targets, sample_labels, bin_labels = self.get_predictions(split="test")
        latent_embeddings = self.model.get_latent()  # [n_bins, latent_dim]
        return {
            "best_val_loss": best_val,
            "predictions": predictions,
            "targets": targets,
            "sample_labels": sample_labels,
            "bin_labels": bin_labels,
            "latent_embeddings": latent_embeddings,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "latent_diagnostics": self.latent_diagnostics,
            # Provide timeline-compatible format for the visualisation script:
            # (phase, cycle, step, loss) — here all are in a single "joint" phase
            "timeline_train_losses": [("joint", 0, e, l) for _, e, l in self.train_losses],
            "timeline_val_losses": [("joint", 0, e, l) for _, e, l in self.val_losses],
            "cycle_train_losses": [],
            "cycle_val_losses": [],
        }

    def _plot_latent_importance(self) -> None:
        """Plot latent importance diagnostics over training epochs."""
        if not self.latent_diagnostics:
            return

        epochs     = [d["epoch"]             for d in self.latent_diagnostics]
        ratios     = [d["weight_norm_ratio"]  for d in self.latent_diagnostics]
        emb_stds   = [d["embedding_std"]      for d in self.latent_diagnostics]

        # Ablation deltas (sparse — only recorded every N epochs)
        ab_epochs  = [d["epoch"]          for d in self.latent_diagnostics if d["ablation_delta"] is not None]
        ab_deltas  = [d["ablation_delta"] for d in self.latent_diagnostics if d["ablation_delta"] is not None]

        n_plots = 3 if ab_epochs else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 2:
            axes = list(axes)

        # Panel 1: Weight norm ratio
        ax = axes[0]
        ax.plot(epochs, ratios, color='steelblue', linewidth=1.8)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='ratio = 1 (equal activity)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Latent / Feature column mean norm')
        ax.set_title('MLP First-Layer: Latent Input Weight Activity\n(ratio < 1 → MLP underusing latent)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2: Embedding std
        ax = axes[1]
        ax.plot(epochs, emb_stds, color='darkorange', linewidth=1.8)
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Std of latent embedding weights')
        ax.set_title('Latent Embedding Activity\n(≈ 0 → embedding collapsed)')
        ax.grid(True, alpha=0.3)

        # Panel 3 (optional): Ablation delta
        if ab_epochs:
            ax = axes[2]
            colors = ['green' if d >= 0 else 'crimson' for d in ab_deltas]
            ax.bar(ab_epochs, ab_deltas, color=colors, alpha=0.75, width=max(1, ab_epochs[1]-ab_epochs[0]) * 0.8 if len(ab_epochs)>1 else 5)
            ax.axhline(0.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Val loss increase when latent zeroed')
            ax.set_title('Latent Ablation Delta\n(> 0 → latent is helping; ≈ 0 → latent ignored)')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        root = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(root, "figures", "latent_importance.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Latent importance plot saved to: {save_path}")
        plt.close()

    def _plot_training_progress(self) -> None:
        """Plot training and validation losses over epochs."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))

        epochs = [e for _, e, _ in self.train_losses]
        train_vals = [l for _, _, l in self.train_losses]
        val_vals = [l for _, _, l in self.val_losses]

        ax.plot(epochs, train_vals, 'b-', alpha=0.8, linewidth=1.5, label='Train Loss (CE)')
        ax.plot(epochs, val_vals, 'r-', alpha=0.8, linewidth=1.5, label='Val Loss (CE)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Joint Training Progress: MLP + Latent')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Save figure
        root = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(root, "figures", "training_progress.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"Training progress plot saved to: {save_path}")
        plt.close()

    def _to_device(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Move batch tensors to device. Returns mask only for sample mode."""
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)
        bin_idx = batch["bin_idx"].to(self.device)
        sample_idx = batch["sample_idx"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)
        return inputs, targets, bin_idx, sample_idx, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metabarcoding training script")
    group_ = parser.add_mutually_exclusive_group(required=False)
    group_.add_argument("--data_path", type=str, default=None,
                        help="Path to raw data CSV file (e.g. data/ecuador_training_data.csv)")
    group_.add_argument("--data_dir", type=str, default=None,
                        help="Path to directory containing processed CSV files (X_*.csv, y_*.csv, taxonomic_data.csv)")
    parser.add_argument("--loss_type", type=str, choices=["cross_entropy", "logistic"], 
                        default="cross_entropy", help="Type of loss function to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()
    
    # Set some generic configurations
    set_seed()
    cfg = Config()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Initialize Weights & Biases (optional)
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if not WANDB_AVAILABLE:
        log.warning("wandb is not installed; continuing without wandb logging")
    else:
        wandb.init(
            project="metabarcoding",
            name=time.strftime("%Y-%m-%d_%H-%M"),
            config=asdict(cfg),
            tags=["cross-entropy", "ecuador", "mlp+latent"],
            dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wandb")
        )

    # Create Trainer and run
    # Default behavior: if neither is provided, fall back to raw CSV in data/
    data_path = args.data_path
    data_dir = args.data_dir
    if data_path is None and data_dir is None:
        data_path = "data/ecuador_training_data.csv"

    trainer = Trainer(cfg, data_path=data_path, data_dir=data_dir, loss_type=args.loss_type)

    log.debug(f"\n   Starting training...")
    results = trainer.run(use_wandb=use_wandb)

    # Save results to pickle for downstream visualization
    os.makedirs(cfg.results_dir, exist_ok=True)
    pkl_path = os.path.join(cfg.results_dir, f"results_{time.strftime('%Y-%m-%d_%H-%M')}.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(results, fh)
    log.info(f"Results saved to: {pkl_path}")

    # Evaluate on test set
    log.debug(f"\n   Final evaluation...")

    test_loss = trainer.validate(split="test")
    log.info(f"Test loss: {test_loss:.6f}")

    log.info("\n" + "="*50)
    log.info("METRICS")
    log.info("="*50)
    for split_name in ["val", "test"]: # FIXME : + train
        loss_ = trainer.validate(split=split_name)  # type: ignore
        metrics = trainer.compute_metrics(split_name)        
        if use_wandb:
            wandb.log({
                f"final_{split_name}_loss": loss_,
                **{f"{split_name}_{k}": v for k, v in metrics.items()}
            })

    # Finish wandb run
    if use_wandb:
        wandb.finish()