from __future__ import annotations

import argparse
import re
import time
from typing import Literal, Optional, List, Tuple, Dict, Any, Union, Callable
import logging as log
from dataclasses import asdict
import os

import pickle
import numpy as np
import pandas as pd
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
from dataset import MBDataset, collate_samples
from loss import Loss
from mlp import MLPModel
from utils import load, load_processed

# Import from local folder (modified for latent-as-input)
from latent_solver import LatentSolver
from model import Model
from config import Config, set_seed

class Trainer:
    """
    EM-style alternation trainer for the V2 (blend) model:
        Phase A: Solve D (output scalar latent) analytically using CG/L-BFGS, given fixed MLP + Z.
        Phase B: Train MLP + Z (input embedding) jointly via gradient descent, given fixed D.

    This is a direct blend of:
      - V1 (latent_as_input): Phase B is identical — MLP and Z are gradient-trained together.
      - Baseline (src):       Phase A is identical — D is solved analytically each cycle.

    The sole architectural difference from V1 is the +D[b] output term.
    The sole architectural difference from baseline is the Z embedding concatenated to MLP input.
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
        
        # Adjust MLP input dimension: original features + latent_dim
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

        # Phase B optimizer: MLP + Z jointly; D is excluded (solved analytically in Phase A)
        self.optimizer = torch.optim.AdamW([
            {"params": self.model.mlp.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": [self.model.latent_embedding.weight], "lr": cfg.latent_lr, "weight_decay": 0.0},
        ])
        
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

        # LR scheduler — total steps = cycles × batches per epoch (one scheduler step per batch)
        # Must be built after train_loader is defined so len(self.train_loader) is available.
        total_steps = self.cfg.max_cycles * len(self.train_loader)
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
        
        # Loss tracking for visualization (EM-style, mirrors baseline src)
        self.train_losses = []              # List of (cycle, epoch, loss) tuples (Phase B only)
        self.val_losses = []                # List of (cycle, epoch, loss) tuples (Phase B only)
        self.cycle_train_losses = []        # End-of-cycle train losses
        self.cycle_val_losses = []          # End-of-cycle val losses
        self.timeline_train_losses = []     # List of (phase, cycle, step, loss) tuples
        self.timeline_val_losses = []       # List of (phase, cycle, step, loss) tuples

        # Latent importance diagnostics for Z (populated during Phase B of each cycle)
        # Each entry: dict with keys epoch, weight_norm_ratio, embedding_std,
        #             ablation_delta (only every diag_ablation_interval epochs)
        self.latent_diagnostics: List[Dict[str, Any]] = []

        # D std per cycle (recorded at end of each Phase A solve)
        self.d_stds_per_cycle: List[Tuple[int, float]] = []
        
        # Prepare save path
        root = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(root, "models", f"{time.strftime('%Y-%m-%d_%H:%M')}.pt")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)


    def _train_batch(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform one MLP+Z gradient step on a single batch. Returns batch CE loss (regularization excluded)."""
        self.model.train()
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

        # Smoothness regularization on Z: λ * ||Z - HZ||^2 (D is fixed, solved analytically)
        Z = self.model.latent_embedding.weight  # [n_bins, latent_dim]
        HZ = torch.sparse.mm(self.H_torch, Z)   # [n_bins, latent_dim]
        smooth_loss_z = self.cfg.latent_smooth_reg * torch.sum((Z - HZ) ** 2)
        norm_loss_z = self.cfg.latent_norm_reg * torch.sum(Z ** 2)

        total_loss = loss + smooth_loss_z + norm_loss_z

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.mlp.parameters()) + [self.model.latent_embedding.weight],
                self.cfg.grad_clip,
            )

        self.optimizer.step()
        return loss.item()

    def solve_latent(self, prox_weight: float = 0.0) -> np.ndarray:
        """Phase A: Solve for scalar output latent D analytically (CG/L-BFGS), MLP + Z fixed.

        The 'intrinsic' vector passed to the solver is mlp([x, z_b]) — the MLP output
        that already incorporates the current Z embedding, so D is solved as:
            output = mlp([x, z_b]) + D[b]

        This mirrors baseline src/train.py solve_latent() but the intrinsic now includes Z.
        """
        self.model.eval()

        intrinsic_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        bin_list: List[np.ndarray] = []
        sample_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch in self.train_loader_bin_ordered:
                x = batch["input"].to(self.device)
                bin_idx_t = batch["bin_idx"].to(self.device)
                y = batch["target"].cpu().numpy()
                bin_idx = batch["bin_idx"].cpu().numpy()
                sample_idx = batch["sample_idx"].cpu().numpy()

                # Augment input with current Z embeddings (Z is fixed during Phase A)
                z = self.model.latent_embedding(bin_idx_t)  # [N, latent_dim]
                x_aug = torch.cat([x, z], dim=-1)           # [N, input_dim + latent_dim]
                intrinsic = self.model.mlp(x_aug).detach().cpu().numpy().reshape(-1)

                intrinsic_list.append(intrinsic)
                y_list.append(y.reshape(-1))
                bin_list.append(bin_idx.reshape(-1))
                sample_list.append(sample_idx.reshape(-1))

        intrinsic_vec = np.concatenate(intrinsic_list, axis=0)
        y_vec = np.concatenate(y_list, axis=0)
        bin_ids = np.concatenate(bin_list, axis=0).astype(np.int64)
        sample_ids = np.concatenate(sample_list, axis=0).astype(np.int64)

        x0_latent = self.model.latent_vec.detach().cpu().numpy()
        latent_vec = self.model.latent_solver.solve(
            y=y_vec,
            intrinsic_vec=intrinsic_vec,
            bin_ids=bin_ids,
            sample_ids=sample_ids,
            loss_type=self.loss_type,
            x0=x0_latent,
            prox_weight=prox_weight,
            x_anchor=x0_latent,
        )
        self.model.set_latent_vec(latent_vec)
        return latent_vec

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_latent_weight_ratio(self) -> float:
        """
        Compute the per-column weight activity ratio of latent Z vs feature inputs
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
        Compute the increase in validation loss when Z is zeroed (D is kept intact).
        A small delta (≈ 0) means the MLP is not using the latent input Z at all.
        """
        # Save Z embeddings
        saved = self.model.latent_embedding.weight.data.clone()
        # Zero out Z only (D is untouched to isolate Z's contribution)
        self.model.latent_embedding.weight.data.zero_()
        loss_no_latent = self.validate(split="val")
        # Restore
        self.model.latent_embedding.weight.data.copy_(saved)
        loss_with_latent = self.validate(split="val")
        return loss_no_latent - loss_with_latent

    def _collect_diagnostics(self, epoch: int, run_ablation: bool = False) -> Dict[str, Any]:
        """Collect one snapshot of Z latent importance diagnostics (Phase B, per epoch)."""
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
        """Run EM-style per-batch alternation: for each batch, solve D then take one MLP+Z gradient step."""
        warmup_cycles = max(1, int(self.cfg.latent_warmup_frac * self.cfg.max_cycles))
        log.debug(f"Starting per-batch alternation training (D warms up over first {warmup_cycles} cycles)...")
        best_val_loss = float('inf')

        for cycle in tqdm(range(self.cfg.max_cycles), desc="Cycles", leave=False):
            # Proximal weight ρ: decays from ρ₀ → 0 over warmup_cycles (damped EM).
            alpha = min(1.0, cycle / warmup_cycles)
            phase_tag = "latent_warmup" if alpha < 1.0 else "latent"
            prox_weight = self.cfg.latent_prox_scale * self.cfg.latent_l2_reg * (1.0 - alpha)

            # Per-batch alternation: solve D then take one MLP+Z gradient step
            latent_vec = self.model.latent_vec.detach().cpu().numpy()  # fallback if loader is empty
            for batch in self.train_loader:
                latent_vec = self.solve_latent(prox_weight=prox_weight)
                self._train_batch(batch)
                self.scheduler.step()

            d_std = float(np.std(latent_vec))
            self.d_stds_per_cycle.append((cycle, d_std))

            # End-of-cycle evaluation
            train_loss = self.validate(split="train")
            val_loss = self.validate(split="val")

            self.timeline_train_losses.append((phase_tag, cycle, 0, train_loss))
            self.timeline_val_losses.append((phase_tag, cycle, 0, val_loss))
            self.train_losses.append((cycle, 0, train_loss))
            self.val_losses.append((cycle, 0, val_loss))
            self.cycle_train_losses.append((cycle, train_loss))
            self.cycle_val_losses.append((cycle, val_loss))

            # Z diagnostics (once per cycle)
            run_ablation = (
                self.cfg.diag_ablation_interval > 0
                and cycle % self.cfg.diag_ablation_interval == 0
            )
            diag = self._collect_diagnostics(cycle, run_ablation=run_ablation)
            self.latent_diagnostics.append(diag)

            if use_wandb and WANDB_AVAILABLE:
                metrics = self.compute_metrics(split="val")
                wandb.log({
                    "cycle": cycle,
                    "final_train_loss": train_loss,
                    "final_val_loss": val_loss,
                    "latent_vec_std": d_std,
                    **{f"val_{k}": v for k, v in metrics.items()},
                    "latent_weight_ratio": diag["weight_norm_ratio"],
                    "latent_embedding_std": diag["embedding_std"],
                    **({"ablation_delta": diag["ablation_delta"]}
                       if diag["ablation_delta"] is not None else {}),
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_model(self.save_path)

            log.info(
                f"Cycle {cycle} (ρ={prox_weight:.4f}): val_loss={val_loss:.6f}, "
                f"train_loss={train_loss:.6f}, D_std={d_std:.4f}"
            )

        # ----------------------------------------------------------------
        # Final evaluation
        # ----------------------------------------------------------------
        if os.path.exists(self.save_path):
            self.model.load_model(self.save_path)
            log.debug(f"Loaded best checkpoint from {self.save_path} for final evaluation")

        self._plot_training_progress()
        self._plot_latent_importance()

        predictions, targets, sample_labels, bin_labels = self.get_predictions(split="test")
        latent_embeddings = self.model.get_latent()
        latent_vec_final = self.model.latent_vec.detach().cpu().numpy()
        return {
            "best_val_loss": best_val_loss,
            "predictions": predictions,
            "targets": targets,
            "sample_labels": sample_labels,
            "bin_labels": bin_labels,
            "latent_embeddings": latent_embeddings,
            "latent_vec": latent_vec_final,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "cycle_train_losses": self.cycle_train_losses,
            "cycle_val_losses": self.cycle_val_losses,
            "timeline_train_losses": self.timeline_train_losses,
            "timeline_val_losses": self.timeline_val_losses,
            "latent_diagnostics": self.latent_diagnostics,
        }

    def _plot_latent_importance(self) -> None:
        """Plot Z latent importance diagnostics (collected in Phase B across all cycles)."""
        if not self.latent_diagnostics:
            return

        epochs   = [d["epoch"]             for d in self.latent_diagnostics]
        ratios   = [d["weight_norm_ratio"]  for d in self.latent_diagnostics]
        emb_stds = [d["embedding_std"]      for d in self.latent_diagnostics]

        # Ablation deltas (sparse — only recorded every diag_ablation_interval epochs)
        ab_epochs  = [d["epoch"]          for d in self.latent_diagnostics if d["ablation_delta"] is not None]
        ab_deltas  = [d["ablation_delta"] for d in self.latent_diagnostics if d["ablation_delta"] is not None]

        n_plots = 3 if ab_epochs else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

        # Panel 1: Weight norm ratio (Z vs features in MLP first layer)
        ax = axes[0]
        ax.plot(epochs, ratios, color='steelblue', linewidth=1.8)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='ratio = 1 (equal activity)')
        ax.set_xlabel('Phase-B epoch (cumulative)')
        ax.set_ylabel('Latent Z / Feature column mean norm')
        ax.set_title('MLP First-Layer: Latent Z Input Weight Activity\n(ratio < 1 → MLP underusing latent)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2: Embedding Z std
        ax = axes[1]
        ax.plot(epochs, emb_stds, color='darkorange', linewidth=1.8)
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlabel('Phase-B epoch (cumulative)')
        ax.set_ylabel('Std of Z embedding weights')
        ax.set_title('Z Embedding Activity\n(≈ 0 → embedding collapsed)')
        ax.grid(True, alpha=0.3)

        # Panel 3 (optional): Z ablation delta
        if ab_epochs:
            ax = axes[2]
            colors = ['green' if d >= 0 else 'crimson' for d in ab_deltas]
            ax.bar(ab_epochs, ab_deltas, color=colors, alpha=0.75,
                   width=max(1, ab_epochs[1]-ab_epochs[0]) * 0.8 if len(ab_epochs)>1 else 5)
            ax.axhline(0.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            ax.set_xlabel('Phase-B epoch (cumulative)')
            ax.set_ylabel('Val loss increase when Z zeroed')
            ax.set_title('Z Ablation Delta\n(> 0 → Z is helping; ≈ 0 → Z ignored)')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        root = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(root, "figures", "latent_importance.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Latent importance plot saved to: {save_path}")
        plt.close()

    def _plot_training_progress(self) -> None:
        """Plot training and validation losses over time (EM cycle view + timeline view)."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Full timeline (init + Phase A markers + Phase B epochs)
        train_vals = [l for _, _, _, l in self.timeline_train_losses]
        val_vals   = [l for _, _, _, l in self.timeline_val_losses]
        steps = list(range(len(train_vals)))

        ax1.plot(steps, train_vals, 'b-', alpha=0.7, linewidth=1.2, label='Train Loss')
        ax1.plot(steps, val_vals,   'r-', alpha=0.7, linewidth=1.2, label='Val Loss')

        # Mark Phase A (latent solve) steps with gray dashed lines
        latent_steps = [
            i for i, (phase, _, _, _) in enumerate(self.timeline_train_losses)
            if phase == "latent"
        ]
        for boundary in latent_steps:
            ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.25, linewidth=1)

        ax1.set_xlabel('Training Step (Init + EM Cycles)')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress: Full Timeline (Phase A markers = dashed)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: End-of-cycle losses
        if self.cycle_train_losses:
            cycle_nums  = [c for c, l in self.cycle_train_losses]
            cycle_train = [l for c, l in self.cycle_train_losses]
            cycle_val   = [l for c, l in self.cycle_val_losses]
            ax2.plot(cycle_nums, cycle_train, 'bo-', linewidth=2, markersize=8, label='Train Loss', alpha=0.7)
            ax2.plot(cycle_nums, cycle_val,   'ro-', linewidth=2, markersize=8, label='Val Loss',   alpha=0.7)
            ax2.set_xlabel('EM Cycle')
            ax2.set_ylabel('Loss')
            ax2.set_title('End-of-Cycle Losses')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
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