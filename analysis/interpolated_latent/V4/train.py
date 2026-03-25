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
    Orchestrates alternation between LatentSolver (Phase A) and MLP training (Phase B).
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
        
        latent_solver = LatentSolver(
            self.cfg, self.neighbour_graph,
            embed_dim=self.cfg.embed_dim,
            gating_fn=self.cfg.gating_fn,
        )
        latent_solver.build_V_and_H(data["train"]["X"], bin_index, method="nw")
        
        # Build model
        self.device = torch.device(self.cfg.device)
        input_dim = data["train"]["X"].shape[1]
        mlp_model = MLPModel(
            input_dim, hidden_dims=[64, 128, 64, 32],
            output_dim=self.cfg.embed_dim,
            dropout=self.cfg.dropout,
        ).to(self.device)
        self.model = Model(
            mlp_model, latent_solver, n_bins=len(bin_index), device=self.device,
            embed_dim=self.cfg.embed_dim,
            gating_fn=self.cfg.gating_fn,
            gating_alpha=self.cfg.gating_alpha,
            gating_kappa=self.cfg.gating_kappa,
            gating_epsilon=self.cfg.gating_epsilon,
        )
        # Mixed latent training ratio: fraction of samples using interpolated latent each epoch.
        self.interpolated_sample_fraction = 0.1
        
        # Initialize optimizer with explicit per-group weight decay.
        if self.cfg.embed_dim > 1:
            optim_params = [
                {"params": self.model.mlp.parameters(), "weight_decay": self.cfg.weight_decay},
                {"params": self.model.final_linear.parameters(), "weight_decay": self.cfg.final_linear_wd},
            ]
        else:
            optim_params = [
                {"params": self.model.mlp.parameters(), "weight_decay": self.cfg.weight_decay},
            ]
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=self.cfg.lr,
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
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=8, pin_memory=True)
        self.val_loader = DataLoader(val, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)
        
        # Non-shuffled loader for stable evaluation/tracking (sample-mode ordering may not match row ordering)
        self.train_loader_ordered = DataLoader(train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)

        # Canonical ordered loader for latent solving (row-aligned with its own targets)
        self.train_loader_bin_ordered = DataLoader(
            train_bin_ordered,
            batch_size=self.cfg.batch_size_bin,
            shuffle=False,
            collate_fn=None,
        )

        # Cache train sample ids to support per-epoch sample-level interpolation selection.
        train_sample_ids: set[int] = set()
        for batch in self.train_loader_bin_ordered:
            s_idx = batch["sample_idx"].detach().cpu().numpy().reshape(-1)
            train_sample_ids.update(int(v) for v in s_idx)
        self.train_sample_ids = np.array(sorted(train_sample_ids), dtype=np.int64)

        # LR scheduler — total steps = epochs × batches per epoch (one scheduler step per batch)
        total_steps = self.cfg.epochs * len(self.train_loader)
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
        
        # Track best validation loss for checkpointing
        self.best_val_loss = float('inf')

        # Loss tracking for visualization
        self.train_losses = []  # List of (epoch, step, loss) tuples
        self.val_losses = []    # List of (epoch, step, loss) tuples
        self.cycle_train_losses = []  # End-of-epoch train losses (kept for compatibility)
        self.cycle_val_losses = []    # End-of-epoch val losses (kept for compatibility)
        self.timeline_train_losses = []  # List of (phase, epoch, step, loss) tuples
        self.timeline_val_losses = []    # List of (phase, epoch, step, loss) tuples
        
        # Prepare save path
        root = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(root, "models", f"{time.strftime('%Y-%m-%d_%H:%M:%S')}.pt")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)


    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
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

            self.optimizer.zero_grad()
            loss.backward()

            if self.cfg.grad_clip is not None:
                params_to_clip = list(self.model.mlp.parameters())
                if self.cfg.embed_dim > 1:
                    params_to_clip += list(self.model.final_linear.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, self.cfg.grad_clip)

            self.optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        epoch_loss = running_loss / max(1, n_samples)
        return epoch_loss


    def _train_batch(
        self,
        batch: Dict[str, torch.Tensor],
        selected_sample_ids: Optional[np.ndarray] = None,
    ) -> float:
        """Perform one MLP gradient step on a single batch. Returns batch loss."""
        self.model.train()
        inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)

        selected_tensor: Optional[torch.Tensor] = None
        if selected_sample_ids is not None and len(selected_sample_ids) > 0:
            selected_tensor = torch.as_tensor(selected_sample_ids, dtype=sample_idx.dtype, device=self.device)

        if self.loss_mode == "sample":
            B, max_bins, n_feat = inputs.shape
            inputs_flat = inputs.view(B * max_bins, n_feat)
            bin_idx_flat = bin_idx.view(B * max_bins)
            obs_interp_mask_flat = None
            if selected_tensor is not None:
                sample_selected = torch.isin(sample_idx.view(-1), selected_tensor)  # [B]
                obs_interp_mask_flat = sample_selected.unsqueeze(1).expand(B, max_bins).reshape(-1)
            outputs_flat = self.model(inputs_flat, bin_idx_flat, interpolated_obs_mask=obs_interp_mask_flat)
            outputs = outputs_flat.view(B, max_bins)
            outputs = outputs.masked_fill(mask == 0, float('-inf'))
            loss = self.criterion(outputs, targets, mask)
        else:
            obs_interp_mask = torch.isin(sample_idx.view(-1), selected_tensor) if selected_tensor is not None else None
            outputs = self.model(inputs, bin_idx, interpolated_obs_mask=obs_interp_mask)
            loss = self.criterion(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()

        if self.cfg.grad_clip is not None:
            params_to_clip = list(self.model.mlp.parameters())
            if self.cfg.embed_dim > 1:
                params_to_clip += list(self.model.final_linear.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, self.cfg.grad_clip)

        self.optimizer.step()
        return loss.item()


    @torch.no_grad()
    def validate(self, split: Literal["train", "val", "test"]) -> float:
        # Always evaluate with normal latent usage.
        self.model.configure_latent_usage(mode="normal", interpolated_bin_mask=None)
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
        # Keep metrics consistent with inference behavior.
        self.model.configure_latent_usage(mode="normal", interpolated_bin_mask=None)
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
                if mask is None:
                    raise ValueError("Mask is required in sample mode")
                B, max_bins, n_feat = inputs.shape
                inputs_flat = inputs.view(B * max_bins, n_feat)
                bin_idx_flat = bin_idx.view(B * max_bins)
                
                outputs_flat = self.model(inputs_flat, bin_idx_flat)
                outputs = outputs_flat.view(B, max_bins)
                outputs = outputs.masked_fill(mask == 0, float('-inf'))
                
                # Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=-1)
                
                # Flatten but only keep valid (non-padded) entries
                mask_flat = mask.view(-1).bool()
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

    def solve_latent(
        self,
        prox_weight: float = 0.0,
        selected_sample_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for latent variable using current MLP predictions and update model."""
        self.model.eval()

        intrinsic_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        bin_list: List[np.ndarray] = []
        sample_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch in self.train_loader_bin_ordered:
                x = batch["input"].to(self.device)
                y = batch["target"].cpu().numpy()
                bin_idx = batch["bin_idx"].cpu().numpy()
                sample_idx = batch["sample_idx"].cpu().numpy()

                intrinsic = self.model.mlp(x).detach().cpu().numpy()
                if self.cfg.embed_dim == 1:
                    intrinsic = intrinsic.reshape(-1)  # (N,)
                # else: keep as (N, d)

                intrinsic_list.append(intrinsic)
                y_list.append(y.reshape(-1))
                bin_list.append(bin_idx.reshape(-1))
                sample_list.append(sample_idx.reshape(-1))

        intrinsic_vec = np.concatenate(intrinsic_list, axis=0)
        y_vec = np.concatenate(y_list, axis=0)
        bin_ids = np.concatenate(bin_list, axis=0).astype(np.int64)
        sample_ids = np.concatenate(sample_list, axis=0).astype(np.int64)
        obs_interp_mask = np.isin(sample_ids, selected_sample_ids) if selected_sample_ids is not None else None

        x0_latent = self.model.latent_vec.detach().cpu().numpy()
        if self.cfg.embed_dim > 1:
            latent_vec = self.model.latent_solver.solve(
                y=y_vec,
                intrinsic_vec=intrinsic_vec,
                final_weights=self.model.final_linear.weight.detach().cpu().numpy().squeeze(),
                bin_ids=bin_ids,
                sample_ids=sample_ids,
                loss_type="cross_entropy" if self.loss_type == "cross_entropy" else "logistic",
                x0=x0_latent,
                prox_weight=prox_weight,
                x_anchor=x0_latent,
                interpolated_obs_mask=obs_interp_mask,
            )
        else:
            latent_vec = self.model.latent_solver.solve(
                y=y_vec,
                intrinsic_vec=intrinsic_vec,
                bin_ids=bin_ids,
                sample_ids=sample_ids,
                loss_type="cross_entropy" if self.loss_type == "cross_entropy" else "logistic",
                x0=x0_latent,
                prox_weight=prox_weight,
                x_anchor=x0_latent,
                interpolated_obs_mask=obs_interp_mask,
            )
        self.model.set_latent(latent_vec)
        return latent_vec

    def _sample_interpolated_sample_ids(self, fraction: Optional[float] = None) -> np.ndarray:
        """Randomly choose which train samples use interpolated latent for the current epoch."""
        frac = float(self.interpolated_sample_fraction if fraction is None else fraction)
        frac = float(np.clip(frac, 0.0, 1.0))
        n_samples = len(self.train_sample_ids)

        if frac <= 0.0:
            return np.array([], dtype=np.int64)
        if frac >= 1.0:
            return self.train_sample_ids.copy()

        n_select = max(1, int(round(frac * n_samples)))
        return np.random.choice(self.train_sample_ids, size=n_select, replace=False).astype(np.int64)

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
        # Always infer with normal latent usage.
        self.model.configure_latent_usage(mode="normal", interpolated_bin_mask=None)
        self.model.eval()
        # Accumulate per-sample lists
        sample_pred: Dict[int, list] = {}
        sample_true: Dict[int, list] = {}
        sample_bins: Dict[int, list] = {}  # integer bin indices (from self.bin_index)
        for batch in data_loader:
            if self.loss_mode == "sample":
                inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
                if mask is None:
                    raise ValueError("Mask is required in sample mode")
                B = inputs.shape[0]
                for b in range(B):
                    s_idx = int(sample_idx[b].item() if hasattr(sample_idx[b], 'item') else sample_idx[b])
                    valid_mask = mask[b].bool()
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
        """Run full alternation training with per-batch latent/MLP alternation."""
        warmup_epochs = max(1, int(self.cfg.latent_warmup_frac * self.cfg.epochs))
        log.debug(f"Starting per-batch alternation training (latent warms up over first {warmup_epochs} epochs)...")
        for epoch in tqdm(range(self.cfg.epochs), desc="Epochs", leave=False):
            # Proximal weight ρ: decays linearly from ρ₀ → 0 over warmup_epochs (proximal/damped EM).
            # At epoch 0: ρ = latent_prox_scale * latent_l2_reg (large anchor near D=0).
            # At epoch = warmup_epochs: ρ = 0 (standard unconstrained solve).
            alpha = min(1.0, epoch / warmup_epochs)
            phase_tag = "latent_warmup" if alpha < 1.0 else "latent"
            prox_weight = self.cfg.latent_prox_scale * self.cfg.latent_l2_reg * (1.0 - alpha)
            epoch_interp_samples = self._sample_interpolated_sample_ids(self.interpolated_sample_fraction)
            self.model.configure_latent_usage(mode="normal", interpolated_bin_mask=None)

            # Alternate: for each batch, solve latent then take one MLP gradient step
            latent_vec = self.model.latent_vec.detach().cpu().numpy()  # fallback if loader is empty
            for batch in self.train_loader:
                # Phase A: solve for latent with fixed MLP
                latent_vec = self.solve_latent(
                    prox_weight=prox_weight,
                    selected_sample_ids=epoch_interp_samples,
                )
                # Phase B: one MLP gradient step on this batch
                self._train_batch(batch, selected_sample_ids=epoch_interp_samples)
                self.scheduler.step()

            # End-of-epoch evaluation
            self.model.configure_latent_usage(mode="normal", interpolated_bin_mask=None)
            train_loss = self.validate(split="train")
            val_loss = self.validate(split="val")

            self.timeline_train_losses.append((phase_tag, epoch, 0, train_loss))
            self.timeline_val_losses.append((phase_tag, epoch, 0, val_loss))
            self.train_losses.append((epoch, 0, train_loss))
            self.val_losses.append((epoch, 0, val_loss))
            self.cycle_train_losses.append((epoch, train_loss))
            self.cycle_val_losses.append((epoch, val_loss))

            if use_wandb and WANDB_AVAILABLE:
                metrics = self.compute_metrics(split="val")
                wandb.log({
                    "epoch": epoch,
                    "cycle": epoch,
                    "final_train_loss": train_loss,
                    "final_val_loss": val_loss,
                    **{f"val_{k}": v for k, v in metrics.items()},
                    "latent_mean": float(latent_vec.mean()),
                    "latent_std": float(latent_vec.std()),
                    "latent_min": float(latent_vec.min()),
                    "latent_max": float(latent_vec.max()),
                    "interpolated_sample_fraction": float(len(epoch_interp_samples) / max(1, len(self.train_sample_ids))),
                    "interpolated_sample_count": int(len(epoch_interp_samples)),
                })

            if val_loss < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss
                self.model.save_model(self.save_path)

            log.info(f"Epoch {epoch} (ρ={prox_weight:.4f}): val_loss={val_loss:.6f}, train_loss={train_loss:.6f}")
        
        # Load best saved model before final evaluation (end-of-training weights may be worse)
        if os.path.exists(self.save_path):
            self.model.load_model(self.save_path)
            log.debug(f"Loaded best checkpoint from {self.save_path} for final evaluation")

        # Plot training progress
        self._plot_training_progress()

        # Return results for external use
        predictions, targets, sample_labels, bin_labels = self.get_predictions(split="test")
        # Save the final latent vector for downstream analysis
        latent_vector = self.model.latent_vec.detach().cpu().numpy()
        return {
            "best_val_loss": self.best_val_loss,
            "predictions": predictions,
            "targets": targets,
            "sample_labels": sample_labels,
            "bin_labels": bin_labels,
            "latent_vector": latent_vector,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "cycle_train_losses": self.cycle_train_losses,
            "cycle_val_losses": self.cycle_val_losses,
            "timeline_train_losses": self.timeline_train_losses,
            "timeline_val_losses": self.timeline_val_losses,
        }

    def _plot_training_progress(self) -> None:
        """Plot training and validation losses over time."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Overall evolution (init + latent warmup + latent + mlp)
        train_vals = [l for _, _, _, l in self.timeline_train_losses]
        val_vals   = [l for _, _, _, l in self.timeline_val_losses]
        steps = list(range(len(train_vals)))

        ax1.plot(steps, train_vals, 'b-', alpha=0.7, linewidth=1.2, label='Train Loss')
        ax1.plot(steps, val_vals,   'r-', alpha=0.7, linewidth=1.2, label='Val Loss')

        # Shade latent warmup region (Phase 2) as a green band
        warmup_indices = [
            i for i, (phase, _, _, _) in enumerate(self.timeline_train_losses)
            if phase == "latent_warmup"
        ]
        if warmup_indices:
            ax1.axvspan(warmup_indices[0], warmup_indices[-1] + 1,
                        color='green', alpha=0.08, label='Latent Warmup (Phase 2)')

        # Mark alternation latent-solve steps with gray dashed lines
        latent_steps = [
            i for i, (phase, _, _, _) in enumerate(self.timeline_train_losses)
            if phase == "latent"
        ]
        for boundary in latent_steps:
            ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.25, linewidth=1)

        ax1.set_xlabel('Training Step (Init + Latent Warmup + Alternation)')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress: Overall Loss Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: End-of-epoch losses (summary view)
        if len(self.cycle_train_losses) > 0:
            epoch_nums  = [e for e, l in self.cycle_train_losses]
            epoch_train = [l for e, l in self.cycle_train_losses]
            epoch_val   = [l for e, l in self.cycle_val_losses]

            ax2.plot(epoch_nums, epoch_train, 'bo-', linewidth=2, markersize=8, label='Train Loss', alpha=0.7)
            ax2.plot(epoch_nums, epoch_val,   'ro-', linewidth=2, markersize=8, label='Val Loss',   alpha=0.7)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Progress: End-of-Epoch Losses')
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
    data_group = parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument("--data_path", type=str, default=None,
                            help="Path to raw data CSV file (e.g. data/ecuador_training_data.csv)")
    data_group.add_argument("--data_dir", type=str, default=None,
                            help="Path to directory containing processed CSV files (X_*.csv, y_*.csv, taxonomic_data.csv)")
    parser.add_argument("--loss_type", type=str, choices=["cross_entropy", "logistic"],
                        default="cross_entropy", help="Type of loss function to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set some generic configurations
    set_seed()
    cfg = Config()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Initialize Weights & Biases (optional)
    use_wandb = WANDB_AVAILABLE
    if not WANDB_AVAILABLE:
        log.warning("wandb is not installed; continuing without wandb logging")
    else:
        wandb.init(
            project="metabarcoding",
            name=time.strftime("%Y-%m-%d_%H-%M"),
            config=asdict(cfg),
            dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wandb")
        )

    # Create Trainer and run
    # Default behavior: if neither is provided, fall back to raw CSV in data/
    data_path = args.data_path
    data_dir = args.data_dir
    if data_path is None and data_dir is None:
        data_path = cfg.data_path

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