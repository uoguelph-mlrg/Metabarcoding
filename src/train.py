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
            data, bins_df, bin_index, sample_index, split_indices = load_processed(data_dir, config=self.cfg)
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
        mlp_model = MLPModel(input_dim, hidden_dims=[128, 64], dropout=self.cfg.dropout).to(self.device)
        self.model = Model(mlp_model, latent_solver, n_bins=len(bin_index), device=self.device)
        
        # Initialize optimizer, scheduler, criterion
        self.optimizer = torch.optim.AdamW(
            self.model.mlp.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=False
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
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0

        # Loss tracking for visualization
        self.train_losses = []  # List of (cycle, epoch, loss) tuples (MLP epochs only)
        self.val_losses = []    # List of (cycle, epoch, loss) tuples (MLP epochs only)
        self.cycle_train_losses = []  # End-of-cycle train losses
        self.cycle_val_losses = []    # End-of-cycle val losses
        self.timeline_train_losses = []  # List of (phase, cycle, step, loss) tuples
        self.timeline_val_losses = []    # List of (phase, cycle, step, loss) tuples
        
        # Prepare save path
        root = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(root, "models", f"{time.strftime('%Y-%m-%d_%H:%M')}.pt")
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
                torch.nn.utils.clip_grad_norm_(self.model.mlp.parameters(), self.cfg.grad_clip)

            self.optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
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

    def solve_latent(self) -> np.ndarray:
        """Solve for latent vector using current MLP predictions and update model."""
        # IMPORTANT: we always use the ordered bin-mode loader so intrinsic/targets/bin_ids/sample_ids
        # are aligned by construction (eliminates the historical sample-mode ordering bug).
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

                intrinsic = self.model.mlp(x).detach().cpu().numpy().reshape(-1)

                intrinsic_list.append(intrinsic)
                y_list.append(y.reshape(-1))
                bin_list.append(bin_idx.reshape(-1))
                sample_list.append(sample_idx.reshape(-1))

        intrinsic_vec = np.concatenate(intrinsic_list, axis=0)
        y_vec = np.concatenate(y_list, axis=0)
        bin_ids = np.concatenate(bin_list, axis=0).astype(np.int64)
        sample_ids = np.concatenate(sample_list, axis=0).astype(np.int64)

        latent_vec = self.model.latent_solver.solve(
            y=y_vec,
            intrinsic_vec=intrinsic_vec,
            bin_ids=bin_ids,
            sample_ids=sample_ids,
            loss_type=self.loss_type,
            x0=self.model.latent_vec.detach().cpu().numpy(),
        )
        self.model.set_latent(latent_vec)
        return latent_vec

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
        """Run full alternation training."""
        # initial stabilize training
        log.debug("Initial stabilization training...")
        best_val = float('inf')
        for epoch in tqdm(range(self.cfg.epochs_init), desc="Init epochs", leave=False):
            self.train_epoch()                              # optimize weights
            train_loss = self.validate(split="train")      # eval-mode — consistent with latent phases
            val_loss   = self.validate(split="val")
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({"epoch": epoch, "init_train_loss": train_loss, "init_val_loss": val_loss})

            # Track losses (cycle=-1 for initialization phase)
            self.train_losses.append((-1, epoch, train_loss))
            self.val_losses.append((-1, epoch, val_loss))
            self.timeline_train_losses.append(("init", -1, epoch, train_loss))
            self.timeline_val_losses.append(("init", -1, epoch, val_loss))
                
            # Early stopping check
            stop, best_val = self.early_stop_and_save(val_loss, best_val, cycle=-1, epoch=epoch)
            if stop:
                break
        
        log.debug("\nStarting alternation cycles...")
        # Reset counter so Phase 1 early-stopping state doesn't bleed into Phase 3
        self.no_improve_epochs = 0
        for cycle in tqdm(range(self.cfg.max_cycles), desc="Cycles", leave=False):
            # Phase A: solve for latent vector with fixed MLP
            latent_vec = self.solve_latent()

            latent_train_loss = self.validate(split="train")
            latent_val_loss = self.validate(split="val")
            self.timeline_train_losses.append(("latent", cycle, 0, latent_train_loss))
            self.timeline_val_losses.append(("latent", cycle, 0, latent_val_loss))
            log.info(f"Cycle {cycle} Phase A: train={latent_train_loss:.6f}, val={latent_val_loss:.6f}")

            # Phase B: train MLP with fixed latent vector
            best_val = float('inf')
            pbar = tqdm(range(self.cfg.epochs), desc=f"Cycle {cycle+1}", leave=False)
            for epoch in pbar:
                self.train_epoch()                             # optimize weights
                val_loss   = self.validate(split="val")
                train_loss = self.validate(split="train")     # eval-mode — consistent with latent phases

                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "cycle": cycle,
                        "epoch": epoch,
                        "train_loss_in_cycle": train_loss,
                        "val_loss_in_cycle": val_loss
                    })

                # Track losses
                self.train_losses.append((cycle, epoch, train_loss))
                self.val_losses.append((cycle, epoch, val_loss))
                self.timeline_train_losses.append(("mlp", cycle, epoch, train_loss))
                self.timeline_val_losses.append(("mlp", cycle, epoch, val_loss))

                # Step scheduler based on validation loss (once per epoch)
                self.scheduler.step(val_loss)

                if epoch % 10 == 0:
                    pbar.set_postfix(
                        {"train loss": f"{train_loss:.6f}", "val loss": f"{val_loss:.6f}"}
                    )

                # Early stopping on small improvements
                stop, best_val = self.early_stop_and_save(val_loss, best_val, cycle=cycle, epoch=epoch)
                if stop:
                    break
            
            train_loss = self.validate(split="train")
            val_loss = self.validate(split="val")

            if use_wandb and WANDB_AVAILABLE:
                metrics = self.compute_metrics(split="val")
                wandb.log({
                    "cycle": cycle,
                    "final_train_loss": train_loss,
                    "final_val_loss": val_loss,
                    **{f"val_{k}": v for k, v in metrics.items()},
                    "latent_mean": float(latent_vec.mean()),
                    "latent_std": float(latent_vec.std()),
                    "latent_min": float(latent_vec.min()),
                    "latent_max": float(latent_vec.max()),
                })

            # Track end-of-cycle losses
            self.cycle_train_losses.append((cycle, train_loss))
            self.cycle_val_losses.append((cycle, val_loss))

            # Early stopping on no improvement after full cycle
            stop, self.best_val_loss = self.early_stop_and_save(
                val_loss, self.best_val_loss, cycle=cycle, epoch=-1
            )
            if stop:
                break

            log.info(f"Cycle {cycle}: val_loss={val_loss:.6f}, train_loss={train_loss:.6f}")
        
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

        # Plot 2: End-of-cycle losses (summary view)
        if len(self.cycle_train_losses) > 0:
            cycle_nums  = [c for c, l in self.cycle_train_losses]
            cycle_train = [l for c, l in self.cycle_train_losses]
            cycle_val   = [l for c, l in self.cycle_val_losses]

            ax2.plot(cycle_nums, cycle_train, 'bo-', linewidth=2, markersize=8, label='Train Loss', alpha=0.7)
            ax2.plot(cycle_nums, cycle_val,   'ro-', linewidth=2, markersize=8, label='Val Loss',   alpha=0.7)
            ax2.set_xlabel('EM Cycle')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Progress: End-of-Cycle Losses')
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
    
    
    def early_stop_and_save(
        self, val_loss: float, best_val: float, cycle: int, epoch: int
    ) -> Tuple[bool, float]:
        """Check if early stopping criterion is met."""
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            if val_loss < self.best_val_loss - 1e-4:
                self.model.save_model(self.save_path)
            self.no_improve_epochs = 0
            return False, best_val
        else:
            self.no_improve_epochs += 1
            if self.cfg.patience is not None and self.no_improve_epochs >= self.cfg.patience:
                log.info(f"Early stopping at cycle {cycle} epoch {epoch} due to no improvement in val loss.")
                return True, best_val
        return False, best_val


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
    results_dir = os.path.join(root_dir, "..", "analysis", "BarcodeBERT", "results")
    os.makedirs(results_dir, exist_ok=True)
    pkl_path = os.path.join(results_dir, "taxonomy_results.pkl")
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