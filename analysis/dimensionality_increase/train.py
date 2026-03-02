"""
Modified Trainer for multiplicative gating architecture.

This module imports the base training infrastructure from src/
but uses local model, latent_solver, mlp, and config with multiplicative gating.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
import importlib.util

# Setup paths explicitly
current_dir = Path(__file__).parent
src_path = str(current_dir.parent.parent / "src")

# Import base utilities from src
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import utilities from src that don't have local overrides
from neighbor_graph import NeighbourGraph
from dataset import MBDataset, collate_samples
from loss import Loss
from utils import load, load_processed

# Remove src from sys.path to avoid conflicts when loading local modules
sys.path.remove(src_path)

# Import local implementations using explicit file paths to avoid module caching
config_spec = importlib.util.spec_from_file_location("local_config_gating", str(current_dir / "config.py"))
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
Config = config_module.Config
set_seed = config_module.set_seed

mlp_spec = importlib.util.spec_from_file_location("local_mlp_gating", str(current_dir / "mlp.py"))
mlp_module = importlib.util.module_from_spec(mlp_spec)
mlp_spec.loader.exec_module(mlp_module)
MLPModel = mlp_module.MLPModel

ls_spec = importlib.util.spec_from_file_location("local_ls_gating", str(current_dir / "latent_solver.py"))
ls_module = importlib.util.module_from_spec(ls_spec)
ls_spec.loader.exec_module(ls_module)
LatentSolver = ls_module.LatentSolver

model_spec = importlib.util.spec_from_file_location("local_model_gating", str(current_dir / "model.py"))
model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)
Model = model_module.Model

# Now import standard libraries
import argparse
import re
import time
from typing import Literal, Optional, List, Tuple, Dict, Any
import logging as log

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """
    Orchestrates alternation between LatentSolver (Phase A) and MLP training (Phase B)
    with multiplicative gating architecture.
    """
    def __init__(
        self,
        cfg: Config,
        data_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        loss_type: Literal["cross_entropy", "logistic"] = "cross_entropy",
        fixed_split_indices: Optional[Dict[str, np.ndarray]] = None,
        model_save_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.model_save_dir = model_save_dir
        
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
        
        latent_solver = LatentSolver(
            self.cfg,
            self.neighbour_graph,
            embed_dim=self.cfg.embed_dim,
            gating_fn=self.cfg.gating_fn,
        )
        latent_solver.build_V_and_H(data["train"]["X"], bin_index, method="nw")
        
        # Build model with vector output
        self.device = torch.device(self.cfg.device)
        input_dim = data["train"]["X"].shape[1]
        mlp_model = MLPModel(
            input_dim,
            hidden_dims=[128, 64],
            output_dim=self.cfg.embed_dim,  # Vector output!
            dropout=self.cfg.dropout
        ).to(self.device)
        
        self.model = Model(
            mlp_model,
            latent_solver,
            n_bins=len(bin_index),
            embed_dim=self.cfg.embed_dim,
            device=self.device,
            gating_fn=self.cfg.gating_fn,
            gating_alpha=self.cfg.gating_alpha,
            gating_kappa=self.cfg.gating_kappa,
            gating_epsilon=self.cfg.gating_epsilon,
        )
        
        # Initialize optimizer with separate weight decay for final_linear
        # Stronger weight decay on w to constrain |w| and avoid scaling issues
        param_groups = [
            {
                'params': self.model.mlp.parameters(),
                'weight_decay': self.cfg.weight_decay,
                'name': 'mlp'
            },
            {
                'params': self.model.final_linear.parameters(),
                'weight_decay': self.cfg.final_linear_wd,
                'name': 'final_linear'
            }
        ]
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.cfg.lr)
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

        train_bin_ordered = MBDataset(data["train"], bin_index, sample_index, loss_mode="bin")
        
        # Initialize data loaders
        batch_size = cfg.batch_size_sample if self.loss_mode == "sample" else cfg.batch_size_bin
        collate_fn = collate_samples if self.loss_mode == "sample" else None
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val, batch_size=1, shuffle=False, collate_fn=collate_fn)
        self.test_loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        self.train_loader_ordered = DataLoader(train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        self.train_loader_bin_ordered = DataLoader(
            train_bin_ordered,
            batch_size=self.cfg.batch_size_bin,
            shuffle=False,
            collate_fn=None,
        )
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0
        
        # Prepare save path
        if self.model_save_dir is not None:
            models_dir = self.model_save_dir
        else:
            root = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(root, "models")
        self.save_path = os.path.join(models_dir, f"{time.strftime('%Y-%m-%d_%H:%M')}.pt")
        os.makedirs(models_dir, exist_ok=True)


    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in self.train_loader:
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

            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            running_loss += loss.item()
            n_samples += 1

        return running_loss / max(1, n_samples)


    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate on given loader. Returns loss and metrics dict."""
        self.model.eval()
        running_loss = 0.0
        n_samples = 0
        
        classification_metrics = {
            "accuracy": [], "precision": [], "recall": [],
            "specificity": [], "f1": []
        }
        mae_metrics = {"mae_when_present": [], "mae_when_absent": []}

        with torch.no_grad():
            for batch in loader:
                inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)

                if self.loss_mode == "sample":
                    B, max_bins, n_feat = inputs.shape
                    inputs_flat = inputs.view(B * max_bins, n_feat)
                    bin_idx_flat = bin_idx.view(B * max_bins)
                    
                    outputs_flat = self.model(inputs_flat, bin_idx_flat)
                    outputs = outputs_flat.view(B, max_bins)
                    outputs = outputs.masked_fill(mask == 0, float('-inf'))
                    
                    loss = self.criterion(outputs, targets, mask)
                    probs = torch.softmax(outputs, dim=-1)
                    mask_flat = mask.view(-1).bool()
                    probs_np = probs.view(-1)[mask_flat].cpu().numpy()
                    targets_np = targets.view(-1)[mask_flat].cpu().numpy()
                else:
                    outputs = self.model(inputs, bin_idx)
                    loss = self.criterion(outputs, targets)
                    probs = torch.sigmoid(outputs)
                    probs_np = probs.cpu().numpy().flatten()
                    targets_np = targets.cpu().numpy().flatten()

                running_loss += loss.item()
                n_samples += 1
                
                # Classification metrics
                epsilon = 1e-2
                tp = sum((p >= epsilon) and (t > 0) for p, t in zip(probs_np, targets_np))
                tn = sum((p < epsilon) and (t == 0) for p, t in zip(probs_np, targets_np))
                fp = sum((p >= epsilon) and (t == 0) for p, t in zip(probs_np, targets_np))
                fn = sum((p < epsilon) and (t > 0) for p, t in zip(probs_np, targets_np))
                total = max(1, tp + tn + fp + fn)
                
                classification_metrics["accuracy"].append((tp + tn) / total)
                classification_metrics["precision"].append(tp / max(1, tp + fp))
                classification_metrics["recall"].append(tp / max(1, tp + fn))
                classification_metrics["specificity"].append(tn / max(1, tn + fp))
                prec_rec = classification_metrics["precision"][-1] + classification_metrics["recall"][-1]
                classification_metrics["f1"].append(
                    2 * classification_metrics["precision"][-1] * classification_metrics["recall"][-1] / max(1e-12, prec_rec)
                )
                
                # MAE
                for p, t in zip(probs_np, targets_np):
                    if t > 0:
                        mae_metrics["mae_when_present"].append(abs(p - t))
                    else:
                        mae_metrics["mae_when_absent"].append(abs(p - t))

        avg_loss = running_loss / max(1, n_samples)
        metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in {**classification_metrics, **mae_metrics}.items()}
        return avg_loss, metrics


    def solve_latent(self) -> np.ndarray:
        """Solve for latent matrix using current MLP predictions and update model."""
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

                intrinsic = self.model.mlp(x).detach().cpu().numpy()  # (N, d)

                intrinsic_list.append(intrinsic)
                y_list.append(y.reshape(-1))
                bin_list.append(bin_idx.reshape(-1))
                sample_list.append(sample_idx.reshape(-1))

        intrinsic_vec = np.concatenate(intrinsic_list, axis=0)  # (N_obs, d)
        y_vec = np.concatenate(y_list, axis=0)
        bin_ids = np.concatenate(bin_list, axis=0).astype(np.int64)
        sample_ids = np.concatenate(sample_list, axis=0).astype(np.int64)

        # Get final_weights from model
        final_weights = self.model.final_linear.weight.detach().cpu().numpy().reshape(-1)  # (d,)

        latent_mat = self.model.latent_solver.solve(
            y=y_vec,
            intrinsic_vec=intrinsic_vec,
            final_weights=final_weights,
            bin_ids=bin_ids,
            sample_ids=sample_ids,
            loss_type=self.loss_type,
            x0=self.model.latent_vec.detach().cpu().numpy(),  # (n_bins, d)
        )
        self.model.set_latent(latent_mat)
        return latent_mat

    def get_predictions(self, split: str = "test") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions and targets for a given split.
        Returns flat 1D arrays for all valid (sample, BIN) pairs:
            - predictions:   float32 (N,) — predicted softmax/sigmoid values
            - targets:       float32 (N,) — true relative abundances
            - sample_labels: (N,) str    — sample ID (from self.sample_index)
            - bin_labels:    (N,) str    — BIN URI (from self.bin_index)
        """
        loader = (
            self.train_loader if split == "train" else
            self.val_loader if split == "val" else
            self.test_loader
        )

        self.model.eval()

        sample_pred: Dict[int, list] = {}
        sample_true: Dict[int, list] = {}
        sample_bins: Dict[int, list] = {}  # integer bin indices (from self.bin_index)

        with torch.no_grad():
            for batch in loader:
                inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)

                if self.loss_mode == "sample":
                    B = inputs.shape[0]
                    for b in range(B):
                        s_idx = sample_idx[b].item() if hasattr(sample_idx[b], "item") else int(sample_idx[b])
                        valid_mask = mask[b].bool()
                        inputs_flat = inputs[b][valid_mask]
                        bin_idx_flat = bin_idx[b][valid_mask]
                        outputs = self.model(inputs_flat, bin_idx_flat)
                        probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                        y_true = targets[b][valid_mask].cpu().numpy()
                        sample_pred.setdefault(s_idx, []).extend(probs.tolist())
                        sample_true.setdefault(s_idx, []).extend(y_true.tolist())
                        sample_bins.setdefault(s_idx, []).extend(bin_idx_flat.cpu().numpy().tolist())
                else:
                    inputs = batch["input"].to(self.device)
                    targets = batch["target"].to(self.device)
                    bin_idx = batch["bin_idx"].to(self.device)
                    sample_idx = batch["sample_idx"].to(self.device)
                    if len(inputs.shape) == 1:
                        inputs = inputs.unsqueeze(0)
                        targets = targets.unsqueeze(0)
                        bin_idx = bin_idx.unsqueeze(0)
                        sample_idx = sample_idx.unsqueeze(0)
                    for i in range(inputs.shape[0]):
                        s_idx = sample_idx[i].item() if hasattr(sample_idx[i], "item") else int(sample_idx[i])
                        b_idx = int(bin_idx[i].item() if hasattr(bin_idx[i], "item") else bin_idx[i])
                        input_i = inputs[i].unsqueeze(0)
                        bin_idx_i = bin_idx[i].unsqueeze(0)
                        output = self.model(input_i, bin_idx_i)
                        prob = torch.sigmoid(output).cpu().numpy().item()
                        y_true = targets[i].cpu().numpy().item()
                        sample_pred.setdefault(s_idx, []).append(prob)
                        sample_true.setdefault(s_idx, []).append(y_true)
                        sample_bins.setdefault(s_idx, []).append(b_idx)

        if self.loss_type == "logistic":
            for s_idx in sample_pred:
                preds = np.array(sample_pred[s_idx])
                pred_sum = preds.sum()
                if pred_sum > 0:
                    sample_pred[s_idx] = (preds / pred_sum).tolist()

        if not sample_pred:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=object),
                np.array([], dtype=object),
            )
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
        """Run full training with Phase A/B alternation."""
        use_wandb = use_wandb and WANDB_AVAILABLE
        
        log.info(f"\n{'='*70}")
        log.info(f"Training with multiplicative gating (embed_dim={self.cfg.embed_dim}, gating_fn={self.cfg.gating_fn})")
        log.info(f"{'='*70}\n")
        
        # Phase 0: Initial MLP training
        log.info("Phase 0: Initial MLP training")
        for epoch in range(self.cfg.epochs_init):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader)
            
            if use_wandb:
                wandb.log({"phase": 0, "epoch": epoch, "train_loss": train_loss,
                          "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
            
            if epoch % 10 == 0:
                log.info(f"  Epoch {epoch}/{self.cfg.epochs_init}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Phase A/B alternation
        for cycle in range(self.cfg.max_cycles):
            log.info(f"\nCycle {cycle+1}/{self.cfg.max_cycles}")
            
            # Phase A: Solve latent
            log.info("  Phase A: Solving latent matrix...")
            latent_mat = self.solve_latent()
            log.debug(f"  Latent H: mean={latent_mat.mean():.3f}, std={latent_mat.std():.3f}")
            
            # Phase B: Train MLP
            log.info(f"  Phase B: Training MLP for {self.cfg.epochs} epochs...")
            for epoch in range(self.cfg.epochs):
                train_loss = self.train_epoch()
                val_loss, val_metrics = self.evaluate(self.val_loader)
                self.scheduler.step(val_loss)
                
                if use_wandb:
                    wandb.log({"phase": f"{cycle+1}B", "epoch": epoch, "train_loss": train_loss,
                              "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
            
            log.info(f"  Cycle {cycle+1} complete: val_loss={val_loss:.4f}, val_f1={val_metrics.get('f1', 0):.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improve_epochs = 0
                self.model.save_model(self.save_path)
            else:
                self.no_improve_epochs += 1
            
            if self.cfg.patience and self.no_improve_epochs >= self.cfg.patience:
                log.info(f"  Early stopping triggered at cycle {cycle+1}")
                break
        
        # Final evaluation
        log.info("\nFinal evaluation on test set...")
        test_loss, test_metrics = self.evaluate(self.test_loader)
        log.info(f"Test loss: {test_loss:.4f}")
        log.info(f"Test metrics: {test_metrics}")
        
        if use_wandb:
            wandb.log({"test_loss": test_loss, **{f"test_{k}": v for k, v in test_metrics.items()}})
        
        # Get predictions for comparison
        log.info("Collecting predictions...")
        predictions, targets, sample_labels, bin_labels = self.get_predictions(split="test")
        
        # Get latent matrix for downstream analysis
        latent_matrix = self.model.latent_vec.detach().cpu().numpy()
        
        return {
            "best_val_loss": self.best_val_loss,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "predictions": predictions,
            "targets": targets,
            "sample_labels": sample_labels,
            "bin_labels": bin_labels,
            "latent_matrix": latent_matrix,
        }


    def _to_device(self, batch: Dict) -> Tuple:
        """Move batch to device and extract components."""
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)
        bin_idx = batch["bin_idx"].to(self.device)
        sample_idx = batch["sample_idx"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)
        return inputs, targets, bin_idx, sample_idx, mask
