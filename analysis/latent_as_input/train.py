from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
import logging as log

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

_HERE = os.path.dirname(__file__)
_SRC = os.path.join(_HERE, '..', '..', 'src')
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _SRC not in sys.path:
    sys.path.insert(1, _SRC)

from dataset import MBDataset, collate_samples
from loss import Loss
from mlp import MLPModel
from neighbor_graph import NeighbourGraph
from utils import load

# Import variant-specific components from this analysis directory.
from config import Config, set_seed
from latent_solver import LatentSolver
from model import Model


def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute regression metrics aligned with visualization outputs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = np.clip(y_pred[valid], 0, 1)
    eps = 1e-10

    rmse_macro = np.nan
    mae_macro = np.nan
    kl_divergence = np.nan

    if sample_labels is not None:
        sample_labels = np.asarray(sample_labels)
        sample_labels_v = sample_labels[valid]
        rmse_per, mae_per, kl_per = [], [], []
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
        if rmse_per:
            rmse_macro = float(np.mean(rmse_per))
            mae_macro = float(np.mean(mae_per))
            kl_divergence = float(np.mean(kl_per))

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


class Trainer:
    """
    Trains MLP and latent embedding jointly in a single forward-backward pass per epoch.
    """
    def __init__(
        self,
        cfg: Config,
        data_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        loss_type: Optional[Literal["cross_entropy", "logistic"]] = None,
        model_name: str = "latent_as_input",
        run_id: Optional[str] = None,
        resume: bool = False,
        fixed_split_indices: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.cfg = cfg
        self.model_name = model_name
        self.run_id = run_id or time.strftime("%Y-%m-%d_%H-%M-%S")
        self.resume = resume

        if data_dir is not None:
            log.warning("`data_dir` is deprecated in Trainer and will be ignored. Raw preprocessing is always used.")
        effective_data_path = data_path or self.cfg.data_path
        effective_loss_type = cast(Literal["cross_entropy", "logistic"], loss_type or self.cfg.loss_type)

        self.start_epoch = 0
        self.current_epoch = -1
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.last_val_metrics: Dict[str, float] = {}

        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses: List[Tuple[int, float]] = []

        self.base_artifact_dir = os.path.abspath(os.path.join(self.cfg.results_dir, self.model_name))
        self.checkpoint_dir = os.path.join(self.base_artifact_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        data, bins_df, bin_index, sample_index, split_indices = load(
            effective_data_path,
            self.cfg,
            save_data=False,
            fixed_split_indices=fixed_split_indices,
        )

        self.data = data
        self.bin_index = bin_index
        self.sample_index = sample_index
        self.split_indices = split_indices

        self.neighbour_graph = NeighbourGraph(self.cfg, bins_df)
        self.neighbour_graph.build()

        latent_solver = LatentSolver(
            self.cfg,
            self.neighbour_graph,
        )
        latent_solver.build_V_and_H(data["train"]["X"], bin_index, method="nw")

        self.device = torch.device(self.cfg.device)
        input_dim = data["train"]["X"].shape[1]
        self.input_dim = input_dim
        self.latent_dim = int(getattr(self.cfg, "latent_dim", getattr(self.cfg, "embed_dim", 1)))
        self.latent_init_std = float(getattr(self.cfg, "latent_init_std", 0.01))
        self.latent_norm_reg = float(getattr(self.cfg, "latent_norm_reg", getattr(self.cfg, "latent_l2_reg", 0.0)))
        
        # In two-phase training, Phase 1 uses input_dim only (no latent).
        # _expand_mlp_for_latent() will expand the first layer after Phase 1.
        # Otherwise, build the full input_dim + latent_dim MLP from the start.
        mlp_input_dim = input_dim + self.latent_dim
        mlp_model = MLPModel(
            mlp_input_dim,
            hidden_dims=[128, 128, 128, 128],
            dropout=self.cfg.dropout,
        ).to(self.device)

        self.model = Model(
            mlp_model,
            latent_solver,
            n_bins=len(bin_index),
            device=self.device,
            latent_dim=self.latent_dim,
            latent_init_std=self.latent_init_std,
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

        self.loss_type: Literal["cross_entropy", "logistic"] = effective_loss_type
        self.loss_mode = "sample" if self.loss_type == "cross_entropy" else "bin"
        self.criterion = Loss(task=self.loss_type)

        train = MBDataset(data["train"], bin_index, sample_index, loss_mode=self.loss_mode)
        val = MBDataset(data["val"], bin_index, sample_index, loss_mode=self.loss_mode)
        test = MBDataset(data["test"], bin_index, sample_index, loss_mode=self.loss_mode)
        train_bin_ordered = MBDataset(data["train"], bin_index, sample_index, loss_mode="bin")

        batch_size = cfg.batch_size_sample if self.loss_mode == "sample" else cfg.batch_size_bin
        collate_fn = collate_samples if self.loss_mode == "sample" else None
        # Safer loader defaults across platforms: multiprocessing + pinned memory can
        # be unstable on macOS/MPS in some environments.
        num_workers = int(getattr(self.cfg, "num_workers", 0 if sys.platform == "darwin" else 8))
        pin_memory = bool(getattr(self.cfg, "pin_memory", self.device.type == "cuda"))

        self.train_loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.val_loader = DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.test_loader = DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.train_loader_bin_ordered = DataLoader(
            train_bin_ordered,
            batch_size=self.cfg.batch_size_bin,
            shuffle=False,
            collate_fn=None,
        )

        total_steps = self.cfg.epochs * len(self.train_loader)
        warmup_steps = max(1, int(0.1 * total_steps))
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=1e-6,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        # ablation_delta (only every diag_ablation_interval epochs)
        self.latent_diagnostics: List[Dict[str, Any]] = []

        if self.resume:
            self._resume_from_latest()

    def _checkpoint_path(self, name: str) -> str:
        return os.path.join(self.checkpoint_dir, name)

    def _build_checkpoint_payload(self, epoch: int, val_loss: float, val_metrics: Dict[str, float]) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "run_id": self.run_id,
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "config": asdict(self.cfg),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "saved_at": time.strftime("%Y-%m-%d_%H-%M-%S"),
        }

    def _save_checkpoint(self, epoch: int, val_loss: float, val_metrics: Dict[str, float], best: bool = False) -> None:
        payload = self._build_checkpoint_payload(epoch=epoch, val_loss=val_loss, val_metrics=val_metrics)

        latest_path = self._checkpoint_path("latest.pt")
        torch.save(payload, latest_path)

        if (epoch + 1) % self.cfg.checkpoint_every == 0:
            periodic_name = f"epoch_{epoch + 1:04d}.pt"
            torch.save(payload, self._checkpoint_path(periodic_name))

        if best:
            torch.save(payload, self._checkpoint_path("best.pt"))

    def _find_latest_checkpoint_path(self) -> Optional[str]:
        latest = self._checkpoint_path("latest.pt")
        if os.path.exists(latest):
            return latest
        if not os.path.exists(self.checkpoint_dir):
            return None
        ckpts = [
            os.path.join(self.checkpoint_dir, p)
            for p in os.listdir(self.checkpoint_dir)
            if p.endswith(".pt")
        ]
        if not ckpts:
            return None
        ckpts.sort(key=os.path.getmtime, reverse=True)
        return ckpts[0]

    def _resume_from_latest(self) -> None:
        ckpt_path = self._find_latest_checkpoint_path()
        if ckpt_path is None:
            log.warning("Resume requested but no checkpoint was found. Starting from epoch 0.")
            return

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        saved_cfg = checkpoint.get("config", {})
        for key in ["latent_dim", "loss_type"]:
            if key in saved_cfg and getattr(self.cfg, key) != saved_cfg[key]:
                raise ValueError(
                    f"Checkpoint/config mismatch for '{key}': "
                    f"checkpoint={saved_cfg[key]} current={getattr(self.cfg, key)}"
                )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = int(checkpoint.get("epoch", -1))
        self.start_epoch = self.current_epoch + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.train_losses = list(checkpoint.get("train_losses", []))
        self.val_losses = list(checkpoint.get("val_losses", []))
        self.last_val_metrics = dict(checkpoint.get("val_metrics", {}))

        rng_numpy = checkpoint.get("rng_numpy")
        rng_torch = checkpoint.get("rng_torch")
        rng_cuda = checkpoint.get("rng_cuda")
        if rng_numpy is not None:
            np.random.set_state(rng_numpy)
        if rng_torch is not None:
            torch.set_rng_state(rng_torch)
        if rng_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_cuda)

        log.info(f"Resumed from checkpoint: {ckpt_path} (epoch {self.current_epoch})")

    def train_epoch(self) -> float:
        """Train for one epoch. Returns average CE loss (regularization excluded)."""
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
                outputs = outputs.masked_fill(mask == 0, float("-inf"))
                loss = self.criterion(outputs, targets, mask)
            else:
                outputs = self.model(inputs, bin_idx)
                loss = self.criterion(outputs, targets)

            # Smoothness regularization: λ * ||Z - HZ||^2
            Z = self.model.latent_embedding.weight  # [n_bins, latent_dim]
            HZ = torch.sparse.mm(self.H_torch, Z)
            smooth_loss = self.cfg.latent_smooth_reg * torch.sum((Z - HZ) ** 2)
            total_loss = loss + smooth_loss

            self.optimizer.zero_grad()
            total_loss.backward()

            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.mlp.parameters()) + [self.model.latent_embedding.weight],
                    self.cfg.grad_clip,
                )

            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size  # track CE loss only
            n_samples += batch_size

        epoch_loss = running_loss / max(1, n_samples)
        return epoch_loss


    @torch.no_grad()
    def validate(self, split: Literal["train", "val", "test"]) -> float:
        data_loader = (
            self.train_loader if split == "train" else
            self.val_loader if split == "val" else
            self.test_loader
        )

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
                outputs = outputs.masked_fill(mask == 0, float("-inf"))
                loss = self.criterion(outputs, targets, mask)
            else:
                outputs = self.model(inputs, bin_idx)
                loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        return float(running_loss / max(1, n_samples))


    @torch.no_grad()
    def get_predictions(
        self,
        split: Literal["train", "val", "test"] = "test",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data_loader = (
            self.train_loader if split == "train" else
            self.val_loader if split == "val" else
            self.test_loader
        )
        self.model.eval()

        sample_pred: Dict[int, List[float]] = {}
        sample_true: Dict[int, List[float]] = {}
        sample_bins: Dict[int, List[int]] = {}

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
                    outputs = self.model(inputs_flat, bin_idx_flat).unsqueeze(0)
                    probs = F.softmax(outputs, dim=-1).squeeze(0).cpu().numpy()
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
                    s_idx = sample_idx[i].item() if hasattr(sample_idx[i], 'item') else int(sample_idx[i])
                    b_idx = int(bin_idx[i].item() if hasattr(bin_idx[i], 'item') else bin_idx[i])
                    input_i = inputs[i].unsqueeze(0)
                    bin_idx_i = bin_idx[i].unsqueeze(0)
                    output = self.model(input_i, bin_idx_i)
                    prob = float(torch.sigmoid(output).cpu().numpy().item())
                    y_true = float(targets[i].cpu().numpy().item())
                    sample_pred.setdefault(s_idx, []).append(prob)
                    sample_true.setdefault(s_idx, []).append(y_true)
                    sample_bins.setdefault(s_idx, []).append(b_idx)

        if self.loss_type == "logistic":
            for s_idx, preds in sample_pred.items():
                pred_arr = np.array(preds)
                pred_sum = pred_arr.sum()
                if pred_sum > 0:
                    sample_pred[s_idx] = (pred_arr / pred_sum).tolist()

        idx_to_sample = {v: k for k, v in self.sample_index.items()}
        idx_to_bin = {v: k for k, v in self.bin_index.items()}

        preds_flat, trues_flat, sample_labels, bin_labels = [], [], [], []
        for s_idx in sorted(sample_pred.keys()):
            n = len(sample_pred[s_idx])
            preds_flat.extend(sample_pred[s_idx])
            trues_flat.extend(sample_true[s_idx])
            sample_labels.extend([idx_to_sample[s_idx]] * n)
            bin_labels.extend([idx_to_bin[int(b)] for b in sample_bins[s_idx]])

        return (
            np.array(preds_flat, dtype=np.float32),
            np.array(trues_flat, dtype=np.float32),
            np.array(sample_labels),
            np.array(bin_labels),
        )

    def compute_metrics(self, split: Literal["train", "val", "test"]) -> Dict[str, float]:
        preds, targets, sample_labels, _ = self.get_predictions(split=split)
        return compute_extended_metrics(y_true=targets, y_pred=preds, sample_labels=sample_labels)

    def _metric_key(self, metric_name: str) -> str:
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

    def _plot_training_progress(self) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        epoch_nums = [e for e, _ in self.train_losses]
        train_vals = [l for _, l in self.train_losses]
        val_vals = [l for _, l in self.val_losses]

        ax.plot(epoch_nums, train_vals, "b-", linewidth=2, alpha=0.8, label="Train Loss")
        ax.plot(epoch_nums, val_vals, "r-", linewidth=2, alpha=0.8, label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Progress")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        root = os.path.dirname(os.path.abspath(__file__))
        fig_dir = os.path.join(root, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        save_path = os.path.join(fig_dir, f"training_progress_{self.model_name}_{self.run_id}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def run(self, use_wandb: bool = True) -> Dict[str, Any]:
        """Run joint MLP+latent training."""

        log.info(
            f"Starting training for model={self.model_name}, epochs={self.cfg.epochs}, resume={self.resume}"
        )
        pbar = tqdm(range(self.start_epoch, self.cfg.epochs), desc="Training", leave=True)
        for epoch in pbar:
            self.current_epoch = epoch
            epoch_start = time.perf_counter()

            self.train_epoch()

            train_eval_start = time.perf_counter()
            train_loss = self.validate(split="train")
            train_eval_s = time.perf_counter() - train_eval_start

            val_eval_start = time.perf_counter()
            val_loss = self.validate(split="val")
            val_eval_s = time.perf_counter() - val_eval_start

            self.train_losses.append((epoch, train_loss))
            self.val_losses.append((epoch, val_loss))

            metric_start = time.perf_counter()
            val_metrics = self.compute_metrics(split="val")
            metric_s = time.perf_counter() - metric_start
            self.last_val_metrics = val_metrics

            if use_wandb and WANDB_AVAILABLE:
                payload = {
                    "train/epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/global_step": self.global_step,
                }
                for metric_name, metric_value in val_metrics.items():
                    payload[f"val/metrics/{self._metric_key(metric_name)}"] = metric_value
                wandb.log(payload)

            if epoch % 10 == 0:
                pbar.set_postfix({
                    "train": f"{train_loss:.6f}",
                    "val": f"{val_loss:.6f}",
                })
                log.info(
                    f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}"
                )

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss

            self._save_checkpoint(epoch=epoch, val_loss=val_loss, val_metrics=val_metrics, best=improved)

            epoch_total_s = time.perf_counter() - epoch_start

            if use_wandb and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "timing/epoch/total_s": float(epoch_total_s),
                        "timing/epoch/train_eval_s": float(train_eval_s),
                        "timing/epoch/val_eval_s": float(val_eval_s),
                        "timing/epoch/metrics_s": float(metric_s),
                    },
                    step=self.global_step,
                )

            log.info(
                f"Epoch {epoch + 1}/{self.cfg.epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"best_val={self.best_val_loss:.6f}, epoch_s={epoch_total_s:.2f}"
            )

        best_ckpt = self._checkpoint_path("best.pt")
        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self._plot_training_progress()
        test_loss = self.validate(split="test")
        test_metrics = self.compute_metrics(split="test")
        predictions, targets, sample_labels, bin_labels = self.get_predictions(split="test")
        latent_vector = self.model.get_latent()  # [n_bins, latent_dim]
        return {
            "model": self.model_name,
            "run_id": self.run_id,
            "best_val_loss": self.best_val_loss,
            "test_loss": test_loss,
            "predictions": predictions,
            "targets": targets,
            "sample_labels": sample_labels,
            "bin_labels": bin_labels,
            "latent_vector": latent_vector,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.last_val_metrics,
            "test_metrics": test_metrics,
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
        latent_col_mean = W_latent.norm(p='fro').item() / self.latent_dim
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


    def _to_device(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)
        bin_idx = batch["bin_idx"].to(self.device)
        sample_idx = batch["sample_idx"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)
        return inputs, targets, bin_idx, sample_idx, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metabarcoding training entrypoint")
    parser.add_argument("--model", type=str, required=True, help="Name of the model variant being trained")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint for this model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    set_seed()
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = Config()

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.abspath(os.path.join(cfg.results_dir, args.model))
    os.makedirs(run_dir, exist_ok=True)

    use_wandb = WANDB_AVAILABLE
    if not WANDB_AVAILABLE:
        log.warning("wandb is not installed; continuing without wandb logging")
    else:
        wandb.init(
            project="metabarcoding",
            name=f"{args.model}_{run_id}",
            config={"model": args.model, **asdict(cfg)},
            dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wandb"),
        )

    trainer = Trainer(cfg=cfg, model_name=args.model, run_id=run_id, resume=args.resume)
    results = trainer.run(use_wandb=use_wandb)

    pkl_name = f"results_{args.model}_{run_id}.pkl"
    pkl_path = os.path.join(run_dir, pkl_name)
    with open(pkl_path, "wb") as fh:
        pickle.dump(results, fh)
    log.info(f"Results saved to: {pkl_path}")

    if use_wandb:
        wandb.finish()
