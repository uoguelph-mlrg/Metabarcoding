from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Tuple
import logging as log

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config, set_seed
from dataset import MBDataset, collate_samples
from Metabarcoding.src.latent_solver import LatentSolver
from loss import Loss
from mlp import MLPModel
from model import Model
from neighbor_graph import NeighbourGraph
from utils import load


class Trainer:
    """
    Trains MLP and latent embedding jointly in a single forward-backward pass per epoch.
    """
    def __init__(
        self,
        cfg: Config,
        model_name: str = "default",
        run_id: Optional[str] = None,
        resume: bool = False,
        fixed_split_indices: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.cfg = cfg
        self.model_name = model_name
        self.run_id = run_id or time.strftime("%Y-%m-%d_%H-%M-%S")
        self.resume = resume

        if self.cfg.use_embedding and self.cfg.barcode_data_path is None and self.cfg.embedding_path is None:
            self.cfg.barcode_data_path = self.cfg.data_path

        self.start_epoch = 0
        self.current_epoch = -1
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.last_val_metrics: Dict[str, float] = {}
        self.latent_global_step = 0

        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses: List[Tuple[int, float]] = []

        self.base_artifact_dir = os.path.abspath(os.path.join(self.cfg.results_dir, self.model_name))
        self.checkpoint_dir = os.path.join(self.base_artifact_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        data, bins_df, bin_index, sample_index, split_indices = load(
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
            embed_dim=self.cfg.embed_dim,
            gating_fn=self.cfg.gating_fn,
        )
        latent_solver.build_interpolation_matrix(method="nw")

        self.device = torch.device(self.cfg.device)
        input_dim = data["train"]["X"].shape[1]
        mlp_model = MLPModel(
            input_dim,
            hidden_dims=[128, 128, 128, 128],
            output_dim=self.cfg.embed_dim,
            dropout=self.cfg.dropout,
        ).to(self.device)

        self.model = Model(
            mlp_model,
            latent_solver,
            n_bins=len(bin_index),
            device=self.device,
            embed_dim=self.cfg.embed_dim,
            gating_fn=self.cfg.gating_fn,
            gating_alpha=self.cfg.gating_alpha,
            gating_kappa=self.cfg.gating_kappa,
            gating_epsilon=self.cfg.gating_epsilon,
        )

        if self.cfg.embed_dim > 1:
            optim_params = [
                {"params": self.model.mlp.parameters(), "weight_decay": self.cfg.weight_decay},
                {"params": self.model.final_linear.parameters(), "weight_decay": self.cfg.final_linear_wd},
            ]
        else:
            optim_params = [
                {"params": self.model.mlp.parameters(), "weight_decay": self.cfg.weight_decay},
            ]
        self.optimizer = torch.optim.AdamW(optim_params, lr=self.cfg.lr)

        self.loss_type: Literal["cross_entropy", "logistic"] = self.cfg.loss_type
        self.loss_mode = "sample" if self.loss_type == "cross_entropy" else "bin"
        self.criterion = Loss(task=self.loss_type)

        train = MBDataset(data["train"], bin_index, sample_index, loss_mode=self.loss_mode)
        val = MBDataset(data["val"], bin_index, sample_index, loss_mode=self.loss_mode)
        test = MBDataset(data["test"], bin_index, sample_index, loss_mode=self.loss_mode)

        batch_size = self.cfg.batch_size_sample if self.loss_mode == "sample" else self.cfg.batch_size_bin
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

        self._latent_total_steps = max(1, self.cfg.epochs * len(self.train_loader))
        self._latent_warmup_steps = max(0, int(self.cfg.latent_lr_warmup_frac * self._latent_total_steps))

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
            "latent_global_step": self.latent_global_step,
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

    def _latent_lr_for_step(self, step_idx: int) -> float:
        total_steps = max(1, self._latent_total_steps)
        base_lr = float(self.cfg.latent_adam_lr)
        start_factor = float(self.cfg.latent_lr_warmup_start_factor)
        eta_min = float(self.cfg.latent_lr_eta_min)
        warmup_steps = min(max(0, int(self._latent_warmup_steps)), total_steps)
        step_idx = max(0, min(int(step_idx), total_steps - 1))

        if warmup_steps > 0 and step_idx < warmup_steps:
            if warmup_steps == 1:
                return max(eta_min, base_lr)
            progress = step_idx / float(warmup_steps - 1)
            factor = start_factor + (1.0 - start_factor) * progress
            return max(eta_min, base_lr * factor)

        cosine_steps = max(1, total_steps - warmup_steps)
        if cosine_steps == 1:
            return max(eta_min, base_lr)

        cosine_step = min(step_idx - warmup_steps, cosine_steps - 1)
        progress = cosine_step / float(cosine_steps - 1)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(eta_min + (base_lr - eta_min) * cosine)

    def _resume_from_latest(self) -> None:
        ckpt_path = self._find_latest_checkpoint_path()
        if ckpt_path is None:
            log.warning("Resume requested but no checkpoint was found. Starting from epoch 0.")
            return

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        saved_cfg = checkpoint.get("config", {})
        for key in ["embed_dim", "gating_fn", "loss_type"]:
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
        self.latent_global_step = int(checkpoint.get("latent_global_step", 0))

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

    def _train_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)

        t0 = time.perf_counter()
        if self.loss_mode == "sample":
            bsz, max_bins, n_feat = inputs.shape
            inputs_flat = inputs.view(bsz * max_bins, n_feat)
            bin_idx_flat = bin_idx.view(bsz * max_bins)
            outputs_flat = self.model(inputs_flat, bin_idx_flat)
            outputs = outputs_flat.view(bsz, max_bins)
            outputs = outputs.masked_fill(mask == 0, float("-inf"))
            loss = self.criterion(outputs, targets, mask)
        else:
            outputs = self.model(inputs, bin_idx)
            loss = self.criterion(outputs, targets)
        t1 = time.perf_counter()

        self.optimizer.zero_grad()
        loss.backward()
        t2 = time.perf_counter()

        if self.cfg.grad_clip is not None:
            params_to_clip = list(self.model.mlp.parameters())
            if self.cfg.embed_dim > 1:
                params_to_clip += list(self.model.final_linear.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, self.cfg.grad_clip)

        self.optimizer.step()
        t3 = time.perf_counter()

        timing = {
            "forward_s": float(t1 - t0),
            "backward_s": float(t2 - t1),
            "optim_s": float(t3 - t2),
            "total_s": float(t3 - t0),
        }
        return float(loss.item()), timing

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
                bsz, max_bins, n_feat = inputs.shape
                inputs_flat = inputs.view(bsz * max_bins, n_feat)
                bin_idx_flat = bin_idx.view(bsz * max_bins)
                outputs_flat = self.model(inputs_flat, bin_idx_flat)
                outputs = outputs_flat.view(bsz, max_bins)
                outputs = outputs.masked_fill(mask == 0, float("-inf"))
                loss = self.criterion(outputs, targets, mask)
            else:
                outputs = self.model(inputs, bin_idx)
                loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        return float(running_loss / max(1, n_samples))

    def solve_latent(self, batch: Dict[str, torch.Tensor], prox_weight: float = 0.0) -> Tuple[np.ndarray, Dict[str, float]]:
        self.model.eval()

        with torch.no_grad():
            inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)

            if self.loss_mode == "sample":
                bsz, max_bins, n_feat = inputs.shape
                inputs_flat = inputs.view(bsz * max_bins, n_feat)
                intrinsic_flat = self.model.mlp(inputs_flat).view(bsz, max_bins, self.cfg.embed_dim)

                sample_grid = sample_idx.unsqueeze(1).expand(-1, max_bins)
                valid = mask.bool() if mask is not None else torch.ones_like(bin_idx, dtype=torch.bool)

                intrinsic_vec = intrinsic_flat[valid].detach().cpu().numpy()
                if self.cfg.embed_dim == 1:
                    intrinsic_vec = intrinsic_vec.reshape(-1)
                y_vec = targets[valid].detach().cpu().numpy().reshape(-1)
                bin_ids = bin_idx[valid].detach().cpu().numpy().reshape(-1).astype(np.int64)
                sample_ids = sample_grid[valid].detach().cpu().numpy().reshape(-1).astype(np.int64)
            else:
                intrinsic = self.model.mlp(inputs)
                if self.cfg.embed_dim == 1:
                    intrinsic = intrinsic.squeeze(-1)
                intrinsic_vec = intrinsic.detach().cpu().numpy()
                y_vec = targets.detach().cpu().numpy().reshape(-1)
                bin_ids = bin_idx.detach().cpu().numpy().reshape(-1).astype(np.int64)
                sample_ids = sample_idx.detach().cpu().numpy().reshape(-1).astype(np.int64)

        latent_anchor = self.model.latent_vec.detach().cpu().numpy()
        latent_lr = self._latent_lr_for_step(self.latent_global_step)


        latent_vec, timings = self.model.latent_solver.solve(
            y=y_vec,
            intrinsic_vec=intrinsic_vec,
            final_weights=self.model.final_linear.weight.detach().cpu().numpy().squeeze() if self.cfg.embed_dim > 1 else None,
            bin_ids=bin_ids,
            sample_ids=sample_ids,
            loss_type=self.loss_type,
            prox_weight=prox_weight,
            x_anchor=latent_anchor,
            latent_lr=latent_lr,
        )

        self.model.set_latent(latent_vec)
        self.latent_global_step += 1
        return latent_vec, timings

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
                bsz = inputs.shape[0]
                for b in range(bsz):
                    s_idx = int(sample_idx[b].item())
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
                    s_idx = int(sample_idx[i].item())
                    b_idx = int(bin_idx[i].item())
                    output = self.model(inputs[i].unsqueeze(0), bin_idx[i].unsqueeze(0))
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
        y_pred, y_true, sample_labels, _ = self.get_predictions(split=split)

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid]
        y_pred = np.clip(y_pred[valid], 0, 1)
        eps = 1e-10
        
        def _shannon_diversity(values: np.ndarray, eps: float = 1e-10) -> float:
            """Compute Shannon diversity from non-negative abundance values."""
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size == 0:
                return np.nan
            probs = (arr + eps) / np.sum(arr + eps)
            return float(-np.sum(probs * np.log(probs + eps)))
        
        def _spearman_rho(x: np.ndarray, y: np.ndarray) -> Optional[float]:
            """Compute Spearman rho and return None if undefined."""
            rank_x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
            rank_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
            if np.std(rank_x) == 0 or np.std(rank_y) == 0:
                return None
            rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
            return rho if np.isfinite(rho) else None
        
        def _r2_and_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
            """Compute R² and intercept from scatter points."""
            if len(y_true) == 0:
                return np.nan, np.nan
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = float(1 - ss_res / (ss_tot + eps))
            
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            intercept = float(intercept) if np.isfinite(intercept) else np.nan
            return r2, intercept

        rmse_macro = np.nan
        mae_macro = np.nan
        kl_divergence = np.nan
        shannon_r2 = np.nan
        shannon_intercept = np.nan
        spearman_macro = np.nan

        if sample_labels is not None:
            sample_labels_v = np.asarray(sample_labels)[valid]
            rmse_per, mae_per, kl_per = [], [], []
            shannon_true_per, shannon_pred_per = [], []
            spearman_per = []
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

                shannon_true = _shannon_diversity(true_s, eps)
                shannon_pred = _shannon_diversity(pred_s, eps)
                if np.isfinite(shannon_true) and np.isfinite(shannon_pred):
                    shannon_true_per.append(shannon_true)
                    shannon_pred_per.append(shannon_pred)

                if len(true_s) > 1:
                    rho = _spearman_rho(true_s, pred_s)
                    if rho is not None:
                        spearman_per.append(rho)

            if rmse_per:
                rmse_macro = float(np.mean(rmse_per))
                mae_macro = float(np.mean(mae_per))
                kl_divergence = float(np.mean(kl_per))

            if len(shannon_true_per) > 1:
                shannon_r2, shannon_intercept = _r2_and_intercept(np.array(shannon_true_per), np.array(shannon_pred_per))

            if spearman_per:
                spearman_macro = float(np.mean(spearman_per))

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
            "R² (Shannon diversity)": shannon_r2,
            "Shannon intercept": shannon_intercept,
            "Spearman Rho (macro)": spearman_macro,
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

        fig_dir = os.path.join(self.base_artifact_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        save_path = os.path.join(fig_dir, f"training_progress_{self.model_name}_{self.run_id}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def run(self, use_wandb: bool = True) -> Dict[str, Any]:
        warmup_epochs = max(1, int(self.cfg.latent_warmup_frac * self.cfg.epochs))
        log.info(
            f"Starting training for model={self.model_name}, epochs={self.cfg.epochs}, resume={self.resume}"
        )

        for epoch in tqdm(range(self.start_epoch, self.cfg.epochs), desc="Epochs", leave=False):
            self.current_epoch = epoch
            epoch_start = time.perf_counter()

            alpha = min(1.0, epoch / warmup_epochs)
            prox_weight = self.cfg.latent_prox_scale * self.cfg.latent_l2_reg * (1.0 - alpha)

            batch_forward, batch_backward, batch_optim, batch_total, batch_latent = [], [], [], [], []
            latent_vec = self.model.latent_vec.detach().cpu().numpy()

            for batch_idx, batch in enumerate(self.train_loader):
                batch_start = time.perf_counter()

                latent_start = time.perf_counter()
                latent_vec, latent_timings = self.solve_latent(batch=batch, prox_weight=prox_weight)
                latent_s = time.perf_counter() - latent_start

                loss_value, mlp_timings = self._train_batch(batch)
                self.scheduler.step()
                self.global_step += 1

                total_s = time.perf_counter() - batch_start
                batch_forward.append(mlp_timings["forward_s"])
                batch_backward.append(mlp_timings["backward_s"])
                batch_optim.append(mlp_timings["optim_s"])
                batch_total.append(total_s)
                batch_latent.append(latent_s)

                if use_wandb and WANDB_AVAILABLE:
                    # timings = {"setup_s": setup_done, "epoch_logits_s": [], "epoch_loss_s": [], "epoch_backward_s": [], "epoch_optimizer_s": []}
                    wandb.log(
                        {
                            "train/batch/loss": loss_value,
                            "train/batch/mlp_lr": self.optimizer.param_groups[0]["lr"],
                            "train/batch/latent_lr": latent_timings["latent_lr"],
                            "timing/batch/mlp/forward_s": mlp_timings["forward_s"],
                            "timing/batch/mlp/backward_s": mlp_timings["backward_s"],
                            "timing/batch/mlp/optim_s": mlp_timings["optim_s"],
                            "timing/batch/latent_solve_s": latent_s,
                            "timing/batch/latent/setup_s": latent_timings["setup_s"],
                            "timing/batch/latent/forward_s": latent_timings["forward_s"],
                            "timing/batch/latent/backward_s": latent_timings["backward_s"],
                            "timing/batch/latent/optim_s": latent_timings["optim_s"],
                            "timing/batch/total_s": total_s,
                            "train/epoch": epoch,
                            "train/batch_idx": batch_idx,
                            "train/global_step": self.global_step,
                            "train/latent_global_step": self.latent_global_step,
                        },
                        step=self.global_step,
                    )

            train_eval_start = time.perf_counter()
            train_loss = self.validate(split="train")
            train_eval_s = time.perf_counter() - train_eval_start

            val_eval_start = time.perf_counter()
            val_loss = self.validate(split="val")
            val_eval_s = time.perf_counter() - val_eval_start
            
            train_metric_start = time.perf_counter()
            train_metrics = self.compute_metrics(split="train")
            train_metric_s = time.perf_counter() - train_metric_start

            val_metric_start = time.perf_counter()
            val_metrics = self.compute_metrics(split="val")
            val_metric_s = time.perf_counter() - val_metric_start
            self.last_val_metrics = val_metrics

            self.train_losses.append((epoch, train_loss))
            self.val_losses.append((epoch, val_loss))

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss

            self._save_checkpoint(epoch=epoch, val_loss=val_loss, val_metrics=val_metrics, best=improved)

            epoch_total_s = time.perf_counter() - epoch_start
            batch_forward_arr = np.array(batch_forward, dtype=float)
            batch_backward_arr = np.array(batch_backward, dtype=float)
            batch_optim_arr = np.array(batch_optim, dtype=float)
            batch_total_arr = np.array(batch_total, dtype=float)
            batch_latent_arr = np.array(batch_latent, dtype=float)

            epoch_log_payload = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/epoch": epoch,
                "train/global_step": self.global_step,
                "latent/mean": float(latent_vec.mean()),
                "latent/std": float(latent_vec.std()),
                "latent/min": float(latent_vec.min()),
                "latent/max": float(latent_vec.max()),
                "timing/epoch/total_s": float(epoch_total_s),
                "timing/epoch/train_eval_s": float(train_eval_s),
                "timing/epoch/val_eval_s": float(val_eval_s),
                "timing/epoch/train_metrics_s": float(train_metric_s),
                "timing/epoch/val_metrics_s": float(val_metric_s),
                "timing/epoch/batch_forward_mean_s": float(np.mean(batch_forward_arr)),
                "timing/epoch/batch_backward_mean_s": float(np.mean(batch_backward_arr)),
                "timing/epoch/batch_optim_mean_s": float(np.mean(batch_optim_arr)),
                "timing/epoch/batch_total_mean_s": float(np.mean(batch_total_arr)),
                "timing/epoch/batch_latent_mean_s": float(np.mean(batch_latent_arr)),
                "timing/epoch/latent_total_s": float(np.sum(batch_latent_arr)),
            }
            for metric_name, metric_value in train_metrics.items():
                epoch_log_payload[f"train/metrics/{self._metric_key(metric_name)}"] = metric_value
            for metric_name, metric_value in val_metrics.items():
                epoch_log_payload[f"val/metrics/{self._metric_key(metric_name)}"] = metric_value

            if use_wandb and WANDB_AVAILABLE:
                wandb.log(epoch_log_payload, step=self.global_step)

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

        if use_wandb and WANDB_AVAILABLE:
            payload = {
                "test/loss": test_loss,
            }
            for metric_name, metric_value in test_metrics.items():
                payload[f"test/metrics/{self._metric_key(metric_name)}"] = metric_value
            wandb.log(payload, step=self.global_step)

        predictions, targets, sample_labels, bin_labels = self.get_predictions(split="test")
        latent_vector = self.model.latent_vec.detach().cpu().numpy()

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
