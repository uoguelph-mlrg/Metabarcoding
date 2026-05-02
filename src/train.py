from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Tuple
import logging as log

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config, set_seed
from dataset import MBDataset, collate_samples
from latent_solver import LatentSolver
from loss import Loss
from metrics import compute_metrics, metric_key
from mlp import MLPModel
from model import Model
from neighbor_graph import NeighbourGraph
from utils import load, PREPROCESSING_STATE_FILENAME


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
        results_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.model_name = model_name
        self.run_id = run_id or time.strftime("%Y-%m-%d_%H-%M-%S")
        self.resume = resume
        self._validate_interpolation_config()

        self.start_epoch = 0
        self.current_epoch = -1
        self.best_val_loss = float("inf")
        self.last_val_metrics: Dict[str, float] = {}

        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses: List[Tuple[int, float]] = []

        effective_results_dir = os.path.abspath(results_dir) if results_dir is not None else self.cfg.results_dir
        self.base_artifact_dir = os.path.abspath(os.path.join(effective_results_dir, self.model_name))
        self.checkpoint_dir = os.path.join(self.base_artifact_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.preprocessing_state_path = self._checkpoint_path(PREPROCESSING_STATE_FILENAME)

        if self.resume and not os.path.exists(self.preprocessing_state_path):
            raise ValueError(
                "Resume requested but preprocessing artifact is missing: "
                f"{self.preprocessing_state_path}."
            )

        data, taxonomy_df, embeddings_array, bins_with_embedding_arr, bin_index, sample_index, split_indices, preprocessing_state_path = load(
            self.cfg,
            fixed_split_indices=fixed_split_indices,
            preprocessing_state_path=self.preprocessing_state_path,
        )

        self.data = data
        self.bin_index = bin_index
        self.sample_index = sample_index
        self.split_indices = split_indices
        self.preprocessing_state_path = preprocessing_state_path

        self.neighbour_graph = NeighbourGraph(
            self.cfg,
            taxonomy_df,
            embeddings=embeddings_array,
            bins_with_embedding=bins_with_embedding_arr,
        )
        self.neighbour_graph.build()

        latent_solver = LatentSolver(
            self.cfg,
            self.neighbour_graph,
            embed_dim=self.cfg.embed_dim,
            gating_fn=self.cfg.gating_fn,
        )
        latent_solver.build_interpolation_matrix()

        self.device = torch.device(self.cfg.device)
        input_dim = data["train"]["X"].shape[1]
        mlp_model = MLPModel(
            input_dim,
            hidden_dims=self.cfg.mlp_hidden_dims,
            output_dim=self.cfg.embed_dim,
            dropout=self.cfg.dropout,
        ).to(self.device)

        self.model = Model(
            mlp_model,
            latent_solver,
            n_bins=len(bin_index),
            device=self.device,
            latent_init_std=self.cfg.latent_init_std,
            embed_dim=self.cfg.embed_dim,
            gating_fn=self.cfg.gating_fn,
            gating_alpha=self.cfg.gating_alpha,
            gating_kappa=self.cfg.gating_kappa,
            gating_epsilon=self.cfg.gating_epsilon,
            interpolation_enabled=bool(self.cfg.train_MLP_with_interpolation or self.cfg.inference_with_interpolation),
        )
        self.model.latent_vec.requires_grad_(False) # latent is optimized separately, not by the main optimizer

        if self.cfg.embed_dim > 1:
            optim_params = [
                {"params": self.model.mlp.parameters(), "weight_decay": self.cfg.weight_decay},
                {"params": self.model.final_linear.parameters(), "weight_decay": self.cfg.final_linear_weight_decay},
            ]
        else:
            optim_params = [
                {"params": self.model.mlp.parameters(), "weight_decay": self.cfg.weight_decay},
            ]
        self.mlp_optimizer = torch.optim.AdamW(optim_params, lr=self.cfg.mlp_lr)
        self.latent_optimizer = torch.optim.AdamW([self.model.latent_vec], lr=self.cfg.latent_lr)

        self.loss_type: Literal["cross_entropy", "logistic"] = self.cfg.loss_type
        self.loss_mode = "sample" if self.loss_type == "cross_entropy" else "bin"
        self.criterion = Loss(task=self.loss_type)

        train = MBDataset(data["train"], bin_index, sample_index, loss_mode=self.loss_mode)
        val = MBDataset(data["val"], bin_index, sample_index, loss_mode=self.loss_mode)
        test = MBDataset(data["test"], bin_index, sample_index, loss_mode=self.loss_mode)
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self._train_sample_ids = np.unique(train.sample_ids).astype(np.int64, copy=False)
        self._epoch_selected_sample_ids = np.empty(0, dtype=np.int64)
        self._epoch_selected_sample_ids_t = torch.empty(0, dtype=torch.long)

        batch_size = self.cfg.batch_size_sample if self.loss_mode == "sample" else self.cfg.batch_size_bin
        collate_fn = collate_samples if self.loss_mode == "sample" else None
        num_workers = int(getattr(self.cfg, "num_workers", 0 if sys.platform == "darwin" else 8))
        pin_memory = bool(getattr(self.cfg, "pin_memory", self.device.type == "cuda"))
        args = {
            "batch_size": batch_size, "collate_fn": collate_fn, "num_workers": num_workers, "pin_memory": pin_memory
        }

        self.train_loader = DataLoader(train, shuffle=True, drop_last=True, **args)
        self.val_loader = DataLoader(val, shuffle=False, **args)
        self.test_loader = DataLoader(test, shuffle=False, **args)

        total_steps = max(1, self.cfg.epochs * len(self.train_loader))
        warmup_steps = max(1, int(self.cfg.mlp_warmup_frac * total_steps))
        warmup_scheduler = LinearLR(self.mlp_optimizer, start_factor=self.cfg.mlp_warmup_start_factor, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.mlp_optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=self.cfg.mlp_lr_eta_min)
        self.mlp_scheduler = SequentialLR(self.mlp_optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

        total_steps = max(1, self.cfg.epochs * len(self.train_loader) * self.cfg.latent_optim_steps)
        warmup_steps = max(1, int(self.cfg.latent_warmup_frac * total_steps))
        warmup_scheduler = LinearLR(self.latent_optimizer, start_factor=self.cfg.latent_warmup_start_factor, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.latent_optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=self.cfg.latent_lr_eta_min)
        self.latent_scheduler = SequentialLR(self.latent_optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

        self.latent_diagnostics: List[Dict[str, Any]] = []

        if self.resume:
            self._resume_from_latest()

    def _validate_interpolation_config(self) -> None:
        fraction = float(self.cfg.interpolated_sample_fraction)
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("interpolated_sample_fraction must be in [0, 1]")

    def _select_epoch_interpolated_samples(self) -> np.ndarray:
        fraction = float(self.cfg.interpolated_sample_fraction)
        if fraction <= 0.0 or self._train_sample_ids.size == 0:
            return np.empty(0, dtype=np.int64)

        n_select = min(self._train_sample_ids.size, int(np.ceil(fraction * self._train_sample_ids.size)))
        if n_select <= 0:
            return np.empty(0, dtype=np.int64)

        selected = np.random.choice(self._train_sample_ids, size=n_select, replace=False)
        selected.sort()
        return selected.astype(np.int64, copy=False)

    def _refresh_epoch_interpolation_selection(self) -> None:
        self._epoch_selected_sample_ids = self._select_epoch_interpolated_samples()
        self._epoch_selected_sample_ids_t = torch.as_tensor(
            self._epoch_selected_sample_ids,
            dtype=torch.long,
            device=self.device,
        )

    def _epoch_sample_selection_mask(self, sample_idx: torch.Tensor) -> Optional[torch.Tensor]:
        """Return per-sample boolean mask for this epoch's interpolation subset.

        The returned mask has the same first dimension as sample_idx and is used to
        decide where interpolated latents are enabled during training/latent solves.
        """
        if self._epoch_selected_sample_ids_t.numel() == 0:
            return None
        sample_idx_np = sample_idx.detach().cpu().numpy().reshape(-1)
        selected_np = np.isin(sample_idx_np, self._epoch_selected_sample_ids)
        return torch.as_tensor(selected_np, dtype=torch.bool, device=sample_idx.device)

    def _checkpoint_path(self, name: str) -> str:
        return os.path.join(self.checkpoint_dir, name)

    def _build_checkpoint_payload(self, epoch: int, val_loss: float, val_metrics: Dict[str, float]) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "run_id": self.run_id,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "config": asdict(self.cfg),
            "model_state_dict": self.model.state_dict(),
            "mlp_optimizer_state_dict": self.mlp_optimizer.state_dict(),
            "mlp_scheduler_state_dict": self.mlp_scheduler.state_dict(),
            "latent_optimizer_state_dict": self.latent_optimizer.state_dict(),
            "latent_scheduler_state_dict": self.latent_scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "latent_diagnostics": self.latent_diagnostics,
            "preprocessing_state_path": self.preprocessing_state_path,
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
        for key in ["embed_dim", "gating_fn", "loss_type"]:
            if key in saved_cfg and getattr(self.cfg, key) != saved_cfg[key]:
                raise ValueError(
                    f"Checkpoint/config mismatch for '{key}': "
                    f"checkpoint={saved_cfg[key]} current={getattr(self.cfg, key)}"
                )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.mlp_optimizer.load_state_dict(checkpoint["mlp_optimizer_state_dict"])
        self.mlp_scheduler.load_state_dict(checkpoint["mlp_scheduler_state_dict"])
        self.latent_optimizer.load_state_dict(checkpoint["latent_optimizer_state_dict"])
        self.latent_scheduler.load_state_dict(checkpoint["latent_scheduler_state_dict"])

        self.current_epoch = int(checkpoint.get("epoch", -1))
        self.start_epoch = self.current_epoch + 1
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.train_losses = list(checkpoint.get("train_losses", []))
        self.val_losses = list(checkpoint.get("val_losses", []))
        self.last_val_metrics = dict(checkpoint.get("val_metrics", {}))
        self.latent_diagnostics = list(checkpoint.get("latent_diagnostics", []))

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

    @staticmethod
    def _has_interpolation_samples(sample_selection: Optional[torch.Tensor]) -> bool:
        """Return True iff sample_selection is non-None and contains at least one True entry."""
        if sample_selection is None or sample_selection.numel() == 0:
            return False
        return bool(torch.any(sample_selection).item())

    def _train_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Run one MLP optimization step and return loss plus timing breakdown.

        In sample mode, inputs are flattened to observation level, then reshaped back
        to [batch_size, max_bins] for masked cross-entropy. In bin mode, observations
        are already independent and no reshape is needed.
        """
        self.model.train()
        inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
        sample_selection = self._epoch_sample_selection_mask(sample_idx)

        t0 = time.perf_counter()
        if self.loss_mode == "sample":
            bsz, max_bins, n_feat = inputs.shape
            # Flatten keeps sample order, so bin/sample alignment is preserved.
            inputs_flat = inputs.view(bsz * max_bins, n_feat)
            bin_idx_flat = bin_idx.view(bsz * max_bins)
            interpolation_mask = None
            if self.cfg.train_MLP_with_interpolation and self._has_interpolation_samples(sample_selection):
                # Expand sample-level selection to observation-level and drop padded bins.
                assert sample_selection is not None
                interpolation_mask = sample_selection.unsqueeze(1).expand(-1, max_bins)
                if mask is not None:
                    interpolation_mask = interpolation_mask & mask.bool()
                interpolation_mask = interpolation_mask.reshape(-1)
            outputs_flat = self.model(
                inputs_flat,
                bin_idx_flat,
                interpolation_mask=interpolation_mask,
            )
            outputs = outputs_flat.view(bsz, max_bins)
            # Padded positions must be -inf so softmax contributes exactly zero mass.
            outputs = outputs.masked_fill(mask == 0, float("-inf"))
            loss = self.criterion(outputs, targets, mask)
        else:
            interpolation_mask = None
            if self.cfg.train_MLP_with_interpolation and self._has_interpolation_samples(sample_selection):
                interpolation_mask = sample_selection
            outputs = self.model(
                inputs,
                bin_idx,
                interpolation_mask=interpolation_mask,
            )
            loss = self.criterion(outputs, targets)
        t1 = time.perf_counter()

        self.mlp_optimizer.zero_grad()
        loss.backward()
        t2 = time.perf_counter()

        if self.cfg.grad_clip is not None:
            params_to_clip = list(self.model.mlp.parameters())
            if self.cfg.embed_dim > 1:
                params_to_clip += list(self.model.final_linear.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, self.cfg.grad_clip)

        self.mlp_optimizer.step()
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
        use_interpolation = bool(self.cfg.inference_with_interpolation)
        running_loss = 0.0
        n_samples = 0
        for batch in data_loader:
            inputs, targets, bin_idx, sample_idx, mask = self._to_device(batch)
            if self.loss_mode == "sample":
                bsz, max_bins, n_feat = inputs.shape
                inputs_flat = inputs.view(bsz * max_bins, n_feat)
                bin_idx_flat = bin_idx.view(bsz * max_bins)
                interpolation_mask = mask.bool().view(-1) if use_interpolation and mask is not None else None
                outputs_flat = self.model(
                    inputs_flat,
                    bin_idx_flat,
                    interpolation_mask=interpolation_mask,
                )
                outputs = outputs_flat.view(bsz, max_bins)
                outputs = outputs.masked_fill(mask == 0, float("-inf"))
                loss = self.criterion(outputs, targets, mask)
            else:
                interpolation_mask = torch.ones_like(sample_idx, dtype=torch.bool) if use_interpolation else None
                outputs = self.model(
                    inputs,
                    bin_idx,
                    interpolation_mask=interpolation_mask,
                )
                loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        return float(running_loss / max(1, n_samples))

    def solve_latent(
        self,
        batch: Dict[str, torch.Tensor],
        prox_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run one latent-variable solve for the current batch.

        The latent solver consumes torch tensors directly on its configured device.
        This method temporarily enables gradient tracking for latent_vec only, then restores its
        previous requires_grad state before returning.
        """
        self.model.eval()

        with torch.no_grad():
            inputs, targets, bin_ids, sample_ids, mask = self._to_device(batch)
            sample_selection = self._epoch_sample_selection_mask(sample_ids)

            if self.loss_mode == "sample":
                bsz, max_bins, n_feat = inputs.shape
                inputs_flat = inputs.view(bsz * max_bins, n_feat)
                intrinsic = self.model.mlp(inputs_flat).view(bsz, max_bins, self.cfg.embed_dim)

                sample_grid = sample_ids.unsqueeze(1).expand(-1, max_bins)
                valid = mask.bool() if mask is not None else torch.ones_like(bin_ids, dtype=torch.bool)
                interpolation_mask = None
                if self._has_interpolation_samples(sample_selection):
                    assert sample_selection is not None
                    interpolation_mask = sample_selection.unsqueeze(1).expand(-1, max_bins)
                    interpolation_mask = interpolation_mask & valid
                    interpolation_mask = interpolation_mask[valid]

                intrinsic = intrinsic[valid]
                targets = targets[valid]
                bin_ids = bin_ids[valid]
                sample_ids = sample_grid[valid]
            else:
                intrinsic = self.model.mlp(inputs)
                interpolation_mask = None
                if self._has_interpolation_samples(sample_selection):
                    assert sample_selection is not None
                    interpolation_mask = sample_selection.to(dtype=torch.bool)

        previous_requires_grad = self.model.latent_vec.requires_grad
        # Invariant: latent is optimized only in this block, not in _train_batch.
        self.model.latent_vec.requires_grad_(True)

        try:
            latent_vec, timings = self.model.latent_solver.solve(
                y=targets,
                intrinsic=intrinsic,
                final_weights=self.model.final_linear.weight if self.cfg.embed_dim > 1 else None,
                bin_ids=bin_ids,
                sample_ids=sample_ids,
                interpolation_mask=interpolation_mask,
                loss_type=self.loss_type,
                prox_weight=prox_weight,
                latent=self.model.latent_vec,
                optimizer=self.latent_optimizer,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.latent_scheduler.step()
        finally:
            self.model.latent_vec.requires_grad_(previous_requires_grad)

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
        use_interpolation = bool(self.cfg.inference_with_interpolation)

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
                    interpolation_mask = torch.ones(len(bin_idx_flat), dtype=torch.bool, device=inputs_flat.device) if use_interpolation else None
                    outputs = self.model(
                        inputs_flat,
                        bin_idx_flat,
                        interpolation_mask=interpolation_mask,
                    ).unsqueeze(0)
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
                    interpolation_mask = torch.ones(1, dtype=torch.bool, device=self.device) if use_interpolation else None
                    output = self.model(
                        inputs[i].unsqueeze(0),
                        bin_idx[i].unsqueeze(0),
                        interpolation_mask=interpolation_mask,
                    )
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


    def compute_metrics(
        self,
        split: Literal["train", "val", "test"],
        predictions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Compute per-observation and per-sample evaluation metrics.

        Args:
            split: Dataset split to evaluate.
            predictions: Optional pre-computed output of get_predictions(). When
                provided the forward pass is skipped.
        """
        y_pred, y_true, sample_labels, _ = predictions if predictions is not None else self.get_predictions(split=split)
        return compute_metrics(y_pred, y_true, sample_labels=sample_labels)

    def run(self, use_wandb: bool = True) -> Dict[str, Any]:
        log.info(
            f"Starting training for model={self.model_name}, epochs={self.cfg.epochs}, resume={self.resume}"
        )
        for epoch in tqdm(range(self.start_epoch, self.cfg.epochs), desc="Epochs", leave=False):
            self.current_epoch = epoch
            epoch_start = time.perf_counter()
            if self.cfg.interpolated_sample_fraction > 0.0:
                self._refresh_epoch_interpolation_selection()
            else:
                self._epoch_selected_sample_ids = np.empty(0, dtype=np.int64)
                self._epoch_selected_sample_ids_t = torch.empty(0, dtype=torch.long)

            # Linearly fade proximal pull as training progresses: early stability,
            # late freedom for latent adaptation.
            alpha = min(1.0, epoch / self.cfg.epochs)
            prox_weight = self.cfg.latent_init_prox_reg * (1.0 - alpha)

            for batch_idx, batch in enumerate(self.train_loader):
                batch_start = time.perf_counter()

                latent_start = time.perf_counter()
                latent_vec, latent_timings = self.solve_latent(batch=batch, prox_weight=prox_weight)
                latent_s = time.perf_counter() - latent_start

                loss_value, mlp_timings = self._train_batch(batch)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self.mlp_scheduler.step()

                total_s = time.perf_counter() - batch_start

                if use_wandb and WANDB_AVAILABLE:
                    # timings = {"setup_s": setup_done, "epoch_logits_s": [], "epoch_loss_s": [], "epoch_backward_s": [], "epoch_optimizer_s": []}
                    wandb.log({
                        "train/batch/loss": loss_value,
                        "train/batch/mlp_lr": self.mlp_optimizer.param_groups[0]["lr"],
                        "train/batch/latent_lr": latent_timings["latent_lr"],
                        "timing/batch/mlp/forward_s": mlp_timings["forward_s"],
                        "timing/batch/mlp/backward_s": mlp_timings["backward_s"],
                        "timing/batch/mlp/optim_s": mlp_timings["optim_s"],
                        "timing/batch/latent/total": latent_s,
                        "timing/batch/latent/setup_s": latent_timings["setup_s"],
                        "timing/batch/latent/forward_s": latent_timings["forward_s"],
                        "timing/batch/latent/backward_s": latent_timings["backward_s"],
                        "timing/batch/latent/optim_s": latent_timings["optim_s"],
                        "train/batch/latent/loss_CE": latent_timings["loss_CE"],
                        "train/batch/latent/loss_l2": latent_timings["loss_l2"],
                        "train/batch/latent/loss_smooth": latent_timings["loss_smooth"],
                        "train/batch/latent/loss_prox": latent_timings["loss_prox"],
                        "train/batch/latent/loss_total": latent_timings["loss_total"],
                        "timing/batch/total_s": total_s,
                        "train/epoch": epoch,
                        "train/batch_idx": batch_idx,
                        "train/interpolated_sample_count": int(self._epoch_selected_sample_ids.size),
                    })

            train_eval_start = time.perf_counter()
            train_preds = self.get_predictions(split="train")
            train_loss = self.validate(split="train")
            train_eval_s = time.perf_counter() - train_eval_start

            val_eval_start = time.perf_counter()
            val_preds = self.get_predictions(split="val")
            val_loss = self.validate(split="val")
            val_eval_s = time.perf_counter() - val_eval_start

            train_metric_start = time.perf_counter()
            train_metrics = self.compute_metrics(split="train", predictions=train_preds)
            train_metric_s = time.perf_counter() - train_metric_start

            val_metric_start = time.perf_counter()
            val_metrics = self.compute_metrics(split="val", predictions=val_preds)
            val_metric_s = time.perf_counter() - val_metric_start
            self.last_val_metrics = val_metrics

            run_ablation = bool(self.cfg.diag_ablation_interval > 0 and (epoch % self.cfg.diag_ablation_interval == 0))
            diag = self._collect_diagnostics(epoch=epoch, run_abl=run_ablation)
            self.latent_diagnostics.append(diag)

            self.train_losses.append((epoch, train_loss))
            self.val_losses.append((epoch, val_loss))

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss

            self._save_checkpoint(epoch=epoch, val_loss=val_loss, val_metrics=val_metrics, best=improved)

            epoch_total_s = time.perf_counter() - epoch_start

            epoch_log_payload = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/epoch": epoch,
                "train/interpolated_sample_count": int(self._epoch_selected_sample_ids.size),
                "timing/epoch/total_s": float(epoch_total_s),
                "timing/epoch/train_eval_s": float(train_eval_s),
                "timing/epoch/val_eval_s": float(val_eval_s),
                "timing/epoch/train_metrics_s": float(train_metric_s),
                "timing/epoch/val_metrics_s": float(val_metric_s),
            }
            for metric_name, metric_value in train_metrics.items():
                epoch_log_payload[f"train/metrics/{metric_key(metric_name)}"] = metric_value
            for metric_name, metric_value in val_metrics.items():
                epoch_log_payload[f"val/metrics/{metric_key(metric_name)}"] = metric_value
            for diag_name, diag_value in diag.items():
                epoch_log_payload[f"diag/{diag_name}"] = diag_value

            if use_wandb and WANDB_AVAILABLE:
                wandb.log(epoch_log_payload)

            log.info(
                f"Epoch {epoch + 1}/{self.cfg.epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"best_val={self.best_val_loss:.6f}, epoch_s={epoch_total_s:.2f}"
            )

        best_ckpt = self._checkpoint_path("best.pt")
        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        _keep = {"best.pt", "latest.pt", PREPROCESSING_STATE_FILENAME}
        for _ckpt_file in os.listdir(self.checkpoint_dir):
            if _ckpt_file not in _keep:
                try:
                    os.remove(os.path.join(self.checkpoint_dir, _ckpt_file))
                except OSError:
                    pass

        test_loss = self.validate(split="test")
        test_preds = self.get_predictions(split="test")
        test_metrics = self.compute_metrics(split="test", predictions=test_preds)

        if use_wandb and WANDB_AVAILABLE:
            payload = {
                "test/loss": test_loss,
            }
            for metric_name, metric_value in test_metrics.items():
                payload[f"test/metrics/{metric_key(metric_name)}"] = metric_value
            wandb.log(payload)

        predictions, targets, sample_labels, bin_labels = test_preds
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
            "latent_diagnostics": self.latent_diagnostics,
            "test_metrics": test_metrics,
        }

    @torch.no_grad()
    def _compute_ablation_loss(self) -> float:
        saved = self.model.latent_vec.data.clone()
        self.model.latent_vec.data.zero_()
        loss_no_latent = self.validate(split="val")
        self.model.latent_vec.data.copy_(saved)
        return float(loss_no_latent)

    def _collect_diagnostics(self, epoch: int, run_abl: bool = False) -> Dict[str, Any]:
        diag: Dict[str, Any] = {
            "epoch": int(epoch),
            # Latent metrics are computed as the average per-dimension to avoid domination by large embed_dim values.
            "latent_mean": float(self.model.latent_vec.data.mean(dim=0).mean().item()),
            "latent_std": float(self.model.latent_vec.data.std(dim=0).mean().item()),
            "latent_min": float(self.model.latent_vec.data.min().item()),
            "latent_max": float(self.model.latent_vec.data.max().item()),
            "ablation_loss": self._compute_ablation_loss() if run_abl else None,
            "final_weight_mean": float(self.model.final_linear.weight.data.mean().item()) if self.cfg.embed_dim > 1 else None,
            "final_weight_std": float(self.model.final_linear.weight.data.std().item()) if self.cfg.embed_dim > 1 else None,
            "final_weight_norm": float(self.model.final_linear.weight.data.norm().item()) if self.cfg.embed_dim > 1 else None,
        }
        return diag

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
    parser.add_argument("--model", type=str, default="default", help="Name of the model variant being trained")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint for this model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--results_dir", type=str, default=None, help="Override results directory (default: cfg.results_dir)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count (default: cfg.epochs)")
    parser.add_argument("--data_path", type=str, default=None, help="Override data CSV path (default: cfg.data_path)")
    args = parser.parse_args()

    set_seed()
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.data_path is not None:
        cfg.data_path = os.path.abspath(args.data_path)
        cfg.preprocessed_dir = None  # don't reuse cached preprocessed data for a different dataset

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    results_base = args.results_dir if args.results_dir else cfg.results_dir
    run_dir = os.path.abspath(os.path.join(results_base, args.model, run_id))
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
