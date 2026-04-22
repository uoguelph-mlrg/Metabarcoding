#!/usr/bin/env python3
"""
Script 5: diagnose_training_verbose.py
----------------------------------------
A modified training run (30 epochs by default) that captures per-batch
gradient and latent diagnostics to identify *when and where* the instability
originates.

Approach: subclasses Trainer from src/train.py and overrides solve_latent()
to add gradient norm logging BEFORE the optimizer step. This is a clean
modification that doesn't duplicate the 1000-line train.py.

Key outputs:
  - instability_batch_diag.csv  : per-batch gradient/latent metrics
  - instability_epoch_diag.csv  : per-epoch loss and latent stats
  - Figures in figures/

Key question: Do gradient norms start growing before epoch 14 (first val loss
rise), or only after? Is the growth concentrated in specific bins?

Usage:
    cd Metabarcoding/analysis/random/instability
    python diagnose_training_verbose.py [--epochs 30] [--verbose]

    Then parse the CSV output or inspect the figures.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: add Metabarcoding/src to path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import logging as log
from config import Config, set_seed   # noqa: E402
from train import Trainer             # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Monitoring Trainer subclass
# ---------------------------------------------------------------------------

class MonitoringTrainer(Trainer):
    """
    Trainer subclass with per-batch gradient and latent monitoring.

    Overrides solve_latent() to capture gradient statistics BEFORE
    the optimizer step. Also logs per-epoch summary to CSV.
    """

    def __init__(self, cfg: Config, model_name: str = "instability_diag",
                 csv_batch_path: Optional[Path] = None,
                 csv_epoch_path: Optional[Path] = None,
                 **kwargs) -> None:
        super().__init__(cfg=cfg, model_name=model_name, **kwargs)

        self._batch_csv_path = csv_batch_path or OUT_DIR / "instability_batch_diag.csv"
        self._epoch_csv_path = csv_epoch_path or OUT_DIR / "instability_epoch_diag.csv"

        # Initialise CSV files with headers
        self._batch_csv_fh = open(self._batch_csv_path, "w", newline="")
        self._epoch_csv_fh = open(self._epoch_csv_path, "w", newline="")

        batch_fields = [
            "epoch", "batch_idx",
            "latent_grad_norm_full",      # gradient norm on full latent_vec (before step)
            "active_set_size",            # number of bins in the active set
            "active_latent_norm_mean",    # mean ||h_b||_2 for active bins
            "active_latent_norm_max",     # max  ||h_b||_2 for active bins
            "active_latent_norm_std",     # std  ||h_b||_2 for active bins
            "loss_CE",                    # CE component (per solve)
            "loss_l2",                    # L2 component
            "loss_smooth",                # smoothness component
            "loss_prox",                  # proximal component
            "loss_total",                 # total latent loss
            "latent_lr",                  # current latent learning rate
            "mlp_batch_loss",             # MLP cross-entropy loss for this batch
        ]
        epoch_fields = [
            "epoch", "train_loss", "val_loss",
            "latent_mean", "latent_std", "latent_min", "latent_max",
            "latent_frac_saturated_high",  # fraction of sigmoid values > 1.9 (h >> 0)
            "latent_frac_saturated_low",   # fraction of sigmoid values < 0.1 (h << 0)
            "mlp_lr",
        ]
        self._batch_writer = csv.DictWriter(self._batch_csv_fh, fieldnames=batch_fields)
        self._epoch_writer = csv.DictWriter(self._epoch_csv_fh, fieldnames=epoch_fields)
        self._batch_writer.writeheader()
        self._epoch_writer.writeheader()
        self._batch_csv_fh.flush()
        self._epoch_csv_fh.flush()

        self._current_batch_idx = 0
        self._batch_rows = []

    def close_csv(self) -> None:
        self._batch_csv_fh.close()
        self._epoch_csv_fh.close()

    # ------------------------------------------------------------------
    # Override solve_latent to capture gradient norms
    # ------------------------------------------------------------------
    def solve_latent(
        self,
        batch: Dict[str, torch.Tensor],
        prox_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Identical to Trainer.solve_latent(), with gradient monitoring hooks
        added INSIDE the latent_solver.solve() call (via monkey-patching the
        optimizer step).
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
                    interpolation_mask = sample_selection.unsqueeze(1).expand(-1, max_bins)
                    interpolation_mask = interpolation_mask & valid
                    interpolation_mask = interpolation_mask[valid]

                intrinsic = intrinsic[valid]
                y = targets[valid]
                bin_ids_solve = bin_ids[valid]
                sample_ids_solve = sample_grid[valid]
            else:
                intrinsic = self.model.mlp(inputs)
                bin_ids_solve = bin_ids
                sample_ids_solve = sample_ids
                interpolation_mask = None
                if self._has_interpolation_samples(sample_selection):
                    interpolation_mask = sample_selection.to(dtype=torch.bool)
                y = targets

        previous_requires_grad = self.model.latent_vec.requires_grad
        self.model.latent_vec.requires_grad_(True)

        # ---- Pre-step gradient capture via a wrapped optimizer ----
        captured_grad_norm = [float("nan")]
        captured_active_ids = [None]

        original_step = self.latent_optimizer.step

        def step_with_capture(*args, **kwargs):
            # Capture gradient norm BEFORE the parameter update
            if self.model.latent_vec.grad is not None:
                captured_grad_norm[0] = float(
                    self.model.latent_vec.grad.norm().item()
                )
            return original_step(*args, **kwargs)

        self.latent_optimizer.step = step_with_capture

        try:
            # Build active set preview to get active bin ids
            if bin_ids_solve is not None:
                batch_bin_ids_np = bin_ids_solve.detach().cpu().numpy()
                active_map = self.model.latent_solver._build_active_set(batch_bin_ids_np)
                captured_active_ids[0] = active_map.global_ids

            latent_vec, timings = self.model.latent_solver.solve(
                y=y,
                intrinsic=intrinsic,
                final_weights=self.model.final_linear.weight if self.cfg.embed_dim > 1 else None,
                bin_ids=bin_ids_solve,
                sample_ids=sample_ids_solve,
                interpolation_mask=interpolation_mask,
                loss_type=self.loss_type,
                prox_weight=prox_weight,
                latent=self.model.latent_vec,
                optimizer=self.latent_optimizer,
            )
            self.latent_scheduler.step()
        finally:
            self.latent_optimizer.step = original_step
            self.model.latent_vec.requires_grad_(previous_requires_grad)

        # ---- Compute active-set latent norm stats ----
        active_ids = captured_active_ids[0]
        active_norm_mean = active_norm_max = active_norm_std = float("nan")
        active_set_size = 0
        if active_ids is not None and len(active_ids) > 0:
            active_set_size = int(len(active_ids))
            with torch.no_grad():
                active_ids_t = torch.as_tensor(active_ids, dtype=torch.long, device=self.model.latent_vec.device)
                active_lat = self.model.latent_vec.detach()[active_ids_t]  # [active, embed_dim] or [active]
                if active_lat.dim() == 1:
                    norms = active_lat.abs()
                else:
                    norms = active_lat.norm(dim=-1)
                active_norm_mean = float(norms.mean().item())
                active_norm_max  = float(norms.max().item())
                active_norm_std  = float(norms.std().item()) if len(norms) > 1 else 0.0

        # Store batch-level stats for later writing in run()
        self._batch_rows.append({
            "epoch":                  self.current_epoch,
            "batch_idx":              self._current_batch_idx,
            "latent_grad_norm_full":  captured_grad_norm[0],
            "active_set_size":        active_set_size,
            "active_latent_norm_mean": active_norm_mean,
            "active_latent_norm_max":  active_norm_max,
            "active_latent_norm_std":  active_norm_std,
            "loss_CE":     timings.get("loss_CE", float("nan")),
            "loss_l2":     timings.get("loss_l2", float("nan")),
            "loss_smooth": timings.get("loss_smooth", float("nan")),
            "loss_prox":   timings.get("loss_prox", float("nan")),
            "loss_total":  timings.get("loss_total", float("nan")),
            "latent_lr":   timings.get("latent_lr", float("nan")),
            "mlp_batch_loss": float("nan"),  # filled after _train_batch below
        })

        return latent_vec, timings

    def _patch_mlp_loss(self, mlp_loss: float) -> None:
        """Update the last batch row with the MLP batch loss."""
        if self._batch_rows:
            self._batch_rows[-1]["mlp_batch_loss"] = mlp_loss

    def _flush_batch_rows(self) -> None:
        for row in self._batch_rows:
            self._batch_writer.writerow(row)
        self._batch_rows.clear()
        self._batch_csv_fh.flush()

    def run(self, use_wandb: bool = True) -> Dict[str, Any]:
        """Overrides Trainer.run() to flush batch CSV and write epoch CSV."""
        import torch.nn.functional as F
        from tqdm import tqdm

        log.info(
            f"MonitoringTrainer: starting training for {self.cfg.epochs} epochs "
            f"(batch CSV: {self._batch_csv_path})"
        )

        for epoch in tqdm(range(self.start_epoch, self.cfg.epochs), desc="Epochs", leave=False):
            self.current_epoch = epoch
            self._current_batch_idx = 0

            if self.cfg.interpolated_sample_fraction > 0.0:
                self._refresh_epoch_interpolation_selection()
            else:
                self._epoch_selected_sample_ids = np.empty(0, dtype=np.int64)
                self._epoch_selected_sample_ids_t = torch.empty(0, dtype=torch.long)

            alpha = min(1.0, epoch / self.cfg.epochs)
            prox_weight = self.cfg.latent_init_prox_reg * (1.0 - alpha)

            for batch in self.train_loader:
                latent_vec, latent_timings = self.solve_latent(batch=batch, prox_weight=prox_weight)
                loss_value, mlp_timings = self._train_batch(batch)
                self.mlp_scheduler.step()
                self._patch_mlp_loss(loss_value)
                self._current_batch_idx += 1

            # Flush batch rows for this epoch
            self._flush_batch_rows()

            # Epoch-level eval
            train_loss = self.validate(split="train")
            val_loss   = self.validate(split="val")

            self.train_losses.append((epoch, train_loss))
            self.val_losses.append((epoch, val_loss))

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss

            # Compute latent saturation: fraction of sigmoid(h) values > 1.9 (h >> 0) or < 0.1 (h << 0)
            with torch.no_grad():
                lv = self.model.latent_vec.detach().cpu().float()
                sigmoid_vals = 2.0 * torch.sigmoid(lv)  # g(h) = 2*sigma(h)
                frac_high = float((sigmoid_vals > 1.9).float().mean().item())
                frac_low  = float((sigmoid_vals < 0.1).float().mean().item())
                lat_mean  = float(lv.mean().item())
                lat_std   = float(lv.std().item())
                lat_min   = float(lv.min().item())
                lat_max   = float(lv.max().item())

            epoch_row = {
                "epoch":      epoch + 1,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "latent_mean": lat_mean,
                "latent_std":  lat_std,
                "latent_min":  lat_min,
                "latent_max":  lat_max,
                "latent_frac_saturated_high": frac_high,
                "latent_frac_saturated_low":  frac_low,
                "mlp_lr": self.mlp_optimizer.param_groups[0]["lr"],
            }
            self._epoch_writer.writerow(epoch_row)
            self._epoch_csv_fh.flush()

            log.info(
                f"Epoch {epoch+1}/{self.cfg.epochs}: "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lat_max={lat_max:.4f}  sat_hi={frac_high:.3f}  sat_lo={frac_low:.3f}"
            )

        self.close_csv()
        self._plot_diagnostics()

        return {
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
            "best_val_loss": self.best_val_loss,
        }

    def _plot_diagnostics(self) -> None:
        """Post-training: load the saved CSVs and produce summary figures."""
        import pandas as pd

        fig_dir = OUT_DIR / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Load batch CSV
        try:
            df_batch = pd.read_csv(self._batch_csv_path)
        except Exception:
            return

        # Load epoch CSV
        try:
            df_epoch = pd.read_csv(self._epoch_csv_path)
        except Exception:
            return

        # --- Figure A: Epoch-level loss + latent max ---
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        ax = axes[0]
        ax.plot(df_epoch["epoch"], df_epoch["train_loss"], "b-o", ms=3, label="train loss")
        ax.plot(df_epoch["epoch"], df_epoch["val_loss"],   "r-o", ms=3, label="val loss")
        ax.set_ylabel("Loss")
        ax.set_title("Monitoring Run — Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(df_epoch["epoch"], df_epoch["latent_max"], "r-", label="latent max")
        ax.plot(df_epoch["epoch"], df_epoch["latent_std"], "m-", label="latent std")
        ax.set_ylabel("Latent magnitude")
        ax.set_title("Latent Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(df_epoch["epoch"], df_epoch["latent_frac_saturated_high"], "r-", label="frac g(h)>1.9 (saturated high)")
        ax.plot(df_epoch["epoch"], df_epoch["latent_frac_saturated_low"],  "b-", label="frac g(h)<0.1 (saturated low)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fraction of latent units")
        ax.set_title("Sigmoid Gating Saturation  g(h)=2σ(h)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(fig_dir / "05a_monitoring_epoch_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- Figure B: Batch-level gradient norms over training ---
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Colour each point by epoch
        epochs_batch = df_batch["epoch"].values
        ax = axes[0]
        sc = ax.scatter(range(len(df_batch)), df_batch["latent_grad_norm_full"],
                        c=epochs_batch, cmap="plasma", s=4, alpha=0.5)
        plt.colorbar(sc, ax=ax, label="Epoch")
        ax.set_ylabel("||grad||_2  (full latent_vec)")
        ax.set_title("Latent Gradient Norm Per Batch\n"
                     "(Expect growth to precede val loss explosion)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        sc = ax.scatter(range(len(df_batch)), df_batch["active_latent_norm_max"],
                        c=epochs_batch, cmap="plasma", s=4, alpha=0.5)
        ax.set_xlabel("Batch index (across all epochs)")
        ax.set_ylabel("max ||h_b||_2 in active set")
        ax.set_title("Active-Set Latent Norm Max Per Batch")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(fig_dir / "05b_batch_gradient_norms.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- Figure C: Per-epoch median gradient norm (smoothed view) ---
        if "epoch" in df_batch.columns and "latent_grad_norm_full" in df_batch.columns:
            epoch_grad = df_batch.groupby("epoch")["latent_grad_norm_full"].median()
            epoch_active_norm = df_batch.groupby("epoch")["active_latent_norm_max"].median()

            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            axes[0].plot(epoch_grad.index + 1, epoch_grad.values, "r-o", ms=3)
            axes[0].set_ylabel("Median gradient norm per epoch")
            axes[0].set_title("Per-Epoch Median Latent Gradient Norm\n"
                              "(Should start growing around the epoch where val loss rises)")
            axes[0].grid(True, alpha=0.3)
            if not df_epoch.empty:
                ax2 = axes[0].twinx()
                ax2.plot(df_epoch["epoch"], df_epoch["val_loss"], "b--", alpha=0.5, label="val loss")
                ax2.set_ylabel("Val loss", color="blue")

            axes[1].plot(epoch_active_norm.index + 1, epoch_active_norm.values, "m-o", ms=3)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Median active-set latent max norm")
            axes[1].set_title("Active-Set Latent Norm Growth")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(fig_dir / "05c_epoch_gradient_summary.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"\nMonitoring figures saved to: {fig_dir}")
        print(f"Batch CSV: {self._batch_csv_path}")
        print(f"Epoch CSV: {self._epoch_csv_path}")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verbose training diagnostic (gradient monitoring)")
    parser.add_argument("--epochs",  type=int, default=30,
                        help="Number of epochs to train (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    set_seed(42)
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = Config()
    cfg.epochs = args.epochs          # short run for diagnosis
    cfg.checkpoint_every = 999        # suppress periodic checkpointing (keeps it tidy)
    cfg.diag_ablation_interval = 5    # run ablation every 5 epochs

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    batch_csv = OUT_DIR / f"instability_batch_diag_{run_id}.csv"
    epoch_csv = OUT_DIR / f"instability_epoch_diag_{run_id}.csv"

    log.info(f"Starting monitoring run for {cfg.epochs} epochs")
    log.info(f"Batch CSV: {batch_csv}")
    log.info(f"Epoch CSV: {epoch_csv}")

    trainer = MonitoringTrainer(
        cfg=cfg,
        model_name="instability_diag",
        run_id=run_id,
        resume=False,
        csv_batch_path=batch_csv,
        csv_epoch_path=epoch_csv,
    )
    results = trainer.run(use_wandb=False)

    print(f"\nDiagnostic run complete.")
    print(f"  Best val loss: {results['best_val_loss']:.6f}")
    print(f"  Batch CSV:    {batch_csv}")
    print(f"  Epoch CSV:    {epoch_csv}")


if __name__ == "__main__":
    main()
