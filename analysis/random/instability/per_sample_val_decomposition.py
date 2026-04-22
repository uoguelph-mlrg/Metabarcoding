#!/usr/bin/env python3
"""
Script 4: per_sample_val_decomposition.py
-------------------------------------------
Decomposes the validation loss by individual sample at two checkpoint epochs:
  - best.pt   (epoch 13 — lowest val loss)
  - epoch_0050.pt (epoch 50 — fully exploded val loss)

For each validation sample, computes its cross-entropy contribution and the
difference between the two checkpoints to identify which samples drive the explosion.

Also checks whether affected samples have more bins shared with the training set
(consistent with latent overfitting hypothesis H1) or are OOD samples (H3).

Usage:
    cd Metabarcoding/analysis/random/instability
    python per_sample_val_decomposition.py [--best PATH] [--late PATH]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config, set_seed
from dataset import MBDataset, collate_samples
from latent_solver import LatentSolver
from mlp import MLPModel
from model import Model
from utils import load

CHECKPOINT_DIR = ROOT / "results" / "default_src" / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "figures"
LARGE_RUN_THRESHOLD = 5_000_000


# ---------------------------------------------------------------------------
# Minimal NeighbourGraph stub (same as Script 2)
# ---------------------------------------------------------------------------
class _StubNeighbourGraph:
    def __init__(self, n_bins: int) -> None:
        self.n_bins = n_bins
        self.neighbours: List = []


def build_inference_model(ckpt_path: Path, input_dim: int, device: str = "cpu") -> Tuple[Model, Config, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    valid_keys = {f.name for f in dc_fields(Config)}
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in valid_keys})
    cfg.device = device

    state = ckpt["model_state_dict"]
    n_bins = state["latent_vec"].shape[0]

    mlp = MLPModel(input_dim, cfg.mlp_hidden_dims, cfg.embed_dim, cfg.dropout)
    stub = _StubNeighbourGraph(n_bins)
    ls = LatentSolver(cfg, stub, cfg.embed_dim, cfg.gating_fn)
    model = Model(
        mlp=mlp, latent_solver=ls, n_bins=n_bins, device=torch.device(device),
        embed_dim=cfg.embed_dim, gating_fn=cfg.gating_fn,
        gating_alpha=cfg.gating_alpha, gating_kappa=cfg.gating_kappa,
        gating_epsilon=cfg.gating_epsilon, latent_init_std=0.0,
        interpolation_enabled=False,
    )
    model.load_state_dict(state)
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def compute_per_sample_losses(
    model: Model,
    val_dataset: MBDataset,
    device: str = "cpu",
) -> Dict[int, float]:
    """
    Compute per-sample cross-entropy loss (sample_idx → loss).
    Uses batch_size=1 for exact per-sample values.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_samples)
    dev = torch.device(device)
    per_sample: Dict[int, float] = {}
    model.eval()

    for batch in loader:
        inputs  = batch["input"].to(dev)    # [1, max_bins, n_feat]
        targets = batch["target"].to(dev)   # [1, max_bins]
        bin_idx = batch["bin_idx"].to(dev)  # [1, max_bins]
        mask    = batch["mask"].to(dev)     # [1, max_bins]
        s_idx   = int(batch["sample_idx"][0].item())

        _, max_bins, n_feat = inputs.shape
        inputs_flat  = inputs.view(max_bins, n_feat)
        bin_idx_flat = bin_idx.view(max_bins)

        outputs_flat = model(inputs_flat, bin_idx_flat, interpolation_mask=None)
        outputs = outputs_flat.view(1, max_bins)
        outputs = outputs.masked_fill(mask == 0, float("-inf"))

        # Per-sample CE: -sum(y * log_softmax(logits))
        log_probs = F.log_softmax(outputs, dim=-1)   # [1, max_bins]
        log_probs_safe = torch.where(mask.bool(), log_probs, torch.zeros_like(log_probs))
        loss_val = float(-(targets * log_probs_safe).sum().item())
        per_sample[s_idx] = loss_val

    return per_sample


def main(best_ckpt_path: Optional[Path], late_ckpt_path: Optional[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Default checkpoint paths
    # ------------------------------------------------------------------
    if best_ckpt_path is None:
        best_ckpt_path = CHECKPOINT_DIR / "best.pt"
    if late_ckpt_path is None:
        # Use epoch_0050.pt if it exists for the failing run, else latest
        cand = CHECKPOINT_DIR / "epoch_0050.pt"
        if cand.exists() and cand.stat().st_size > LARGE_RUN_THRESHOLD:
            late_ckpt_path = cand
        else:
            all_large = sorted(
                [p for p in CHECKPOINT_DIR.glob("epoch_*.pt") if p.stat().st_size > LARGE_RUN_THRESHOLD]
            )
            late_ckpt_path = all_large[-1] if all_large else None

    if not best_ckpt_path.exists():
        print(f"Best checkpoint not found: {best_ckpt_path}")
        return
    if late_ckpt_path is None or not late_ckpt_path.exists():
        print("Late checkpoint not found.")
        return

    print(f"Early (best) checkpoint: {best_ckpt_path}")
    print(f"Late  checkpoint:        {late_ckpt_path}")

    # ------------------------------------------------------------------
    # Load data with reproducible split
    # ------------------------------------------------------------------
    print("\nLoading validation data...")
    # Need config to load data: read from best checkpoint
    sample_cfg_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    valid_keys = {f.name for f in dc_fields(Config)}
    cfg = Config(**{k: v for k, v in sample_cfg_ckpt["config"].items() if k in valid_keys})

    set_seed(42)
    data, _, bin_index, sample_index, _ = load(cfg, save_data=False)
    input_dim = data["val"]["X"].shape[1]

    val_dataset  = MBDataset(data["val"],   bin_index, sample_index, loss_mode="sample")
    train_dataset = MBDataset(data["train"], bin_index, sample_index, loss_mode="sample")

    idx_to_sample = {v: k for k, v in sample_index.items()}

    # ------------------------------------------------------------------
    # Compute bins per sample and training-overlap stats
    # ------------------------------------------------------------------
    # Build set of training bins
    train_bin_set = set(train_dataset.bin_uris.tolist())

    # For each val sample: count bins and count shared with training
    val_sample_info: Dict[int, Dict] = {}
    for s_idx in val_dataset._sample_list:
        row_ids = val_dataset._sample_to_indices[s_idx]
        bin_ids = val_dataset.bin_uris[row_ids].tolist()
        n_bins_sample = len(bin_ids)
        n_shared = sum(1 for b in bin_ids if b in train_bin_set)
        val_sample_info[s_idx] = {
            "n_bins": n_bins_sample,
            "n_shared_with_train": n_shared,
            "frac_shared": n_shared / max(1, n_bins_sample),
        }

    # ------------------------------------------------------------------
    # Compute per-sample losses for both checkpoints
    # ------------------------------------------------------------------
    print("\nBuilding early (best) model and computing per-sample losses...")
    model_early, _, ckpt_early = build_inference_model(best_ckpt_path, input_dim)
    losses_early = compute_per_sample_losses(model_early, val_dataset)
    epoch_early = ckpt_early.get("epoch", -1) + 1

    print("Building late model and computing per-sample losses...")
    model_late, _, ckpt_late = build_inference_model(late_ckpt_path, input_dim)
    losses_late = compute_per_sample_losses(model_late, val_dataset)
    epoch_late = ckpt_late.get("epoch", -1) + 1

    # ------------------------------------------------------------------
    # Build results dataframe
    # ------------------------------------------------------------------
    rows = []
    for s_idx in sorted(set(losses_early) | set(losses_late)):
        le = losses_early.get(s_idx, np.nan)
        ll = losses_late.get(s_idx, np.nan)
        info = val_sample_info.get(s_idx, {})
        rows.append({
            "sample_idx":     s_idx,
            "sample_id":      idx_to_sample.get(s_idx, f"idx_{s_idx}"),
            "loss_early":     le,
            "loss_late":      ll,
            "delta_loss":     ll - le,
            "n_bins":         info.get("n_bins", 0),
            "n_shared":       info.get("n_shared_with_train", 0),
            "frac_shared":    info.get("frac_shared", 0),
        })

    df = pd.DataFrame(rows).sort_values("delta_loss", ascending=False)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    avg_early = np.nanmean([r["loss_early"] for r in rows])
    avg_late  = np.nanmean([r["loss_late"]  for r in rows])
    print(f"\nAggregate val loss — epoch {epoch_early}: {avg_early:.4f}, epoch {epoch_late}: {avg_late:.4f}")
    print(f"Average per-sample delta: {avg_late - avg_early:+.4f}")

    n_exploded = (df["delta_loss"] > 5).sum()
    print(f"Samples with delta_loss > 5:  {n_exploded}/{len(df)} ({100*n_exploded/len(df):.0f}%)")
    n_neg = (df["delta_loss"] < 0).sum()
    print(f"Samples that IMPROVED:        {n_neg}/{len(df)} ({100*n_neg/len(df):.0f}%)")

    print(f"\nTop 15 worst-affected samples (largest loss increase from epoch {epoch_early} to {epoch_late}):")
    print(df.head(15)[[
        "sample_id", "loss_early", "loss_late", "delta_loss",
        "n_bins", "n_shared", "frac_shared"
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Correlation of delta_loss with n_shared and n_bins
    if len(df) > 5:
        corr_shared = df[["delta_loss", "n_shared"]].dropna().corr().iloc[0, 1]
        corr_frac   = df[["delta_loss", "frac_shared"]].dropna().corr().iloc[0, 1]
        corr_bins   = df[["delta_loss", "n_bins"]].dropna().corr().iloc[0, 1]
        print(f"\nCorrelation(delta_loss, n_shared_with_train) = {corr_shared:.4f}")
        print(f"Correlation(delta_loss, frac_shared)          = {corr_frac:.4f}")
        print(f"Correlation(delta_loss, n_bins_in_sample)     = {corr_bins:.4f}")
        if corr_shared > 0.3 or corr_frac > 0.3:
            print("  → Samples with MORE training-overlapping bins suffer larger loss increases.")
            print("    Consistent with H1: shared bins have miscalibrated latents.")
        else:
            print("  → Loss increase is NOT strongly correlated with bin overlap.")
            print("    May point to OOD samples (H3) or a systemic latent effect.")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    # Fig 1: Per-sample loss comparison (sorted by delta)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    x = np.arange(len(df))

    ax = axes[0]
    ax.bar(x, df["loss_early"].values, alpha=0.6, color="steelblue", label=f"Epoch {epoch_early} (best)")
    ax.bar(x, df["loss_late"].values,  alpha=0.5, color="red",       label=f"Epoch {epoch_late} (exploded)")
    ax.set_ylabel("Per-sample CE loss")
    ax.set_title(f"Per-Sample Validation Loss: Epoch {epoch_early} vs Epoch {epoch_late}\n"
                 f"(Sorted by loss increase; n={len(df)} validation samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = ["red" if d > 0 else "green" for d in df["delta_loss"]]
    ax.bar(x, df["delta_loss"].values, color=colors, alpha=0.7)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Validation sample (sorted by loss increase)")
    ax.set_ylabel(f"Δ loss (epoch {epoch_late} − epoch {epoch_early})")
    ax.set_title("Loss Change Per Sample")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "04a_per_sample_loss_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: delta_loss vs frac_shared
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(df["frac_shared"], df["delta_loss"], alpha=0.7, s=40, c=df["n_bins"],
               cmap="viridis", norm=matplotlib.colors.LogNorm(
                   vmin=max(1, df["n_bins"].min()), vmax=df["n_bins"].max()
               ))
    ax.set_xlabel("Fraction of sample's bins shared with training set")
    ax.set_ylabel(f"Δ loss (epoch {epoch_late} − epoch {epoch_early})")
    ax.set_title("Loss Increase vs Bin Overlap With Training\n"
                 "(H1: more overlap → larger increase)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(df["n_bins"], df["delta_loss"], alpha=0.7, s=40, c=df["frac_shared"],
               cmap="plasma")
    ax.set_xlabel("Number of bins in validation sample")
    ax.set_ylabel(f"Δ loss (epoch {epoch_late} − epoch {epoch_early})")
    ax.set_title("Loss Increase vs Sample Size")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "04b_delta_loss_vs_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures saved to: {out_dir}")
    print("Script 4 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-sample validation loss decomposition")
    parser.add_argument("--best", type=Path, default=None, help="Best checkpoint path")
    parser.add_argument("--late", type=Path, default=None, help="Late (exploded) checkpoint path")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    main(args.best, args.late, args.out_dir)
