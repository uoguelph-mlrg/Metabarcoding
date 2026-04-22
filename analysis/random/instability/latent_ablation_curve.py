#!/usr/bin/env python3
"""
Script 2: latent_ablation_curve.py
------------------------------------
For each saved epoch checkpoint, compute:
  (a) full model val loss  (MLP + latent)
  (b) MLP-only val loss    (latent zeroed out)
  (c) latent_help = mlp_only - full  (positive = latent helps, negative = hurts)

Does NOT rebuild the neighbor graph (49-minute build). Instead it builds a
minimal inference-only model from the checkpoint state dict.

Key question: At which epoch does the latent start hurting validation more
than helping? Is MLP-only val loss stable across epochs?

If MLP-only loss is flat AND full-model loss explodes → latent is the sole culprit.

Usage:
    cd Metabarcoding/analysis/random/instability
    python latent_ablation_curve.py [--ckpt-dir PATH] [--data-dir PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup — resolve Metabarcoding/src from analysis/random/instability/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]   # Metabarcoding/
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config, set_seed       # noqa: E402
from dataset import MBDataset, collate_samples  # noqa: E402
from mlp import MLPModel                  # noqa: E402
from model import Model                   # noqa: E402
from latent_solver import LatentSolver    # noqa: E402
from utils import load                    # noqa: E402
from loss import Loss                     # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

CHECKPOINT_DIR = ROOT / "results" / "default_src" / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "figures"
LARGE_RUN_THRESHOLD = 5_000_000  # 5 MB, used to distinguish the two runs


# ---------------------------------------------------------------------------
# Minimal NeighbourGraph stub (no graph build — inference doesn't need it)
# ---------------------------------------------------------------------------
class _StubNeighbourGraph:
    """Minimal stub with just n_bins, sufficient for LatentSolver.__init__."""
    def __init__(self, n_bins: int) -> None:
        self.n_bins = n_bins
        self.neighbours: List = []


def build_inference_model(ckpt_path: Path, input_dim: int, device: str = "cpu") -> Tuple[Model, Config]:
    """
    Build a Model usable for inference-only from a checkpoint.

    Skips the NeighbourGraph build entirely — the forward pass only needs
    latent_vec, mlp, and final_linear (no graph operations during inference
    when inference_with_interpolation=False).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]

    # Reconstruct Config, tolerating extra/missing keys from older checkpoints
    valid_keys = {f.name for f in dc_fields(Config)}
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in valid_keys})
    cfg.device = device  # always CPU for fast inference

    # Derive n_bins from latent_vec shape
    state = ckpt["model_state_dict"]
    n_bins = state["latent_vec"].shape[0]

    # Build components
    mlp = MLPModel(
        input_dim=input_dim,
        hidden_dims=cfg.mlp_hidden_dims,
        output_dim=cfg.embed_dim,
        dropout=cfg.dropout,
    )
    stub_graph = _StubNeighbourGraph(n_bins)
    ls = LatentSolver(cfg, stub_graph, cfg.embed_dim, cfg.gating_fn)

    model = Model(
        mlp=mlp,
        latent_solver=ls,
        n_bins=n_bins,
        device=torch.device(device),
        embed_dim=cfg.embed_dim,
        gating_fn=cfg.gating_fn,
        gating_alpha=cfg.gating_alpha,
        gating_kappa=cfg.gating_kappa,
        gating_epsilon=cfg.gating_epsilon,
        latent_init_std=0.0,
        interpolation_enabled=False,  # no graph ops needed
    )
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def compute_val_loss(
    model: Model,
    val_loader: DataLoader,
    criterion: Loss,
    device: str = "cpu",
) -> float:
    """Replicates Trainer.validate() logic for sample-mode cross-entropy."""
    model.eval()
    running_loss = 0.0
    n_samples = 0
    dev = torch.device(device)
    for batch in val_loader:
        inputs   = batch["input"].to(dev)
        targets  = batch["target"].to(dev)
        bin_idx  = batch["bin_idx"].to(dev)
        mask     = batch["mask"].to(dev)

        bsz, max_bins, n_feat = inputs.shape
        inputs_flat  = inputs.view(bsz * max_bins, n_feat)
        bin_idx_flat = bin_idx.view(bsz * max_bins)

        outputs_flat = model(inputs_flat, bin_idx_flat, interpolation_mask=None)
        outputs = outputs_flat.view(bsz, max_bins)
        outputs = outputs.masked_fill(mask == 0, float("-inf"))
        loss = criterion(outputs, targets, mask)

        running_loss += loss.item() * bsz
        n_samples += bsz

    return float(running_loss / max(1, n_samples))


def main(ckpt_dir: Path, out_dir: Path, data_dir: Optional[Path] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Identify failing-run checkpoints (large dataset ~8 MB files)
    # ------------------------------------------------------------------
    all_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    failing_ckpts = [p for p in all_ckpts if p.stat().st_size > LARGE_RUN_THRESHOLD]
    if not failing_ckpts:
        print("No failing-run checkpoints found. Exiting.")
        return

    print(f"Found {len(failing_ckpts)} failing-run checkpoints: "
          f"{failing_ckpts[0].name} → {failing_ckpts[-1].name}")

    # ------------------------------------------------------------------
    # Load a single checkpoint to get the config and determine input_dim
    # ------------------------------------------------------------------
    sample_ckpt = torch.load(failing_ckpts[0], map_location="cpu", weights_only=False)
    cfg_dict    = sample_ckpt["config"]
    valid_keys  = {f.name for f in dc_fields(Config)}
    cfg_ref     = Config(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

    # ------------------------------------------------------------------
    # Load data (fast — no neighbor graph)
    # set_seed(42) + load() reproduces the same train/val split as the run
    # ------------------------------------------------------------------
    print("\nLoading validation data (no neighbor graph, should be fast)...")
    set_seed(42)
    data, _, bin_index, sample_index, _ = load(cfg_ref, save_data=False)

    input_dim = data["val"]["X"].shape[1]
    print(f"  Val set: {len(data['val']['X'])} observations, {input_dim} features")

    val_dataset = MBDataset(data["val"], bin_index, sample_index, loss_mode="sample")
    val_loader  = DataLoader(
        val_dataset,
        batch_size=cfg_ref.batch_size_sample,
        shuffle=False,
        collate_fn=collate_samples,
        num_workers=0,
    )
    criterion = Loss("cross_entropy")

    # ------------------------------------------------------------------
    # For each checkpoint: compute full-model and MLP-only val loss
    # ------------------------------------------------------------------
    results: List[Dict] = []

    for ckpt_path in failing_ckpts:
        epoch_num = int(ckpt_path.stem.replace("epoch_", ""))
        print(f"\nEpoch {epoch_num}: loading {ckpt_path.name} ...", end=" ", flush=True)

        try:
            model, _ = build_inference_model(ckpt_path, input_dim, device="cpu")
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        # Full-model val loss
        full_loss = compute_val_loss(model, val_loader, criterion, device="cpu")

        # MLP-only val loss: zero out latent_vec temporarily
        saved_latent = model.latent_vec.data.clone()
        model.latent_vec.data.zero_()
        mlp_only_loss = compute_val_loss(model, val_loader, criterion, device="cpu")
        model.latent_vec.data.copy_(saved_latent)

        latent_help = mlp_only_loss - full_loss  # positive = latent helps
        print(f"full={full_loss:.4f}  mlp_only={mlp_only_loss:.4f}  help={latent_help:+.4f}")

        results.append({
            "epoch": epoch_num,
            "full_loss": full_loss,
            "mlp_only_loss": mlp_only_loss,
            "latent_help": latent_help,
        })

    if not results:
        print("No results collected.")
        return

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'Epoch':>6}  {'full_loss':>12}  {'mlp_only':>12}  {'latent_help':>14}")
    print("-"*70)
    for r in results:
        flag = "  ← latent HURTS" if r["latent_help"] < 0 else ""
        print(f"{r['epoch']:>6}  {r['full_loss']:>12.4f}  {r['mlp_only_loss']:>12.4f}  {r['latent_help']:>+14.4f}{flag}")
    print("="*70)

    # Find crossover epoch (where latent switches from helping to hurting)
    crossover = [r["epoch"] for r in results if r["latent_help"] < 0]
    if crossover:
        print(f"\n→ Latent starts HURTING validation at epoch {crossover[0]}")
    else:
        print("\n→ Latent always HELPS validation in the measured range.")

    # ------------------------------------------------------------------
    # Figure: full-model vs MLP-only loss + latent_help curve
    # ------------------------------------------------------------------
    epochs    = [r["epoch"] for r in results]
    full_vals = [r["full_loss"] for r in results]
    mlp_vals  = [r["mlp_only_loss"] for r in results]
    help_vals = [r["latent_help"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(epochs, full_vals, "r-o", ms=4, lw=1.5, label="Full model (MLP + latent)")
    ax1.plot(epochs, mlp_vals,  "b-o", ms=4, lw=1.5, label="MLP only (latent zeroed)")
    ax1.set_ylabel("Val cross-entropy loss")
    ax1.set_title("Ablation: Full Model vs MLP-Only Validation Loss\n"
                  "(If MLP-only is flat but full-model explodes → latent is the cause)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(epochs, help_vals, color=["green" if v >= 0 else "red" for v in help_vals], alpha=0.7)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Latent help = mlp_only − full\n(green = helps, red = hurts)")
    ax2.set_title("Latent Contribution to Validation Loss")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = out_dir / "02_latent_ablation_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {save_path}")
    print("Script 2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent ablation curve across epochs")
    parser.add_argument("--ckpt-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--out-dir",  type=Path, default=OUT_DIR)
    args = parser.parse_args()
    main(args.ckpt_dir, args.out_dir)
