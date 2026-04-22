#!/usr/bin/env python3
"""
Script 1: inspect_checkpoints.py
---------------------------------
Passive analysis of already-saved checkpoint files from the failing run
(results/default_src/checkpoints/epoch_0005.pt through epoch_0055.pt).

Extracts stored latent_diagnostics and loss curves from each checkpoint
without re-loading data or rebuilding the model.

Key question: Does latent_max grow monotonically alongside the val loss explosion?
If yes → latent overfitting (H1) is confirmed.

Usage:
    python inspect_checkpoints.py [--ckpt-dir PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "results" / "default_src" / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "figures"


def load_checkpoint_metadata(path: Path) -> Optional[Dict]:
    """Load only the non-tensor metadata from a checkpoint (fast, avoids large tensors)."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt
    except Exception as e:
        print(f"  Warning: could not load {path.name}: {e}")
        return None


def extract_diagnostics(ckpt: Dict) -> Dict:
    """Pull the accumulated diagnostics lists from a checkpoint."""
    train_losses = ckpt.get("train_losses", [])  # [(epoch, loss), ...]
    val_losses   = ckpt.get("val_losses", [])    # [(epoch, loss), ...]
    latent_diag  = ckpt.get("latent_diagnostics", [])  # [{"epoch": .., "latent_max": ..}, ...]
    return {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "latent_diag":  latent_diag,
        "epoch":        ckpt.get("epoch", -1),
        "best_val":     ckpt.get("best_val_loss", float("inf")),
    }


def main(ckpt_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Locate epoch checkpoints from the failing run (big dataset files ~8 MB)
    # and from the previous stable run (small dataset files ~2.5 MB)
    # -----------------------------------------------------------------
    epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not epoch_ckpts:
        print(f"No epoch checkpoints found in {ckpt_dir}")
        return

    # Separate by file size to distinguish the two runs:
    #   failing run (data_merged, 485 samples)  → ~8 MB
    #   stable run (small dataset, 99 samples)  → ~2.5 MB
    LARGE_RUN_THRESHOLD = 5_000_000  # 5 MB

    failing_ckpts = [p for p in epoch_ckpts if p.stat().st_size > LARGE_RUN_THRESHOLD]
    stable_ckpts  = [p for p in epoch_ckpts if p.stat().st_size <= LARGE_RUN_THRESHOLD]

    print(f"Found {len(failing_ckpts)} checkpoints from failing run (large dataset)")
    print(f"Found {len(stable_ckpts)} checkpoints from stable run (small dataset)")

    # -----------------------------------------------------------------
    # Load the latest failing-run checkpoint (has full accumulated history)
    # -----------------------------------------------------------------
    if not failing_ckpts:
        print("No failing-run checkpoints found.")
        return

    latest_failing = failing_ckpts[-1]
    print(f"\nLoading diagnostics from: {latest_failing.name}")
    ckpt = load_checkpoint_metadata(latest_failing)
    if ckpt is None:
        return

    diags = extract_diagnostics(ckpt)
    print(f"  Checkpoint covers epochs 0 to {diags['epoch']} ({len(diags['latent_diag'])} entries in latent_diagnostics)")
    print(f"  Best val loss: {diags['best_val']:.6f}")

    # Convert to arrays for plotting
    train_epochs = [e for e, _ in diags["train_losses"]]
    train_loss   = [l for _, l in diags["train_losses"]]
    val_epochs   = [e for e, _ in diags["val_losses"]]
    val_loss     = [l for _, l in diags["val_losses"]]

    # Build per-epoch latent stats from diagnostics list
    # (each entry corresponds to one epoch in order)
    lat_epochs  = [d["epoch"] for d in diags["latent_diag"]]
    lat_mean    = [d.get("latent_mean", np.nan) for d in diags["latent_diag"]]
    lat_std     = [d.get("latent_std",  np.nan) for d in diags["latent_diag"]]
    lat_min     = [d.get("latent_min",  np.nan) for d in diags["latent_diag"]]
    lat_max     = [d.get("latent_max",  np.nan) for d in diags["latent_diag"]]
    lat_abl     = [d.get("ablation_delta", np.nan) for d in diags["latent_diag"]]

    # -----------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------
    print("\n" + "="*80)
    print(f"{'Epoch':>6}  {'train_loss':>12}  {'val_loss':>12}  {'lat_mean':>10}  {'lat_std':>10}  {'lat_max':>10}")
    print("-"*80)
    for i, (e, tl) in enumerate(diags["train_losses"]):
        vl  = diags["val_losses"][i][1] if i < len(diags["val_losses"]) else np.nan
        lm  = lat_mean[i]  if i < len(lat_mean) else np.nan
        ls  = lat_std[i]   if i < len(lat_std)  else np.nan
        lmx = lat_max[i]   if i < len(lat_max)  else np.nan
        # Flag epochs where val loss explodes (>2x best)
        best = diags["best_val"]
        flag = "  *** EXPLOSION" if vl > 2 * best else ""
        print(f"{e+1:>6}  {tl:>12.4f}  {vl:>12.4f}  {lm:>10.4f}  {ls:>10.4f}  {lmx:>10.4f}{flag}")
    print("="*80)

    # Also collect ablation deltas if any were computed
    abl_epochs = [lat_epochs[i] for i, v in enumerate(lat_abl) if v is not None]
    abl_vals   = [v for v in lat_abl if v is not None]
    if abl_vals:
        print("\nAblation deltas (MLP-only_loss - full_model_loss; positive = latent helps):")
        for ep, av in zip(abl_epochs, abl_vals):
            print(f"  Epoch {ep+1}: {av:+.4f}")

    # -----------------------------------------------------------------
    # Figure 1: Loss curves + latent evolution
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Panel 1: Loss curves
    ax = axes[0]
    ax.plot([e+1 for e in train_epochs], train_loss, "b-o", ms=3, lw=1.5, label="train loss")
    ax.plot([e+1 for e in val_epochs],   val_loss,   "r-o", ms=3, lw=1.5, label="val loss")
    best_epoch = val_epochs[int(np.argmin(val_loss))] + 1 if val_loss else 0
    best_vl = min(val_loss) if val_loss else 0
    ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"best val (epoch {best_epoch})")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training Stability Diagnosis — Loss Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Panel 2: Latent max and std
    ax = axes[1]
    ax.plot([e+1 for e in lat_epochs], lat_max, "r-o", ms=3, lw=1.5, label="|latent| max")
    ax.plot([e+1 for e in lat_epochs], lat_std, "m-o", ms=3, lw=1.5, label="latent std (mean over dims)")
    ax.axvline(best_epoch, color="gray", ls="--", lw=1)
    ax.set_ylabel("Latent magnitude")
    ax.set_title("Latent Vector Evolution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Latent mean (should stay near 0 if no drift)
    ax = axes[2]
    ax.plot([e+1 for e in lat_epochs], lat_mean, "g-o", ms=3, lw=1.5, label="latent mean")
    ax.plot([e+1 for e in lat_epochs], lat_min,  "b-o", ms=3, lw=1.5, label="|latent| min")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(best_epoch, color="gray", ls="--", lw=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Latent value")
    ax.set_title("Latent Mean and Min")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = out_dir / "01_loss_and_latent_evolution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {save_path}")

    # -----------------------------------------------------------------
    # Figure 2: Val loss vs latent_max scatter (to show correlation)
    # -----------------------------------------------------------------
    if lat_max and val_loss:
        n = min(len(val_loss), len(lat_max))
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter_epochs = list(range(n))
        sc = ax.scatter(lat_max[:n], val_loss[:n], c=scatter_epochs, cmap="plasma", s=30)
        plt.colorbar(sc, ax=ax, label="Epoch (darker = later)")
        ax.set_xlabel("latent_max (max absolute latent value)")
        ax.set_ylabel("Validation cross-entropy loss")
        ax.set_title("Val Loss vs Latent Max\n(H1: expect strong positive correlation)")
        ax.grid(True, alpha=0.3)
        save_path2 = out_dir / "01b_val_loss_vs_latent_max.png"
        fig.savefig(save_path2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved: {save_path2}")

        # Compute correlation
        corr = np.corrcoef(lat_max[:n], val_loss[:n])[0, 1]
        print(f"\nCorrelation(latent_max, val_loss) = {corr:.4f}")
        if corr > 0.8:
            print("  → STRONG positive correlation. H1 (latent overfitting) is very likely the cause.")
        elif corr > 0.5:
            print("  → Moderate correlation. Consistent with H1 but not conclusive.")
        else:
            print("  → Weak correlation. H1 is less likely; investigate secondary hypotheses.")

    # -----------------------------------------------------------------
    # Figure 3: If stable-run checkpoints exist, compare latent evolution
    # -----------------------------------------------------------------
    if stable_ckpts:
        latest_stable = stable_ckpts[-1]
        print(f"\nLoading stable-run diagnostics from: {latest_stable.name}")
        ckpt_s = load_checkpoint_metadata(latest_stable)
        if ckpt_s is not None:
            diags_s = extract_diagnostics(ckpt_s)
            lat_diag_s = diags_s["latent_diag"]
            if lat_diag_s:
                s_epochs = [d["epoch"] for d in lat_diag_s]
                s_max    = [d.get("latent_max", np.nan) for d in lat_diag_s]
                s_vloss  = [l for _, l in diags_s["val_losses"]]

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes[0].plot([e+1 for e in s_epochs], s_max, "b-o", ms=3, lw=1.5, label="stable run (small)")
                axes[0].plot([e+1 for e in lat_epochs], lat_max, "r-o", ms=3, lw=1.5, label="failing run (large)")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("latent_max")
                axes[0].set_title("Latent Max: Stable vs Failing Run")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                axes[1].plot([e+1 for e in range(len(s_vloss))],   s_vloss,  "b-o", ms=3, lw=1.5, label="stable run")
                axes[1].plot([e+1 for e in range(len(val_loss))],   val_loss, "r-o", ms=3, lw=1.5, label="failing run")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Val loss")
                axes[1].set_title("Val Loss: Stable vs Failing Run")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                save_path3 = out_dir / "01c_stable_vs_failing_comparison.png"
                fig.savefig(save_path3, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Figure saved: {save_path3}")

    print("\nScript 1 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect checkpoint diagnostics for latent evolution")
    parser.add_argument("--ckpt-dir", type=Path, default=CHECKPOINT_DIR, help="Path to checkpoint directory")
    parser.add_argument("--out-dir",  type=Path, default=OUT_DIR, help="Output directory for figures")
    args = parser.parse_args()
    main(args.ckpt_dir, args.out_dir)
