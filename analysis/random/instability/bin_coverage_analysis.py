#!/usr/bin/env python3
"""
Script 3: bin_coverage_analysis.py
-------------------------------------
Investigates whether latent overfitting is driven by sparse bin coverage
and training/validation target discrepancy.

For each BIN, computes:
  - n_train: number of training samples it appears in
  - mean_y_train: mean relative abundance in training samples
  - n_val: number of validation samples it appears in
  - mean_y_val: mean relative abundance in validation samples
  - target_discrepancy = |mean_y_train - mean_y_val|
  - latent_norm = ||h_b||_2 (from latest checkpoint)

Then correlates latent_norm with target_discrepancy and n_train.

Key question: Are the bins with the largest latent norms the ones with
  - fewest training appearances (low n_train → high memorization risk), AND/OR
  - highest train/val target discrepancy (conflicting supervision)?

Usage:
    cd Metabarcoding/analysis/random/instability
    python bin_coverage_analysis.py [--ckpt PATH]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config, set_seed   # noqa: E402
from utils import load                # noqa: E402

CHECKPOINT_DIR = ROOT / "results" / "default_src" / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "figures"
LARGE_RUN_THRESHOLD = 5_000_000


def main(ckpt_path: Optional[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Select checkpoint: prefer explicit path, else latest large-run ckpt
    # ------------------------------------------------------------------
    if ckpt_path is None:
        all_ckpts = sorted(CHECKPOINT_DIR.glob("epoch_*.pt"))
        failing = [p for p in all_ckpts if p.stat().st_size > LARGE_RUN_THRESHOLD]
        if not failing:
            print("No failing-run checkpoints found.")
            return
        ckpt_path = failing[-1]  # most recent from failing run
    print(f"Using checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    valid_keys = {f.name for f in dc_fields(Config)}
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

    # Extract latent_vec
    latent_vec = ckpt["model_state_dict"]["latent_vec"].cpu()  # [n_bins, embed_dim] or [n_bins]
    embed_dim  = cfg.embed_dim
    if latent_vec.dim() == 1:
        latent_norms = latent_vec.abs()   # scalar mode
    else:
        latent_norms = latent_vec.norm(dim=-1)   # [n_bins]
    latent_norms_np = latent_norms.numpy()

    # ------------------------------------------------------------------
    # Load data with the same seed used during training
    # ------------------------------------------------------------------
    print("\nLoading data (reproducible split with seed=14)...")
    set_seed()
    data, taxonomy_df, bin_index, sample_index, split_indices = load(cfg, save_data=False)

    # Build reverse index: integer index → bin_uri
    idx_to_bin = {v: k for k, v in bin_index.items()}
    n_bins = len(bin_index)
    assert n_bins == latent_norms_np.shape[0], (
        f"Mismatch: {n_bins} bins in data vs {latent_norms_np.shape[0]} in checkpoint"
    )

    # ------------------------------------------------------------------
    # Compute per-bin statistics
    # ------------------------------------------------------------------
    print("Computing per-bin train/val statistics...")

    # Flatten to observation level for train and val
    def obs_stats(split_data: Dict) -> pd.DataFrame:
        X = split_data["X"]   # MultiIndex (sample_id, bin_uri)
        y = split_data["y"]   # aligned series
        df = X.reset_index()[["sample_id", "bin_uri"]]
        df["y"] = y.values
        return df

    train_obs = obs_stats(data["train"])
    val_obs   = obs_stats(data["val"])

    # Map bin_uri → integer index
    train_obs["bin_idx"] = train_obs["bin_uri"].map(bin_index)
    val_obs["bin_idx"]   = val_obs["bin_uri"].map(bin_index)

    # Per-bin aggregations for training set
    train_agg = train_obs.groupby("bin_idx")["y"].agg(
        n_train_appearances="count",
        mean_y_train="mean",
        std_y_train="std",
    ).reindex(range(n_bins), fill_value=0)
    train_agg["std_y_train"] = train_agg["std_y_train"].fillna(0.0)

    # Per-bin aggregations for validation set
    val_agg = val_obs.groupby("bin_idx")["y"].agg(
        n_val_appearances="count",
        mean_y_val="mean",
    ).reindex(range(n_bins), fill_value=0)

    # Combine into a single dataframe
    df_bins = pd.DataFrame({
        "bin_idx":            range(n_bins),
        "bin_uri":            [idx_to_bin.get(i, f"unknown_{i}") for i in range(n_bins)],
        "n_train":            train_agg["n_train_appearances"].values,
        "mean_y_train":       train_agg["mean_y_train"].values,
        "std_y_train":        train_agg["std_y_train"].values,
        "n_val":              val_agg["n_val_appearances"].values,
        "mean_y_val":         val_agg["mean_y_val"].values,
        "latent_norm":        latent_norms_np,
    })
    df_bins["target_discrepancy"] = (df_bins["mean_y_train"] - df_bins["mean_y_val"]).abs()

    # Only include bins that appear in at least one split for the analysis
    df_obs = df_bins[(df_bins["n_train"] > 0) | (df_bins["n_val"] > 0)].copy()
    df_shared = df_bins[(df_bins["n_train"] > 0) & (df_bins["n_val"] > 0)].copy()

    print(f"\nTotal bins: {n_bins}")
    print(f"Bins in training set:                  {(df_bins['n_train'] > 0).sum()}")
    print(f"Bins in validation set:                {(df_bins['n_val'] > 0).sum()}")
    print(f"Bins in BOTH (shared):                 {len(df_shared)}")
    print(f"Bins only in training:                 {((df_bins['n_train']>0) & (df_bins['n_val']==0)).sum()}")
    print(f"Bins only in validation:               {((df_bins['n_train']==0) & (df_bins['n_val']>0)).sum()}")
    print(f"Bins in neither (zero latent, unused): {((df_bins['n_train']==0) & (df_bins['n_val']==0)).sum()}")

    # Coverage statistics
    train_cov = df_bins.loc[df_bins["n_train"] > 0, "n_train"]
    print(f"\nTraining appearances per bin (bins that appear at least once):")
    print(f"  mean={train_cov.mean():.2f}, median={train_cov.median():.2f}, "
          f"min={train_cov.min()}, max={train_cov.max()}")
    print(f"  Bins with n_train == 1: {(train_cov == 1).sum()} "
          f"({100*(train_cov==1).sum()/len(train_cov):.1f}%)")
    print(f"  Bins with n_train <= 2: {(train_cov <= 2).sum()} "
          f"({100*(train_cov<=2).sum()/len(train_cov):.1f}%)")
    print(f"  Bins with n_train <= 5: {(train_cov <= 5).sum()} "
          f"({100*(train_cov<=5).sum()/len(train_cov):.1f}%)")

    # Correlation analysis
    if len(df_shared) > 10:
        corr_norm_disc = df_shared["latent_norm"].corr(df_shared["target_discrepancy"])
        corr_norm_ntrain = df_shared["latent_norm"].corr(df_shared["n_train"])
        print(f"\nCorrelation(latent_norm, target_discrepancy) = {corr_norm_disc:.4f}")
        print(f"Correlation(latent_norm, n_train) = {corr_norm_ntrain:.4f}")
        if corr_norm_ntrain < -0.2:
            print("  → Latent norms are LARGER for bins with fewer training appearances. "
                  "Consistent with memorization of rare training events.")
        if corr_norm_disc > 0.2:
            print("  → Latent norms are LARGER for bins with higher train/val discrepancy. "
                  "These bins drive the validation loss explosion.")

    # Top bins by latent norm
    print(f"\nTop 20 bins by latent_norm (checkpoint at epoch {ckpt.get('epoch', '?')+1}):")
    top20 = df_bins.nlargest(20, "latent_norm")[
        ["bin_uri", "latent_norm", "n_train", "mean_y_train", "n_val", "mean_y_val", "target_discrepancy"]
    ]
    print(top20.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    # Fig 1: Distribution of training appearances per bin
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    bins_with_train = df_bins.loc[df_bins["n_train"] > 0, "n_train"]
    ax.hist(bins_with_train, bins=range(1, min(int(bins_with_train.max())+2, 50)), alpha=0.7, color="steelblue")
    ax.set_xlabel("Number of training samples bin appears in")
    ax.set_ylabel("Number of bins")
    ax.set_title(f"Training Coverage Per Bin\n"
                 f"(Total bins: {n_bins}, bins seen: {len(bins_with_train)}, "
                 f"median appearances: {bins_with_train.median():.1f})")
    ax.axvline(bins_with_train.median(), color="red", ls="--", label=f"median={bins_with_train.median():.1f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fig 1b: Latent norm distribution
    ax = axes[1]
    ax.hist(df_bins["latent_norm"], bins=50, alpha=0.7, color="coral")
    ax.set_xlabel("||h_b||_2 (latent norm per bin)")
    ax.set_ylabel("Number of bins")
    ax.set_title(f"Distribution of Latent Norms at Epoch {ckpt.get('epoch','?')+1}\n"
                 f"max={df_bins['latent_norm'].max():.3f}, "
                 f"mean={df_bins['latent_norm'].mean():.3f}, "
                 f"std={df_bins['latent_norm'].std():.3f}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "03a_bin_coverage_and_latent_norms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: Latent norm vs n_train (scatter)
    if len(df_shared) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        sc = ax.scatter(df_shared["n_train"], df_shared["latent_norm"],
                        c=df_shared["target_discrepancy"], cmap="hot_r",
                        alpha=0.4, s=8, norm=matplotlib.colors.LogNorm(
                            vmin=max(1e-8, df_shared["target_discrepancy"].quantile(0.01)),
                            vmax=df_shared["target_discrepancy"].quantile(0.99) + 1e-8
                        ))
        plt.colorbar(sc, ax=ax, label="target discrepancy |train-val|")
        ax.set_xlabel("n_train (appearances in training set)")
        ax.set_ylabel("latent_norm ||h_b||_2")
        ax.set_title("Latent Norm vs Training Coverage\n(colored by train/val target discrepancy)")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        # Bin by n_train buckets to see average latent norm
        df_shared_copy = df_shared.copy()
        df_shared_copy["n_train_bucket"] = pd.cut(
            df_shared_copy["n_train"],
            bins=[0, 1, 2, 3, 5, 10, 20, df_shared_copy["n_train"].max()+1],
            labels=["1", "2", "3", "4-5", "6-10", "11-20", "21+"],
            right=True
        )
        avg_norm = df_shared_copy.groupby("n_train_bucket")["latent_norm"].mean()
        avg_norm.plot(kind="bar", ax=ax, color="coral", alpha=0.8)
        ax.set_xlabel("Training appearances bucket")
        ax.set_ylabel("Average latent norm")
        ax.set_title("Average Latent Norm by Training Coverage\n"
                     "(H1 predicts: fewer appearances → larger norm)")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        fig.savefig(out_dir / "03b_latent_norm_vs_coverage.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Fig 3: Target discrepancy histogram for shared bins
    if len(df_shared) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_shared["target_discrepancy"], bins=50, alpha=0.7, color="purple")
        ax.set_xlabel("Target discrepancy |mean_y_train - mean_y_val|")
        ax.set_ylabel("Number of shared bins")
        ax.set_title(f"Train/Val Target Discrepancy for Shared Bins (n={len(df_shared)})\n"
                     f"mean={df_shared['target_discrepancy'].mean():.4f}, "
                     f"max={df_shared['target_discrepancy'].max():.4f}")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / "03c_target_discrepancy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nFigures saved to: {out_dir}")
    print("Script 3 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bin coverage and latent norm analysis")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="Path to checkpoint (default: latest large-run epoch checkpoint)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    main(args.ckpt, args.out_dir)
