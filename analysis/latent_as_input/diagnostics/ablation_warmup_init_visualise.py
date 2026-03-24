"""
Visualisation for the warmup-vs-init ablation study.

Loads results from ablation_warmup_init.pkl and generates:
  - ablation_metrics.png      : side-by-side RMSE / Correlation / KL bar charts
  - ablation_training.png     : training + validation loss curves for all variants
  - ablation_latent_diag.png  : weight-norm ratio, embedding std, ablation delta per variant
  - ablation_summary.png      : summary table

Usage
-----
  python ablation_warmup_init_visualise.py \
      --results_path results/ablation_warmup_init.pkl \
      --output_dir figures/ablation
"""
from __future__ import annotations

import argparse
import os
import pickle
import logging as log
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.stats import pearsonr


# ── colour palette (one colour per variant key, consistent across plots) ──────
VARIANT_COLORS = {
    "A_warmup20_std0":    "#1f77b4",   # blue
    "B_no_warmup_std0":   "#d62728",   # red
    "C_no_warmup_std001": "#2ca02c",   # green
    "D_no_warmup_std005": "#ff7f0e",   # orange
}
DEFAULT_COLOR = "#888888"


def _color(key: str) -> str:
    return VARIANT_COLORS.get(key, DEFAULT_COLOR)


def _label(results: Dict[str, Any], key: str) -> str:
    return results[key].get("variant_label", key)


# ── metric helpers ─────────────────────────────────────────────────────────────

def _flatten_valid(arr: np.ndarray) -> np.ndarray:
    """Flatten a (n_samples, n_bins) array and drop NaN padding."""
    flat = arr.flatten()
    return flat[~np.isnan(flat)]


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    p = _flatten_valid(preds)
    t = _flatten_valid(targets)
    mask = ~np.isnan(p) & ~np.isnan(t)
    p, t = p[mask], t[mask]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae  = float(np.mean(np.abs(p - t)))
    corr = float(pearsonr(p, t)[0]) if len(p) > 1 else 0.0
    # KL divergence (per sample then averaged)
    eps = 1e-12
    # sum across bins per sample for KL
    n_samples = preds.shape[0] if preds.ndim == 2 else 1
    kl_vals = []
    arr_p, arr_t = (preds, targets) if preds.ndim == 2 else (preds[None], targets[None])
    for i in range(arr_p.shape[0]):
        pi = arr_p[i]
        ti = arr_t[i]
        mask_i = ~np.isnan(pi) & ~np.isnan(ti)
        if mask_i.sum() < 2:
            continue
        pi, ti = pi[mask_i], ti[mask_i]
        pi = np.clip(pi, eps, None);  pi /= (pi.sum() + eps)
        ti = np.clip(ti, eps, None);  ti /= (ti.sum() + eps)
        kl_vals.append(float(np.sum(ti * np.log(ti / pi))))
    kl = float(np.mean(kl_vals)) if kl_vals else float("nan")
    return {"RMSE": rmse, "MAE": mae, "Correlation": corr, "KL Divergence": kl}


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_metrics(results: Dict[str, Any], output_dir: str) -> None:
    keys = list(results.keys())
    metric_names = ["RMSE", "MAE", "Correlation", "KL Divergence"]
    higher_better = {"Correlation"}

    metrics: Dict[str, Dict[str, float]] = {}
    for k in keys:
        metrics[k] = compute_metrics(
            np.array(results[k]["predictions"]),
            np.array(results[k]["targets"]),
        )

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
    x = np.arange(len(keys))
    width = 0.6

    for ax, mname in zip(axes, metric_names):
        vals = [metrics[k][mname] for k in keys]
        colors = [_color(k) for k in keys]
        bars = ax.bar(x, vals, width=width, color=colors, alpha=0.82, edgecolor='white', linewidth=0.8)

        # Highlight best bar
        best_idx = int(np.argmax(vals)) if mname in higher_better else int(np.argmin(vals))
        bars[best_idx].set_edgecolor("#222222")
        bars[best_idx].set_linewidth(2.0)

        ax.set_xticks(x)
        ax.set_xticklabels([_label(results, k) for k in keys], rotation=18, ha='right', fontsize=8)
        ax.set_title(mname, fontweight='bold')
        ax.set_ylabel(mname)
        ax.grid(True, axis='y', alpha=0.3)

        # Annotate bar values
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f"{val:.5f}", ha='center', va='bottom', fontsize=7, rotation=0)

    fig.suptitle("Ablation: Warmup vs Init Strategy — Prediction Metrics", fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "ablation_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  ✓ Saved: ablation_metrics.png")


def plot_training_curves(results: Dict[str, Any], output_dir: str) -> None:
    keys = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_train, ax_val = axes

    for k in keys:
        color = _color(k)
        label = _label(results, k)
        r = results[k]

        # Warmup segment (negative x-axis) ──────────────────────────────────
        wu_losses = r.get("warmup_train_losses", [])
        n_wu = r.get("latent_warmup_epochs", 0)
        if wu_losses and n_wu > 0:
            wu_x = [-(n_wu - wu_e) for wu_e, _ in wu_losses]
            wu_y = [l for _, l in wu_losses]
            ax_train.plot(wu_x, wu_y, color=color, linestyle=':', linewidth=1.2, alpha=0.55)

        # Joint training segment ──────────────────────────────────────────────
        train_vals = [l for _, _, l in r.get("train_losses", [])]
        val_vals   = [l for _, _, l in r.get("val_losses",   [])]
        epochs     = list(range(len(train_vals)))

        ax_train.plot(epochs, train_vals, color=color, linewidth=1.8, label=label, alpha=0.9)
        ax_val.plot(  epochs, val_vals,   color=color, linewidth=1.8, label=label, alpha=0.9)

        # Mark warmup/joint boundary
        if wu_losses and n_wu > 0:
            ax_train.axvline(0, color=color, linestyle='--', linewidth=0.8, alpha=0.4)

    for ax, title in [(ax_train, "Training Loss"), (ax_val, "Validation Loss")]:
        ax.set_xlabel("Epoch (joint training)")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=8, frameon=True)
        ax.grid(True, alpha=0.3)

    # Shared warmup annotation on training ax
    any_warmup = any(results[k].get("latent_warmup_epochs", 0) > 0 for k in keys)
    if any_warmup:
        ax_train.text(0.02, 0.97, "← warmup (dotted) | joint (solid) →",
                      transform=ax_train.transAxes, fontsize=8, va='top', color='#555555',
                      bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.7))

    fig.suptitle("Ablation: Training Curves", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_training.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  ✓ Saved: ablation_training.png")


def plot_latent_diagnostics(results: Dict[str, Any], output_dir: str) -> None:
    """
    For each variant: weight-norm ratio, embedding std, ablation delta — all in
    one grid (variants as rows, 3 diagnostic panels as columns).
    """
    keys = list(results.keys())
    n = len(keys)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4 * n), squeeze=False)
    fig.suptitle("Ablation: Latent Diagnostics per Variant", fontsize=13, y=1.01)

    for row, k in enumerate(keys):
        color = _color(k)
        label = _label(results, k)
        diag  = results[k].get("latent_diagnostics", [])
        if not diag:
            for col in range(3):
                axes[row, col].set_visible(False)
            continue

        epochs   = [d["epoch"]            for d in diag]
        ratios   = [d["weight_norm_ratio"] for d in diag]
        emb_stds = [d["embedding_std"]     for d in diag]
        ab_epochs = [d["epoch"]          for d in diag if d.get("ablation_delta") is not None]
        ab_deltas = [d["ablation_delta"] for d in diag if d.get("ablation_delta") is not None]

        # Weight norm ratio ──────────────────────────────────────────────────
        ax = axes[row, 0]
        ax.plot(epochs, ratios, color=color, linewidth=1.8)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.fill_between(epochs, 0, ratios, alpha=0.10, color=color)
        ax.set_ylabel("latent/feat norm ratio")
        ax.set_title(f"{label}\nWeight norm ratio (>1 = MLP uses latent more)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        if epochs:
            ax.annotate(f"{ratios[0]:.2f}", xy=(epochs[0], ratios[0]), xytext=(6, 6),
                        textcoords='offset points', fontsize=8, color=color)
            ax.annotate(f"{ratios[-1]:.2f}", xy=(epochs[-1], ratios[-1]), xytext=(-24, 6),
                        textcoords='offset points', fontsize=8, color=color)

        # Embedding std ──────────────────────────────────────────────────────
        ax = axes[row, 1]
        ax.plot(epochs, emb_stds, color=color, linewidth=1.8)
        ax.fill_between(epochs, 0, emb_stds, alpha=0.10, color=color)
        ax.set_ylabel("std of embedding weights")
        ax.set_title(f"{label}\nEmbedding std (≈0 = latent inactive)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        if epochs:
            ax.annotate(f"{emb_stds[-1]:.4f}", xy=(epochs[-1], emb_stds[-1]),
                        xytext=(-32, 6), textcoords='offset points', fontsize=8, color=color)

        # Ablation delta ─────────────────────────────────────────────────────
        ax = axes[row, 2]
        if ab_epochs:
            bar_colors = ['#2ca02c' if d >= 0 else '#d62728' for d in ab_deltas]
            width = max(1, (ab_epochs[-1] - ab_epochs[0]) / max(len(ab_epochs), 1) * 0.8) if len(ab_epochs) > 1 else 5
            ax.bar(ab_epochs, ab_deltas, color=bar_colors, alpha=0.75, width=width)
            ax.axhline(0.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            ax.set_ylabel("Δ val loss (zeroed latent)")
            ax.set_title(f"{label}\nAblation Δ (>0 = latent helps)")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No ablation data", ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=10)
            ax.set_visible(False)

    for ax_row in axes:
        ax_row[0].set_xlabel("Epoch")
        ax_row[1].set_xlabel("Epoch")
        ax_row[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_latent_diag.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  ✓ Saved: ablation_latent_diag.png")


def plot_summary_table(results: Dict[str, Any], output_dir: str) -> pd.DataFrame:
    keys      = list(results.keys())
    rows      = []
    metric_names = ["RMSE", "MAE", "Correlation", "KL Divergence"]
    higher_better = {"Correlation"}

    for k in keys:
        m = compute_metrics(
            np.array(results[k]["predictions"]),
            np.array(results[k]["targets"]),
        )
        diag = results[k].get("latent_diagnostics", [])
        final_ratio = diag[-1]["weight_norm_ratio"] if diag else float("nan")
        final_delta = next(
            (d["ablation_delta"] for d in reversed(diag) if d.get("ablation_delta") is not None),
            float("nan"),
        )
        rows.append({
            "Variant": _label(results, k),
            "init_std": results[k].get("variant_init_std", "?"),
            "warmup": results[k].get("variant_warmup_epochs", "?"),
            **{name: m[name] for name in metric_names},
            "Final latent ratio": final_ratio,
            "Final ablation Δ": final_delta,
        })

    df = pd.DataFrame(rows)

    # ── figure ──────────────────────────────────────────────────────────────
    display_cols = ["Variant", "init_std", "warmup"] + metric_names + ["Final latent ratio", "Final ablation Δ"]
    display_df   = df[display_cols].copy()
    for col in metric_names + ["Final latent ratio", "Final ablation Δ"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.5f}" if not np.isnan(float(x)) else "—")

    fig, ax = plt.subplots(figsize=(18, 2 + 0.5 * len(keys)))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=list(display_df.columns),
        cellLoc="center",
        loc="center",
        colColours=["#e8e8e8"] * len(display_df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Header bold
    for c in range(len(display_df.columns)):
        table[(0, c)].set_text_props(fontweight="bold")

    # Colour first column with variant colour
    for r_idx, k in enumerate(keys):
        cell = table[(r_idx + 1, 0)]
        cell.set_facecolor(_color(k))
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight best value in each metric column
    for c_idx, col in enumerate(display_df.columns):
        if col not in metric_names:
            continue
        vals = [float(display_df.iloc[r, c_idx].replace("—", "nan")) for r in range(len(keys))]
        valid_vals = [v for v in vals if not np.isnan(v)]
        if not valid_vals:
            continue
        best = max(valid_vals) if col in higher_better else min(valid_vals)
        for r_idx, v in enumerate(vals):
            if not np.isnan(v) and abs(v - best) < 1e-12:
                table[(r_idx + 1, c_idx)].set_facecolor("#d5f5e3")
                table[(r_idx + 1, c_idx)].set_text_props(fontweight="bold", color="#1a7a40")

    plt.title("Ablation Summary: Warmup vs Init Strategy", fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: ablation_summary.png")
    df.to_csv(os.path.join(output_dir, "ablation_summary.csv"), index=False)
    log.info("  ✓ Saved: ablation_summary.csv")
    return df


def create_all(results: Dict[str, Any], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log.info("\n" + "=" * 60)
    log.info("ABLATION VISUALISATIONS")
    log.info("=" * 60)

    log.info("1. Metrics comparison...")
    plot_metrics(results, output_dir)

    log.info("2. Training curves...")
    plot_training_curves(results, output_dir)

    log.info("3. Latent diagnostics grid...")
    plot_latent_diagnostics(results, output_dir)

    log.info("4. Summary table...")
    df = plot_summary_table(results, output_dir)

    log.info(f"\n✅ Ablation figures saved to: {output_dir}/")
    log.info(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise warmup/init ablation results")
    parser.add_argument("--results_path", type=str,
                        default="results/ablation_warmup_init.pkl")
    parser.add_argument("--output_dir", type=str, default="figures/ablation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log.basicConfig(
        level=log.DEBUG if args.verbose else log.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    root = os.path.dirname(os.path.abspath(__file__))
    results_path = args.results_path if os.path.isabs(args.results_path) \
        else os.path.join(root, args.results_path)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) \
        else os.path.join(root, args.output_dir)

    log.info(f"Loading results from {results_path}...")
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    create_all(results, output_dir)
