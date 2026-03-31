#!/usr/bin/env python
"""
Visualization for latent present-only vs all-BINs comparison.

Usage:
    python latent_present_only_visualize.py \
        --results_path results/latent_present_only_results.pkl \
        --output_dir figures
"""
from __future__ import annotations

import argparse
import os
import pickle
import logging as log
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize

# Add analysis folder to path to import centralized visualization module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from visualize_results import create_all_visualizations as create_viz_centralized


# ============================================================================
# Style / Labels
# ============================================================================

VARIANT_COLORS = {
    "present_only": "#e74c3c",  # Red
    "all_bins":     "#3498db",  # Blue
}

VARIANT_LABELS = {
    "present_only": "Present BINs only",
    "all_bins":     "All BINs",
}


def get_label(key: str, results: Dict[str, Any]) -> str:
    return results[key].get("label", VARIANT_LABELS.get(key, key))


def get_color(key: str) -> str:
    return VARIANT_COLORS.get(key, "#333333")


def set_style():
    sns.set_theme(style="white", font_scale=1.1)
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = np.clip(y_pred[valid], 0, 1)

    mae   = np.mean(np.abs(y_true - y_pred))
    rmse  = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    zero_mask    = y_true == 0
    nonzero_mask = y_true > 0

    mae_zeros    = np.mean(np.abs(y_true[zero_mask]    - y_pred[zero_mask]))    if zero_mask.sum()    > 0 else 0.0
    mae_nonzeros = np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask])) if nonzero_mask.sum() > 0 else 0.0
    rmse_zeros   = np.sqrt(np.mean((y_true[zero_mask]    - y_pred[zero_mask])    ** 2)) if zero_mask.sum()    > 0 else 0.0
    rmse_nonzeros= np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask]) ** 2)) if nonzero_mask.sum() > 0 else 0.0

    rel_err = np.zeros_like(y_true)
    rel_err[nonzero_mask] = np.abs(y_pred[nonzero_mask] - y_true[nonzero_mask]) / y_true[nonzero_mask]
    are = np.mean(rel_err[nonzero_mask]) if nonzero_mask.sum() > 0 else 0.0

    eps = 1e-10
    p = y_true + eps; p /= p.sum()
    q = y_pred + eps; q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))

    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    if np.isnan(corr):
        corr = 0.0

    return {
        "MAE":                  mae,
        "RMSE":                 rmse,
        "R²":                   r2,
        "MAE (zeros)":          mae_zeros,
        "MAE (non-zeros)":      mae_nonzeros,
        "RMSE (zeros)":         rmse_zeros,
        "RMSE (non-zeros)":     rmse_nonzeros,
        "Abs. Relative Error":  are,
        "KL Divergence":        kl,
        "Correlation":          corr,
        "n_zeros":              int(zero_mask.sum()),
        "n_nonzeros":           int(nonzero_mask.sum()),
    }


# ============================================================================
# Plot helpers
# ============================================================================

def _density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    try:
        xy = np.vstack([x, y]) + np.random.default_rng(0).normal(0, 1e-8, (2, len(x)))
        return gaussian_kde(xy)(xy)
    except Exception:
        return np.ones(len(x))


# ============================================================================
# Plot: metrics bar chart
# ============================================================================

def plot_metrics_comparison(results: Dict[str, Any], output_dir: str):
    set_style()
    keys = list(results.keys())
    metrics_all = {k: compute_metrics(results[k]["targets"], results[k]["predictions"]) for k in keys}

    to_plot = ["MAE", "RMSE", "Abs. Relative Error", "KL Divergence",
               "MAE (zeros)", "MAE (non-zeros)", "Correlation"]

    fig, axes = plt.subplots(1, len(to_plot), figsize=(26, 4))
    for ax, metric in zip(axes, to_plot):
        vals   = [metrics_all[k][metric] for k in keys]
        labels = [get_label(k, results)  for k in keys]
        colors = [get_color(k)           for k in keys]
        errors = [v * 0.05 for v in vals]

        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5,
                      yerr=errors, capsize=4, error_kw={"elinewidth": 1.5})
        for bar, val, err in zip(bars, vals, errors):
            ax.text(bar.get_x() + bar.get_width() / 2, val + err + max(vals) * 0.02,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.3)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        sns.despine(ax=ax)

    plt.suptitle("Present-Only vs All-BINs Latent – Performance Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ metrics_comparison.png")


# ============================================================================
# Plot: scatter actual vs predicted
# ============================================================================

def plot_scatter(results: Dict[str, Any], output_dir: str,
                 suffix: str = "", zoom: float | None = None, loglog: bool = False):
    set_style()
    keys = list(results.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 6))
    if len(keys) == 1:
        axes = [axes]

    scatter_data = []
    all_densities: List[float] = []
    global_min = float("inf")
    global_max = float("-inf")

    for k in keys:
        preds   = results[k]["predictions"].flatten()
        targets = results[k]["targets"].flatten()
        valid   = np.isfinite(preds) & np.isfinite(targets)
        preds, targets = preds[valid], targets[valid]

        if zoom is not None:
            mask = targets < zoom
            preds, targets = preds[mask], targets[mask]

        if loglog:
            eps = 1e-4
            x_ = np.log10(targets + eps)
            y_ = np.log10(preds   + eps)
        else:
            x_, y_ = targets, preds

        d = _density(x_, y_)
        all_densities.extend(d)
        scatter_data.append((k, x_, y_, d))
        global_min = min(global_min, x_.min(), y_.min())
        global_max = max(global_max, x_.max(), y_.max())

    norm = Normalize(vmin=min(all_densities), vmax=max(all_densities))
    sc = None

    for ax, (k, x_, y_, d) in zip(axes, scatter_data):
        idx = d.argsort()
        sc = ax.scatter(x_[idx], y_[idx], c=d[idx], cmap="viridis", norm=norm,
                        s=8, alpha=0.6, edgecolors="none")
        ax.plot([global_min, global_max], [global_min, global_max],
                "r--", lw=1.5, alpha=0.7)
        corr = np.corrcoef(x_, y_)[0, 1]
        xlabel = ("Log10 Actual" if loglog else "Actual")
        ylabel = ("Log10 Predicted" if loglog else "Predicted")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{get_label(k, results)}\n(r = {corr:.3f})", fontsize=13, fontweight="bold")
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    if sc is not None:
        fig.colorbar(sc, cax=cbar_ax).set_label("Point Density", fontsize=10)

    fname = f"scatter{suffix}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  ✓ {fname}")


# ============================================================================
# Plot: MAE / RAE by abundance range
# ============================================================================

BINS = [
    ("zero",  None,   None,  "Zero"),
    (None,    0,      0.001, "0.01-0.1%"),
    (None,    0.001,  0.01,  "0.1-1%"),
    (None,    0.01,   0.1,   "1-10%"),
    (None,    0.1,    1.0,   ">10%"),
]


def _range_errors(results: Dict[str, Any], metric: str) -> Tuple[pd.DataFrame, List[str]]:
    keys = list(results.keys())
    rows = []
    for k in keys:
        y_pred = np.clip(results[k]["predictions"].flatten(), 0, 1)
        y_true = results[k]["targets"].flatten()
        valid  = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred, y_true = y_pred[valid], y_true[valid]

        for tag, lo, hi, label in BINS:
            if tag == "zero":
                mask = y_true == 0
            else:
                mask = (y_true > lo) & (y_true <= hi)
            if mask.sum() == 0:
                continue
            if metric == "MAE":
                val = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            else:  # RAE
                val = np.mean(np.abs(y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-10)) if tag != "zero" else 0.0
            rows.append({"key": k, "label": get_label(k, results), "Range": label,
                         "value": val, "Count": int(mask.sum())})

    df = pd.DataFrame(rows)
    range_order = ["Zero", "0.01-0.1%", "0.1-1%", "1-10%", ">10%"]
    return df, [r for r in range_order if r in df["Range"].values]


def plot_error_by_range(results: Dict[str, Any], output_dir: str, metric: str = "MAE"):
    set_style()
    df, range_order = _range_errors(results, metric)
    if df.empty:
        return

    keys = list(results.keys())
    first_key = keys[0]
    count_df = df[df["key"] == first_key].set_index("Range")["Count"]

    pivot = df.pivot(index="Range", columns="label", values="value").reindex(range_order)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [get_color(k) for k in keys]
    pivot.plot(kind="bar", ax=ax, color=colors, width=0.7, edgecolor="white")

    xlabels = [f"{r}\n(n={count_df[r]})" if r in count_df else r for r in range_order]
    ax.set_xticklabels(xlabels, rotation=0)
    ax.set_xlabel("Abundance Range", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"{metric} by Abundance Range", fontsize=14, fontweight="bold")
    ax.legend(title="Latent mode", frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()

    fname = f"{'mae' if metric == 'MAE' else 'rae'}_by_range.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  ✓ {fname}")


# ============================================================================
# Plot: zero vs non-zero MAE/RMSE
# ============================================================================

def plot_zero_vs_nonzero(results: Dict[str, Any], output_dir: str):
    set_style()
    keys = list(results.keys())
    metrics_all = {k: compute_metrics(results[k]["targets"], results[k]["predictions"]) for k in keys}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(keys))
    w = 0.35

    shades = {
        "present_only": {"zero": "#f1948a", "nonzero": "#e74c3c"},
        "all_bins":     {"zero": "#85c1e9", "nonzero": "#3498db"},
    }

    for col, (metric_z, metric_nz, ylabel, title) in enumerate([
        ("MAE (zeros)",  "MAE (non-zeros)",  "MAE",  "MAE: Zero vs Non-Zero"),
        ("RMSE (zeros)", "RMSE (non-zeros)", "RMSE", "RMSE: Zero vs Non-Zero"),
    ]):
        ax = axes[col]
        for i, k in enumerate(keys):
            s = shades.get(k, {"zero": "#aaa", "nonzero": "#555"})
            ax.bar(x[i] - w / 2, metrics_all[k][metric_z],  w, color=s["zero"],    edgecolor="white")
            ax.bar(x[i] + w / 2, metrics_all[k][metric_nz], w, color=s["nonzero"], edgecolor="white")

        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#ccc",   label="Zero (lighter)"),
                            Patch(facecolor="#555",   label="Non-zero (darker)")],
                  frameon=False)
        ax.set_xticks(x)
        ax.set_xticklabels([get_label(k, results) for k in keys], rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zero_vs_nonzero.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ zero_vs_nonzero.png")


# ============================================================================
# Plot: val loss curves
# ============================================================================

def plot_val_loss_curves(results: Dict[str, Any], output_dir: str):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    for k, res in results.items():
        timeline = res.get("timeline_val_losses", [])
        if not timeline:
            continue
        steps  = list(range(len(timeline)))
        losses = [t[-1] for t in timeline]
        ax.plot(steps, losses, label=get_label(k, results), color=get_color(k), lw=2)

    ax.set_xlabel("Training step (init + cycles × epochs)", fontsize=12)
    ax.set_ylabel("Validation loss", fontsize=12)
    ax.set_title("Validation Loss Over Training", fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ val_loss_curves.png")


# ============================================================================
# Plot: residual distribution
# ============================================================================

def plot_residuals(results: Dict[str, Any], output_dir: str):
    set_style()
    keys = list(results.keys())
    all_res: List[np.ndarray] = []

    for k in keys:
        t = results[k]["targets"].flatten()
        p = results[k]["predictions"].flatten()
        valid = np.isfinite(t) & np.isfinite(p)
        all_res.append(t[valid] - p[valid])

    g_min = min(r.min() for r in all_res)
    g_max = max(r.max() for r in all_res)

    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5))
    if len(keys) == 1:
        axes = [axes]

    for ax, k, res in zip(axes, keys, all_res):
        ax.hist(res, bins=60, color=get_color(k), alpha=0.75, edgecolor="white", density=True)
        ax.axvline(0, color="red", ls="--", lw=1.5)
        ax.set_xlabel("Residual (Actual − Predicted)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"{get_label(k, results)}\n(μ={res.mean():.4f}, σ={res.std():.4f})",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(g_min, g_max)
        sns.despine(ax=ax)

    plt.suptitle("Residual Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ residuals.png")


# ============================================================================
# Plot: summary table
# ============================================================================

def plot_summary_table(results: Dict[str, Any], output_dir: str):
    set_style()
    keys  = list(results.keys())
    mets  = {k: compute_metrics(results[k]["targets"], results[k]["predictions"]) for k in keys}

    display_metrics = ["MAE", "RMSE", "R²", "Abs. Relative Error", "KL Divergence",
                       "Correlation", "MAE (zeros)", "MAE (non-zeros)"]

    rows = [[f"{mets[k][m]:.5f}" for k in keys] for m in display_metrics]
    col_labels = [get_label(k, results) for k in keys]

    fig, ax = plt.subplots(figsize=(8, len(display_metrics) * 0.55 + 1.5))
    ax.axis("off")
    tbl = ax.table(
        cellText   = rows,
        rowLabels  = display_metrics,
        colLabels  = col_labels,
        cellLoc    = "center",
        loc        = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.4)

    # Highlight header
    for j, k in enumerate(keys):
        tbl[(0, j)].set_facecolor(get_color(k))
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight best value per row (lower is better except R² and Correlation)
    higher_is_better = {"R²", "Correlation"}
    for i, m in enumerate(display_metrics):
        vals = [float(mets[k][m]) for k in keys]
        best_idx = int(np.argmax(vals)) if m in higher_is_better else int(np.argmin(vals))
        tbl[(i + 1, best_idx)].set_facecolor("#d5f5e3")

    plt.title("Summary – Latent Present-Only vs All-BINs", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ summary_table.png")


# ============================================================================
# Main
# ============================================================================

def run_all(results: Dict[str, Any], output_dir: str):
    """Create all visualizations using centralized module plus analysis-specific plots."""
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Saving figures to {output_dir}")
    
    # Use centralized visualization function
    create_viz_centralized(
        results=results,
        output_dir=output_dir,
        labels=VARIANT_LABELS,
        colors=VARIANT_COLORS,
        title="Present-Only vs All-BINs Latent",
    )
    
    # Add analysis-specific plot for validation loss curves
    plot_val_loss_curves(results, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise latent present-only comparison")
    parser.add_argument("--results_path", type=str, default="results/latent_present_only_results.pkl",
                        help="Path to results pickle from latent_present_only_comparison.py")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory for output figures")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # Setup logging
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Handle paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = args.results_path
    if not os.path.isabs(results_path):
        results_path = os.path.join(script_dir, results_path)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)

    log.info(f"Loading results from {results_path}...")
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    # Create visualizations
    run_all(results, output_dir)

    log.info(f"All visualizations saved to {output_dir}")
