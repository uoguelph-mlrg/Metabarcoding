#!/usr/bin/env python
"""
Unified visualization module for model comparison results.

Creates clean, presentation-ready plots that automatically adapt to any number
of models/configurations being compared (2 models, 8 architectures, etc.).

Usage
-----
    python visualize_results.py \\
        --results_path path/to/results.pkl \\
        --output_dir figures \\
        --colors '{"baseline": "#95a5a6", "exp": "#e74c3c"}' \\
        --labels '{"baseline": "Baseline", "exp": "Exponential"}' \\
        --title "My Experiment Comparison"

The results pickle must be a dict with one key per model/variant:
    {
        "model_a": {"predictions": np.ndarray, "targets": np.ndarray, ...},
        "model_b": {"predictions": np.ndarray, "targets": np.ndarray, ...},
        ...
    }

Required per-model keys:
    - "predictions":    float32 array (N,)  — predicted values (flat)
    - "targets":        float32 array (N,)  — true relative abundances (flat)
    - "sample_labels":  str array (N,)      — sample ID for each entry
    - "bin_labels":     str array (N,)      — BIN URI for each entry
    where N = total valid (sample, BIN) pairs in the test split.

    Backward compatible: old result files with 2-D NaN-padded arrays are
    still accepted (sample_labels / bin_labels keys will be absent; macro
    metrics fall back to the 2-D row-wise computation).

Optional per-model keys (used when present):
    - "train_losses", "val_losses": list of (cycle, epoch, loss) tuples
    - "timeline_train_losses", "timeline_val_losses": list of (phase, cycle, step, loss)
    - "cycle_train_losses", "cycle_val_losses": list of (cycle, loss) tuples
    - "latent_diagnostics": list of dicts with 'epoch', 'weight_norm_ratio',
                            'embedding_std', 'ablation_delta' keys
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import logging as log
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

# ============================================================================
# Default color palette (used as fallback for unknown keys)
# ============================================================================

def _default_colors(labels: Dict[str, str]) -> Dict[str, str]:
    """Generate a default color mapping for *n* models, using *labels* if available."""
    palette = np.concatenate([["#95a5a6"], sns.color_palette("tab10").as_hex(), np.column_stack([
        sns.color_palette("pastel").as_hex(), sns.color_palette("dark").as_hex()
    ]).flatten()])
    keys = list(labels.keys())
    return {k: palette[i % len(palette)] for i, k in enumerate(keys)}

# ============================================================================
# Helpers
# ============================================================================

def get_color(key: str, colors: Optional[Dict[str, str]]) -> str:
    """Return a hex color for *key*, falling back to the default palette."""
    if colors and key in colors:
        return colors[key]


def get_label(key: str, labels: Optional[Dict[str, str]] = None) -> str:
    """Return a human-readable label for *key*."""
    if labels and key in labels:
        return labels[key]
    return key.replace("_", " ").title()


def _contrasting_text_color(hex_color: str) -> str:
    """Return 'white' or 'black' for a hex background color."""
    r, g, b = mc.to_rgb(hex_color)
    return "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.85 else "black"


def _ci_tuple_to_errorbar(mean_val: float, ci_tuple) -> List[float]:
    """Convert CI tuple (lower, upper) to error-bar format [lower_err, upper_err]."""
    if isinstance(ci_tuple, tuple) and len(ci_tuple) == 2:
        ci_lower, ci_upper = ci_tuple
        return [mean_val - ci_lower, ci_upper - mean_val]
    return [0.0, 0.0]


def _scatter_grid(n: int) -> Tuple[int, int]:
    """Return (n_rows, n_cols) for a scatter subplot grid.

    Rule: n_rows = ceil(n / 4)
    """
    n_rows = math.ceil(n / 4)
    n_cols = n - 4 * (n_rows - 1) if n_rows > 1 else n
    return n_rows, n_cols


def _colorbar_axes(n_rows: int) -> tuple[float, float, float, float]:
    """Return [left, bottom, width, height] for a right-side colorbar."""
    if n_rows == 1:
        return (0.94, 0.15, 0.02, 0.70)
    return (0.95, 0.10, 0.015, 0.80)


# ============================================================================
# Bootstrap CI
# ============================================================================

def compute_95ci_bootstrap(errors: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float]:
    """Compute 95 % CI of the mean via bootstrap resampling.

    Returns
    -------
    (ci_lower, ci_upper): 2.5 and 97.5 percentiles of the bootstrap distribution.
    """
    n = len(errors)
    if n < 2:
        return (0.0, 0.0)
    bootstrap_means = [
        np.mean(np.random.choice(errors, size=n, replace=True))
        for _ in range(n_bootstrap)
    ]
    return (float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)))


# ============================================================================
# Style
# ============================================================================

def set_style() -> None:
    """Apply a clean, minimal Seaborn/Matplotlib style."""
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

def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_labels: Optional[np.ndarray] = None,
    bin_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive prediction metrics.

    Accepts flat 1-D arrays (the canonical format produced by every
    ``get_predictions`` implementation) plus optional label arrays that enable
    rigorous grouped metrics.  For backward compatibility with older result
    pickles that still carry 2-D NaN-padded arrays, the 2-D path is preserved.

    Args:
        y_true:         Ground-truth relative abundances — shape (N,) or
                        (n_samples, max_bins) for old-style pickles.
        y_pred:         Predicted values — same shape as y_true.
        sample_labels:  (N,) string array identifying which sample each entry
                        belongs to.  When provided, per-sample macro metrics
                        (RMSE_macro, MAE_macro, KL Divergence) are computed
                        via groupby then averaged — the only rigorous way to
                        measure distributional error.
        bin_labels:     (N,) string array of BIN URIs.  Not used in scalar
                        summary metrics but passed through for callers that
                        want per-BIN breakdown via ``groupby(bin_labels)``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = np.clip(y_pred[valid], 0, 1)

    rmse_macro: Optional[float] = np.nan
    mae_macro: Optional[float] = np.nan
    kl_divergence: Optional[float] = np.nan
    eps = 1e-10

    # ------------------------------------------------------------------
    # Per sample metrics (macro-averaged)
    # ------------------------------------------------------------------
    if sample_labels is not None:
        sample_labels = np.asarray(sample_labels)
        sample_labels_v = sample_labels[valid]

        rmse_per, mae_per, kl_per = [], [], []
        for s in np.unique(sample_labels_v):
            mask = sample_labels_v == s
            true_s = y_true[mask]
            pred_s = y_pred[mask]
            if len(true_s) == 0:
                continue
            rmse_per.append(float(np.sqrt(np.mean((true_s - pred_s) ** 2))))
            mae_per.append(float(np.mean(np.abs(true_s - pred_s))))
            # KL per sample: each sample's values form a probability distribution
            true_s_norm = (true_s + eps) / (true_s + eps).sum()
            pred_s_norm = (pred_s + eps) / (pred_s + eps).sum()
            kl_per.append(float(np.sum(true_s_norm * np.log(true_s_norm / pred_s_norm))))

        if rmse_per:
            rmse_macro = float(np.mean(rmse_per))
            mae_macro = float(np.mean(mae_per))
            kl_divergence = float(np.mean(kl_per))
    
    elif y_true.ndim == 2:
        rmse_per, mae_per, kl_per = [], [], []
        for i in range(y_true.shape[0]):
            row_true = y_true[i]
            row_pred = y_pred[i]
            valid_i = np.isfinite(row_true) & np.isfinite(row_pred)
            if valid_i.sum() == 0:
                continue
            row_true = row_true[valid_i]
            row_pred = np.clip(row_pred[valid_i], 0, 1)
            rmse_per.append(float(np.sqrt(np.mean((row_true - row_pred) ** 2))))
            mae_per.append(float(np.mean(np.abs(row_true - row_pred))))
            row_true_norm = (row_true + eps) / (row_true + eps).sum()
            row_pred_norm = (row_pred + eps) / (row_pred + eps).sum()
            kl_per.append(float(np.sum(row_true_norm * np.log(row_true_norm / row_pred_norm))))
        if rmse_per:
            rmse_macro = float(np.mean(rmse_per))
            mae_macro = float(np.mean(mae_per))
            kl_divergence = float(np.mean(kl_per))
    
    # ------------------------------------------------------------------
    # Overall micro-averaged metrics (treating all entries as a single vector)
    # ------------------------------------------------------------------

    mse = np.mean((y_true - y_pred) ** 2)
    rmse_micro = float(np.sqrt(mse))
    mae_micro = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    
    # Log-transform metrics to better capture performance on low-abundance bins
    y_true_log = np.log(y_true + 1)
    y_pred_log = np.log(y_pred + 1)
    r2_log = float(1 - np.sum((y_true_log - y_pred_log) ** 2) / (np.sum((y_true_log - np.mean(y_true_log)) ** 2) + 1e-10))

    # Compute metrics for zero and non-zero abundance bins
    zero_mask = y_true == 0
    nonzero_mask = y_true > 0

    if zero_mask.sum() > 0:
        rmse_zeros = float(np.sqrt(np.mean((y_true[zero_mask] - y_pred[zero_mask]) ** 2)))
        mae_zeros = float(np.mean(np.abs(y_true[zero_mask] - y_pred[zero_mask])))
    else:
        rmse_zeros = mae_zeros = np.nan

    if nonzero_mask.sum() > 0:
        rmse_nonzeros = float(np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask]) ** 2)))
        mae_nonzeros = float(np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask])))
    else:
        rmse_nonzeros = mae_nonzeros = np.nan
        
    # Pearson correlation between true and predicted values
    corr = np.corrcoef(y_true, y_pred)[0, 1]
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
        "n_zeros": int(zero_mask.sum()),
        "n_nonzeros": int(nonzero_mask.sum()),
    }


# ============================================================================
# Plot 1 – Metric bar chart
# ============================================================================

def plot_metrics_comparison(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    title: str = "Performance Comparison",
) -> None:
    """Bar plots of key metrics, one bar per model, with 95 % bootstrap CIs."""
    set_style()
    models = list(results.keys())

    ext = {
        model: compute_extended_metrics(
            results[model]["targets"], 
            results[model]["predictions"],
            sample_labels=results[model].get("sample_labels"), 
            bin_labels=results[model].get("bin_labels"),
        ) for model in models
    }

    metrics_to_plot = [
        "MAE (macro)", "MAE (micro)", "Absolute Relative Error", "KL Divergence", 
        "MAE (zeros)", "MAE (non-zeros)", "R²", "R² (log + 1)"
    ]
    n_metrics = len(metrics_to_plot)

    # Compute bootstrap CIs
    cis: Dict[str, Dict[str, Any]] = {}
    for model in models:
        y_true = results[model]["targets"]
        y_pred = results[model]["predictions"]
        sample_labels = results[model].get("sample_labels")
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[valid], np.clip(y_pred[valid], 0, 1)
        if len(y_true) < 2:
            cis[model] = {m: None for m in metrics_to_plot}
            continue
        
        # Get macro metrics for CI computation
        mae_per_s, kl_div_per_s = [], []
        for s in np.unique(sample_labels[valid]) if sample_labels is not None else [0]:
            mask = sample_labels[valid] == s if sample_labels is not None else np.arange(len(y_true))
            true_s = y_true[mask]
            pred_s = y_pred[mask]
            if len(true_s) == 0:
                continue
            mae_per_s.append(float(np.mean(np.abs(true_s - pred_s))))
            kl_div_per_s.append(float(np.sum((true_s + 1e-10) * np.log((true_s + 1e-10) / (pred_s + 1e-10)))))
        mae_per_s, kl_div_per_s = np.array(mae_per_s), np.array(kl_div_per_s)
        
        # Get masks for zero vs non-zero true values (used for subgroup CI computation)
        nz = y_true != 0
        zero_m = y_true == 0
        nonzero_m = y_true > 0
        
        # Get micro metrics for CI computation
        abs_err = np.abs(y_true - y_pred)
        
        cis[model] = {
            'MAE (macro)': compute_95ci_bootstrap(mae_per_s),
            "MAE (micro)": compute_95ci_bootstrap(abs_err),
            "Absolute Relative Error": compute_95ci_bootstrap(abs_err[nz] / np.abs(y_true[nz])),
            "KL Divergence": compute_95ci_bootstrap(kl_div_per_s),
            "MAE (zeros)": compute_95ci_bootstrap(abs_err[zero_m]),
            "MAE (non-zeros)": compute_95ci_bootstrap(abs_err[nonzero_m]),
            "R²": None,
            "R² (log + 1)": None,
        }
        
    n_cols_max = np.clip(np.floor(50 / (len(models) + 2)), 1, None)  # avoid division by zero
    n_rows = np.ceil(n_metrics / n_cols_max).astype(int)
    n_cols = np.ceil(n_metrics / n_rows).astype(int)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten() if n_metrics > 1 else [axes]

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [ext[model][metric] for model in models]
        ci_info = [cis[model].get(metric) for model in models]
        bar_colors = [get_color(model, colors) for model in models]
        x_pos = np.arange(len(models))

        if all(c is None for c in ci_info):
            yerr = None
        else:
            ci_lower = [c[0] if isinstance(c, tuple) else 0.0 for c in ci_info]
            ci_upper = [c[1] if isinstance(c, tuple) else 0.0 for c in ci_info]
            yerr = [
                np.abs(np.array(values) - np.array(ci_lower)),
                np.abs(np.array(ci_upper) - np.array(values)),
            ]

        ax.bar(
            x_pos, values, color=bar_colors, edgecolor="white", linewidth=1.5,
            yerr=yerr, capsize=4,
            error_kw={"elinewidth": 1.5} if yerr is not None else {},
        )
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [get_label(model, labels) for model in models], rotation=45, ha="right", fontsize=9
        )
        sns.despine(ax=ax)

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: metrics_comparison.png")


# ============================================================================
# Plot 2 – Scatter: predicted vs actual (full range)
# ============================================================================

def plot_scatter_actual_vs_predicted(
    results: Dict[str, Any],
    output_dir: str,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Scatter plots (actual vs predicted) with density colouring, one panel per model."""
    set_style()
    models = list(results.keys())
    n = len(models)
    n_rows, n_cols = _scatter_grid(n)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    all_densities: List[float] = []
    scatter_data = []
    global_max_target = global_max_pred = 0.0

    for model in models:
        preds = results[model]["predictions"].flatten()
        targets = results[model]["targets"].flatten()
        vm = np.isfinite(preds) & np.isfinite(targets)
        preds, targets = preds[vm], targets[vm]
        global_max_target = max(global_max_target, float(targets.max()))
        global_max_pred = max(global_max_pred, float(preds.max()))
        try:
            xy = np.vstack([preds, targets]) + np.random.normal(0, 1e-8, (2, len(preds)))
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {model}: {e}")
            density = np.ones(len(preds))
        all_densities.extend(density)
        scatter_data.append((model, preds, targets, density))

    norm = Normalize(vmin=min(all_densities), vmax=max(all_densities))
    sc = None
    axis_max = max(global_max_target, global_max_pred)

    for idx, (model, preds, targets, density) in enumerate(scatter_data):
        ax = axes[idx]
        order = density.argsort()
        sc = ax.scatter(
            targets[order], preds[order], c=density[order],
            cmap="viridis", norm=norm, s=8, alpha=0.6, edgecolors="none",
        )
        ax.plot([0, axis_max], [0, axis_max], "r--", lw=1.5, alpha=0.7)
        corr = float(np.corrcoef(targets, preds)[0, 1])
        ax.set_xlabel("Actual", fontsize=11)
        ax.set_ylabel("Predicted", fontsize=11)
        ax.set_title(f"{get_label(model, labels)}\n(Pearson r = {corr:.3f})", fontsize=12, fontweight="bold")
        ax.set_xlim(0, global_max_target)
        ax.set_ylim(0, global_max_pred)
        sns.despine(ax=ax)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    right_margin = 0.92 if n_rows == 1 else 0.93
    fig.subplots_adjust(right=right_margin)
    if sc is not None:
        cbar_ax = fig.add_axes(_colorbar_axes(n_rows))
        fig.colorbar(sc, cax=cbar_ax).set_label("Point Density", fontsize=10)
    plt.savefig(os.path.join(output_dir, "scatter_predicted_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: scatter_predicted_vs_actual.png")


# ============================================================================
# Plot 3 – Scatter: zoomed on ground truth < 1 %
# ============================================================================

def plot_scatter_zoomed(
    results: Dict[str, Any],
    output_dir: str,
    labels: Optional[Dict[str, str]] = None,
    max_actual: float = 0.01,
) -> None:
    """Scatter plots zoomed on ground-truth values below *max_actual*."""
    set_style()
    models = list(results.keys())
    n = len(models)
    n_rows, n_cols = _scatter_grid(n)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    all_densities: List[float] = []
    scatter_data = []
    global_max_target = global_max_pred = 0.0

    for model in models:
        targets = results[model]["targets"].flatten()
        preds = results[model]["predictions"].flatten()
        vm = np.isfinite(targets) & np.isfinite(preds)
        targets, preds = targets[vm], preds[vm]
        zm = targets < max_actual
        tgt_z, prd_z = targets[zm], preds[zm]
        if len(tgt_z) > 0:
            global_max_target = max(global_max_target, float(tgt_z.max()))
            global_max_pred = max(global_max_pred, float(prd_z.max()))
        try:
            xy = np.vstack([prd_z, tgt_z]) + np.random.normal(0, 1e-8, (2, len(prd_z)))
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {model}: {e}")
            density = np.ones(len(prd_z)) if len(prd_z) > 0 else np.array([])
        all_densities.extend(density)
        scatter_data.append((model, prd_z, tgt_z, density))

    vmin = min(all_densities) if all_densities else 0
    vmax = max(all_densities) if all_densities else 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    axis_max = max(global_max_target, global_max_pred)
    sc = None

    for idx, (model, prd_z, tgt_z, density) in enumerate(scatter_data):
        ax = axes[idx]
        if len(prd_z) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(get_label(model, labels), fontsize=12, fontweight="bold")
            sns.despine(ax=ax)
            continue
        order = density.argsort()
        sc = ax.scatter(
            tgt_z[order], prd_z[order], c=density[order],
            cmap="viridis", norm=norm, s=8, alpha=0.6, edgecolors="none",
        )
        ax.plot([0, axis_max], [0, axis_max], "r--", lw=1.5, alpha=0.7)
        corr = float(np.corrcoef(tgt_z, prd_z)[0, 1]) if len(tgt_z) > 1 else 0.0
        ax.set_xlabel("Actual", fontsize=11)
        ax.set_ylabel("Predicted", fontsize=11)
        ax.set_title(f"{get_label(model, labels)}\n(Pearson r = {corr:.3f})", fontsize=12, fontweight="bold")
        ax.set_xlim(0, global_max_target)
        ax.set_ylim(0, global_max_pred)
        sns.despine(ax=ax)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Predicted vs Actual (Ground Truth <1%)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    right_margin = 0.92 if n_rows == 1 else 0.93
    fig.subplots_adjust(right=right_margin)
    if sc is not None:
        cbar_ax = fig.add_axes(_colorbar_axes(n_rows))
        fig.colorbar(sc, cax=cbar_ax).set_label("Point Density", fontsize=10)
    plt.savefig(os.path.join(output_dir, "scatter_zoomed.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: scatter_zoomed.png")


# ============================================================================
# Plot 4 – Log-log scatter
# ============================================================================

def plot_loglog_scatter_actual_vs_predicted(
    results: Dict[str, Any],
    output_dir: str,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Log-log scatter plots of actual vs predicted."""
    set_style()
    keys = list(results.keys())
    n = len(keys)
    n_rows, n_cols = _scatter_grid(n)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes = np.array(axes).flatten()

    eps = 10 ** -3.5
    all_densities: List[float] = []
    scatter_data = []
    global_min, global_max_val = float("inf"), float("-inf")

    for model in keys:
        preds = results[model]["predictions"].flatten()
        targets = results[model]["targets"].flatten()
        vm = np.isfinite(preds) & np.isfinite(targets)
        preds, targets = preds[vm], targets[vm]
        preds_log = np.log10(preds + eps)
        targets_log = np.log10(targets + eps)
        try:
            xy = np.vstack([targets_log, preds_log]) + np.random.normal(0, 1e-8, (2, len(preds_log)))
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {model}: {e}")
            density = np.ones(len(preds_log))
        all_densities.extend(density)
        scatter_data.append((model, preds_log, targets_log, density))
        global_min = min(global_min, float(targets_log.min()), float(preds_log.min()))
        global_max_val = max(global_max_val, float(targets_log.max()), float(preds_log.max()))

    norm = Normalize(vmin=min(all_densities), vmax=max(all_densities))
    sc = None

    for idx, (model, preds_log, targets_log, density) in enumerate(scatter_data):
        ax = axes[idx]
        order = density.argsort()
        sc = ax.scatter(
            targets_log[order], preds_log[order], c=density[order],
            cmap="viridis", norm=norm, s=8, alpha=0.6, edgecolors="none",
        )
        ax.plot([global_min, global_max_val], [global_min, global_max_val], "r--", lw=1.5, alpha=0.7)
        corr = float(np.corrcoef(targets_log, preds_log)[0, 1])
        ax.set_xlabel("Log₁₀ Actual", fontsize=12)
        ax.set_ylabel("Log₁₀ Predicted", fontsize=12)
        ax.set_title(f"{get_label(model, labels)}\n(Log-Log Pearson r = {corr:.3f})", fontsize=12, fontweight="bold")
        sns.despine(ax=ax)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    right_margin = 0.92 if n_rows == 1 else 0.92
    fig.subplots_adjust(right=right_margin)
    if sc is not None:
        cbar_ax = fig.add_axes(_colorbar_axes(n_rows))
        fig.colorbar(sc, cax=cbar_ax).set_label("Point Density", fontsize=10)
    plt.savefig(os.path.join(output_dir, "scatter_loglog_predicted_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: scatter_loglog_predicted_vs_actual.png")


# ============================================================================
# Shared helper: grouped bar chart for range-based error data
# ============================================================================

def _grouped_range_bar(
    error_df: pd.DataFrame,
    value_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    model_col: str,
    range_order: List[str],
    count_df: pd.DataFrame,
    models: List[str],
    colors: Optional[Dict[str, str]],
    labels: Optional[Dict[str, str]],
    xlabel: str,
    ylabel: str,
    title: str,
    legend_title: str,
    filename: str,
    output_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    label_to_model = {get_label(model, labels): model for model in models}

    pivot = error_df.pivot(index="Range", columns=model_col, values=value_col)
    pivot_lo = error_df.pivot(index="Range", columns=model_col, values=ci_lower_col)
    pivot_hi = error_df.pivot(index="Range", columns=model_col, values=ci_upper_col)

    range_order = [r for r in range_order if r in pivot.index]
    ordered_labels = [get_label(model, labels) for model in models]
    for piv in (pivot, pivot_lo, pivot_hi):
        piv = piv.reindex(range_order)

    pivot = pivot.reindex(range_order)
    pivot_lo = pivot_lo.reindex(range_order).fillna(0)
    pivot_hi = pivot_hi.reindex(range_order).fillna(0)
    pivot = pivot[[c for c in ordered_labels if c in pivot.columns]]
    pivot_lo = pivot_lo[[c for c in ordered_labels if c in pivot_lo.columns]]
    pivot_hi = pivot_hi[[c for c in ordered_labels if c in pivot_hi.columns]]

    mean_vals = pivot.values
    lower_err = np.abs(mean_vals - pivot_lo.values)
    upper_err = np.abs(pivot_hi.values - mean_vals)

    x = np.arange(len(range_order))
    width = 0.7 / len(pivot.columns)
    bar_colors = [get_color(label_to_model.get(col, col), colors) for col in pivot.columns]

    for i, col in enumerate(pivot.columns):
        offset = (i - len(pivot.columns) / 2 + 0.5) * width
        ax.bar(
            x + offset, pivot[col], width, label=col, color=bar_colors[i],
            edgecolor="white",
            yerr=[lower_err[:, i], upper_err[:, i]],
            capsize=5, error_kw={"elinewidth": 2},
        )

    xtick_labels = [
        f"{r}\n(n={int(count_df.loc[r, 'Count']):,})" if r in count_df.index else r
        for r in range_order
    ]
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title=legend_title, frameon=False, loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  ✓ Saved: {filename}")


# ============================================================================
# Plot 5 – MAE per abundance range
# ============================================================================

def plot_mae_per_range(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Grouped bar chart of MAE per abundance range with 95 % CIs."""
    set_style()
    models = list(results.keys())
    bins = [
        ("zero", "Zero"),
        (0, 0.001, ">0% to 0.1%"),
        (0.001, 0.01, "0.1-1%"),
        (0.01, 0.1, "1-10%"),
        (0.1, 1.0, ">10%"),
    ]
    rows = []
    for model in models:
        y_pred = results[model]["predictions"].flatten()
        y_true = results[model]["targets"].flatten()
        vm = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred, y_true = y_pred[vm], y_true[vm]
        for b in bins:
            if b[0] == "zero":
                mask, rlabel = y_true == 0, b[1]
            else:
                lo, hi, rlabel = b
                mask = (y_true > lo) & (y_true <= hi)
            if mask.sum() > 0:
                errs = np.abs(y_true[mask] - y_pred[mask])
                mae = float(np.mean(errs))
                ci = compute_95ci_bootstrap(errs) if mask.sum() > 1 else (mae, mae)
                rows.append({"Model": get_label(model, labels), "Range": rlabel,
                             "MAE": mae, "MAE_CI_Lower": ci[0], "MAE_CI_Upper": ci[1],
                             "Count": int(mask.sum())})

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No data for MAE per range plot")
        return
    count_df = df[df["Model"] == get_label(models[0], labels)][["Range", "Count"]].set_index("Range")
    _grouped_range_bar(
        df, "MAE", "MAE_CI_Lower", "MAE_CI_Upper", "Model",
        ["Zero", ">0% to 0.1%", "0.1-1%", "1-10%", ">10%"],
        count_df, models, colors, labels,
        xlabel="Abundance Range", ylabel="Mean Absolute Error",
        title="Prediction Error by Abundance Range",
        legend_title="Model", filename="error_by_range.png", output_dir=output_dir,
    )


# ============================================================================
# Plot 6 – MAE per range (zoomed, fine bins for <1 %)
# ============================================================================

def plot_mae_per_range_zoomed(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Grouped bar chart of MAE over fine-grained bins in the <1 % range."""
    set_style()
    models = list(results.keys())
    bins = [
        ("zero", "Zero"),
        (0, 0.0011, "0-0.11%"),
        (0.0011, 0.0015, "0.11-0.15%"),
        (0.0015, 0.0022, "0.15-0.22%"),
        (0.0022, 0.01, "0.22-1%"),
    ]
    rows = []
    for model in models:
        y_pred = results[model]["predictions"].flatten()
        y_true = results[model]["targets"].flatten()
        vm = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred, y_true = y_pred[vm], y_true[vm]
        for b in bins:
            if b[0] == "zero":
                mask, rlabel = y_true == 0, b[1]
            else:
                lo, hi, rlabel = b
                mask = (y_true > lo) & (y_true <= hi)
            if mask.sum() > 0:
                errs = np.abs(y_true[mask] - y_pred[mask])
                mae = float(np.mean(errs))
                ci = compute_95ci_bootstrap(errs) if mask.sum() > 1 else (mae, mae)
                rows.append({"Model": get_label(model, labels), "Range": rlabel,
                             "MAE": mae, "MAE_CI_Lower": ci[0], "MAE_CI_Upper": ci[1],
                             "Count": int(mask.sum())})

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No data for zoomed MAE per range plot")
        return
    count_df = df[df["Model"] == get_label(models[0], labels)][["Range", "Count"]].set_index("Range")
    _grouped_range_bar(
        df, "MAE", "MAE_CI_Lower", "MAE_CI_Upper", "Model",
        ["Zero", "0-0.11%", "0.11-0.15%", "0.15-0.22%", "0.22-1%"],
        count_df, models, colors, labels,
        xlabel="Abundance Range", ylabel="Mean Absolute Error",
        title="Prediction Error by Abundance Range (Zoomed: <1%)",
        legend_title="Model", filename="error_by_range_zoomed.png", output_dir=output_dir,
    )


# ============================================================================
# Plot 7 – Relative Absolute Error per range
# ============================================================================

def plot_RAE_per_range(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Grouped bar chart of Relative Absolute Error per abundance range (non-zero only)."""
    set_style()
    models = list(results.keys())
    bins = [
        (0, 0.001, ">0% to 0.1%"),
        (0.001, 0.01, "0.1-1%"),
        (0.01, 0.1, "1-10%"),
        (0.1, 1.0, ">10%"),
    ]
    rows = []
    for model in models:
        y_pred = results[model]["predictions"].flatten()
        y_true = results[model]["targets"].flatten()
        vm = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred, y_true = y_pred[vm], y_true[vm]
        for lo, hi, rlabel in bins:
            mask = (y_true > lo) & (y_true <= hi) & (y_true != 0)
            if mask.sum() > 0:
                rae_arr = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
                rae = float(np.mean(rae_arr))
                ci = compute_95ci_bootstrap(rae_arr) if mask.sum() > 1 else (rae, rae)
                rows.append({"Model": get_label(model, labels), "Range": rlabel,
                             "RAE": rae, "RAE_CI_Lower": ci[0], "RAE_CI_Upper": ci[1],
                             "Count": int(mask.sum())})

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No data for RAE per range plot")
        return
    count_df = df[df["Model"] == get_label(models[0], labels)][["Range", "Count"]].set_index("Range")
    _grouped_range_bar(
        df, "RAE", "RAE_CI_Lower", "RAE_CI_Upper", "Model",
        [">0% to 0.1%", "0.1-1%", "1-10%", ">10%"],
        count_df, models, colors, labels,
        xlabel="Abundance Range", ylabel="Relative Absolute Error",
        title="Relative Absolute Error by Abundance Range",
        legend_title="Model", filename="relative_err_by_range.png", output_dir=output_dir,
    )


# ============================================================================
# Plot 8 – Residual distribution
# ============================================================================

def plot_residual_distribution(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Overlapping residual histograms + KDE, one series per model."""
    set_style()
    models = list(results.keys())
    fig, ax = plt.subplots(figsize=(9, 5))

    all_res: List[float] = []
    per_model = {}
    for model in models:
        t = results[model]["targets"].flatten()
        p = results[model]["predictions"].flatten()
        vm = np.isfinite(t) & np.isfinite(p)
        res = (t - p)[vm]
        per_model[model] = res
        all_res.extend(res.tolist())

    if not all_res:
        log.warning("No residuals to plot")
        return

    x_kde = np.linspace(min(all_res), max(all_res), 300)
    legend_handles = []
    max_count = 0

    for model, res in per_model.items():
        color = get_color(model, colors)
        lbl = get_label(model, labels)
        counts, _, _ = ax.hist(res, bins=60, color=color, alpha=0.3, edgecolor="none")
        max_count = max(max_count, counts.max())
        if len(res) > 1:
            kde_vals = gaussian_kde(res)(x_kde)
            ax.plot(x_kde, kde_vals, color=color, linewidth=2)
        legend_handles.append(Line2D(
            [0], [0], color=color, linewidth=2,
            label=f"{lbl} (μ={np.mean(res):.4f}, σ={np.std(res):.4f})",
        ))

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Residual (Actual − Predicted)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(0.5, max_count * 1.2)
    ax.set_title("Residual Distributions", fontsize=13, fontweight="bold")
    ax.legend(handles=legend_handles, frameon=False, fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: residual_distribution.png")


# ============================================================================
# Plot 9 – Zero vs non-zero MAE
# ============================================================================

def plot_zero_vs_nonzero_comparison(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Paired bars showing MAE on zero vs non-zero ground-truth values."""
    set_style()
    models = list(results.keys())
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(models) + 2), 5))
    x = np.arange(len(models))
    width = 0.35

    def _blend_white(color: str, alpha: float = 0.6) -> Tuple[float, float, float]:
        r, g, b = mc.to_rgb(color)
        return (r * (1 - alpha) + alpha, g * (1 - alpha) + alpha, b * (1 - alpha) + alpha)

    for i, model in enumerate(models):
        t = results[model]["targets"].flatten()
        p = results[model]["predictions"].flatten()
        vm = np.isfinite(t) & np.isfinite(p)
        t, p = t[vm], p[vm]
        zm, nzm = t == 0, t != 0
        e_z = np.abs(t[zm] - p[zm]) if zm.sum() > 0 else np.array([0.0])
        e_nz = np.abs(t[nzm] - p[nzm]) if nzm.sum() > 0 else np.array([0.0])
        mae_z, mae_nz = float(np.mean(e_z)), float(np.mean(e_nz))
        base = get_color(model, colors)
        ci_z = _ci_tuple_to_errorbar(mae_z, compute_95ci_bootstrap(e_z))
        ci_nz = _ci_tuple_to_errorbar(mae_nz, compute_95ci_bootstrap(e_nz))
        ax.bar(
            x[i] - width / 2, mae_z, width, color=_blend_white(base), edgecolor="white",
            yerr=[[ci_z[0]], [ci_z[1]]], capsize=4, error_kw={"elinewidth": 1.5}
        )
        ax.bar(
            x[i] + width / 2, mae_nz, width, color=base, edgecolor="white",
            yerr=[[ci_nz[0]], [ci_nz[1]]], capsize=4, error_kw={"elinewidth": 1.5}
        )

    ax.legend(handles=[
        Patch(facecolor="#cccccc", edgecolor="white", label="Zero GT (lighter)"),
        Patch(facecolor="#666666", edgecolor="white", label="Non-zero GT (darker)"),
    ], frameon=False, loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels([get_label(model, labels) for model in models], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title("MAE: Zero vs Non-Zero Values", fontsize=14, fontweight="bold")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zero_vs_nonzero_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: zero_vs_nonzero_comparison.png")


# ============================================================================
# Plot 10 – Training progress (optional – only if training data present)
# ============================================================================

def plot_training_progress_comparison(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Loss evolution and end-of-cycle summaries for all models that have training data.

    Silently skipped if no training data is found in *results*.
    """
    def _has_training_data(model_results: Dict[str, Any]) -> bool:
        return bool(
            model_results.get("timeline_train_losses") or
            model_results.get("train_losses")
        )

    def _extract(model_results: Dict[str, Any]) -> Tuple[List[float], List[float]]:
        tl = model_results.get("timeline_train_losses")
        vl = model_results.get("timeline_val_losses")
        if tl and vl:
            return [l for *_, l in tl], [l for *_, l in vl]
        return (
            [l for *_, l in model_results.get("train_losses", [])],
            [l for *_, l in model_results.get("val_losses", [])],
        )

    models = [model for model in results if _has_training_data(results[model])]
    if not models:
        log.info("  (Skipping training progress plot — no training data found.)")
        return

    set_style()
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(8 * n, 10), sharey="row")
    if n == 1:
        axes = np.array([
            [axes[0, 0] if axes.ndim == 2 else axes[0]], [axes[1, 0] if axes.ndim == 2 else axes[1]]
        ])

    for idx, model in enumerate(models):
        lbl = get_label(model, labels)
        color = get_color(model, colors)
        train_vals, val_vals = _extract(results[model])
        cycle_train = results[model].get("cycle_train_losses", [])
        cycle_val = results[model].get("cycle_val_losses", [])

        ax1 = axes[0, idx]
        ax1.plot(train_vals, color=color, lw=1.6, label="Train", alpha=0.85)
        ax1.plot(val_vals, color=color, lw=1.6, ls="--", label="Val", alpha=0.85)
        # Vertical lines at EM cycle boundaries
        if results[model].get("timeline_train_losses"):
            for i, (phase, *_) in enumerate(results[model]["timeline_train_losses"]):
                if phase == "latent" and i > 0:
                    ax1.axvline(x=i, color="gray", ls=":", alpha=0.5, lw=1)
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{lbl}: Loss Evolution")
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=True, fontsize=10, loc="upper right")

        ax2 = axes[1, idx]
        if cycle_train:
            cyc = [c + 1 for c, _ in cycle_train]
            ax2.plot(
                cyc, [l for _, l in cycle_train], color=color, ls="-", 
                marker="o", lw=1.8, ms=6, label="Train", alpha=0.9
            )
        if cycle_val:
            cyc = [c + 1 for c, _ in cycle_val]
            ax2.plot(
                cyc, [l for _, l in cycle_val], color=color, ls="--",
                marker="o", lw=1.8, ms=6, label="Val", alpha=0.9
            )
        ax2.set_xlabel("EM Cycle")
        ax2.set_ylabel("Loss")
        ax2.set_title(f"{lbl}: End-of-Cycle Losses")
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=True, fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: training_progress.png")


# ============================================================================
# Plot 11 – Latent importance diagnostics (optional, specific to latent models)
# ============================================================================

def plot_latent_importance_diagnostics(
    results: Dict[str, Any],
    output_dir: str,
    model_key: str = "latent_as_input",
) -> None:
    """Visualise whether the MLP actually uses the latent embedding.

    Three panels:
      1. Weight norm ratio (latent columns vs feature columns in MLP first layer).
      2. Latent embedding std over training.
      3. Ablation delta (optional) — val-loss increase when latent is zeroed.

    Silently skipped when *model_key* is absent or has no ``latent_diagnostics``.
    """
    if model_key not in results:
        log.info(f"  (Skipping latent diagnostics — key '{model_key}' not in results.)")
        return
    diagnostics = results[model_key].get("latent_diagnostics", [])
    if not diagnostics:
        log.info("  (Skipping latent diagnostics — no 'latent_diagnostics' data recorded.)")
        return

    set_style()
    epochs = [d["epoch"] for d in diagnostics]
    ratios = [d["weight_norm_ratio"] for d in diagnostics]
    emb_stds = [d["embedding_std"] for d in diagnostics]
    ab_epochs = [d["epoch"] for d in diagnostics if d.get("ablation_delta") is not None]
    ab_deltas = [d["ablation_delta"] for d in diagnostics if d.get("ablation_delta") is not None]

    n_panels = 3 if ab_epochs else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 2:
        axes = list(axes)
    fig.suptitle("Latent-as-Input: Does the MLP Actually Use the Latent?", fontsize=13, y=1.01)

    ax = axes[0]
    ax.plot(epochs, ratios, color="steelblue", lw=1.8, label="latent/feat ratio")
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="ratio = 1")
    ax.fill_between(epochs, 0, ratios, alpha=0.12, color="steelblue")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Per-column mean weight norm ratio")
    ax.set_title("Latent vs Feature Weight Activity\nin MLP First Layer")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.annotate(f"{ratios[0]:.3f}", xy=(epochs[0], ratios[0]), xytext=(8, 8),
                textcoords="offset points", fontsize=8, color="steelblue")
    ax.annotate(f"{ratios[-1]:.3f}", xy=(epochs[-1], ratios[-1]), xytext=(-30, 8),
                textcoords="offset points", fontsize=8, color="steelblue")

    ax = axes[1]
    ax.plot(epochs, emb_stds, color="darkorange", lw=1.8, label="embedding std")
    ax.fill_between(epochs, 0, emb_stds, alpha=0.12, color="darkorange")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Std of latent embedding weights")
    ax.set_title("Latent Embedding Activity\n(≈ 0 → embedding collapsed / not used)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.annotate(f"{emb_stds[0]:.4f}", xy=(epochs[0], emb_stds[0]), xytext=(8, 8),
                textcoords="offset points", fontsize=8, color="darkorange")
    ax.annotate(f"{emb_stds[-1]:.4f}", xy=(epochs[-1], emb_stds[-1]), xytext=(-40, 8),
                textcoords="offset points", fontsize=8, color="darkorange")

    if ab_epochs:
        ax = axes[2]
        bar_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in ab_deltas]
        w = max(1, (ab_epochs[-1] - ab_epochs[0]) / max(len(ab_epochs), 1) * 0.8) if len(ab_epochs) > 1 else 5
        ax.bar(ab_epochs, ab_deltas, color=bar_colors, alpha=0.75, width=w)
        ax.axhline(0.0, color="gray", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Δ Val loss (zeroed latent − normal)")
        ax.set_title("Latent Ablation Delta\n(> 0 → latent reduces loss; ≈ 0 → MLP ignores it)")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[
            Rectangle((0, 0), 1, 1, color="#2ca02c", alpha=0.75, label="latent helps (Δ > 0)"),
            Rectangle((0, 0), 1, 1, color="#d62728", alpha=0.75, label="latent hurts (Δ < 0)"),
        ], fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_importance_diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: latent_importance_diagnostics.png")


# ============================================================================
# Plot 12 – Summary table
# ============================================================================

def plot_summary_table(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    title: str = "Model Comparison Summary",
    csv_filename: str = "comparison_results.csv",
) -> pd.DataFrame:
    """Render a summary metrics table as a PNG (with best values highlighted) and a CSV."""
    set_style()
    models = list(results.keys())
    ext = {
        model: compute_extended_metrics(
                results[model]["targets"], results[model]["predictions"],
                sample_labels=results[model].get("sample_labels"),
                bin_labels=results[model].get("bin_labels"),
            ) for model in models
    }

    metrics = [
        "MAE (macro)", "MAE (micro)", "Absolute Relative Error", "KL Divergence", 
        "MAE (zeros)", "MAE (non-zeros)", "R²", "R² (log + 1)"
    ]
    best_is_high = {
        "MAE (macro)": False, "MAE (micro)": False, "Absolute Relative Error": False, "KL Divergence": False,
        "MAE (zeros)": False, "MAE (non-zeros)": False, "R²": True, "R² (log + 1)": True,
    }

    data = [{"Model": get_label(model, labels), **{m: ext[model][m] for m in metrics}} for model in models]
    df = pd.DataFrame(data)

    # Best value per metric
    best_rows: Dict[str, List[int]] = {}
    for col in metrics:
        best_val = df[col].max() if best_is_high[col] else df[col].min()
        best_rows[col] = df[df[col] == best_val].index.tolist()

    display_df = df.copy()
    for col in metrics:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}")

    n_cols_table = len(display_df.columns)
    fig_w = max(14, n_cols_table * 2.0)
    fig_h = max(1.5, (len(models) + 1) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=list(display_df.columns),
        cellLoc="center",
        loc="center",
        colColours=["#f0f0f0"] * n_cols_table,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for i in range(n_cols_table):
        table[(0, i)].set_text_props(fontweight="bold")
        table[(0, i)].set_facecolor("#d0d0d0")

    model_col_idx = list(display_df.columns).index("Model")
    for row_idx, model in enumerate(models):
        cell = table[(row_idx + 1, model_col_idx)]
        hex_color = get_color(model, colors)
        cell.set_facecolor(hex_color)
        cell.set_text_props(fontweight="bold", color=_contrasting_text_color(hex_color))
        #cell.set_height(0.15)

    for col_idx, col in enumerate(display_df.columns):
        if col == "Model":
            continue
        for row_idx in best_rows.get(col, []):
            table[(row_idx + 1, col_idx)].set_facecolor("#d5f5e3")
            table[(row_idx + 1, col_idx)].set_text_props(fontweight="bold", color="#1a7a40")

    plt.title(title, fontweight="bold", fontsize=14, y=1.10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: summary_table.png")

    df.to_csv(os.path.join(output_dir, csv_filename), index=False)
    log.info(f"  ✓ Saved: {csv_filename}")
    return df


# ============================================================================
# Master orchestration
# ============================================================================

def create_all_visualizations(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    title: str = "Model Comparison",
    latent_model_key: str = "latent_as_input",
    csv_filename: str = "comparison_results.csv",
) -> None:
    """Generate all standard comparison plots and save them to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    log.info("\n" + "=" * 60)
    log.info("CREATING VISUALIZATIONS")
    log.info("=" * 60)

    log.info("\n 1. Metric comparison bar plots...")
    plot_metrics_comparison(results, output_dir, colors, labels, title=title)

    log.info(" 2. Scatter plots (full range)...")
    plot_scatter_actual_vs_predicted(results, output_dir, labels)

    log.info(" 3. Scatter plots (ground truth <1%)...")
    plot_scatter_zoomed(results, output_dir, labels)

    log.info(" 4. Log-log scatter plots...")
    plot_loglog_scatter_actual_vs_predicted(results, output_dir, labels)

    log.info(" 5. MAE by abundance range...")
    plot_mae_per_range(results, output_dir, colors, labels)

    log.info(" 6. MAE by abundance range (zoomed, <1%)...")
    plot_mae_per_range_zoomed(results, output_dir, colors, labels)

    log.info(" 7. Relative Absolute Error by range...")
    plot_RAE_per_range(results, output_dir, colors, labels)

    log.info(" 8. Residual distribution...")
    plot_residual_distribution(results, output_dir, colors, labels)

    log.info(" 9. Zero vs non-zero MAE comparison...")
    plot_zero_vs_nonzero_comparison(results, output_dir, colors, labels)

    log.info("10. Training progress (if available)...")
    plot_training_progress_comparison(results, output_dir, colors, labels)

    log.info("11. Latent importance diagnostics (if available)...")
    plot_latent_importance_diagnostics(results, output_dir, model_key=latent_model_key)

    log.info("12. Summary table...")
    plot_summary_table(results, output_dir, colors, labels, title=title, csv_filename=csv_filename)

    log.info(f"\n✅ All visualizations saved to: {output_dir}/")


# ============================================================================
# Console print comparison
# ============================================================================

def print_comparison(
    results: Dict[str, Any],
    labels: Optional[Dict[str, str]] = None,
    title: str = "MODEL COMPARISON RESULTS",
) -> None:
    """Print a comparison table and win summary to the console."""
    models = list(results.keys())
    ext = {
        model: compute_extended_metrics(
            results[model]["targets"], results[model]["predictions"],
            sample_labels=results[model].get("sample_labels"),
            bin_labels=results[model].get("bin_labels"),
        ) for model in models
    }

    metrics_cfg = [
        ("RMSE (micro)", False),
        ("RMSE (macro)", False),
        ("MAE (micro)", False),
        ("MAE (macro)", False),
        ("Absolute Relative Error", False),
        ("R²", True),
        ("R² (log + 1)", True),
        ("KL Divergence", False),
        ("RMSE (zeros)", False),
        ("MAE (zeros)", False),
        ("RMSE (non-zeros)", False),
        ("MAE (non-zeros)", False),
        ("Correlation", True),
    ]

    col_w = 22
    log.info("\n" + "=" * (col_w * (len(models) + 2)))
    log.info(title)
    log.info("=" * (col_w * (len(models) + 2)))

    header = f"{'Metric':<{col_w}}" + "".join(f"{get_label(model, labels):<{col_w}}" for model in models) + f"{'Best':<{col_w}}"
    log.info(header)
    log.info("-" * len(header))

    wins = {model: 0 for model in models}
    for metric, higher_better in metrics_cfg:
        vals = [ext[model][metric] for model in models]
        best_idx = vals.index(max(vals) if higher_better else min(vals))
        wins[models[best_idx]] += 1
        row = f"{metric:<{col_w}}"
        row += "".join(f"{v:<{col_w}.6f}" for v in vals)
        row += f"{get_label(models[best_idx], labels):<{col_w}}"
        log.info(row)

    log.info("-" * len(header))
    log.info("\nWin summary:")
    for model in models:
        log.info(f"  {get_label(model, labels)}: {wins[model]} wins")
    overall = max(wins, key=wins.get)
    log.info(f"\n✓ Best overall: {get_label(overall, labels)} ({wins[overall]}/{len(metrics_cfg)} metrics)")

    # Improvement over first model (treated as baseline)
    baseline = models[0]
    if len(models) > 1:
        log.info(f"\nImprovement over {get_label(baseline, labels)}:")
        for model in models[1:]:
            log.info(f"\n  {get_label(model, labels)}:")
            for metric, higher_better in metrics_cfg:
                bv, kv = ext[baseline][metric], ext[model][metric]
                if higher_better:
                    pct = ((kv - bv) / abs(bv)) * 100 if bv != 0 else 0.0
                    sym = "↑" if pct > 0 else "↓"
                else:
                    pct = ((bv - kv) / abs(bv)) * 100 if bv != 0 else 0.0
                    sym = "↓" if pct < 0 else "↑"
                log.info(f"    {metric:<32}: {pct:>7.2f}% {sym}")


# ============================================================================
# I/O helpers
# ============================================================================

def load_results(results_path: str) -> Dict[str, Any]:
    """Load a results dict from a pickle file."""
    with open(results_path, "rb") as f:
        return pickle.load(f)


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified visualization for model comparison results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results_path", required=True,
                        help="Path to the results pickle file.")
    parser.add_argument("--output_dir", default="figures",
                        help="Directory where plots will be saved.")
    parser.add_argument("--colors", default=None,
                        help='JSON dict mapping result keys to hex colors, e.g. \'{"a": "#e74c3c"}\'')
    parser.add_argument("--labels", default=None,
                        help='JSON dict mapping result keys to display names, e.g. \'{"a": "Model A"}\'')
    parser.add_argument("--title", default="Model Comparison",
                        help="Title used on aggregate plots and the summary table.")
    parser.add_argument("--latent_key", default="latent_as_input",
                        help="Key in results that holds latent diagnostics data (default: 'latent_as_input').")
    parser.add_argument("--csv", default="comparison_results.csv",
                        help="Filename for the output CSV summary table.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log.basicConfig(
        level=log.DEBUG if args.verbose else log.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    
    # Resolve paths relative to cwd
    results_path = os.path.abspath(args.results_path)
    output_dir = os.path.abspath(args.output_dir)

    log.info(f"Loading results from {results_path} ...")
    results = load_results(results_path)
    log.info(f"Found {len(results)} model(s): {list(results.keys())}")

    labels = json.loads(args.labels) if args.labels else {k: get_label(k) for k in results.keys()}
    colors = json.loads(args.colors) if args.colors else _default_colors(labels)

    print_comparison(results, labels=labels, title=args.title.upper())
    create_all_visualizations(
        results, output_dir,
        colors=colors, labels=labels,
        title=args.title,
        latent_model_key=args.latent_key,
        csv_filename=args.csv,
    )
