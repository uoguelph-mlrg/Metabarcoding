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
from matplotlib.patches import Patch


# ============================================================================
# Default color palette (used as fallback for unknown keys)
# ============================================================================

_DEFAULT_PALETTE = [
    "#2ecc71", "#e74c3c", "#3498db", "#e67e22", "#9b59b6",
    "#1abc9c", "#f39c12", "#95a5a6", "#34495e", "#16a085",
    "#27ae60", "#d35400", "#c0392b", "#8e44ad", "#2980b9",
]


# ============================================================================
# Helpers
# ============================================================================

def get_color(key: str, colors: Optional[Dict[str, str]] = None) -> str:
    """Return a hex color for *key*, falling back to the default palette."""
    if colors and key in colors:
        return colors[key]
    return _DEFAULT_PALETTE[hash(key) % len(_DEFAULT_PALETTE)]


def get_label(key: str, labels: Optional[Dict[str, str]] = None) -> str:
    """Return a human-readable label for *key*."""
    if labels and key in labels:
        return labels[key]
    return key.replace("_", " ").title()


def _contrasting_text_color(hex_color: str) -> str:
    """Return 'white' or 'black' for a hex background color."""
    r, g, b = mc.to_rgb(hex_color)
    return "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.45 else "black"


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


def _colorbar_axes(fig: plt.Figure, n_rows: int) -> List[float]:
    """Return [left, bottom, width, height] for a right-side colorbar."""
    if n_rows == 1:
        return [0.94, 0.15, 0.02, 0.70]
    return [0.95, 0.10, 0.015, 0.80]


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
        y_true:        Ground-truth relative abundances — shape (N,) or
                       (n_samples, max_bins) for old-style pickles.
        y_pred:        Predicted values — same shape as y_true.
        sample_labels: (N,) string array identifying which sample each entry
                       belongs to.  When provided, per-sample macro metrics
                       (RMSE_macro, MAE_macro, KL Divergence) are computed
                       via groupby then averaged — the only rigorous way to
                       measure distributional error.
        bin_labels:    (N,) string array of BIN URIs.  Not used in scalar
                       summary metrics but passed through for callers that
                       want per-BIN breakdown via ``groupby(bin_labels)``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse_macro: Optional[float] = None
    mae_macro: Optional[float] = None
    kl_divergence_macro: Optional[float] = None
    eps = 1e-10

    # ------------------------------------------------------------------
    # Path A: flat 1-D arrays with sample_labels (rigorous per-sample metrics)
    # ------------------------------------------------------------------
    if y_true.ndim == 1 and sample_labels is not None:
        sl = np.asarray(sample_labels)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        yt_v = y_true[valid]
        yp_v = np.clip(y_pred[valid], 0, 1)
        sl_v = sl[valid]

        rmse_per, mae_per, kl_per = [], [], []
        for s in np.unique(sl_v):
            mask = sl_v == s
            rt = yt_v[mask]
            rp = yp_v[mask]
            if len(rt) == 0:
                continue
            rmse_per.append(float(np.sqrt(np.mean((rt - rp) ** 2))))
            mae_per.append(float(np.mean(np.abs(rt - rp))))
            # KL per sample: each sample's values form a probability distribution
            rt_norm = (rt + eps) / (rt + eps).sum()
            rp_norm = (rp + eps) / (rp + eps).sum()
            kl_per.append(float(np.sum(rt_norm * np.log(rt_norm / rp_norm))))

        if rmse_per:
            rmse_macro = float(np.mean(rmse_per))
            mae_macro = float(np.mean(mae_per))
            kl_divergence_macro = float(np.mean(kl_per))

    # ------------------------------------------------------------------
    # Path B: 2-D NaN-padded arrays (backward compat with old result files)
    # ------------------------------------------------------------------
    elif y_true.ndim == 2:
        rmse_per, mae_per, kl_per = [], [], []
        for i in range(y_true.shape[0]):
            row_t = y_true[i]
            row_p = y_pred[i]
            valid_i = np.isfinite(row_t) & np.isfinite(row_p)
            if valid_i.sum() == 0:
                continue
            rt = row_t[valid_i]
            rp = np.clip(row_p[valid_i], 0, 1)
            rmse_per.append(float(np.sqrt(np.mean((rt - rp) ** 2))))
            mae_per.append(float(np.mean(np.abs(rt - rp))))
            rt_norm = (rt + eps) / (rt + eps).sum()
            rp_norm = (rp + eps) / (rp + eps).sum()
            kl_per.append(float(np.sum(rt_norm * np.log(rt_norm / rp_norm))))
        if rmse_per:
            rmse_macro = float(np.mean(rmse_per))
            mae_macro = float(np.mean(mae_per))
            kl_divergence_macro = float(np.mean(kl_per))

    # ------------------------------------------------------------------
    # Flatten for micro (global) metrics
    # ------------------------------------------------------------------
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    valid = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true = y_true_flat[valid]
    y_pred = np.clip(y_pred_flat[valid], 0, 1)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse_micro = float(np.sqrt(mse))
    mae_micro = float(np.mean(np.abs(y_true - y_pred)))

    # Fall back to micro when no grouping info was available
    if rmse_macro is None:
        rmse_macro = rmse_micro
        mae_macro = mae_micro

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))

    zero_mask = y_true == 0
    nonzero_mask = y_true > 0

    if zero_mask.sum() > 0:
        rmse_zeros = float(np.sqrt(np.mean((y_true[zero_mask] - y_pred[zero_mask]) ** 2)))
        mae_zeros = float(np.mean(np.abs(y_true[zero_mask] - y_pred[zero_mask])))
    else:
        rmse_zeros = mae_zeros = 0.0

    if nonzero_mask.sum() > 0:
        rmse_nonzeros = float(np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask]) ** 2)))
        mae_nonzeros = float(np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask])))
    else:
        rmse_nonzeros = mae_nonzeros = 0.0

    # Micro KL — fallback when no sample grouping is available
    y_tn = (y_true + eps) / (y_true + eps).sum()
    y_pn = (y_pred + eps) / (y_pred + eps).sum()
    kl_divergence_micro = float(np.sum(y_tn * np.log(y_tn / y_pn)))

    # Prefer per-sample (macro) KL when available
    kl_divergence = kl_divergence_macro if kl_divergence_macro is not None else kl_divergence_micro

    corr = np.corrcoef(y_true, y_pred)[0, 1]
    correlation = 0.0 if np.isnan(corr) else float(corr)

    nz = y_true != 0
    rel_error = np.zeros_like(y_true, dtype=float)
    rel_error[nz] = np.abs(y_pred[nz] - y_true[nz]) / np.abs(y_true[nz])
    absolute_relative_error = float(np.mean(rel_error[nz])) if nz.sum() > 0 else 0.0

    return {
        "RMSE_micro": rmse_micro,
        "RMSE_macro": rmse_macro,
        "MAE_micro": mae_micro,
        "MAE_macro": mae_macro,
        "Absolute Relative Error": absolute_relative_error,
        "R²": r2,
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
    keys = list(results.keys())

    ext = {k: compute_extended_metrics(
               results[k]["targets"], results[k]["predictions"],
               sample_labels=results[k].get("sample_labels"),
               bin_labels=results[k].get("bin_labels"),
           )
           for k in keys}

    metrics_to_plot = [
        "RMSE_micro", "MAE_micro", "Absolute Relative Error",
        "KL Divergence", "MAE (zeros)", "MAE (non-zeros)", "Correlation",
    ]
    n_metrics = len(metrics_to_plot)

    # Compute bootstrap CIs
    cis: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        y_true = results[k]["targets"].flatten()
        y_pred = results[k]["predictions"].flatten()
        vm = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[vm], np.clip(y_pred[vm], 0, 1)
        if len(y_true) < 2:
            cis[k] = {m: None for m in metrics_to_plot}
            continue
        abs_err = np.abs(y_true - y_pred)
        nz = y_true != 0
        zero_m = y_true == 0
        nonzero_m = y_true > 0
        rel_err = np.zeros_like(y_true)
        rel_err[nz] = abs_err[nz] / np.abs(y_true[nz])

        sq_err = (y_true - y_pred) ** 2
        rmse_ci = tuple(np.sqrt(v) for v in compute_95ci_bootstrap(sq_err))
        cis[k] = {
            "RMSE_micro": rmse_ci,
            "MAE_micro": compute_95ci_bootstrap(abs_err),
            "Absolute Relative Error": (
                compute_95ci_bootstrap(rel_err[nz]) if nz.sum() > 1 else (0.0, 0.0)
            ),
            "KL Divergence": None,
            "MAE (zeros)": (
                compute_95ci_bootstrap(abs_err[zero_m]) if zero_m.sum() > 1 else (0.0, 0.0)
            ),
            "MAE (non-zeros)": (
                compute_95ci_bootstrap(abs_err[nonzero_m]) if nonzero_m.sum() > 1 else (0.0, 0.0)
            ),
            "Correlation": None,
        }

    n_cols = min(4, n_metrics)
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten() if n_metrics > 1 else [axes]

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [ext[k][metric] for k in keys]
        ci_info = [cis[k].get(metric) for k in keys]
        bar_colors = [get_color(k, colors) for k in keys]
        x_pos = np.arange(len(keys))

        if metric in ("KL Divergence", "Correlation") or all(c is None for c in ci_info):
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
            [get_label(k, labels) for k in keys], rotation=45, ha="right", fontsize=9
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
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Scatter plots (actual vs predicted) with density colouring, one panel per model."""
    set_style()
    keys = list(results.keys())
    n = len(keys)
    n_rows, n_cols = _scatter_grid(n)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    all_densities: List[float] = []
    scatter_data = []
    global_max_target = global_max_pred = 0.0

    for k in keys:
        preds = results[k]["predictions"].flatten()
        targets = results[k]["targets"].flatten()
        vm = np.isfinite(preds) & np.isfinite(targets)
        preds, targets = preds[vm], targets[vm]
        global_max_target = max(global_max_target, float(targets.max()))
        global_max_pred = max(global_max_pred, float(preds.max()))
        try:
            xy = np.vstack([preds, targets]) + np.random.normal(0, 1e-8, (2, len(preds)))
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {k}: {e}")
            density = np.ones(len(preds))
        all_densities.extend(density)
        scatter_data.append((k, preds, targets, density))

    norm = Normalize(vmin=min(all_densities), vmax=max(all_densities))
    sc = None
    axis_max = max(global_max_target, global_max_pred)

    for idx, (k, preds, targets, density) in enumerate(scatter_data):
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
        ax.set_title(f"{get_label(k, labels)}\n(Pearson r = {corr:.3f})", fontsize=12, fontweight="bold")
        ax.set_xlim(0, global_max_target)
        ax.set_ylim(0, global_max_pred)
        sns.despine(ax=ax)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    right_margin = 0.92 if n_rows == 1 else 0.93
    fig.subplots_adjust(right=right_margin)
    if sc is not None:
        cbar_ax = fig.add_axes(_colorbar_axes(fig, n_rows))
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
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    max_actual: float = 0.01,
) -> None:
    """Scatter plots zoomed on ground-truth values below *max_actual*."""
    set_style()
    keys = list(results.keys())
    n = len(keys)
    n_rows, n_cols = _scatter_grid(n)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    all_densities: List[float] = []
    scatter_data = []
    global_max_target = global_max_pred = 0.0

    for k in keys:
        targets = results[k]["targets"].flatten()
        preds = results[k]["predictions"].flatten()
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
            log.warning(f"Could not compute density for {k}: {e}")
            density = np.ones(len(prd_z)) if len(prd_z) > 0 else np.array([])
        all_densities.extend(density)
        scatter_data.append((k, prd_z, tgt_z, density))

    vmin = min(all_densities) if all_densities else 0
    vmax = max(all_densities) if all_densities else 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    axis_max = max(global_max_target, global_max_pred)
    sc = None

    for idx, (k, prd_z, tgt_z, density) in enumerate(scatter_data):
        ax = axes[idx]
        if len(prd_z) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(get_label(k, labels), fontsize=12, fontweight="bold")
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
        ax.set_title(f"{get_label(k, labels)}\n(Pearson r = {corr:.3f})", fontsize=12, fontweight="bold")
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
        cbar_ax = fig.add_axes(_colorbar_axes(fig, n_rows))
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
    colors: Optional[Dict[str, str]] = None,
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

    for k in keys:
        preds = results[k]["predictions"].flatten()
        targets = results[k]["targets"].flatten()
        vm = np.isfinite(preds) & np.isfinite(targets)
        preds, targets = preds[vm], targets[vm]
        preds_log = np.log10(preds + eps)
        targets_log = np.log10(targets + eps)
        try:
            xy = np.vstack([targets_log, preds_log]) + np.random.normal(0, 1e-8, (2, len(preds_log)))
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {k}: {e}")
            density = np.ones(len(preds_log))
        all_densities.extend(density)
        scatter_data.append((k, preds_log, targets_log, density))
        global_min = min(global_min, float(targets_log.min()), float(preds_log.min()))
        global_max_val = max(global_max_val, float(targets_log.max()), float(preds_log.max()))

    norm = Normalize(vmin=min(all_densities), vmax=max(all_densities))
    sc = None

    for idx, (k, preds_log, targets_log, density) in enumerate(scatter_data):
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
        ax.set_title(f"{get_label(k, labels)}\n(Log-Log r = {corr:.3f})", fontsize=12, fontweight="bold")
        sns.despine(ax=ax)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    right_margin = 0.92 if n_rows == 1 else 0.92
    fig.subplots_adjust(right=right_margin)
    if sc is not None:
        cbar_ax = fig.add_axes(_colorbar_axes(fig, n_rows))
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
    keys: List[str],
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
    label_to_key = {get_label(k, labels): k for k in keys}

    pivot = error_df.pivot(index="Range", columns=model_col, values=value_col)
    pivot_lo = error_df.pivot(index="Range", columns=model_col, values=ci_lower_col)
    pivot_hi = error_df.pivot(index="Range", columns=model_col, values=ci_upper_col)

    range_order = [r for r in range_order if r in pivot.index]
    ordered_labels = [get_label(k, labels) for k in keys]
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
    bar_colors = [get_color(label_to_key.get(col, col), colors) for col in pivot.columns]

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
    keys = list(results.keys())
    bins = [
        ("zero", "Zero"),
        (0, 0.001, ">0% to 0.1%"),
        (0.001, 0.01, "0.1-1%"),
        (0.01, 0.1, "1-10%"),
        (0.1, 1.0, ">10%"),
    ]
    rows = []
    for k in keys:
        y_pred = results[k]["predictions"].flatten()
        y_true = results[k]["targets"].flatten()
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
                rows.append({"Model": get_label(k, labels), "Range": rlabel,
                             "MAE": mae, "MAE_CI_Lower": ci[0], "MAE_CI_Upper": ci[1],
                             "Count": int(mask.sum())})

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No data for MAE per range plot")
        return
    count_df = df[df["Model"] == get_label(keys[0], labels)][["Range", "Count"]].set_index("Range")
    _grouped_range_bar(
        df, "MAE", "MAE_CI_Lower", "MAE_CI_Upper", "Model",
        ["Zero", ">0% to 0.1%", "0.1-1%", "1-10%", ">10%"],
        count_df, keys, colors, labels,
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
    keys = list(results.keys())
    bins = [
        ("zero", "Zero"),
        (0, 0.0011, "0-0.11%"),
        (0.0011, 0.0015, "0.11-0.15%"),
        (0.0015, 0.0022, "0.15-0.22%"),
        (0.0022, 0.01, "0.22-1%"),
    ]
    rows = []
    for k in keys:
        y_pred = results[k]["predictions"].flatten()
        y_true = results[k]["targets"].flatten()
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
                rows.append({"Model": get_label(k, labels), "Range": rlabel,
                             "MAE": mae, "MAE_CI_Lower": ci[0], "MAE_CI_Upper": ci[1],
                             "Count": int(mask.sum())})

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No data for zoomed MAE per range plot")
        return
    count_df = df[df["Model"] == get_label(keys[0], labels)][["Range", "Count"]].set_index("Range")
    _grouped_range_bar(
        df, "MAE", "MAE_CI_Lower", "MAE_CI_Upper", "Model",
        ["Zero", "0-0.11%", "0.11-0.15%", "0.15-0.22%", "0.22-1%"],
        count_df, keys, colors, labels,
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
    keys = list(results.keys())
    bins = [
        (0, 0.001, ">0% to 0.1%"),
        (0.001, 0.01, "0.1-1%"),
        (0.01, 0.1, "1-10%"),
        (0.1, 1.0, ">10%"),
    ]
    rows = []
    for k in keys:
        y_pred = results[k]["predictions"].flatten()
        y_true = results[k]["targets"].flatten()
        vm = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred, y_true = y_pred[vm], y_true[vm]
        for lo, hi, rlabel in bins:
            mask = (y_true > lo) & (y_true <= hi) & (y_true != 0)
            if mask.sum() > 0:
                rae_arr = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
                rae = float(np.mean(rae_arr))
                ci = compute_95ci_bootstrap(rae_arr) if mask.sum() > 1 else (rae, rae)
                rows.append({"Model": get_label(k, labels), "Range": rlabel,
                             "RAE": rae, "RAE_CI_Lower": ci[0], "RAE_CI_Upper": ci[1],
                             "Count": int(mask.sum())})

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No data for RAE per range plot")
        return
    count_df = df[df["Model"] == get_label(keys[0], labels)][["Range", "Count"]].set_index("Range")
    _grouped_range_bar(
        df, "RAE", "RAE_CI_Lower", "RAE_CI_Upper", "Model",
        [">0% to 0.1%", "0.1-1%", "1-10%", ">10%"],
        count_df, keys, colors, labels,
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
    keys = list(results.keys())
    fig, ax = plt.subplots(figsize=(9, 5))

    all_res: List[float] = []
    per_key = {}
    for k in keys:
        t = results[k]["targets"].flatten()
        p = results[k]["predictions"].flatten()
        vm = np.isfinite(t) & np.isfinite(p)
        res = (t - p)[vm]
        per_key[k] = res
        all_res.extend(res.tolist())

    if not all_res:
        log.warning("No residuals to plot")
        return

    x_kde = np.linspace(min(all_res), max(all_res), 300)
    legend_handles = []
    max_count = 0

    for k, res in per_key.items():
        color = get_color(k, colors)
        lbl = get_label(k, labels)
        counts, _, _ = ax.hist(res, bins=60, color=color, alpha=0.3, edgecolor="none")
        max_count = max(max_count, counts.max())
        if len(res) > 1:
            kde_vals = gaussian_kde(res)(x_kde)
            ax.plot(x_kde, kde_vals, color=color, linewidth=2)
        legend_handles.append(plt.Line2D(
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
    keys = list(results.keys())
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(keys) + 2), 5))
    x = np.arange(len(keys))
    width = 0.35

    def _blend_white(color: str, alpha: float = 0.6) -> Tuple[float, float, float]:
        r, g, b = mc.to_rgb(color)
        return (r * (1 - alpha) + alpha, g * (1 - alpha) + alpha, b * (1 - alpha) + alpha)

    for i, k in enumerate(keys):
        t = results[k]["targets"].flatten()
        p = results[k]["predictions"].flatten()
        vm = np.isfinite(t) & np.isfinite(p)
        t, p = t[vm], p[vm]
        zm, nzm = t == 0, t != 0
        e_z = np.abs(t[zm] - p[zm]) if zm.sum() > 0 else np.array([0.0])
        e_nz = np.abs(t[nzm] - p[nzm]) if nzm.sum() > 0 else np.array([0.0])
        mae_z, mae_nz = float(np.mean(e_z)), float(np.mean(e_nz))
        base = get_color(k, colors)
        ci_z = _ci_tuple_to_errorbar(mae_z, compute_95ci_bootstrap(e_z))
        ci_nz = _ci_tuple_to_errorbar(mae_nz, compute_95ci_bootstrap(e_nz))
        ax.bar(x[i] - width / 2, mae_z, width, color=_blend_white(base), edgecolor="white",
               yerr=[[ci_z[0]], [ci_z[1]]], capsize=4, error_kw={"elinewidth": 1.5})
        ax.bar(x[i] + width / 2, mae_nz, width, color=base, edgecolor="white",
               yerr=[[ci_nz[0]], [ci_nz[1]]], capsize=4, error_kw={"elinewidth": 1.5})

    ax.legend(handles=[
        Patch(facecolor="#cccccc", edgecolor="white", label="Zero GT (lighter)"),
        Patch(facecolor="#666666", edgecolor="white", label="Non-zero GT (darker)"),
    ], frameon=False, loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels([get_label(k, labels) for k in keys], rotation=45, ha="right", fontsize=9)
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

    keys = [k for k in results if _has_training_data(results[k])]
    if not keys:
        log.info("  (Skipping training progress plot — no training data found.)")
        return

    set_style()
    n = len(keys)
    fig, axes = plt.subplots(2, n, figsize=(8 * n, 10), sharey="row")
    if n == 1:
        axes = np.array([[axes[0, 0] if axes.ndim == 2 else axes[0]],
                         [axes[1, 0] if axes.ndim == 2 else axes[1]]])

    for idx, k in enumerate(keys):
        lbl = get_label(k, labels)
        color = get_color(k, colors)
        train_vals, val_vals = _extract(results[k])
        cycle_train = results[k].get("cycle_train_losses", [])
        cycle_val = results[k].get("cycle_val_losses", [])

        ax1 = axes[0, idx]
        ax1.plot(train_vals, color=color, lw=1.6, label="Train", alpha=0.85)
        ax1.plot(val_vals, color=color, lw=1.6, ls="--", label="Val", alpha=0.85)
        # Vertical lines at EM cycle boundaries
        if results[k].get("timeline_train_losses"):
            for i, (phase, *_) in enumerate(results[k]["timeline_train_losses"]):
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
            ax2.plot(cyc, [l for _, l in cycle_train], color=color, ls="-",
                     marker="o", lw=1.8, ms=6, label="Train", alpha=0.9)
        if cycle_val:
            cyc = [c + 1 for c, _ in cycle_val]
            ax2.plot(cyc, [l for _, l in cycle_val], color=color, ls="--",
                     marker="o", lw=1.8, ms=6, label="Val", alpha=0.9)
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
            plt.Rectangle((0, 0), 1, 1, color="#2ca02c", alpha=0.75, label="latent helps (Δ > 0)"),
            plt.Rectangle((0, 0), 1, 1, color="#d62728", alpha=0.75, label="latent hurts (Δ < 0)"),
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
    keys = list(results.keys())
    ext = {k: compute_extended_metrics(
               results[k]["targets"], results[k]["predictions"],
               sample_labels=results[k].get("sample_labels"),
               bin_labels=results[k].get("bin_labels"),
           )
           for k in keys}

    metrics = [
        "RMSE_micro", "MAE_micro", "Absolute Relative Error",
        "KL Divergence", "MAE (zeros)", "MAE (non-zeros)", "Correlation",
    ]

    data = [{"Model": get_label(k, labels), **{m: ext[k][m] for m in metrics}} for k in keys]
    df = pd.DataFrame(data)

    # Best value per metric
    best_rows: Dict[str, List[int]] = {}
    for col in metrics:
        best_val = df[col].max() if col == "Correlation" else df[col].min()
        best_rows[col] = df[df[col] == best_val].index.tolist()

    display_df = df.copy()
    for col in metrics:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}")

    n_cols_table = len(display_df.columns)
    fig_w = max(14, n_cols_table * 2.0)
    fig_h = max(1.5, len(keys) * 0.6 + 1.0)
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
    for row_idx, k in enumerate(keys):
        cell = table[(row_idx + 1, model_col_idx)]
        hex_color = get_color(k, colors)
        cell.set_facecolor(hex_color)
        cell.set_text_props(fontweight="bold", color=_contrasting_text_color(hex_color))
        cell.set_height(0.15)

    for col_idx, col in enumerate(display_df.columns):
        if col == "Model":
            continue
        for row_idx in best_rows.get(col, []):
            table[(row_idx + 1, col_idx)].set_facecolor("#d5f5e3")
            table[(row_idx + 1, col_idx)].set_text_props(fontweight="bold", color="#1a7a40")

    plt.title(title, fontweight="bold", fontsize=14, y=1.02)
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
    plot_scatter_actual_vs_predicted(results, output_dir, colors, labels)

    log.info(" 3. Scatter plots (ground truth <1%)...")
    plot_scatter_zoomed(results, output_dir, colors, labels)

    log.info(" 4. Log-log scatter plots...")
    plot_loglog_scatter_actual_vs_predicted(results, output_dir, colors, labels)

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
    keys = list(results.keys())
    ext = {k: compute_extended_metrics(
               results[k]["targets"], results[k]["predictions"],
               sample_labels=results[k].get("sample_labels"),
               bin_labels=results[k].get("bin_labels"),
           )
           for k in keys}

    metrics_cfg = [
        ("RMSE_micro", False),
        ("MAE_micro", False),
        ("MAE_macro", False),
        ("Absolute Relative Error", False),
        ("KL Divergence", False),
        ("MAE (zeros)", False),
        ("MAE (non-zeros)", False),
        ("Correlation", True),
    ]

    col_w = 22
    log.info("\n" + "=" * (col_w * (len(keys) + 2)))
    log.info(title)
    log.info("=" * (col_w * (len(keys) + 2)))

    header = f"{'Metric':<{col_w}}" + "".join(f"{get_label(k, labels):<{col_w}}" for k in keys) + f"{'Best':<{col_w}}"
    log.info(header)
    log.info("-" * len(header))

    wins = {k: 0 for k in keys}
    for metric, higher_better in metrics_cfg:
        vals = [ext[k][metric] for k in keys]
        best_idx = vals.index(max(vals) if higher_better else min(vals))
        wins[keys[best_idx]] += 1
        row = f"{metric:<{col_w}}"
        row += "".join(f"{v:<{col_w}.6f}" for v in vals)
        row += f"{get_label(keys[best_idx], labels):<{col_w}}"
        log.info(row)

    log.info("-" * len(header))
    log.info("\nWin summary:")
    for k in keys:
        log.info(f"  {get_label(k, labels)}: {wins[k]} wins")
    overall = max(wins, key=wins.get)
    log.info(f"\n✓ Best overall: {get_label(overall, labels)} ({wins[overall]}/{len(metrics_cfg)} metrics)")

    # Improvement over first model (treated as baseline)
    baseline = keys[0]
    if len(keys) > 1:
        log.info(f"\nImprovement over {get_label(baseline, labels)}:")
        for k in keys[1:]:
            log.info(f"\n  {get_label(k, labels)}:")
            for metric, higher_better in metrics_cfg:
                bv, kv = ext[baseline][metric], ext[k][metric]
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

    colors = json.loads(args.colors) if args.colors else None
    labels = json.loads(args.labels) if args.labels else None

    # Resolve paths relative to cwd
    results_path = os.path.abspath(args.results_path)
    output_dir = os.path.abspath(args.output_dir)

    log.info(f"Loading results from {results_path} ...")
    results = load_results(results_path)
    log.info(f"Found {len(results)} model(s): {list(results.keys())}")

    print_comparison(results, labels=labels, title=args.title.upper())
    create_all_visualizations(
        results, output_dir,
        colors=colors, labels=labels,
        title=args.title,
        latent_model_key=args.latent_key,
        csv_filename=args.csv,
    )
