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

    python visualize_results.py \\
        --results_paths path/to/a.pkl path/to/b.pkl path/to/c.pkl \\
        --output_dir figures

    python visualize_results.py \\
        --results_path path/to/results_folder \\
        --output_dir figures

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

Optional per-model keys (used when present):
    - "latent_vector":  np.ndarray with latent values per BIN (used for latent
                        comparison plots when at least two models provide it)
    - "train_losses", "val_losses": list of (epoch, loss) tuples
    - "timeline_train_losses", "timeline_val_losses": list of (phase, cycle, step, loss)
    - "cycle_train_losses", "cycle_val_losses": list of (cycle, loss) tuples
    - "latent_diagnostics": schema depends on trainer variant:
        - src/train.py style keys include: 'epoch', 'latent_mean', 'latent_std', 'ablation_loss', 
        and optional final-weight stats.
        - analysis/latent_as_in_and_output/train.py style keys include: 'epoch', 'latent_z_mean', 
        'latent_z_std', optional 'latent_d_mean', optional 'latent_d_std', 'z_weight_norm_ratio', 
        and ablation losses ('z_ablation_loss', 'd_ablation_loss', 'joint_ablation_loss').
"""
from __future__ import annotations

import argparse
import glob
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
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D


# ============================================================================
# Helpers
# ============================================================================

DEFAULT_PALETTE = np.concatenate([["#95a5a6"], sns.color_palette("tab10").as_hex(), np.column_stack([
    sns.color_palette("pastel").as_hex(), sns.color_palette("dark").as_hex()
]).flatten()])

def _sorted_models(results: Dict[str, Any], labels: Optional[Dict[str, str]] = None) -> List[str]:
    """Return model keys in display order.

    Priority:
    1. If *labels* is provided, follow its insertion order (keys not in labels
       fall back to baseline-first then alphabetical and are appended at the end).
    2. Otherwise: "baseline" first, then the remaining keys in alphabetical order.
    """
    keys = list(results.keys())
    if labels:
        ordered = [k for k in labels if k in results]
        remainder = sorted([k for k in keys if k not in labels], key=lambda k: (k != "baseline", k.lower()))
        return ordered + remainder
    return sorted(keys, key=lambda k: (k != "baseline", k.lower()))


def _default_colors(labels: Dict[str, str]) -> Dict[str, str]:
    """Generate a default color mapping for *n* models, using *labels* if available."""
    keys = list(labels.keys())
    return {k: DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i, k in enumerate(keys)}


def get_color(key: str, colors: Optional[Dict[str, str]]) -> str:
    """Return a hex color for *key*, falling back to the default palette."""
    if colors and key in colors:
        return colors[key]
    return DEFAULT_PALETTE[abs(hash(key)) % len(DEFAULT_PALETTE)]


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
    n_cols = 4 if n > 3 else n
    return n_rows, n_cols


def _colorbar_axes(n_rows: int) -> tuple[float, float, float, float]:
    """Return [left, bottom, width, height] for a right-side colorbar."""
    if n_rows == 1:
        return (0.94, 0.15, 0.02, 0.70)
    return (0.95, 0.10, 0.015, 0.80)


def _shannon_diversity(values: np.ndarray, eps: float = 1e-10) -> float:
    """Compute Shannon diversity from non-negative abundance values."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.nan
    probs = (arr + eps) / np.sum(arr + eps)
    return float(-np.sum(probs * np.log(probs + eps)))


def _fit_r2_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Return (r2, intercept) from linear fit y ~ x; None when undefined."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[valid], y_arr[valid]
    if x_arr.size < 2 or np.allclose(x_arr, x_arr[0]):
        return None, None

    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    i0 = float(intercept) if np.isfinite(intercept) else None
    
    ss_res = np.sum((y_arr - (slope * x_arr + intercept)) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10)) if np.isfinite(ss_tot) else None
    return r2, i0


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Compute Spearman rho and return None if undefined."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[valid], y_arr[valid]
    if x_arr.size < 2:
        return None

    rank_x = pd.Series(x_arr).rank(method="average").to_numpy(dtype=float)
    rank_y = pd.Series(y_arr).rank(method="average").to_numpy(dtype=float)
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        return None
    rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return rho if np.isfinite(rho) else None


def _bootstrap_r2_intercept_ci(
    shannon_true: np.ndarray,
    shannon_pred: np.ndarray,
    n_bootstrap: int = 1000,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Bootstrap CIs for Shannon fit R^2 and intercept across samples."""
    x = np.asarray(shannon_true, dtype=float)
    y = np.asarray(shannon_pred, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return None, None

    r2_vals: List[float] = []
    intercept_vals: List[float] = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(x.size, size=x.size, replace=True)
        xb, yb = x[idx], y[idx]
        r2, intercept = _fit_r2_intercept(xb, yb)
        if r2 is not None:
            r2_vals.append(r2)
        if intercept is not None:
            intercept_vals.append(intercept)

    def _ci(vals: List[float]) -> Optional[Tuple[float, float]]:
        if len(vals) < 2:
            return None
        arr = np.asarray(vals, dtype=float)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return _ci(r2_vals), _ci(intercept_vals)


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

    Accepts flat 1-D arrays (the canonical format produced by
    ``Trainer.get_predictions``) with label arrays that enable rigorous
    grouped metrics.

    Args:
        y_true:         Ground-truth relative abundances — shape (N,).
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
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("'targets' and 'predictions' must have the same shape")

    if sample_labels is not None:
        sample_labels = np.asarray(sample_labels).reshape(-1)
        if sample_labels.shape[0] != y_true.shape[0]:
            raise ValueError("'sample_labels' length must match 'targets' and 'predictions'")

    if bin_labels is not None:
        bin_labels = np.asarray(bin_labels).reshape(-1)
        if bin_labels.shape[0] != y_true.shape[0]:
            raise ValueError("'bin_labels' length must match 'targets' and 'predictions'")

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = np.clip(y_pred[valid], 0, 1)

    rmse_macro: Optional[float] = np.nan
    mae_macro: Optional[float] = np.nan
    kl_divergence: Optional[float] = np.nan
    shannon_r2: Optional[float] = np.nan
    shannon_intercept: Optional[float] = np.nan
    spearman_macro: Optional[float] = np.nan
    eps = 1e-10

    # ------------------------------------------------------------------
    # Per sample metrics (macro-averaged)
    # ------------------------------------------------------------------
    if sample_labels is not None:
        sample_labels_v = sample_labels[valid]

        rmse_per, mae_per, kl_per = [], [], []
        shannon_true_per, shannon_pred_per = [], []
        spearman_per = []
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

            shannon_true = _shannon_diversity(true_s, eps=eps)
            shannon_pred = _shannon_diversity(pred_s, eps=eps)
            if np.isfinite(shannon_true) and np.isfinite(shannon_pred):
                shannon_true_per.append(shannon_true)
                shannon_pred_per.append(shannon_pred)

            if len(true_s) > 1:
                rho = _spearman_rho(true_s, pred_s)
                if rho is not None:
                    spearman_per.append(rho)

        if rmse_per:
            rmse_macro = float(np.mean(rmse_per))
            mae_macro = float(np.mean(mae_per))
            kl_divergence = float(np.mean(kl_per))

        if len(shannon_true_per) >= 2:
            fit_r2, fit_intercept = _fit_r2_intercept(
                np.asarray(shannon_true_per, dtype=float),
                np.asarray(shannon_pred_per, dtype=float),
            )
            if fit_r2 is not None:
                shannon_r2 = fit_r2
            if fit_intercept is not None:
                shannon_intercept = fit_intercept

        if spearman_per:
            spearman_macro = float(np.mean(spearman_per))
    
    # ------------------------------------------------------------------
    # Overall micro-averaged metrics (treating all entries as a single vector)
    # ------------------------------------------------------------------

    mse = np.mean((y_true - y_pred) ** 2)
    rmse_micro = float(np.sqrt(mse))
    mae_micro = float(np.mean(np.abs(y_true - y_pred)))

    if np.isnan(rmse_macro):
        rmse_macro = rmse_micro
        mae_macro = mae_micro
        y_tn = (y_true + eps) / (y_true + eps).sum()
        y_pn = (y_pred + eps) / (y_pred + eps).sum()
        kl_divergence = float(np.sum(y_tn * np.log(y_tn / y_pn)))

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
        "R² (Shannon diversity)": shannon_r2,
        "Shannon intercept": shannon_intercept,
        "Spearman Rho (macro)": spearman_macro,
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
    models = _sorted_models(results, labels)

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
        "MAE (zeros)", "MAE (non-zeros)", "R² (log + 1)"
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
        
        # Get macro metrics for CI computation (sample-wise, then averaged)
        mae_per_s, kl_div_per_s = [], []
        shannon_true_per_s, shannon_pred_per_s = [], []
        spearman_per_s = []
        if sample_labels is not None:
            sample_labels_v = np.asarray(sample_labels).reshape(-1)[valid]
            for s in np.unique(sample_labels_v):
                mask = sample_labels_v == s
                true_s = y_true[mask]
                pred_s = y_pred[mask]
                if len(true_s) == 0:
                    continue
                mae_per_s.append(float(np.mean(np.abs(true_s - pred_s))))
                true_s_norm = (true_s + 1e-10) / (true_s + 1e-10).sum()
                pred_s_norm = (pred_s + 1e-10) / (pred_s + 1e-10).sum()
                kl_div_per_s.append(float(np.sum(true_s_norm * np.log(true_s_norm / pred_s_norm))))
                shannon_true = _shannon_diversity(true_s)
                shannon_pred = _shannon_diversity(pred_s)
                if np.isfinite(shannon_true) and np.isfinite(shannon_pred):
                    shannon_true_per_s.append(shannon_true)
                    shannon_pred_per_s.append(shannon_pred)
                if len(true_s) > 1:
                    rho = _spearman_rho(true_s, pred_s)
                    if rho is not None:
                        spearman_per_s.append(rho)
        else:
            mae_per_s.append(float(np.mean(np.abs(y_true - y_pred))))
            y_tn = (y_true + 1e-10) / (y_true + 1e-10).sum()
            y_pn = (y_pred + 1e-10) / (y_pred + 1e-10).sum()
            kl_div_per_s.append(float(np.sum(y_tn * np.log(y_tn / y_pn))))
        mae_per_s, kl_div_per_s, spearman_per_s = np.array(mae_per_s), np.array(kl_div_per_s), np.array(spearman_per_s)
        
        # Get masks for zero vs non-zero true values (used for subgroup CI computation)
        nz = y_true != 0
        zero_m = y_true == 0
        nonzero_m = y_true > 0
        
        # Get micro metrics for CI computation
        abs_err = np.abs(y_true - y_pred)

        shannon_r2_ci, shannon_intercept_ci = _bootstrap_r2_intercept_ci(
            np.asarray(shannon_true_per_s, dtype=float),
            np.asarray(shannon_pred_per_s, dtype=float),
        )
        
        cis[model] = {
            'MAE (macro)': compute_95ci_bootstrap(mae_per_s),
            "MAE (micro)": compute_95ci_bootstrap(abs_err),
            "Absolute Relative Error": compute_95ci_bootstrap(abs_err[nz] / np.abs(y_true[nz])),
            "KL Divergence": compute_95ci_bootstrap(kl_div_per_s),
            "MAE (zeros)": compute_95ci_bootstrap(abs_err[zero_m]),
            "MAE (non-zeros)": compute_95ci_bootstrap(abs_err[nonzero_m]),
            "Spearman Rho (macro)": compute_95ci_bootstrap(spearman_per_s),
            "R² (Shannon diversity)": shannon_r2_ci,
            "Shannon intercept": shannon_intercept_ci,
            "R² (log + 1)": None,
        }
        
    n_cols_max = np.clip(np.floor(50 / (len(models) + 2)), 1, None)  # avoid division by zero
    n_rows = np.ceil(n_metrics / n_cols_max).astype(int)
    n_cols = np.ceil(n_metrics / n_rows).astype(int)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten() if n_metrics > 1 else [axes]

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = np.asarray([ext[model][metric] for model in models], dtype=float)
        values_plot = np.where(np.isfinite(values), values, 0.0)
        ci_info = [cis[model].get(metric) for model in models]
        bar_colors = [get_color(model, colors) for model in models]
        x_pos = np.arange(len(models))

        if all(c is None for c in ci_info):
            yerr = None
        else:
            ci_lower = [c[0] if isinstance(c, tuple) else values_plot[i] for i, c in enumerate(ci_info)]
            ci_upper = [c[1] if isinstance(c, tuple) else values_plot[i] for i, c in enumerate(ci_info)]
            yerr = [
                np.abs(np.array(values_plot) - np.array(ci_lower)),
                np.abs(np.array(ci_upper) - np.array(values_plot)),
            ]

        ax.bar(
            x_pos, values_plot, color=bar_colors, edgecolor="white", linewidth=1.5,
            yerr=yerr, capsize=4,
            error_kw={"elinewidth": 1.5} if yerr is not None else {},
        )
        ax.set_title(metric, fontsize=11, fontweight="bold")
        finite_values = values[np.isfinite(values)]
        if finite_values.size > 0:
            max_val = float(np.max(finite_values))
            min_val = float(np.min(finite_values))
            if min_val < 0:
                lower = min_val * 1.25
                upper = max_val * 1.25 if max_val > 0 else 1.0
                ax.set_ylim(lower, upper)
            else:
                ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1)
        else:
            ax.set_ylim(0, 1)
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
    models = _sorted_models(results, labels)
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
    models = _sorted_models(results, labels)
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
    keys = _sorted_models(results, labels)
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
        f"{r}\n(n={int(np.asarray(count_df.loc[r, 'Count']).item()):,})" if r in count_df.index else r
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
    models = _sorted_models(results, labels)
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
    models = _sorted_models(results, labels)
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
    models = _sorted_models(results, labels)
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
    models = _sorted_models(results, labels)
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
        counts_arr = np.asarray(counts, dtype=float)
        if counts_arr.size > 0:
            max_count = max(max_count, float(counts_arr.max()))
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
    models = _sorted_models(results, labels)
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

    models = [model for model in _sorted_models(results, labels) if _has_training_data(results[model])]
    if not models:
        log.info("  (Skipping training progress plot — no training data found.)")
        return

    set_style()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), sharey=True)
    axes = np.array(axes).flatten() if n > 1 else np.array([axes])

    for idx, model in enumerate(models):
        lbl = get_label(model, labels)
        color = get_color(model, colors)
        train_vals, val_vals = _extract(results[model])

        ax = axes[idx]
        ax.plot(train_vals, color=color, lw=1.6, label="Train", alpha=0.85)
        ax.plot(val_vals, color=color, lw=1.6, ls="--", label="Val", alpha=0.85)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{lbl}: Loss Evolution")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fontsize=10, loc="upper right")
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: training_progress.png")


# ============================================================================
# Plot 11 – Latent importance diagnostics (optional, specific to latent models)
# ============================================================================

def plot_latent_diagnostics(
    results: Dict[str, Any],
    output_dir: str,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Visualise whether the MLP actually uses the latent embedding.

    Each model gets its own row. Three to four panels per row depending on available diagnostics:
    1. Latent distribution over epochs (mean line with std band).
    2. Z weight-ratio trend (only if the ratio key exists).
    3. Validation and ablation losses over epochs.
    4. Delta bars (validation loss minus ablated loss).
    """
    model_keys = [
        key for key in _sorted_models(results, labels)
        if isinstance(results.get(key), dict) and results[key].get("latent_diagnostics")
    ]
    if not model_keys:
        log.info("  (Skipping latent diagnostics — no model provides latent_diagnostics.)")
        return

    def _epoch_loss_map(items: Any) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for item in items or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                epoch = int(item[0])
                loss = float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(loss):
                out[epoch] = loss
        return out

    def _series(diags: List[Dict[str, Any]], keys: List[str]) -> List[Tuple[int, float]]:
        series: List[Tuple[int, float]] = []
        for diag in diags:
            epoch_raw = diag.get("epoch")
            if epoch_raw is None:
                continue
            try:
                epoch = int(epoch_raw)
            except (TypeError, ValueError):
                continue
            value = None
            for key in keys:
                raw = diag.get(key)
                if raw is None:
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    break
                value = None
            if value is not None:
                series.append((epoch, value))
        series.sort(key=lambda pair: pair[0])
        return series

    def _with_std(mean_series: List[Tuple[int, float]], std_series: List[Tuple[int, float]]) -> Tuple[List[int], List[float], List[float]]:
        std_map = {epoch: value for epoch, value in std_series}
        xs: List[int] = []
        means: List[float] = []
        stds: List[float] = []
        for epoch, mean in mean_series:
            std = std_map.get(epoch)
            if std is not None:
                xs.append(epoch)
                means.append(mean)
                stds.append(std)
        return xs, means, stds

    def _delta_series(val_map: Dict[int, float], ab_map: Dict[int, float]) -> List[Tuple[int, float]]:
        return [(epoch, ab_map[epoch] - val_map[epoch]) for epoch in sorted(val_map) if epoch in ab_map]

    rows: List[Dict[str, Any]] = []
    any_ratio = False
    for key in model_keys:
        model_results = results[key]
        diags = [diag for diag in model_results.get("latent_diagnostics", []) if isinstance(diag, dict)]
        if not diags:
            continue

        val_map = _epoch_loss_map(model_results.get("val_losses", []))
        ratio = _series(diags, ["z_weight_norm_ratio"])
        any_ratio = any_ratio or bool(ratio)

        z_mean = _series(diags, ["latent_z_mean"])
        z_std = _series(diags, ["latent_z_std"])
        d_mean = _series(diags, ["latent_d_mean", "latent_mean"])
        d_std = _series(diags, ["latent_d_std", "latent_std"])

        ablation_z = _series(diags, ["z_ablation_loss"])
        ablation_d = _series(diags, ["d_ablation_loss"])
        ablation_joint = _series(diags, ["joint_ablation_loss", "ablation_loss"])

        rows.append({
            "model": key,
            "label": get_label(key, labels),
            "val_map": val_map,
            "ratio": ratio,
            "z": _with_std(z_mean, z_std),
            "d": _with_std(d_mean, d_std),
            "ablation_z": ablation_z,
            "ablation_d": ablation_d,
            "ablation_joint": ablation_joint,
        })

    if not rows:
        log.info("  (Skipping latent diagnostics — no valid latent diagnostics rows found.)")
        return

    n_cols = 4 if any_ratio else 3
    set_style()
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(6 * n_cols, 4.2 * len(rows)), squeeze=False)
    fig.suptitle("Latent Importance Diagnostics", fontsize=13, y=0.995)

    for row_idx, row in enumerate(rows):
        label = row["label"]
        val_map: Dict[int, float] = row["val_map"]
        ratio = row["ratio"]
        z_x, z_m, z_s = row["z"]
        d_x, d_m, d_s = row["d"]
        ablation_z = row["ablation_z"]
        ablation_d = row["ablation_d"]
        ablation_joint = row["ablation_joint"]

        col = 0

        ax = axes[row_idx, col]
        col += 1
        if z_x:
            ax.plot(z_x, z_m, color="steelblue", lw=1.8, label="latent z mean")
            ax.fill_between(z_x, np.array(z_m) - np.array(z_s), np.array(z_m) + np.array(z_s),
                            color="steelblue", alpha=0.18, label="latent z ±1 std")
        if d_x:
            ax.plot(d_x, d_m, color="#2ca02c", lw=1.8, label="latent d mean")
            ax.fill_between(d_x, np.array(d_m) - np.array(d_s), np.array(d_m) + np.array(d_s),
                            color="#2ca02c", alpha=0.18, label="latent d ±1 std")
        ax.set_title(f"{label}: latent distributions")
        ax.set_ylabel("latent value")
        ax.grid(True, alpha=0.3)
        if row_idx == len(rows) - 1:
            ax.set_xlabel("epoch")
        if row_idx == 0:
            ax.legend(fontsize=8)

        if n_cols == 4:
            ax = axes[row_idx, col]
            col += 1
            if ratio:
                rx = [epoch for epoch, _ in ratio]
                ry = [value for _, value in ratio]
                ax.plot(rx, ry, color="purple", lw=1.8, label="z_weight_norm_ratio")
                ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
                ax.fill_between(rx, 0, ry, color="purple", alpha=0.12)
                ax.set_ylim(bottom=0)
                ax.set_title("z weight norm ratio")
                ax.grid(True, alpha=0.3)
                if row_idx == len(rows) - 1:
                    ax.set_xlabel("epoch")
                if row_idx == 0:
                    ax.legend(fontsize=8)
            else:
                ax.axis("off")

        ax = axes[row_idx, col]
        col += 1
        has_both_z_and_d = bool(ablation_z) and bool(ablation_d)
        if val_map:
            epochs = sorted(val_map)
            ax.plot(epochs, [val_map[e] for e in epochs], color="black", lw=2.0, label="validation loss")
        if has_both_z_and_d:
            # Show all three ablation lines when both z and d are available
            ax.plot([e for e, _ in ablation_z], [v for _, v in ablation_z], color="steelblue", lw=1.6,
                    marker="o", ms=3, label="z-only ablation loss")
            ax.plot([e for e, _ in ablation_d], [v for _, v in ablation_d], color="#2ca02c", lw=1.6,
                    marker="s", ms=3, label="d-only ablation loss")
            if ablation_joint:
                ax.plot([e for e, _ in ablation_joint], [v for _, v in ablation_joint], color="#d62728", lw=1.9,
                        marker="^", ms=4, label="joint (z+d) ablation loss")
        elif ablation_joint:
            ax.plot([e for e, _ in ablation_joint], [v for _, v in ablation_joint], color="#d62728", lw=1.9,
                    marker="o", ms=4, label="latent ablation loss")
        elif ablation_z:
            ax.plot([e for e, _ in ablation_z], [v for _, v in ablation_z], color="steelblue", lw=1.9,
                    marker="o", ms=4, label="z-only ablation loss")
        elif ablation_d:
            ax.plot([e for e, _ in ablation_d], [v for _, v in ablation_d], color="#2ca02c", lw=1.9,
                    marker="o", ms=4, label="d-only ablation loss")
        ax.set_title("validation and ablation losses")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        if row_idx == len(rows) - 1:
            ax.set_xlabel("epoch")
        if row_idx == 0:
            ax.legend(fontsize=8)

        ax = axes[row_idx, col]
        # Choose the primary ablation series to use for delta bars.
        # Priority: joint > z-only > d-only (whichever is available).
        if ablation_joint:
            primary_ablation = ablation_joint
            delta_label = "validation − joint ablation"
        elif ablation_z:
            primary_ablation = ablation_z
            delta_label = "validation − z ablation"
        elif ablation_d:
            primary_ablation = ablation_d
            delta_label = "validation − d ablation"
        else:
            primary_ablation = []
            delta_label = "delta"

        delta_pairs: List[Tuple[int, float]] = []
        if primary_ablation and val_map:
            ab_map = {epoch: loss for epoch, loss in primary_ablation}
            delta_pairs = _delta_series(val_map, ab_map)
            delta_label = "latent ablation − validation (↑ = latent helps)"
        if delta_pairs:
            dx = [epoch for epoch, _ in delta_pairs]
            dy = [value for _, value in delta_pairs]
            colors = ["#2ca02c" if value >= 0 else "#d62728" for value in dy]
            width = max(1, (dx[-1] - dx[0]) / max(len(dx), 1) * 0.8) if len(dx) > 1 else 5
            ax.bar(dx, dy, width=width, color=bar_colors, alpha=0.8)
            ax.axhline(0.0, color="gray", ls="--", lw=1, alpha=0.8)
        ax.set_title("delta bars")
        ax.set_ylabel(delta_label)
        ax.grid(True, alpha=0.3)
        if row_idx == len(rows) - 1:
            ax.set_xlabel("epoch")
        if row_idx == 0 and delta_pairs:
            ax.legend(handles=[
                Patch(facecolor="#2ca02c", edgecolor="white", label="latent helps (delta > 0)"),
                Patch(facecolor="#d62728", edgecolor="white", label="latent hurts (delta ≤ 0)"),
            ], fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(os.path.join(output_dir, "latent_importance_diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: latent_importance_diagnostics.png")


def plot_latent_comparison(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Compare latent vectors across models.

    Layout strategy:
    - ≤4 models with latent vectors: one row per unique pair (all-vs-all).
    - >4 models: one row per model comparing the baseline (key "baseline" or
      the first model) against every other model.

    Each row has three panels: overlapping distributions, difference distribution,
    and a per-bin scatter coloured by point density.

    Performance optimizations:
    - Samples large datasets (>10k points) for KDE computation.
    - Reduces histogram bins for faster rendering.
    - Skips KDE for very large datasets (>100k points).
    """
    def _has_valid_latent(m: str) -> bool:
        if not isinstance(results.get(m), dict):
            return False
        lv = results[m].get("latent_vector")
        if lv is None:
            return False
        arr = np.asarray(lv)
        return arr.ndim >= 1 and arr.size > 0

    latent_models = [m for m in _sorted_models(results, labels) if _has_valid_latent(m)]
    if len(latent_models) < 2:
        log.info("  (Skipping latent comparison — fewer than two models provide latent vectors.)")
        return

    # Build the list of (m1, m2) pairs to plot.
    if len(latent_models) <= 4:
        from itertools import combinations
        pairs = list(combinations(latent_models, 2))
    else:
        anchor = latent_models[0]
        pairs = [(anchor, m) for m in latent_models if m != anchor]

    scatter_handles: List[Any] = []

    def _render_row(ax_row, m1: str, m2: str) -> None:
        lv1 = np.asarray(results[m1]["latent_vector"]).flatten()
        lv2 = np.asarray(results[m2]["latent_vector"]).flatten()
        if lv1.shape[0] != lv2.shape[0]:
            log.warning(
                f"Latent vectors have different shapes after norm reduction "
                f"({m1}: {lv1.shape}, {m2}: {lv2.shape}); skipping pair."
            )
            for ax in ax_row:
                ax.axis("off")
            return

        is_vector_latent = is_vec1 or is_vec2
        scatter_xlabel = f"{get_label(m1, labels)} latent norm" if is_vector_latent else get_label(m1, labels)
        scatter_ylabel = f"{get_label(m2, labels)} latent norm" if is_vector_latent else get_label(m2, labels)
        dist_xlabel = "Latent norm" if is_vector_latent else "Latent value"
        diff_xlabel_suffix = "latent norm" if is_vector_latent else "latent value"

        n_points = len(lv1)
        use_kde = n_points <= 100_000
        kde_sample_size = min(10_000, n_points) if n_points > 10_000 else n_points

        if n_points > 10_000 and use_kde:
            sample_idx = np.random.choice(n_points, size=kde_sample_size, replace=False)
            lv1_kde = lv1[sample_idx]
            lv2_kde = lv2[sample_idx]
        else:
            lv1_kde = lv1
            lv2_kde = lv2

        label1 = get_label(m1, labels)
        label2 = get_label(m2, labels)
        color1 = get_color(m1, colors)
        color2 = get_color(m2, colors)
        bins = 30 if n_points > 50_000 else 40

        # Panel 1: overlapping distributions.
        ax = ax_row[0]
        sns.histplot(lv1, bins=bins, kde=use_kde, stat="density", color=color1, alpha=0.35, edgecolor="none", ax=ax, label=label1)
        sns.histplot(lv2, bins=bins, kde=use_kde, stat="density", color=color2, alpha=0.35, edgecolor="none", ax=ax, label=label2)
        ax.axvline(lv1.mean(), color=color1, linestyle="--", linewidth=1.5, alpha=0.9)
        ax.axvline(lv2.mean(), color=color2, linestyle="--", linewidth=1.5, alpha=0.9)
        ax.set_xlabel(dist_xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{label1} vs {label2}: distributions", fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
        sns.despine(ax=ax)

        # Panel 2: difference distribution.
        ax = ax_row[1]
        diff = lv2 - lv1
        sns.histplot(diff, bins=bins, kde=use_kde, stat="density", color="#4a90d9", alpha=0.8, edgecolor="none", ax=ax)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(diff.mean(), color="#e74c3c", linestyle="-", linewidth=1.5, alpha=0.9)
        ax.set_xlabel(f"Difference in {diff_xlabel_suffix} ({label2} − {label1})")
        ax.set_ylabel("Density")
        ax.set_title("Latent Difference Distribution", fontsize=11, fontweight="bold")
        sns.despine(ax=ax)

        # Panel 3: per-bin scatter coloured by density.
        ax = ax_row[2]
        # Always compute density on the full dataset (subsample only for KDE fitting)
        if len(lv1_kde) > 1:
            try:
                xy_kde = np.vstack([lv1_kde, lv2_kde]) + np.random.normal(0, 1e-8, (2, len(lv1_kde)))
                kde_fn = gaussian_kde(xy_kde)
                # Evaluate density on ALL points for accurate coloring
                xy_all = np.vstack([lv1, lv2])
                density = kde_fn(xy_all)
                order = np.argsort(density)
            except Exception as e:
                log.debug(f"KDE computation failed: {e}; using uniform density")
                density = np.ones(len(lv1))
                order = np.arange(len(lv1))
        else:
            density = np.ones(len(lv1))
            order = np.arange(len(lv1))

        # Use PowerNorm(gamma<1) to stretch low-density region and show full colormap
        d_min, d_max = density.min(), density.max()
        if d_max > d_min:
            norm = PowerNorm(gamma=0.4, vmin=d_min, vmax=d_max)
        else:
            norm = Normalize(vmin=d_min, vmax=d_max + 1e-10)

        sc = ax.scatter(lv1[order], lv2[order], c=density[order], cmap="viridis",
                        norm=norm, s=10, alpha=0.75, edgecolors="none")
        scatter_handles.append(sc)
        lo = min(lv1.min(), lv2.min())
        hi = max(lv1.max(), lv2.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, alpha=0.7)
        corr = float(np.corrcoef(lv1, lv2)[0, 1]) if len(lv1) > 1 else 0.0
        ax.set_xlabel(scatter_xlabel)
        ax.set_ylabel(scatter_ylabel)
        title_suffix = " (norm)" if is_vector_latent else ""
        ax.set_title(f"Per-BIN Latent{title_suffix} Comparison (r={corr:.3f})", fontsize=11, fontweight="bold")
        sns.despine(ax=ax)

    n_rows = len(pairs)
    set_style()
    fig, axes = plt.subplots(n_rows, 3, figsize=(21, 5 * n_rows), squeeze=False)
    fig.suptitle("Latent Vector Comparison", fontsize=13, y=0.995)

    for row_idx, (m1, m2) in enumerate(pairs):
        _render_row(axes[row_idx], m1, m2)

    plt.tight_layout(rect=(0, 0, 0.96, 0.985))
    # Single shared colorbar on the right for all scatter panels
    if scatter_handles:
        cbar_ax = fig.add_axes((0.965, 0.15, 0.012, 0.70))
        fig.colorbar(scatter_handles[0], cax=cbar_ax).set_label("Point Density", fontsize=10)
    plt.savefig(os.path.join(output_dir, "latent_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: latent_comparison.png")


def plot_top_models_overview(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    top_n: int = 5,
) -> None:
    """Create a radar-chart overview for top models (only meaningful for many-model comparisons)."""
    if len(results) <= 4:
        log.info("  (Skipping top-model overview — requires more than 4 models.)")
        return

    metrics = [
        "MAE (micro)",
        "MAE (macro)",
        "Absolute Relative Error",
        "KL Divergence",
        "MAE (zeros)",
        "MAE (non-zeros)",
        "Correlation",
    ]

    ext = {
        model: compute_extended_metrics(
            results[model]["targets"],
            results[model]["predictions"],
            sample_labels=results[model].get("sample_labels"),
            bin_labels=results[model].get("bin_labels"),
        )
        for model in results
    }

    score_df = pd.DataFrame(
        [{"Model": m, **{k: ext[m][k] for k in metrics}} for m in results]
    )
    score_df = score_df.sort_values("MAE (micro)", ascending=True).head(top_n)

    normalized = score_df[metrics].copy()
    for col in metrics:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max <= col_min:
            normalized[col] = 1.0
        elif col == "Correlation":
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 1.0 - (normalized[col] - col_min) / (col_max - col_min)

    set_style()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for i, model in enumerate(score_df["Model"].tolist()):
        values = normalized.iloc[i].tolist()
        values += values[:1]
        color = get_color(model, colors)
        label = get_label(model, labels)
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title("Top Models Overview (larger area = better)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_models_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: top_models_radar.png")


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
    models = _sorted_models(results, labels)
    ext = {
        model: compute_extended_metrics(
                results[model]["targets"], results[model]["predictions"],
                sample_labels=results[model].get("sample_labels"),
                bin_labels=results[model].get("bin_labels"),
            ) for model in models
    }

    metrics = [
        "MAE (macro)", "MAE (micro)", "Absolute Relative Error", "KL Divergence", 
        "MAE (zeros)", "MAE (non-zeros)", "Spearman Rho (macro)",
        "R² (Shannon diversity)", "Shannon intercept", "R² (log + 1)"
    ]
    best_is_high = {
        "MAE (macro)": False, "MAE (micro)": False, "Absolute Relative Error": False, "KL Divergence": False,
        "MAE (zeros)": False, "MAE (non-zeros)": False, "Spearman Rho (macro)": True,
        "R² (Shannon diversity)": True, "Shannon intercept": False, "R² (log + 1)": True,
    }

    data = [{"Model": get_label(model, labels), **{m: ext[model][m] for m in metrics}} for model in models]
    df = pd.DataFrame(data)

    # Best value per metric
    best_rows: Dict[str, List[int]] = {}
    for col in metrics:
        col_vals = pd.to_numeric(df[col], errors="coerce")
        valid_vals = col_vals.dropna()
        if valid_vals.empty:
            best_rows[col] = []
            continue
        if col == "Shannon intercept":
            best_abs = float(valid_vals.abs().min())
            best_rows[col] = col_vals[(col_vals.abs() == best_abs)].index.tolist()
        else:
            best_val = float(valid_vals.max() if best_is_high[col] else valid_vals.min())
            best_rows[col] = col_vals[(col_vals == best_val)].index.tolist()

    display_df = df.copy()
    for col in metrics:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}")

    n_cols_table = len(display_df.columns)
    fig_w = max(14, n_cols_table * 2.0)
    fig_h = max(1.5, (len(models) + 1) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values.tolist(),
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

def _top_n_by_kl(results: Dict[str, Any], n: int, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Return a subset of *results* containing the *n* models with lowest KL Divergence."""
    models = _sorted_models(results, labels)
    kl_scores = {}
    for m in models:
        metrics = compute_extended_metrics(
            results[m]["targets"], results[m]["predictions"],
            sample_labels=results[m].get("sample_labels"),
            bin_labels=results[m].get("bin_labels"),
        )
        kl_scores[m] = metrics["KL Divergence"]
    top = sorted(kl_scores, key=lambda m: kl_scores[m])[:n]
    return {m: results[m] for m in top}


def create_all_visualizations(
    results: Dict[str, Any],
    output_dir: str,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    title: str = "Model Comparison",
    latent_model_key: str = "latent_as_input",
    csv_filename: str = "comparison_results.csv",
    subset_top_n: Optional[int] = None,
) -> None:
    """Generate all standard comparison plots and save them to *output_dir*.

    When *subset_top_n* is set, the 9 detail plots (scatter, range-error,
    residual, zero-vs-nonzero, top-model radar) use only the top-N models
    ranked by KL Divergence; metrics_comparison and summary_table always
    show all models.
    """
    os.makedirs(output_dir, exist_ok=True)

    log.info("\n" + "=" * 60)
    log.info("CREATING VISUALIZATIONS")
    log.info("=" * 60)

    # Subset used for detail plots when requested
    if subset_top_n is not None and len(results) > subset_top_n:
        results_detail = _top_n_by_kl(results, subset_top_n, labels)
        log.info(f"  (Detail plots restricted to top-{subset_top_n} by KL Divergence: {list(results_detail.keys())})")
    else:
        results_detail = results

    log.info("\n 1. Metric comparison bar plots...")
    plot_metrics_comparison(results, output_dir, colors, labels, title=title)

    log.info(" 2. Scatter plots (full range)...")
    plot_scatter_actual_vs_predicted(results_detail, output_dir, labels)

    log.info(" 3. Scatter plots (ground truth <1%)...")
    plot_scatter_zoomed(results_detail, output_dir, labels)

    log.info(" 4. Log-log scatter plots...")
    plot_loglog_scatter_actual_vs_predicted(results_detail, output_dir, labels)

    log.info(" 5. MAE by abundance range...")
    plot_mae_per_range(results_detail, output_dir, colors, labels)

    log.info(" 6. MAE by abundance range (zoomed, <1%)...")
    plot_mae_per_range_zoomed(results_detail, output_dir, colors, labels)

    log.info(" 7. Relative Absolute Error by range...")
    plot_RAE_per_range(results_detail, output_dir, colors, labels)

    log.info(" 8. Residual distribution...")
    plot_residual_distribution(results_detail, output_dir, colors, labels)

    log.info(" 9. Zero vs non-zero MAE comparison...")
    plot_zero_vs_nonzero_comparison(results_detail, output_dir, colors, labels)

    log.info("10. Training progress (if available)...")
    plot_training_progress_comparison(results, output_dir, colors, labels)

    log.info("11. Latent importance diagnostics (if available)...")
    plot_latent_diagnostics(results, output_dir, labels)

    log.info("12. Latent vector comparison (if available)...")
    plot_latent_comparison(results, output_dir, colors, labels)

    log.info("13. Top-model radar overview (>4 models only)...")
    plot_top_models_overview(results_detail, output_dir, colors, labels)

    log.info("14. Summary table...")
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
    models = _sorted_models(results, labels)
    ext = {
        model: compute_extended_metrics(
            results[model]["targets"], results[model]["predictions"],
            sample_labels=results[model].get("sample_labels"),
            bin_labels=results[model].get("bin_labels"),
        ) for model in models
    }

    metrics_cfg = [
        ("RMSE (micro)", "min"),
        ("RMSE (macro)", "min"),
        ("MAE (micro)", "min"),
        ("MAE (macro)", "min"),
        ("Absolute Relative Error", "min"),
        ("R² (Shannon diversity)", "max"),
        ("Shannon intercept", "absmin"),
        ("Spearman Rho (macro)", "max"),
        ("R²", "max"),
        ("R² (log + 1)", "max"),
        ("KL Divergence", "min"),
        ("RMSE (zeros)", "min"),
        ("MAE (zeros)", "min"),
        ("RMSE (non-zeros)", "min"),
        ("MAE (non-zeros)", "min"),
        ("Correlation", "max"),
    ]

    col_w = 22
    log.info("\n" + "=" * (col_w * (len(models) + 2)))
    log.info(title)
    log.info("=" * (col_w * (len(models) + 2)))

    header = f"{'Metric':<{col_w}}" + "".join(f"{get_label(model, labels):<{col_w}}" for model in models) + f"{'Best':<{col_w}}"
    log.info(header)
    log.info("-" * len(header))

    wins = {model: 0 for model in models}
    for metric, criterion in metrics_cfg:
        vals = np.asarray([ext[model][metric] for model in models], dtype=float)
        if criterion == "max":
            score = np.where(np.isfinite(vals), vals, -np.inf)
            best_idx = int(np.argmax(score))
        elif criterion == "absmin":
            score = np.where(np.isfinite(vals), np.abs(vals), np.inf)
            best_idx = int(np.argmin(score))
        else:
            score = np.where(np.isfinite(vals), vals, np.inf)
            best_idx = int(np.argmin(score))
        wins[models[best_idx]] += 1
        row = f"{metric:<{col_w}}"
        row += "".join(f"{v:<{col_w}.6f}" for v in vals)
        row += f"{get_label(models[best_idx], labels):<{col_w}}"
        log.info(row)

    log.info("-" * len(header))
    log.info("\nWin summary:")
    for model in models:
        log.info(f"  {get_label(model, labels)}: {wins[model]} wins")
    overall = max(wins.items(), key=lambda item: item[1])[0]
    log.info(f"\n✓ Best overall: {get_label(overall, labels)} ({wins[overall]}/{len(metrics_cfg)} metrics)")

    # Improvement over first model (treated as baseline)
    baseline = models[0]
    if len(models) > 1:
        log.info(f"\nImprovement over {get_label(baseline, labels)}:")
        for model in models[1:]:
            log.info(f"\n  {get_label(model, labels)}:")
            for metric, criterion in metrics_cfg:
                bv, kv = ext[baseline][metric], ext[model][metric]
                if criterion == "max":
                    pct = ((kv - bv) / abs(bv)) * 100 if bv != 0 else 0.0
                    sym = "↑" if pct > 0 else "↓"
                elif criterion == "absmin":
                    b_abs, k_abs = abs(bv), abs(kv)
                    pct = ((b_abs - k_abs) / b_abs) * 100 if b_abs != 0 else 0.0
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


def _is_model_payload(obj: Any) -> bool:
    required = {"predictions", "targets", "sample_labels", "bin_labels"}
    return isinstance(obj, dict) and required.issubset(set(obj.keys()))


def _validate_model_payload(model_key: str, payload: Dict[str, Any]) -> None:
    required = ("predictions", "targets", "sample_labels", "bin_labels")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Model entry '{model_key}' is missing required keys: {missing}")

    preds = np.asarray(payload["predictions"])
    trues = np.asarray(payload["targets"])
    samples = np.asarray(payload["sample_labels"])
    bins = np.asarray(payload["bin_labels"])

    if preds.ndim != 1 or trues.ndim != 1:
        raise ValueError(
            f"Model entry '{model_key}' must use flat 1-D arrays for 'predictions' and 'targets'"
        )
    if preds.shape != trues.shape:
        raise ValueError(
            f"Model entry '{model_key}' has mismatched shapes: predictions {preds.shape}, targets {trues.shape}"
        )
    if samples.ndim != 1 or bins.ndim != 1:
        raise ValueError(
            f"Model entry '{model_key}' must use flat 1-D arrays for 'sample_labels' and 'bin_labels'"
        )
    if samples.shape[0] != preds.shape[0] or bins.shape[0] != preds.shape[0]:
        raise ValueError(
            f"Model entry '{model_key}' has inconsistent lengths among predictions/targets/sample_labels/bin_labels"
        )


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _load_one_pickle_as_results_dict(results_path: str) -> Dict[str, Any]:
    loaded = load_results(results_path)
    if not isinstance(loaded, dict):
        raise ValueError(f"Pickle must contain a dict: {results_path}")

    if _is_model_payload(loaded):
        return {_stem(results_path): loaded}

    # Standard case: combined dict keyed by model name.
    return loaded


def _merge_results_dicts(results_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for d in results_dicts:
        for key, value in d.items():
            if key in merged:
                raise ValueError(
                    f"Duplicate model key '{key}' while merging results. "
                    "Use --labels to rename display names or avoid duplicate file/model keys."
                )
            if not isinstance(value, dict):
                raise ValueError(f"Model entry '{key}' must be a dict")
            _validate_model_payload(key, value)
            merged[key] = value
    return merged


def load_results_multi(results_path: Optional[str], results_paths: Optional[List[str]]) -> Dict[str, Any]:
    """Load results from one file, many files, or a folder containing pickle files."""
    if not results_path and not results_paths:
        raise ValueError("Provide --results_path or --results_paths")

    candidate_files: List[str] = []

    if results_paths:
        candidate_files.extend(os.path.abspath(p) for p in results_paths)

    if results_path:
        abs_path = os.path.abspath(results_path)
        if os.path.isdir(abs_path):
            found = sorted(glob.glob(os.path.join(abs_path, "*.pkl")))
            if not found:
                raise ValueError(f"No .pkl files found in folder: {abs_path}")
            candidate_files.extend(found)
        else:
            candidate_files.append(abs_path)

    if not candidate_files:
        raise ValueError("No result pickle files to load")

    # De-duplicate while preserving order.
    seen = set()
    unique_files: List[str] = []
    for p in candidate_files:
        if p not in seen:
            seen.add(p)
            unique_files.append(p)

    for p in unique_files:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Results file not found: {p}")

    loaded = [_load_one_pickle_as_results_dict(p) for p in unique_files]
    merged = _merge_results_dicts(loaded)
    if not merged:
        raise ValueError("Loaded results are empty")
    return merged


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified visualization for model comparison results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results_path", required=False,
                        help="Path to one results pickle file OR a folder containing .pkl files.")
    parser.add_argument("--results_paths", nargs="+", default=None,
                        help="Explicit list of results pickle files to compare.")
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
    
    # Resolve output path relative to cwd
    output_dir = os.path.abspath(args.output_dir)

    if args.results_path:
        log.info(f"Loading results from --results_path={os.path.abspath(args.results_path)} ...")
    if args.results_paths:
        log.info(f"Loading results from --results_paths ({len(args.results_paths)} file(s)) ...")

    results = load_results_multi(args.results_path, args.results_paths)
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
