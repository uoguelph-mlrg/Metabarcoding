#!/usr/bin/env python
"""
Visualization module for model comparison results.
Creates clean, presentation-ready plots comparing the baseline model and the
interpolated-latent variant.

This script is separate from training to allow re-generating visualizations
without re-running the expensive training.

Usage:
    python interpolated_latent_visualisation.py --results_path results/model_comparison_results.pkl --output_dir figures
"""
from __future__ import annotations

import argparse
import os
import pickle
import logging as log
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def _contrasting_text_color(hex_color: str) -> str:
    """Return 'white' or 'black' for a hex background color."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    return 'white' if (0.2126*r + 0.7152*g + 0.0722*b) < 0.45 else 'black'
def _ci_tuple_to_errorbar(mean_val, ci_tuple):
    """Convert CI tuple (lower, upper) to error bar format [lower_err, upper_err]."""
    if isinstance(ci_tuple, tuple) and len(ci_tuple) == 2:
        ci_lower, ci_upper = ci_tuple
        return [mean_val - ci_lower, ci_upper - mean_val]
    return [0.0, 0.0]


def _format_count_label(value: Any) -> str:
    """Format a pandas/numpy scalar count for plot labels."""
    count = int(np.asarray(value).item())
    return f"{count:,}"



def compute_95ci_bootstrap(errors: np.ndarray, n_bootstrap: int = 1000) -> tuple:
    """Compute 95% confidence interval using bootstrap resampling.
    
    Returns:
        tuple: (lower, upper) percentiles (2.5, 97.5) of bootstrap distribution
    """
    n = len(errors)
    if n < 2:
        return (0.0, 0.0)
    
    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(errors, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # 95% CI from bootstrap distribution
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    return (ci_lower, ci_upper)


def compute_95ci_bootstrap_metric(values: np.ndarray) -> tuple:
    """Compute 95% CI with n_bootstrap = len(values).
    
    Used for KL Divergence and Correlation. Computes metric for each
    sample and takes 2.5/97.5 percentiles of the distribution.
    
    Returns:
        tuple: (lower, upper) percentiles (2.5, 97.5)
    """
    if len(values) < 2:
        return (0.0, 0.0)
    n = len(values)
    return compute_95ci_bootstrap(values, n_bootstrap=n)
    return 1.96 * np.std(errors, ddof=1) / np.sqrt(n)


# ============================================================================
# Style Configuration
# ============================================================================

# Model colors
LOSS_COLORS = {
    "baseline": "#1f6feb",              # Blue
    "interpolated_latent": "#d97706",   # Amber
}

LOSS_LABELS = {
    "baseline": "Baseline",
    "interpolated_latent": "Interpolated Latent",
}


def get_loss_label(loss_type: str) -> str:
    """Get display label for a model type."""
    return LOSS_LABELS.get(loss_type, loss_type)


def get_loss_color(loss_type: str) -> str:
    """Get consistent color for a model type."""
    return LOSS_COLORS.get(loss_type, "#333333")


def set_style():
    """Set clean, minimal style for all plots."""
    sns.set_theme(style="white", font_scale=1.1)
    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
    })


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_labels: Optional[np.ndarray] = None,
    bin_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive metrics with proper per-sample handling.
    
    If sample_labels is provided (new flat format), computes per-sample KL
    divergence and macro MAE/RMSE rigorously. Otherwise falls back to
    micro (global) metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse_macro: Optional[float] = None
    mae_macro: Optional[float] = None
    kl_divergence_macro: Optional[float] = None
    eps = 1e-10

    # ==================== Path A: flat 1D + sample_labels ====================
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

    # ==================== Path B: 2D NaN-padded (backward compat) ====================
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

    # ===================== Flatten for micro (global) metrics =====================
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    valid = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true = y_true_flat[valid]
    y_pred = np.clip(y_pred_flat[valid], 0, 1)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse_micro = float(np.sqrt(mse))
    mae_micro = float(np.mean(np.abs(y_true - y_pred)))

    # Fall back to micro when no grouping available
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

    # Micro KL — fallback when no sample grouping available
    y_tn = (y_true + eps) / (y_true + eps).sum()
    y_pn = (y_pred + eps) / (y_pred + eps).sum()
    kl_divergence_micro = float(np.sum(y_tn * np.log(y_tn / y_pn)))
    kl_divergence = kl_divergence_macro if kl_divergence_macro is not None else kl_divergence_micro

    corr = np.corrcoef(y_true, y_pred)[0, 1]
    correlation = 0.0 if np.isnan(corr) else float(corr)

    nz = y_true != 0
    rel_error = np.zeros_like(y_true, dtype=float)
    rel_error[nz] = np.abs(y_pred[nz] - y_true[nz]) / np.abs(y_true[nz])
    absolute_relative_error = float(np.mean(rel_error[nz])) if nz.sum() > 0 else 0.0

    return {
        'RMSE_micro': rmse_micro,
        'RMSE_macro': rmse_macro,
        'MAE_micro': mae_micro,
        'MAE_macro': mae_macro,
        'Absolute Relative Error': absolute_relative_error,
        'R²': r2,
        'RMSE (zeros)': rmse_zeros,
        'MAE (zeros)': mae_zeros,
        'RMSE (non-zeros)': rmse_nonzeros,
        'MAE (non-zeros)': mae_nonzeros,
        'KL Divergence': kl_divergence,
        'Correlation': correlation,
        'n_zeros': int(zero_mask.sum()),
        'n_nonzeros': int(nonzero_mask.sum()),
    }

def plot_metrics_comparison(results: Dict[str, Any], output_dir: str):
    """Create bar plots comparing key metrics between the two models."""
    set_style()
    
    loss_types = list(results.keys())
    
    # Compute extended metrics for both models
    extended_metrics = {}
    for lt in loss_types:
        preds = results[lt]["predictions"]
        targets = results[lt]["targets"]
        extended_metrics[lt] = compute_extended_metrics(
            targets, preds,
            sample_labels=results[lt].get("sample_labels"),
            bin_labels=results[lt].get("bin_labels"),
        )
    
    # Metrics to plot
    metrics_to_plot = ['RMSE_micro', 'MAE_micro', 'Absolute Relative Error', 'KL Divergence', 'MAE (zeros)', 'MAE (non-zeros)', 'Correlation']
    
    fig, axes = plt.subplots(1, 7, figsize=(24, 4))
    axes = axes.flatten()

    metric_cis: Dict[str, Dict[str, Any]] = {}
    for lt in loss_types:
        y_true = results[lt]["targets"].flatten()
        y_pred = results[lt]["predictions"].flatten()
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        if len(y_true) < 2:
            metric_cis[lt] = {k: None for k in metrics_to_plot}
            continue
        y_pred = np.clip(y_pred, 0, 1)
        abs_err = np.abs(y_true - y_pred)
        sq_err = (y_true - y_pred) ** 2
        zero_mask = y_true == 0
        nonzero_mask = y_true > 0
        nonzero_rel = y_true != 0
        rel_err = np.zeros_like(y_true)
        rel_err[nonzero_rel] = abs_err[nonzero_rel] / np.abs(y_true[nonzero_rel])
        rmse_ci_lower, rmse_ci_upper = compute_95ci_bootstrap(sq_err)
        rmse_ci_tuple = (np.sqrt(rmse_ci_lower), np.sqrt(rmse_ci_upper))
        mae_ci_tuple = compute_95ci_bootstrap(abs_err)
        are_ci_tuple = compute_95ci_bootstrap(rel_err[nonzero_rel]) if nonzero_rel.sum() > 1 else (0.0, 0.0)
        mae_zeros_ci_tuple = compute_95ci_bootstrap(abs_err[zero_mask]) if zero_mask.sum() > 1 else (0.0, 0.0)
        mae_nz_ci_tuple = compute_95ci_bootstrap(abs_err[nonzero_mask]) if nonzero_mask.sum() > 1 else (0.0, 0.0)

        metric_cis[lt] = {
            'RMSE_micro': rmse_ci_tuple,
            'MAE_micro': mae_ci_tuple,
            'Absolute Relative Error': are_ci_tuple,
            'KL Divergence': None,
            'MAE (zeros)': mae_zeros_ci_tuple,
            'MAE (non-zeros)': mae_nz_ci_tuple,
            'Correlation': None,
        }
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [extended_metrics[lt][metric] for lt in loss_types]
        ci_info = [metric_cis[lt].get(metric, None) for lt in loss_types]
        colors = [get_loss_color(lt) for lt in loss_types]
        labels = [get_loss_label(lt) for lt in loss_types]

        if metric in ['KL Divergence', 'Correlation']:
            yerr = None
        else:
            ci_lower = [ci[0] if isinstance(ci, tuple) else 0.0 for ci in ci_info]
            ci_upper = [ci[1] if isinstance(ci, tuple) else 0.0 for ci in ci_info]
            lower_err = np.abs(np.array(values) - np.array(ci_lower))
            upper_err = np.abs(np.array(ci_upper) - np.array(values))
            yerr = [lower_err, upper_err]
        
        bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5,
                      yerr=yerr, capsize=4, error_kw={'elinewidth': 1.5} if yerr is not None else None)
        
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        sns.despine(ax=ax)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved: metrics_comparison.png")


def plot_scatter_actual_vs_predicted(results: Dict[str, Any], output_dir: str):
    """Create scatter plots of actual vs predicted values with dynamic axis limits."""
    set_style()
    
    loss_types = list(results.keys())
    
    fig, axes = plt.subplots(1, len(loss_types), figsize=(7 * len(loss_types), 6))
    if len(loss_types) == 1:
        axes = [axes]
    
    # First pass: compute densities and find global max for axis limits
    all_densities = []
    scatter_data = []
    global_max_target = 0
    global_max_pred = 0
    
    for lt in loss_types:
        preds = results[lt]["predictions"].flatten()
        targets = results[lt]["targets"].flatten()
        
        # Filter out NaN values (padding)
        valid_mask = np.isfinite(preds) & np.isfinite(targets)
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        # Track global max for axis limits
        global_max_target = max(global_max_target, targets.max())
        global_max_pred = max(global_max_pred, preds.max())
        
        try:
            xy = np.vstack([preds, targets])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {lt}: {e}")
            density = np.ones(len(preds))
        
        all_densities.extend(density)
        scatter_data.append((lt, preds, targets, density))
    
    # Create shared normalization
    vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    sc = None
    for ax, (lt, preds, targets, density) in zip(axes, scatter_data):
        # Sort by density so densest points are plotted last
        idx = density.argsort()
        preds_sorted, targets_sorted, density_sorted = preds[idx], targets[idx], density[idx]
        
        sc = ax.scatter(targets_sorted, preds_sorted, c=density_sorted, 
                        cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        
        # Add diagonal line (perfect predictions)
        axis_max = max(global_max_target, global_max_pred)
        ax.plot([0, axis_max], [0, axis_max], 'r--', lw=1.5, alpha=0.7, label='Perfect')
        
        # Compute Pearson correlation
        corr = np.corrcoef(targets, preds)[0, 1]
        
        ax.set_xlabel("Actual", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title(f"{get_loss_label(lt)}\n(Pearson r = {corr:.3f})", fontsize=14, fontweight='bold')
        
        # Dynamic axis limits
        ax.set_xlim(0, global_max_target)
        ax.set_ylim(0, global_max_pred)
        
        sns.despine(ax=ax)
    
    # Add shared colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes((0.94, 0.15, 0.02, 0.7))
    if sc is not None:
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('Point Density', fontsize=10)
    
    plt.savefig(os.path.join(output_dir, "scatter_predicted_vs_actual.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved: scatter_predicted_vs_actual.png")


def plot_scatter_zoomed(results: Dict[str, Any], output_dir: str, max_actual: float = 0.01):
    """Create scatter plots zoomed on ground truth <1% range with dynamic axis limits."""
    set_style()
    
    loss_types = list(results.keys())
    
    fig, axes = plt.subplots(1, len(loss_types), figsize=(7 * len(loss_types), 6))
    if len(loss_types) == 1:
        axes = [axes]
    
    # First pass: compute densities and find max in zoomed region
    all_densities = []
    scatter_data = []
    global_max_target = 0
    global_max_pred = 0
    
    for lt in loss_types:
        targets = results[lt]["targets"].flatten()
        preds = results[lt]["predictions"].flatten()
        
        # Filter out NaN values (padding)
        valid_mask = np.isfinite(targets) & np.isfinite(preds)
        targets = targets[valid_mask]
        preds = preds[valid_mask]
        
        # Filter to actual values < max_actual
        zoom_mask = targets < max_actual
        targets_zoomed = targets[zoom_mask]
        preds_zoomed = preds[zoom_mask]
        
        # Track max for axis limits
        if len(targets_zoomed) > 0:
            global_max_target = max(global_max_target, targets_zoomed.max())
            global_max_pred = max(global_max_pred, preds_zoomed.max())
        
        try:
            xy = np.vstack([preds_zoomed, targets_zoomed])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {lt}: {e}")
            density = np.ones(len(preds_zoomed))
        
        all_densities.extend(density)
        scatter_data.append((lt, preds_zoomed, targets_zoomed, density))
    
    # Create shared normalization
    if len(all_densities) > 0:
        vmin, vmax = min(all_densities), max(all_densities)
    else:
        vmin, vmax = 0, 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    sc = None
    for ax, (lt, preds_zoomed, targets_zoomed, density) in zip(axes, scatter_data):
        # Sort by density so densest points are plotted last
        idx = density.argsort()
        preds_sorted, targets_sorted, density_sorted = preds_zoomed[idx], targets_zoomed[idx], density[idx]
        
        sc = ax.scatter(targets_sorted, preds_sorted, c=density_sorted, 
                        cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        
        # Add diagonal line
        axis_max = max(global_max_target, global_max_pred)
        ax.plot([0, axis_max], [0, axis_max], 'r--', lw=1.5, alpha=0.7)
        
        # Compute Pearson correlation on zoomed data
        corr = np.corrcoef(targets_zoomed, preds_zoomed)[0, 1]
        
        ax.set_xlabel("Actual", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title(f"{get_loss_label(lt)}\n(Pearson r = {corr:.3f})", fontsize=14, fontweight='bold')
        
        # Dynamic axis limits
        ax.set_xlim(0, global_max_target)
        ax.set_ylim(0, global_max_pred)
        
        sns.despine(ax=ax)
    
    # Add shared colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes((0.94, 0.15, 0.02, 0.7))
    if sc is not None:
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('Point Density', fontsize=10)
    
    plt.suptitle('Predicted vs Actual (Ground Truth <1%)', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_dir, "scatter_zoomed.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved: scatter_zoomed.png")
    
    
def plot_loglog_scatter_actual_vs_predicted(results: Dict[str, Any], output_dir: str):
    """Create log-log scatter plots of actual vs predicted values."""
    set_style()
    loss_types = list(results.keys())
    fig, axes = plt.subplots(1, len(loss_types), figsize=(7 * len(loss_types), 6))
    if len(loss_types) == 1:
        axes = [axes]

    # First pass: compute densities and find global min/max for color normalization
    all_densities = []
    scatter_data = []
    global_min = float('inf')
    global_max = float('-inf')

    for lt in loss_types:
        preds = results[lt]["predictions"].flatten()
        targets = results[lt]["targets"].flatten()
        
        # Filter out NaN values (padding)
        valid_mask = np.isfinite(preds) & np.isfinite(targets)
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        epsilon = 10**(-3.5)  # 1e-3.5
        preds_log = np.log10(preds + epsilon)
        targets_log = np.log10(targets + epsilon)
        # Compute density
        try:
            xy = np.vstack([targets_log, preds_log])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {lt}: {e}")
            density = np.ones(len(preds_log))
        all_densities.extend(density)
        scatter_data.append((lt, preds_log, targets_log, density))
        # Track min/max for axis
        global_min = min(global_min, np.min(targets_log), np.min(preds_log))
        global_max = max(global_max, np.max(targets_log), np.max(preds_log))

    # Create shared normalization for color
    vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)

    sc = None
    for ax, (lt, preds_log, targets_log, density) in zip(axes, scatter_data):
        # Sort by density so densest points are plotted last
        idx = density.argsort()
        preds_sorted, targets_sorted, density_sorted = preds_log[idx], targets_log[idx], density[idx]
        sc = ax.scatter(targets_sorted, preds_sorted, c=density_sorted, cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        # Diagonal line
        ax.plot([global_min, global_max], [global_min, global_max], 'r--', lw=1.5, alpha=0.7, label='Perfect')
        corr = np.corrcoef(targets_log, preds_log)[0, 1]
        ax.set_xlabel("Log10 Actual", fontsize=12)
        ax.set_ylabel("Log10 Predicted", fontsize=12)
        ax.set_title(f"{get_loss_label(lt)}\n(Log-Log Pearson r = {corr:.3f})", fontsize=14, fontweight='bold')
        sns.despine(ax=ax)

    # Add shared colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes((0.94, 0.15, 0.02, 0.7))
    if sc is not None:
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('Point Density', fontsize=10)

    plt.savefig(os.path.join(output_dir, "scatter_loglog_predicted_vs_actual.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved: scatter_loglog_predicted_vs_actual.png")



def plot_mae_per_range(results: Dict[str, Any], output_dir: str):
    """Create bar plot of MAE per target value range using specified bins."""
    set_style()
    
    loss_types = list(results.keys())
    
    bins = [
        ('zero', 'Zero'),
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%'),
    ]
    error_data = []
    for lt in loss_types:
        y_pred = results[lt]["predictions"].flatten()
        y_true = results[lt]["targets"].flatten()
        valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        for bin_def in bins:
            if bin_def[0] == 'zero':
                mask = y_true == 0
                range_label = bin_def[1]
            else:
                low, high, range_label = bin_def
                mask = (y_true > low) & (y_true <= high)
            if mask.sum() > 0:
                abs_errors = np.abs(y_true[mask] - y_pred[mask])
                mae = np.mean(abs_errors)
                ci_lower, ci_upper = compute_95ci_bootstrap(abs_errors) if mask.sum() > 1 else (0.0, 0.0)
                error_data.append({'Loss': get_loss_label(lt), 'Range': range_label,
                                   'MAE': mae, 'MAE_CI_Lower': ci_lower,
                    'MAE_CI_Upper': ci_upper, 'Count': int(mask.sum()), 'loss_key': lt})
    error_df = pd.DataFrame(error_data)
    if len(error_df) == 0:
        log.warning("No data for MAE per range plot")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = error_df.pivot(index='Range', columns='Loss', values='MAE')
    pivot_ci_lower = error_df.pivot(index='Range', columns='Loss', values='MAE_CI_Lower')
    pivot_ci_upper = error_df.pivot(index='Range', columns='Loss', values='MAE_CI_Upper')
    pivot_count_df = error_df.pivot(index='Range', columns='Loss', values='Count')
    count_df = error_df[error_df['Loss'] == get_loss_label(loss_types[0])][['Range', 'Count']].set_index('Range')
    range_order = ['Zero', '>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_ci_lower = pivot_ci_lower.reindex(range_order)
    pivot_ci_upper = pivot_ci_upper.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    loss_labels_ordered = [get_loss_label(lt) for lt in loss_types]
    pivot_df = pivot_df[[col for col in loss_labels_ordered if col in pivot_df.columns]]
    pivot_ci_lower = pivot_ci_lower[[col for col in loss_labels_ordered if col in pivot_ci_lower.columns]]
    pivot_ci_upper = pivot_ci_upper[[col for col in loss_labels_ordered if col in pivot_ci_upper.columns]]
    pivot_count_df = pivot_count_df[[col for col in loss_labels_ordered if col in pivot_count_df.columns]]

    mean_vals = pivot_df.values
    pivot_ci_lower = pivot_ci_lower.fillna(0)
    pivot_ci_upper = pivot_ci_upper.fillna(0)
    lower_err = np.abs(mean_vals - pivot_ci_lower.values)
    upper_err = np.abs(pivot_ci_upper.values - mean_vals)

    colors = [get_loss_color(lt) for lt in loss_types]
    x = np.arange(len(range_order))
    width = 0.7 / len(pivot_df.columns)
    for i, col in enumerate(pivot_df.columns):
        offset = (i - len(pivot_df.columns) / 2 + 0.5) * width
        ax.bar(x + offset, pivot_df[col], width, label=col, color=colors[i],
               edgecolor='white', yerr=[lower_err[:, i], upper_err[:, i]],
               capsize=5, error_kw={'elinewidth': 2})
    range_labels_with_counts = [f"{r}\n(n={_format_count_label(count_df.loc[r, 'Count'])})" if r in count_df.index else r
                                 for r in range_order]
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Abundance Range', fontsize=14, fontweight='bold')
    ax.legend(title='Model', frameon=False, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_range.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved: error_by_range.png")
    

def plot_RAE_per_range(results: Dict[str, Any], output_dir: str):
    """Create bar plot of Relative Absolute Error per range (excluding zeros)."""
    set_style()
    loss_types = list(results.keys())
    bins = [
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%'),
    ]
    error_data = []
    for lt in loss_types:
        y_pred = results[lt]["predictions"].flatten()
        y_true = results[lt]["targets"].flatten()
        valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        for low, high, range_label in bins:
            mask = (y_true > low) & (y_true <= high) & (y_true != 0)
            if mask.sum() > 0:
                rel_error = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
                rae = np.mean(rel_error)
                ci_lower, ci_upper = compute_95ci_bootstrap(rel_error) if mask.sum() > 1 else (0.0, 0.0)
                error_data.append({'Loss': get_loss_label(lt), 'Range': range_label,
                                   'RAE': rae, 'RAE_CI_Lower': ci_lower,
                    'RAE_CI_Upper': ci_upper, 'Count': int(mask.sum()), 'loss_key': lt})
    error_df = pd.DataFrame(error_data)
    if len(error_df) == 0:
        log.warning("No data for RAE per range plot")
        return
    count_df = error_df[error_df['Loss'] == get_loss_label(loss_types[0])][['Range', 'Count']].set_index('Range')
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = error_df.pivot(index='Range', columns='Loss', values='RAE')
    pivot_ci_lower = error_df.pivot(index='Range', columns='Loss', values='RAE_CI_Lower')
    pivot_ci_upper = error_df.pivot(index='Range', columns='Loss', values='RAE_CI_Upper')
    pivot_count_df = error_df.pivot(index='Range', columns='Loss', values='Count')
    range_order = ['>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_ci_lower = pivot_ci_lower.reindex(range_order)
    pivot_ci_upper = pivot_ci_upper.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    loss_labels_ordered = [get_loss_label(lt) for lt in loss_types]
    pivot_df = pivot_df[[col for col in loss_labels_ordered if col in pivot_df.columns]]
    pivot_ci_lower = pivot_ci_lower[[col for col in loss_labels_ordered if col in pivot_ci_lower.columns]]
    pivot_ci_upper = pivot_ci_upper[[col for col in loss_labels_ordered if col in pivot_ci_upper.columns]]
    pivot_count_df = pivot_count_df[[col for col in loss_labels_ordered if col in pivot_count_df.columns]]

    mean_vals = pivot_df.values
    pivot_ci_lower = pivot_ci_lower.fillna(0)
    pivot_ci_upper = pivot_ci_upper.fillna(0)
    lower_err = np.abs(mean_vals - pivot_ci_lower.values)
    upper_err = np.abs(pivot_ci_upper.values - mean_vals)

    colors = [get_loss_color(lt) for lt in loss_types]
    x = np.arange(len(range_order))
    width = 0.7 / len(pivot_df.columns)
    for i, col in enumerate(pivot_df.columns):
        offset = (i - len(pivot_df.columns) / 2 + 0.5) * width
        ax.bar(x + offset, pivot_df[col], width, label=col, color=colors[i],
               edgecolor='white', yerr=[lower_err[:, i], upper_err[:, i]],
               capsize=5, error_kw={'elinewidth': 2})
    range_labels_with_counts = [f"{r}\n(n={_format_count_label(count_df.loc[r, 'Count'])})" if r in count_df.index else r
                                 for r in range_order]
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Relative Absolute Error', fontsize=12)
    ax.set_title('Relative Absolute Error by Abundance Range', fontsize=14, fontweight='bold')
    ax.legend(title='Model', frameon=False, loc='upper left')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "relative_err_by_range.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved: relative_err_by_range.png")

def plot_mae_per_range_zoomed(results: Dict[str, Any], output_dir: str):
    """Create bar plot of MAE per range zoomed with quantile bins and sample counts."""
    set_style()
    
    loss_types = list(results.keys())
    
    # Zoomed bins for <1% range
    bins = [
        ('zero', 'Zero'),
        (0, 0.0011, '0-0.11%'),
        (0.0011, 0.0015, '0.11-0.15%'),
        (0.0015, 0.0022, '0.15-0.22%'),
        (0.0022, 0.01, '0.22-1%'),
    ]
    
    # Compute MAE for each range
    error_data = []
    
    for lt in loss_types:
        y_pred = results[lt]["predictions"].flatten()
        y_true = results[lt]["targets"].flatten()
        
        # Filter out NaN values (padding)
        valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        
        for bin_def in bins:
            if bin_def[0] == 'zero':
                mask = y_true == 0
                range_label = bin_def[1]
            else:
                low, high, range_label = bin_def
                mask = (y_true > low) & (y_true <= high)
            if mask.sum() > 0:
                abs_errors = np.abs(y_true[mask] - y_pred[mask])
                mae = np.mean(abs_errors)
                ci_lower, ci_upper = compute_95ci_bootstrap(abs_errors) if mask.sum() > 1 else (0.0, 0.0)
                error_data.append({'Loss': get_loss_label(lt), 'Range': range_label,
                                   'MAE': mae, 'MAE_CI_Lower': ci_lower,
                    'MAE_CI_Upper': ci_upper, 'Count': int(mask.sum()), 'loss_key': lt})
    error_df = pd.DataFrame(error_data)
    if len(error_df) == 0:
        log.warning("No data for zoomed MAE per range plot")
        return
    count_df = error_df[error_df['Loss'] == get_loss_label(loss_types[0])][['Range', 'Count']].set_index('Range')
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = error_df.pivot(index='Range', columns='Loss', values='MAE')
    pivot_ci_lower = error_df.pivot(index='Range', columns='Loss', values='MAE_CI_Lower')
    pivot_ci_upper = error_df.pivot(index='Range', columns='Loss', values='MAE_CI_Upper')
    pivot_count_df = error_df.pivot(index='Range', columns='Loss', values='Count')
    range_order = ['Zero', '0-0.11%', '0.11-0.15%', '0.15-0.22%', '0.22-1%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_ci_lower = pivot_ci_lower.reindex(range_order)
    pivot_ci_upper = pivot_ci_upper.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    loss_labels_ordered = [get_loss_label(lt) for lt in loss_types]
    pivot_df = pivot_df[[col for col in loss_labels_ordered if col in pivot_df.columns]]
    pivot_ci_lower = pivot_ci_lower[[col for col in loss_labels_ordered if col in pivot_ci_lower.columns]]
    pivot_ci_upper = pivot_ci_upper[[col for col in loss_labels_ordered if col in pivot_ci_upper.columns]]
    pivot_count_df = pivot_count_df[[col for col in loss_labels_ordered if col in pivot_count_df.columns]]

    mean_vals = pivot_df.values
    pivot_ci_lower = pivot_ci_lower.fillna(0)
    pivot_ci_upper = pivot_ci_upper.fillna(0)
    lower_err = np.abs(mean_vals - pivot_ci_lower.values)
    upper_err = np.abs(pivot_ci_upper.values - mean_vals)

    colors = [get_loss_color(lt) for lt in loss_types]
    x = np.arange(len(range_order))
    width = 0.7 / len(pivot_df.columns)
    for i, col in enumerate(pivot_df.columns):
        offset = (i - len(pivot_df.columns) / 2 + 0.5) * width
        ax.bar(x + offset, pivot_df[col], width, label=col, color=colors[i],
               edgecolor='white', yerr=[lower_err[:, i], upper_err[:, i]],
               capsize=5, error_kw={'elinewidth': 2})
    range_labels_with_counts = [f"{r}\n(n={_format_count_label(count_df.loc[r, 'Count'])})" if r in count_df.index else r
                                 for r in range_order]
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Abundance Range (Zoomed: <1%)', fontsize=14, fontweight='bold')
    ax.legend(title='Model', frameon=False, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_range_zoomed.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved: error_by_range_zoomed.png")


def plot_residual_distribution(results: Dict[str, Any], output_dir: str):
    """Create overlapping residual distribution histograms with KDE for both models."""
    set_style()
    loss_types = list(results.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    all_res_flat = []
    lt_residuals = {}
    for lt in loss_types:
        targets = results[lt]["targets"].flatten()
        preds = results[lt]["predictions"].flatten()
        valid_mask = np.isfinite(targets) & np.isfinite(preds)
        residuals = (targets - preds)[valid_mask]
        lt_residuals[lt] = residuals
        all_res_flat.extend(residuals.tolist())
    if len(all_res_flat) == 0:
        log.warning("No residuals available for plotting")
        return
    x_kde = np.linspace(min(all_res_flat), max(all_res_flat), 300)
    legend_handles = []
    max_count = 0
    for lt, residuals in lt_residuals.items():
        color = get_loss_color(lt)
        label = get_loss_label(lt)
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        counts, _, _ = ax.hist(residuals, bins=60, color=color, alpha=0.3, edgecolor='none')
        counts = np.asarray(counts)
        max_count = max(max_count, float(np.max(counts)))
        if len(residuals) > 1:
            kde = gaussian_kde(residuals)
            kde_vals = kde(x_kde)
            ax.plot(x_kde, kde_vals, color=color, linewidth=2)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2,
                              label=f'{label} (μ={mean_res:.4f}, σ={std_res:.4f})'))
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(0.5, max_count * 1.2)
    ax.set_title('Residual Distributions', fontsize=13, fontweight='bold')
    ax.legend(handles=legend_handles, frameon=False, fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved: residual_distribution.png")


def plot_zero_vs_nonzero_comparison(results: Dict[str, Any], output_dir: str):
    """Create MAE bar chart comparing zero vs non-zero values."""
    set_style()
    loss_types = list(results.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(loss_types))
    width = 0.35
    def _blend_with_white(color, alpha=0.6):
        r, g, b = mc.to_rgb(color)
        return (r * (1 - alpha) + alpha, g * (1 - alpha) + alpha, b * (1 - alpha) + alpha)
    for i, lt in enumerate(loss_types):
        targets = results[lt]["targets"].flatten()
        preds = results[lt]["predictions"].flatten()
        valid_mask = np.isfinite(targets) & np.isfinite(preds)
        targets = targets[valid_mask]
        preds = preds[valid_mask]
        zero_mask = targets == 0
        nonzero_mask = targets != 0
        errs_z = np.abs(targets[zero_mask] - preds[zero_mask]) if zero_mask.sum() > 0 else np.array([0.0])
        errs_nz = np.abs(targets[nonzero_mask] - preds[nonzero_mask]) if nonzero_mask.sum() > 0 else np.array([0.0])
        mae_z = np.mean(errs_z)
        mae_nz = np.mean(errs_nz)
        base_color = get_loss_color(lt)
        zero_color = _blend_with_white(base_color, alpha=0.6)
        nonzero_color = base_color
        ci_z = _ci_tuple_to_errorbar(mae_z, compute_95ci_bootstrap(errs_z))
        ci_nz = _ci_tuple_to_errorbar(mae_nz, compute_95ci_bootstrap(errs_nz))
        # Reshape error bars for single bar plotting
        yerr_z = [[ci_z[0]], [ci_z[1]]] if ci_z else None
        yerr_nz = [[ci_nz[0]], [ci_nz[1]]] if ci_nz else None
        ax.bar(x[i] - width/2, mae_z, width, color=zero_color, edgecolor='white',
               yerr=yerr_z, capsize=4, error_kw={'elinewidth': 1.5})
        ax.bar(x[i] + width/2, mae_nz, width, color=nonzero_color, edgecolor='white',
               yerr=yerr_nz, capsize=4, error_kw={'elinewidth': 1.5})
    # Generic legend colors (lighter and darker gray)
    legend_elements = [
        Patch(facecolor='#cccccc', edgecolor='white', label='Zero GT (lighter)'),
        Patch(facecolor='#666666', edgecolor='white', label='Non-zero GT (darker)')
    ]
    ax.legend(handles=legend_elements, frameon=False, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels([get_loss_label(lt) for lt in loss_types], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE: Zero vs Non-Zero Values', fontsize=14, fontweight='bold')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zero_vs_nonzero_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved: zero_vs_nonzero_comparison.png")


def plot_training_progress_comparison(results: Dict[str, Any], output_dir: str) -> None:
    """Plot overall training/validation loss evolution and end-of-cycle summaries for both models."""
    set_style()

    def extract_timeline(model_results: Dict[str, Any]) -> Tuple[List[float], List[float]]:
        timeline_train = model_results.get("timeline_train_losses")
        timeline_val = model_results.get("timeline_val_losses")

        if timeline_train and timeline_val:
            train_vals = [l for _, _, _, l in timeline_train]
            val_vals = [l for _, _, _, l in timeline_val]
            return train_vals, val_vals

        train_vals = [l for _, _, l in model_results.get("train_losses", [])]
        val_vals = [l for _, _, l in model_results.get("val_losses", [])]
        return train_vals, val_vals


    # Plot each model's training/validation loss on a separate subplot for direct comparison
    model_keys = list(results.keys())
    n_models = len(model_keys)
    fig, axes = plt.subplots(2, n_models, figsize=(8 * n_models, 10), sharey='row')
    if n_models == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for idx, model_key in enumerate(model_keys):
        label = get_loss_label(model_key)
        color = get_loss_color(model_key)
        train_vals, val_vals = extract_timeline(results[model_key])
        cycle_train = results[model_key].get("cycle_train_losses", [])
        cycle_val = results[model_key].get("cycle_val_losses", [])
        cfg = results[model_key].get("config", {})
        # Top row: full loss evolution with vertical lines for EM cycles
        ax1 = axes[0, idx]
        ax1.plot(list(range(len(train_vals))), train_vals, color=color, linestyle='-', linewidth=1.6, label=f"Train", alpha=0.85)
        ax1.plot(list(range(len(val_vals))), val_vals, color=color, linestyle='--', linewidth=1.6, label=f"Val", alpha=0.85)
        # Add vertical lines for EM cycles
        if cycle_train:
            # Try to infer EM cycle boundaries from timeline
            em_boundaries = []
            last_idx = 0
            for i, (phase, cycle, step, loss) in enumerate(results[model_key].get("timeline_train_losses", [])):
                if phase == "latent" and i > 0:
                    em_boundaries.append(i)
            for boundary in em_boundaries:
                ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax1.set_xlabel('Training Step (Latent + MLP)')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{label}: Loss Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=True, fontsize=10, loc='upper right')

        # Bottom row: End-of-cycle loss curves
        ax2 = axes[1, idx]
        if cycle_train:
            cycles = [c + 1 for c, _ in cycle_train]
            train_cycle_vals = [l for _, l in cycle_train]
            ax2.plot(cycles, train_cycle_vals, color=color, linestyle='-', marker='o', linewidth=1.8, markersize=6, label=f"Train", alpha=0.9)
        if cycle_val:
            cycles = [c + 1 for c, _ in cycle_val]
            val_cycle_vals = [l for _, l in cycle_val]
            ax2.plot(cycles, val_cycle_vals, color=color, linestyle='--', marker='o', linewidth=1.8, markersize=6, label=f"Val", alpha=0.9)
        ax2.set_xlabel('EM Cycle')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{label}: End-of-Cycle Losses')
        ax2.grid(True, alpha=0.3)
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            ax2.legend(frameon=True, fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  ✓ Saved: training_progress.png")


def plot_summary_table(results: Dict[str, Any], output_dir: str):
    """Create a clean summary table as an image with best values highlighted."""
    set_style()
    
    loss_types = list(results.keys())
    
    # Compute extended metrics
    extended_metrics = {}
    for lt in loss_types:
        preds = results[lt]["predictions"]
        targets = results[lt]["targets"]
        extended_metrics[lt] = compute_extended_metrics(
            targets, preds,
            sample_labels=results[lt].get("sample_labels"),
            bin_labels=results[lt].get("bin_labels"),
        )
    
    # Metrics to include in table
    metrics = ['RMSE_micro', 'MAE_micro', 'Absolute Relative Error', 'KL Divergence', 'MAE (zeros)', 'MAE (non-zeros)', 'Correlation']
    
    # Build table data
    data = []
    for lt in loss_types:
        row = {'Model': get_loss_label(lt)}
        for m in metrics:
            row[m] = extended_metrics[lt][m]
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Determine best values for each metric (lower is better except Correlation)
    best_indices = {}  # col -> list of row indices with best value
    for col in metrics:
        if col == 'Correlation':
            best_value = df[col].max()
            best_indices[col] = df[df[col] == best_value].index.tolist()
        else:
            best_value = df[col].min()
            best_indices[col] = df[df[col] == best_value].index.tolist()
    
    # Format numeric columns
    display_df = df.copy()
    for col in metrics:
        display_df[col] = display_df[col].apply(lambda x: f'{x:.6f}')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 2.5))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=display_df.astype(str).values.tolist(),
        colLabels=list(display_df.columns),
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(display_df.columns)
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header row: light gray bold
    for i in range(len(display_df.columns)):
        table[(0, i)].set_text_props(fontweight='bold')
        table[(0, i)].set_facecolor('#d0d0d0')

    # Color first column (Model) with each model's color
    model_col_idx = list(display_df.columns).index('Model')
    for row_idx, lt in enumerate(loss_types):
        cell = table[(row_idx + 1, model_col_idx)]
        hex_color = get_loss_color(lt)
        cell.set_facecolor(hex_color)
        cell.set_text_props(fontweight='bold', color='white')
        # Prevent title overlap by adjusting cell height
        cell.set_height(0.15)

    # Best value: bold dark green text + light green background
    for col_idx, col in enumerate(display_df.columns):
        if col == 'Model':
            continue
        if col in best_indices:
            # Highlight all rows that have the best value for this metric
            for best_row_idx in best_indices[col]:
                table[(best_row_idx + 1, col_idx)].set_facecolor('#d5f5e3')
                table[(best_row_idx + 1, col_idx)].set_text_props(fontweight='bold', color='#1a7a40')

    plt.title('Model Comparison Summary', fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved: summary_table.png")
    
    # Also save as CSV
    df.to_csv(os.path.join(output_dir, "model_comparison_results.csv"), index=False)
    log.info(f"  ✓ Saved: model_comparison_results.csv")
    
    return df


def plot_latent_comparison(results: Dict[str, Any], output_dir: str) -> None:
    """Compare learned latent vectors between baseline and interpolated-latent models."""
    set_style()

    baseline_lv = results.get("baseline", {}).get("latent_vector")
    interp_lv = results.get("interpolated_latent", {}).get("latent_vector")

    if baseline_lv is None and interp_lv is None:
        log.warning("No latent vectors found; skipping latent comparison plot")
        return

    baseline_flat = np.asarray(baseline_lv).flatten() if baseline_lv is not None else None
    interp_flat = np.asarray(interp_lv).flatten() if interp_lv is not None else None

    shapes_match = (
        baseline_flat is not None
        and interp_flat is not None
        and baseline_flat.shape == interp_flat.shape
    )

    n_panels = 1 + (1 if shapes_match else 0) + (1 if shapes_match else 0)
    if baseline_flat is not None and interp_flat is not None and not shapes_match:
        n_panels = 2

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    base_color = get_loss_color("baseline")
    interp_color = get_loss_color("interpolated_latent")
    base_label = get_loss_label("baseline")
    interp_label = get_loss_label("interpolated_latent")

    panel = 0

    # Panel 1: overlapping latent value distributions
    ax = axes[panel]
    panel += 1
    if baseline_flat is not None:
        sns.histplot(
            baseline_flat, bins=50, kde=True, stat="density",
            color=base_color, alpha=0.35, edgecolor="none", ax=ax, label=base_label,
        )
    if interp_flat is not None:
        sns.histplot(
            interp_flat, bins=50, kde=True, stat="density",
            color=interp_color, alpha=0.35, edgecolor="none", ax=ax, label=interp_label,
        )
    ax.set_xlabel("Latent value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Latent Value Distributions", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    for lv, color in [(baseline_flat, base_color), (interp_flat, interp_color)]:
        if lv is not None:
            ax.axvline(lv.mean(), color=color, linestyle="--", linewidth=1.5, alpha=0.8)
    sns.despine(ax=ax)

    if shapes_match:
        assert baseline_flat is not None and interp_flat is not None
        diff = interp_flat - baseline_flat

        # Panel 2: difference distribution
        ax = axes[panel]
        panel += 1
        sns.histplot(
            diff, bins=50, kde=True, stat="density",
            color="#4a90d9", alpha=0.7, edgecolor="none", ax=ax,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Zero")
        ax.axvline(diff.mean(), color="#e74c3c", linestyle="-", linewidth=1.5, alpha=0.9,
                   label=f"Mean = {diff.mean():.4f}")
        ax.set_xlabel(f"Latent difference ({interp_label} - {base_label})", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Per-BIN Latent Difference Distribution", fontsize=13, fontweight="bold")
        n_pos = int((diff > 0).sum())
        n_neg = int((diff < 0).sum())
        ax.text(
            0.97, 0.95,
            f"std = {diff.std():.4f}\n"
            f"pos: {n_pos} ({100*n_pos/len(diff):.1f}%)\n"
            f"neg: {n_neg} ({100*n_neg/len(diff):.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )
        ax.legend(frameon=False, fontsize=9)
        sns.despine(ax=ax)

        # Panel 3: per-bin scatter with density
        ax = axes[panel]
        panel += 1
        try:
            xy = np.vstack([baseline_flat, interp_flat])
            xy = xy + np.random.default_rng(0).normal(0, 1e-9, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception:
            density = np.ones(len(baseline_flat))
        idx_sort = density.argsort()
        sc = ax.scatter(
            baseline_flat[idx_sort], interp_flat[idx_sort],
            c=density[idx_sort], cmap="viridis", s=6, alpha=0.6, edgecolors="none",
        )
        vmin = min(baseline_flat.min(), interp_flat.min())
        vmax = max(baseline_flat.max(), interp_flat.max())
        ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.5, alpha=0.7, label="y = x")
        corr = float(np.corrcoef(baseline_flat, interp_flat)[0, 1])
        ax.set_xlabel(f"{base_label} latent", fontsize=11)
        ax.set_ylabel(f"{interp_label} latent", fontsize=11)
        ax.set_title(f"Per-BIN Latent Scatter\n(Pearson r = {corr:.3f})", fontsize=13, fontweight="bold")
        plt.colorbar(sc, ax=ax, label="Point density", fraction=0.046, pad=0.04)
        ax.legend(frameon=False, fontsize=9)
        sns.despine(ax=ax)

    elif baseline_flat is not None and interp_flat is not None and not shapes_match:
        ax = axes[panel]
        panel += 1
        log.warning(
            f"Latent vectors have different lengths "
            f"({len(baseline_flat)} vs {len(interp_flat)}); skipping diff/scatter panels"
        )
        ax.text(
            0.5, 0.5,
            f"Shapes differ:\n{base_label}: {baseline_flat.shape}\n{interp_label}: {interp_flat.shape}",
            ha="center", va="center", fontsize=11, transform=ax.transAxes,
        )
        ax.set_axis_off()

    fig.suptitle("Latent Vector Comparison: Baseline vs Interpolated-Latent", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✓ Saved: latent_comparison.png")


def create_all_visualizations(results: Dict[str, Any], output_dir: str):
    """Create all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    log.info("\n" + "="*60)
    log.info("CREATING VISUALIZATIONS")
    log.info("="*60)
    
    # 1. Bar plots for key metrics
    log.info("\n1. Creating metric comparison plots...")
    plot_metrics_comparison(results, output_dir)
    
    # 2. Scatter plots: predicted vs actual (with density colormap)
    log.info("2. Creating scatter plots (full range)...")
    plot_scatter_actual_vs_predicted(results, output_dir)
    
    # 3. Scatter plots zoomed on <1% ground truth
    log.info("3. Creating zoomed scatter plots (<1%)...")
    plot_scatter_zoomed(results, output_dir)
    
    # 3b. Scatter plots on log-log scale
    log.info("3b. Creating log-log scatter plots...")
    plot_loglog_scatter_actual_vs_predicted(results, output_dir)
    
    # 4. Error by range (standard bins)
    log.info("4. Creating error by range plot...")
    plot_mae_per_range(results, output_dir)
    
    # 5. Error by range (zoomed quantile bins with counts)
    log.info("5. Creating zoomed error by range plot...")
    plot_mae_per_range_zoomed(results, output_dir)
    
    # 5b. Relative Absolute Error by range (standard bins)
    log.info("5b. Creating relative absolute error by range plot...")
    plot_RAE_per_range(results, output_dir)
    
    # 6. Residual distribution
    log.info("6. Creating residual distribution plot...")
    plot_residual_distribution(results, output_dir)
    
    # 7. Zero vs non-zero comparison
    log.info("7. Creating zero vs non-zero comparison...")
    plot_zero_vs_nonzero_comparison(results, output_dir)
    
    # 8. Training progress comparison
    log.info("8. Creating training progress comparison plot...")
    plot_training_progress_comparison(results, output_dir)

    # 9. Summary table
    log.info("9. Creating summary table...")
    results_df = plot_summary_table(results, output_dir)

    # 10. Latent vector comparison
    log.info("10. Creating latent vector comparison...")
    plot_latent_comparison(results, output_dir)
    
    log.info(f"\n✅ All visualizations saved to: {output_dir}/")
    log.info("   - metrics_comparison.png")
    log.info("   - scatter_predicted_vs_actual.png")
    log.info("   - scatter_zoomed.png")
    log.info("   - scatter_loglog_predicted_vs_actual.png")
    log.info("   - error_by_range.png")
    log.info("   - error_by_range_zoomed.png")
    log.info("   - relative_err_by_range.png")
    log.info("   - residual_distribution.png")
    log.info("   - zero_vs_nonzero_comparison.png")
    log.info("   - training_progress.png")
    log.info("   - summary_table.png")
    log.info("   - model_comparison_results.csv")
    log.info("   - latent_comparison.png")


def print_comparison(results: Dict[str, Any]):
    """Print comparison table to console."""
    log.info("\n" + "="*80)
    log.info("MODEL COMPARISON RESULTS")
    log.info("="*80)
    
    loss_types = list(results.keys())
    
    # Compute extended metrics
    extended_metrics = {}
    for lt in loss_types:
        preds = results[lt]["predictions"]
        targets = results[lt]["targets"]
        extended_metrics[lt] = compute_extended_metrics(
            targets, preds,
            sample_labels=results[lt].get("sample_labels"),
            bin_labels=results[lt].get("bin_labels"),
        )
    
    # Print comparison table
    headers = ["Metric", get_loss_label(loss_types[0]), get_loss_label(loss_types[1]), "Winner"]
    row_format = "{:<25} {:<18} {:<18} {:<15}"
    
    log.info(row_format.format(*headers))
    log.info("-" * 76)
    
    metrics_to_compare = [
        ("RMSE_micro", "RMSE_micro", False),
        ("MAE_micro", "MAE_micro", False),
        ("MAE_macro", "MAE_macro", False),
        ("Absolute Relative Error", "Absolute Relative Error", False),
        ("KL Divergence", "KL Divergence", False),
        ("MAE (zeros)", "MAE (zeros)", False),
        ("MAE (non-zeros)", "MAE (non-zeros)", False),
        ("Correlation", "Correlation", True),
    ]
    
    wins = {lt: 0 for lt in loss_types}
    
    for metric_name, metric_key, higher_better in metrics_to_compare:
        val1 = extended_metrics[loss_types[0]][metric_key]
        val2 = extended_metrics[loss_types[1]][metric_key]
        
        if higher_better:
            winner = loss_types[0] if val1 > val2 else loss_types[1]
        else:
            winner = loss_types[0] if val1 < val2 else loss_types[1]
        
        wins[winner] += 1
        
        log.info(row_format.format(
            metric_name,
            f"{val1:.6f}",
            f"{val2:.6f}",
            get_loss_label(winner)
        ))
    
    log.info("-" * 76)
    log.info(f"\nOverall: {get_loss_label(loss_types[0])} wins {wins[loss_types[0]]}, "
             f"{get_loss_label(loss_types[1])} wins {wins[loss_types[1]]}")
    
    overall_winner = max(wins, key=lambda k: wins[k])
    log.info(f"Recommended model: {get_loss_label(overall_winner)}")


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from pickle file."""
    with open(results_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Model Comparison Results")
    parser.add_argument("--results_path", type=str, default="results/model_comparison_results.pkl",
                        help="Path to results pickle file")
    parser.add_argument("--output_dir", type=str, default="figures", 
                        help="Output directory for plots")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    # Setup logging
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = args.results_path
    if not os.path.isabs(results_path):
        results_path = os.path.join(script_dir, results_path)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)

    log.info(f"Loading results from {results_path}...")
    results = load_results(results_path)
    
    # Print comparison
    print_comparison(results)
    
    # Create visualizations
    create_all_visualizations(results, output_dir)
