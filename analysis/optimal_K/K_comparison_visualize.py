#!/usr/bin/env python
"""
Visualization module for K comparison results.
Creates clean, presentation-ready plots comparing K=13 vs K=78.
Creates clean, presentation-ready plots comparing K=13 vs K=197 (or any optimal K).

This script is separate from K_comparison.py to allow re-generating visualizations
without re-running the expensive training.

K_COLORS = {
    "K=13": "#3498db",   # Blue
    "K=78": "#e67e22",   # Orange
    "K=197": "#e67e22",  # Orange (or pick a new color if desired)
}
K_LABELS = {
    "K=13": "K=13 (Previous)",
    "K=78": "K=78 (Old Optimal)",
    "K=197": "K=197 (Optimal)"
}
"""
from __future__ import annotations
import argparse
import os
import logging as log
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.patches import Patch


def _contrasting_text_color(hex_color: str) -> str:
    """Return 'white' or 'black' for a hex background color."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    return 'white' if (0.2126*r + 0.7152*g + 0.0722*b) < 0.45 else 'black'


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


# Style Configuration
K_COLORS = {
    "K=13": "#3498db",   # Blue
    "K=78": "#e67e22",   # Orange
}
K_LABELS = {
    "K=13": "K=13",
    "K=78": "K=78 (Optimal)"
}

def get_k_label(k_type: str) -> str:
    return K_LABELS.get(k_type, k_type)


# Dynamically assign colors for any K value
def get_dynamic_k_colors(k_types):
    base_colors = [
        '#3498db', # Blue
        '#e67e22', # Orange
        '#ff6f61', # Red
        '#1b9e77', # Green
        '#f9a602', # Yellow/Orange
        '#9b59b6', # Purple
        '#e84393', # Pink
        '#00b894', # Teal
    ]
    return {kt: base_colors[i % len(base_colors)] for i, kt in enumerate(k_types)}

def set_style():
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


# --- Metrics computation and plotting functions (adapted from loss_comparison_visualize.py) ---
import pandas as pd

def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    # Remove all NaNs (from padding)
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    y_pred = np.clip(y_pred, 0, 1)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse_micro = np.sqrt(mse)
    mae_micro = np.mean(np.abs(y_true - y_pred))
    rmse_macro = None
    mae_macro = None
    if hasattr(y_true, 'shape') and len(y_true.shape) == 2:
        pass  # Macro metrics not implemented for 1D
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    zero_mask = y_true == 0
    nonzero_mask = y_true > 0
    if zero_mask.sum() > 0:
        rmse_zeros = np.sqrt(np.mean((y_true[zero_mask] - y_pred[zero_mask]) ** 2))
        mae_zeros = np.mean(np.abs(y_true[zero_mask] - y_pred[zero_mask]))
    else:
        rmse_zeros = np.nan
        mae_zeros = np.nan
    if nonzero_mask.sum() > 0:
        rmse_nonzeros = np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask]) ** 2))
        mae_nonzeros = np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]))
    else:
        rmse_nonzeros = np.nan
        mae_nonzeros = np.nan
    epsilon = 1e-10
    y_true_smooth = y_true + epsilon
    y_pred_smooth = y_pred + epsilon
    y_true_norm = y_true_smooth / y_true_smooth.sum()
    y_pred_norm = y_pred_smooth / y_pred_smooth.sum()
    kl_divergence = np.sum(y_true_norm * np.log(y_true_norm / y_pred_norm))
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    median_ae = np.median(np.abs(y_true - y_pred))
    rel_error = np.zeros_like(y_true, dtype=float)
    nonzero_mask = y_true != 0
    rel_error[nonzero_mask] = np.abs(y_pred[nonzero_mask] - y_true[nonzero_mask]) / np.abs(y_true[nonzero_mask])
    absolute_relative_error = np.mean(rel_error[nonzero_mask]) if np.any(nonzero_mask) else 0.0
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
def _ci_tuple_to_errorbar(mean_val, ci_tuple):
    """Convert CI tuple (lower, upper) to error bar format [lower_err, upper_err]."""
    if isinstance(ci_tuple, tuple) and len(ci_tuple) == 2:
        ci_lower, ci_upper = ci_tuple
        return [mean_val - ci_lower, ci_upper - mean_val]
    return [0.0, 0.0]


def plot_metrics_comparison(results: Dict[str, Any], output_dir: str):
    set_style()
    k_types = list(results.keys())
    extended_metrics = {}
    for kt in k_types:
        y_true = results[kt]['targets']
        y_pred = results[kt]['predictions']
        extended_metrics[kt] = compute_extended_metrics(y_true, y_pred)
    metrics_to_plot = ['RMSE_micro', 'MAE_micro', 'Absolute Relative Error', 'KL Divergence', 'MAE (zeros)', 'MAE (non-zeros)', 'Correlation']
    fig, axes = plt.subplots(1, 7, figsize=(24, 4))
    axes = axes.flatten()
    k_colors = get_dynamic_k_colors(k_types)
    metric_cis: Dict[str, Dict[str, Any]] = {}
    for kt in k_types:
        y_true = np.array(results[kt]["targets"]).flatten()
        y_pred = np.array(results[kt]["predictions"]).flatten()
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        if len(y_true) < 2:
            metric_cis[kt] = {k: None for k in metrics_to_plot}
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

        metric_cis[kt] = {
            'RMSE_micro': rmse_ci_tuple,
            'MAE_micro': mae_ci_tuple,
            'Absolute Relative Error': are_ci_tuple,
            'KL Divergence': None,
            'MAE (zeros)': mae_zeros_ci_tuple,
            'MAE (non-zeros)': mae_nz_ci_tuple,
            'Correlation': None,
        }

    for idx, metric in enumerate(metrics_to_plot):
        vals = [extended_metrics[kt][metric] for kt in k_types]
        ci_info = [metric_cis[kt].get(metric, None) for kt in k_types]
        colors = [k_colors[kt] for kt in k_types]
        labels = [get_k_label(kt) for kt in k_types]
        if metric in ['KL Divergence', 'Correlation']:
            yerr = None
        else:
            ci_lower = [ci[0] if isinstance(ci, tuple) else 0.0 for ci in ci_info]
            ci_upper = [ci[1] if isinstance(ci, tuple) else 0.0 for ci in ci_info]
            lower_err = np.abs(np.array(vals) - np.array(ci_lower))
            upper_err = np.abs(np.array(ci_upper) - np.array(vals))
            yerr = [lower_err, upper_err]
        bars = axes[idx].bar(labels, vals, color=colors, edgecolor='white', linewidth=1.5,
                            yerr=yerr, capsize=4, error_kw={'elinewidth': 1.5} if yerr is not None else None)
        axes[idx].set_title(metric, fontsize=12, fontweight='bold')
        axes[idx].set_ylim(0, max(vals) * 1.25)
        axes[idx].set_xticks(range(len(labels)))
        axes[idx].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        sns.despine(ax=axes[idx])
    plt.suptitle('KNN Neighbors Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

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

    # 8. Summary table
    log.info("8. Creating summary table...")
    plot_summary_table(results, output_dir)

def plot_scatter_actual_vs_predicted(results: Dict[str, Any], output_dir: str):
    set_style()
    k_types = list(results.keys())
    fig, axes = plt.subplots(1, len(k_types), figsize=(7 * len(k_types), 6))
    if len(k_types) == 1:
        axes = [axes]

    # Compute densities and global max for axis limits
    all_densities = []
    scatter_data = []
    global_max_target = 0
    global_max_pred = 0
    from scipy.stats import gaussian_kde
    from matplotlib.colors import Normalize

    for kt in k_types:
        y_true = results[kt]['targets'].flatten()
        y_pred = results[kt]['predictions'].flatten()
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        global_max_target = max(global_max_target, y_true.max() if len(y_true) > 0 else 0)
        global_max_pred = max(global_max_pred, y_pred.max() if len(y_pred) > 0 else 0)
        try:
            xy = np.vstack([y_pred, y_true])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {kt}: {e}")
            density = np.ones(len(y_pred))
        all_densities.extend(density)
        scatter_data.append((kt, y_pred, y_true, density))

    vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sc = None
    for ax, (kt, y_pred, y_true, density) in zip(axes, scatter_data):
        idx = density.argsort()
        y_pred_sorted, y_true_sorted, density_sorted = y_pred[idx], y_true[idx], density[idx]
        sc = ax.scatter(y_true_sorted, y_pred_sorted, c=density_sorted, cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        axis_max = max(global_max_target, global_max_pred)
        ax.plot([0, axis_max], [0, axis_max], 'r--', lw=1.5, alpha=0.7, label='Perfect')
        corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 0 else 0.0
        ax.set_xlabel("Actual", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title(f"{get_k_label(kt)}\n(Pearson r = {corr:.3f})", fontsize=14, fontweight='bold')
        ax.set_xlim(0, global_max_target)
        ax.set_ylim(0, global_max_pred)
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point Density', fontsize=10)
    plt.savefig(os.path.join(output_dir, "scatter_predicted_vs_actual.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_scatter_zoomed(results: Dict[str, Any], output_dir: str, max_actual: float = 0.01):
    set_style()
    k_types = list(results.keys())
    fig, axes = plt.subplots(1, len(k_types), figsize=(7 * len(k_types), 6))
    if len(k_types) == 1:
        axes = [axes]

    all_densities = []
    scatter_data = []
    global_max_target = 0
    global_max_pred = 0
    from scipy.stats import gaussian_kde
    from matplotlib.colors import Normalize

    for kt in k_types:
        y_true = results[kt]['targets'].flatten()
        y_pred = results[kt]['predictions'].flatten()
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        zoom_mask = y_true < max_actual
        y_true_zoomed = y_true[zoom_mask]
        y_pred_zoomed = y_pred[zoom_mask]
        if len(y_true_zoomed) > 0:
            global_max_target = max(global_max_target, y_true_zoomed.max())
            global_max_pred = max(global_max_pred, y_pred_zoomed.max())
        try:
            xy = np.vstack([y_pred_zoomed, y_true_zoomed])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {kt}: {e}")
            density = np.ones(len(y_pred_zoomed))
        all_densities.extend(density)
        scatter_data.append((kt, y_pred_zoomed, y_true_zoomed, density))

    vmin, vmax = min(all_densities) if all_densities else 0, max(all_densities) if all_densities else 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    sc = None
    for ax, (kt, y_pred_zoomed, y_true_zoomed, density) in zip(axes, scatter_data):
        idx = density.argsort()
        y_pred_sorted, y_true_sorted, density_sorted = y_pred_zoomed[idx], y_true_zoomed[idx], density[idx]
        sc = ax.scatter(y_true_sorted, y_pred_sorted, c=density_sorted, cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        axis_max = max(global_max_target, global_max_pred)
        ax.plot([0, axis_max], [0, axis_max], 'r--', lw=1.5, alpha=0.7)
        corr = np.corrcoef(y_true_zoomed, y_pred_zoomed)[0, 1] if len(y_true_zoomed) > 0 else 0.0
        ax.set_xlabel("Actual (<1%)", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title(f"{get_k_label(kt)}\n(Pearson r = {corr:.3f})", fontsize=14, fontweight='bold')
        ax.set_xlim(0, global_max_target)
        ax.set_ylim(0, global_max_pred)
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point Density', fontsize=10)
    plt.suptitle('Predicted vs Actual (Ground Truth <1%)', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_dir, "scatter_zoomed.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_loglog_scatter_actual_vs_predicted(results: Dict[str, Any], output_dir: str):
    set_style()
    k_types = list(results.keys())
    fig, axes = plt.subplots(1, len(k_types), figsize=(7 * len(k_types), 6))
    if len(k_types) == 1:
        axes = [axes]

    all_densities = []
    scatter_data = []
    global_min = float('inf')
    global_max = float('-inf')
    from scipy.stats import gaussian_kde
    from matplotlib.colors import Normalize

    for kt in k_types:
        y_true = results[kt]['targets'].flatten()
        y_pred = results[kt]['predictions'].flatten()
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        epsilon = 10**(-3.5)  # 1e-3.5
        y_true_log = np.log10(y_true + epsilon)
        y_pred_log = np.log10(y_pred + epsilon)
        try:
            xy = np.vstack([y_true_log, y_pred_log])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {kt}: {e}")
            density = np.ones(len(y_pred_log))
        all_densities.extend(density)
        scatter_data.append((kt, y_pred_log, y_true_log, density))
        global_min = min(global_min, np.min(y_true_log), np.min(y_pred_log))
        global_max = max(global_max, np.max(y_true_log), np.max(y_pred_log))

    vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sc = None
    for ax, (kt, y_pred_log, y_true_log, density) in zip(axes, scatter_data):
        idx = density.argsort()
        y_pred_sorted, y_true_sorted, density_sorted = y_pred_log[idx], y_true_log[idx], density[idx]
        sc = ax.scatter(y_true_sorted, y_pred_sorted, c=density_sorted, cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        ax.plot([global_min, global_max], [global_min, global_max], 'r--', lw=1.5, alpha=0.7, label='Perfect')
        corr = np.corrcoef(y_true_log, y_pred_log)[0, 1] if len(y_true_log) > 0 else 0.0
        ax.set_xlabel("Log10 Actual", fontsize=12)
        ax.set_ylabel("Log10 Predicted", fontsize=12)
        ax.set_title(f"{get_k_label(kt)}\n(Log-Log Pearson r = {corr:.3f})", fontsize=14, fontweight='bold')
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point Density', fontsize=10)
    plt.savefig(os.path.join(output_dir, "scatter_loglog_predicted_vs_actual.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_mae_per_range(results: Dict[str, Any], output_dir: str):
    set_style()
    k_types = list(results.keys())
    bins = [
        ('zero', 'Zero'),
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%'),
    ]
    error_data = []
    for kt in k_types:
        y_true = np.array(results[kt]['targets']).flatten()
        y_pred = np.array(results[kt]['predictions']).flatten()
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        for b in bins:
            if b[0] == 'zero':
                mask = y_true == 0
            else:
                mask = (y_true > b[0]) & (y_true <= b[1])
                mae_arr = np.abs(y_true[mask] - y_pred[mask]) if np.any(mask) else np.array([np.nan])
                mae = np.mean(mae_arr) if np.any(mask) else np.nan
                ci_lower, ci_upper = compute_95ci_bootstrap(mae_arr) if np.sum(mask) > 1 else (0.0, 0.0)
                error_data.append({'K': get_k_label(kt), 'Range': b[-1], 'MAE': mae, 'MAE_CI_Lower': ci_lower,
                    'MAE_CI_Upper': ci_upper, 'Count': int(np.sum(mask))})
    error_df = pd.DataFrame(error_data)
    count_df = error_df[error_df['K'] == get_k_label(k_types[0])][['Range', 'Count']].set_index('Range')
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = error_df.pivot(index='Range', columns='K', values='MAE')
    pivot_ci_lower = error_df.pivot(index='Range', columns='K', values='MAE_CI_Lower')
    pivot_ci_upper = error_df.pivot(index='Range', columns='K', values='MAE_CI_Upper')
    pivot_count_df = error_df.pivot(index='Range', columns='K', values='Count')
    range_order = ['Zero', '>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_ci_lower = pivot_ci_lower.reindex(range_order)
    pivot_ci_upper = pivot_ci_upper.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    k_colors = get_dynamic_k_colors(k_types)
    k_labels_ordered = [get_k_label(kt) for kt in k_types]
    pivot_df = pivot_df[[col for col in k_labels_ordered if col in pivot_df.columns]]
    pivot_ci_lower = pivot_ci_lower[[col for col in k_labels_ordered if col in pivot_ci_lower.columns]]
    pivot_ci_upper = pivot_ci_upper[[col for col in k_labels_ordered if col in pivot_ci_upper.columns]]
    pivot_count_df = pivot_count_df[[col for col in k_labels_ordered if col in pivot_count_df.columns]]

    mean_vals = pivot_df.values
    pivot_ci_lower = pivot_ci_lower.fillna(0)
    pivot_ci_upper = pivot_ci_upper.fillna(0)
    lower_err = np.abs(mean_vals - pivot_ci_lower.values)
    upper_err = np.abs(pivot_ci_upper.values - mean_vals)

    x = np.arange(len(range_order))
    width = 0.7 / len(pivot_df.columns)
    for i, col in enumerate(pivot_df.columns):
        offset = (i - len(pivot_df.columns) / 2 + 0.5) * width
        ax.bar(x + offset, pivot_df[col], width, label=col, color=[k_colors[kt] for kt in k_types][i],
               edgecolor='white', yerr=[lower_err[:, i], upper_err[:, i]],
               capsize=5, error_kw={'elinewidth': 2})
    range_labels_with_counts = [f"{r}\n(n={int(count_df.loc[r, 'Count']):,})" if r in count_df.index else r
                                 for r in range_order]
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Abundance Range', fontsize=14, fontweight='bold')
    ax.legend(title='K', frameon=False, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_range.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_mae_per_range_zoomed(results: Dict[str, Any], output_dir: str):
    set_style()
    k_types = list(results.keys())
    bins = [
        ('zero', 'Zero'),
        (0, 0.0011, '0-0.11%'),
        (0.0011, 0.0015, '0.11-0.15%'),
        (0.0015, 0.0022, '0.15-0.22%'),
        (0.0022, 0.01, '0.22-1%'),
    ]
    error_data = []
    for kt in k_types:
        y_true = np.array(results[kt]['targets']).flatten()
        y_pred = np.array(results[kt]['predictions']).flatten()
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        for b in bins:
            if b[0] == 'zero':
                mask = y_true == 0
            else:
                mask = (y_true > b[0]) & (y_true <= b[1])
                mae_arr = np.abs(y_true[mask] - y_pred[mask]) if np.any(mask) else np.array([np.nan])
                mae = np.mean(mae_arr) if np.any(mask) else np.nan
                ci_lower, ci_upper = compute_95ci_bootstrap(mae_arr) if np.sum(mask) > 1 else (0.0, 0.0)
                error_data.append({'K': get_k_label(kt), 'Range': b[-1], 'MAE': mae, 'MAE_CI_Lower': ci_lower,
                    'MAE_CI_Upper': ci_upper, 'Count': int(np.sum(mask))})
    error_df = pd.DataFrame(error_data)
    count_df = error_df[error_df['K'] == get_k_label(k_types[0])][['Range', 'Count']].set_index('Range')
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = error_df.pivot(index='Range', columns='K', values='MAE')
    pivot_ci_lower = error_df.pivot(index='Range', columns='K', values='MAE_CI_Lower')
    pivot_ci_upper = error_df.pivot(index='Range', columns='K', values='MAE_CI_Upper')
    pivot_count_df = error_df.pivot(index='Range', columns='K', values='Count')
    range_order = ['Zero', '0-0.11%', '0.11-0.15%', '0.15-0.22%', '0.22-1%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_ci_lower = pivot_ci_lower.reindex(range_order)
    pivot_ci_upper = pivot_ci_upper.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    k_colors = get_dynamic_k_colors(k_types)
    k_labels_ordered = [get_k_label(kt) for kt in k_types]
    pivot_df = pivot_df[[col for col in k_labels_ordered if col in pivot_df.columns]]
    pivot_ci_lower = pivot_ci_lower[[col for col in k_labels_ordered if col in pivot_ci_lower.columns]]
    pivot_ci_upper = pivot_ci_upper[[col for col in k_labels_ordered if col in pivot_ci_upper.columns]]
    pivot_count_df = pivot_count_df[[col for col in k_labels_ordered if col in pivot_count_df.columns]]

    mean_vals = pivot_df.values
    pivot_ci_lower = pivot_ci_lower.fillna(0)
    pivot_ci_upper = pivot_ci_upper.fillna(0)
    lower_err = np.abs(mean_vals - pivot_ci_lower.values)
    upper_err = np.abs(pivot_ci_upper.values - mean_vals)

    x = np.arange(len(range_order))
    width = 0.7 / len(pivot_df.columns)
    for i, col in enumerate(pivot_df.columns):
        offset = (i - len(pivot_df.columns) / 2 + 0.5) * width
        ax.bar(x + offset, pivot_df[col], width, label=col, color=[k_colors[kt] for kt in k_types][i],
               edgecolor='white', yerr=[lower_err[:, i], upper_err[:, i]],
               capsize=5, error_kw={'elinewidth': 2})
    range_labels_with_counts = [f"{r}\n(n={int(count_df.loc[r, 'Count']):,})" if r in count_df.index else r
                                 for r in range_order]
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Abundance Range (Zoomed: <1%)', fontsize=14, fontweight='bold')
    ax.legend(title='K', frameon=False, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_range_zoomed.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_RAE_per_range(results: Dict[str, Any], output_dir: str):
    set_style()
    k_types = list(results.keys())
    bins = [
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%'),
    ]
    error_data = []
    for kt in k_types:
        y_true = np.array(results[kt]['targets']).flatten()
        y_pred = np.array(results[kt]['predictions']).flatten()
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        for low, high, range_label in bins:
            mask = (y_true > low) & (y_true <= high) & (y_true != 0)
            if np.any(mask):
                rel_error = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
                rae = np.mean(rel_error)
                ci_lower, ci_upper = compute_95ci_bootstrap(rel_error) if np.sum(mask) > 1 else (0.0, 0.0)
                error_data.append({'K': get_k_label(kt), 'Range': range_label, 'RAE': rae, 'RAE_CI_Lower': ci_lower,
                    'RAE_CI_Upper': ci_upper, 'Count': int(np.sum(mask))})
    error_df = pd.DataFrame(error_data)
    if len(error_df) == 0:
        return
    count_df = error_df[error_df['K'] == get_k_label(k_types[0])][['Range', 'Count']].set_index('Range')
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = error_df.pivot(index='Range', columns='K', values='RAE')
    pivot_ci_lower = error_df.pivot(index='Range', columns='K', values='RAE_CI_Lower')
    pivot_ci_upper = error_df.pivot(index='Range', columns='K', values='RAE_CI_Upper')
    pivot_count_df = error_df.pivot(index='Range', columns='K', values='Count')
    range_order = ['>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_ci_lower = pivot_ci_lower.reindex(range_order)
    pivot_ci_upper = pivot_ci_upper.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    k_colors = get_dynamic_k_colors(k_types)
    k_labels_ordered = [get_k_label(kt) for kt in k_types]
    pivot_df = pivot_df[[col for col in k_labels_ordered if col in pivot_df.columns]]
    pivot_ci_lower = pivot_ci_lower[[col for col in k_labels_ordered if col in pivot_ci_lower.columns]]
    pivot_ci_upper = pivot_ci_upper[[col for col in k_labels_ordered if col in pivot_ci_upper.columns]]
    pivot_count_df = pivot_count_df[[col for col in k_labels_ordered if col in pivot_count_df.columns]]

    mean_vals = pivot_df.values
    pivot_ci_lower = pivot_ci_lower.fillna(0)
    pivot_ci_upper = pivot_ci_upper.fillna(0)
    lower_err = np.abs(mean_vals - pivot_ci_lower.values)
    upper_err = np.abs(pivot_ci_upper.values - mean_vals)

    x = np.arange(len(range_order))
    width = 0.7 / len(pivot_df.columns)
    for i, col in enumerate(pivot_df.columns):
        offset = (i - len(pivot_df.columns) / 2 + 0.5) * width
        ax.bar(x + offset, pivot_df[col], width, label=col, color=[k_colors[kt] for kt in k_types][i],
               edgecolor='white', yerr=[lower_err[:, i], upper_err[:, i]],
               capsize=5, error_kw={'elinewidth': 2})
    range_labels_with_counts = [f"{r}\n(n={int(count_df.loc[r, 'Count']):,})" if r in count_df.index else r
                                 for r in range_order]
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Relative Absolute Error', fontsize=12)
    ax.set_title('Relative Absolute Error by Abundance Range', fontsize=14, fontweight='bold')
    ax.legend(title='K', frameon=False, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "relative_err_by_range.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_residual_distribution(results: Dict[str, Any], output_dir: str):
    set_style()
    k_types = list(results.keys())
    k_colors = get_dynamic_k_colors(k_types)
    fig, ax = plt.subplots(figsize=(9, 5))
    all_res_flat = []
    kt_residuals = {}
    for kt in k_types:
        y_true = np.array(results[kt]['targets']).flatten()
        y_pred = np.array(results[kt]['predictions']).flatten()
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        residuals = (y_true - y_pred)[valid]
        kt_residuals[kt] = residuals
        all_res_flat.extend(residuals.tolist())
    if len(all_res_flat) == 0:
        return
    x_kde = np.linspace(min(all_res_flat), max(all_res_flat), 300)
    legend_handles = []
    density_values = []
    for kt, residuals in kt_residuals.items():
        color = k_colors[kt]
        label = get_k_label(kt)
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax.hist(residuals, bins=60, color=color, alpha=0.3, density=True, edgecolor='none')
        if len(residuals) > 1:
            kde = gaussian_kde(residuals)
            kde_vals = kde(x_kde)
            density_values.extend(kde_vals[kde_vals > 0])
            ax.plot(x_kde, kde_vals, color=color, linewidth=2)
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2,
                              label=f'{label} (μ={mean_res:.4f}, σ={std_res:.4f})'))
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_yscale('log')
    if density_values:
        max_density = max(density_values)
        min_density = min(density_values)
        lower = max(min_density, max_density * 1e-6)
        ax.set_ylim(lower, max_density * 1.2)
    ax.set_title('Residual Distributions', fontsize=13, fontweight='bold')
    ax.legend(handles=legend_handles, frameon=False, fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_zero_vs_nonzero_comparison(results: Dict[str, Any], output_dir: str):
    set_style()
    k_types = list(results.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(k_types))
    width = 0.35
    def _blend_with_white(color, alpha=0.6):
        r, g, b = mc.to_rgb(color)
        return (r * (1 - alpha) + alpha, g * (1 - alpha) + alpha, b * (1 - alpha) + alpha)
    for i, kt in enumerate(k_types):
        y_true = np.array(results[kt]['targets']).flatten()
        y_pred = np.array(results[kt]['predictions']).flatten()
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        zero_mask = y_true == 0
        nonzero_mask = y_true != 0
        errs_z = np.abs(y_true[zero_mask] - y_pred[zero_mask]) if zero_mask.sum() > 0 else np.array([0.0])
        errs_nz = np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]) if nonzero_mask.sum() > 0 else np.array([0.0])
        mae_z = np.mean(errs_z)
        mae_nz = np.mean(errs_nz)
         base_color = k_colors[kt]
         zero_color = _blend_with_white(base_color, alpha=0.6)
         nonzero_color = base_color
         ci_z = _ci_tuple_to_errorbar(mae_z, compute_95ci_bootstrap(errs_z))
         ci_nz = _ci_tuple_to_errorbar(mae_nz, compute_95ci_bootstrap(errs_nz))
         ax.bar(x[i] - width/2, mae_z, width, color=zero_color, edgecolor='white',
             yerr=ci_z, capsize=4, error_kw={'elinewidth': 1.5})
         ax.bar(x[i] + width/2, mae_nz, width, color=nonzero_color, edgecolor='white',
             yerr=ci_nz, capsize=4, error_kw={'elinewidth': 1.5})
    legend_elements = [
        Patch(facecolor='#bbbbbb', edgecolor='white', label='Zero GT (lighter)'),
        Patch(facecolor='#444444', edgecolor='white', label='Non-zero GT (darker)')
    ]
    ax.legend(handles=legend_elements, frameon=False, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels([get_k_label(kt) for kt in k_types], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE: Zero vs Non-Zero Values', fontsize=14, fontweight='bold')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zero_vs_nonzero_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_summary_table(results: Dict[str, Any], output_dir: str):
    set_style()
    k_types = list(results.keys())
    extended_metrics = {}
    for kt in k_types:
        y_true = np.array(results[kt]['targets'])
        y_pred = np.array(results[kt]['predictions'])
        extended_metrics[kt] = compute_extended_metrics(y_true, y_pred)
    metrics = ['RMSE_micro', 'MAE_micro', 'Absolute Relative Error', 'KL Divergence', 'MAE (zeros)', 'MAE (non-zeros)', 'Correlation']
    data = []
    for kt in k_types:
        row = {'K': get_k_label(kt)}
        for m in metrics:
            row[m] = extended_metrics[kt][m]
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
    fig, ax = plt.subplots(figsize=(16, 2.5))
    ax.axis('off')
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(display_df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for i in range(len(display_df.columns)):
        table[(0, i)].set_text_props(fontweight='bold')
        table[(0, i)].set_facecolor('#d0d0d0')
    k_col_idx = list(display_df.columns).index('K')
    k_colors = get_dynamic_k_colors(k_types)
    for row_idx, kt in enumerate(k_types):
        cell = table[(row_idx + 1, k_col_idx)]
        hex_color = k_colors[kt]
        cell.set_facecolor(hex_color)
        cell.set_text_props(fontweight='bold', color='white')
    for col_idx, col in enumerate(display_df.columns):
        if col == 'K':
            continue
        if col in best_indices:
            # Highlight all rows that have the best value for this metric
            for best_row_idx in best_indices[col]:
                table[(best_row_idx + 1, col_idx)].set_facecolor('#d5f5e3')
                table[(best_row_idx + 1, col_idx)].set_text_props(fontweight='bold', color='#1a7a40')
    plt.title('K Comparison Summary', fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150, bbox_inches='tight')
    plt.close()
    # Also save as CSV
    df.to_csv(os.path.join(output_dir, "K_comparison_results.csv"), index=False)
def plot_latent_difference_distribution(results: Dict[str, Any], output_dir: str):
    """
    Plot the distribution of differences in the latent vectors between the two available K values.
    """
    # Find the two K keys with latent vectors
    k_keys = [k for k in results.keys() if isinstance(results[k], dict) and "latent_vector" in results[k]]
    if len(k_keys) != 2:
        print(f"Expected exactly 2 K values with latent vectors, found {len(k_keys)}: {k_keys}. Skipping latent difference plot.")
        return
    # Prefer K=13 and K=972 if present, otherwise use whatever is present
    if "K=13" in k_keys and "K=972" in k_keys:
        k1, k2 = "K=13", "K=972"
    else:
        k1, k2 = k_keys
    latent_1 = results[k1]["latent_vector"]
    latent_2 = results[k2]["latent_vector"]
    if latent_1 is None or latent_2 is None:
        print("Latent vectors not found in results. Skipping latent difference plot.")
        return
    if latent_1.shape != latent_2.shape:
        print(f"Latent vectors have different shapes: {latent_1.shape} vs {latent_2.shape}. Skipping plot.")
        return
    diff = latent_2 - latent_1
    plt.figure(figsize=(8, 5))
    sns.histplot(diff, bins=40, kde=True, color="#4a90d9", edgecolor=None, alpha=0.8)
    plt.axvline(0, color="black", linestyle="--", lw=1)
    plt.title(f"Distribution of Latent Vector Differences ({k2} - {k1})", fontsize=14, fontweight="bold")
    plt.xlabel(f"Latent Value Difference ({k2} - {k1})")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_difference_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

def create_all_visualizations(results: Dict[str, Any], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log.info("\n" + "="*60)
    log.info("CREATING VISUALIZATIONS")
    log.info("="*60)
    log.info("\n1. Creating metric comparison plots...")
    plot_metrics_comparison(results, output_dir)
    log.info("2. Creating scatter plots (full range)...")
    plot_scatter_actual_vs_predicted(results, output_dir)
    log.info("3. Creating zoomed scatter plots (<1%)...")
    plot_scatter_zoomed(results, output_dir)
    log.info("3b. Creating log-log scatter plots...")
    plot_loglog_scatter_actual_vs_predicted(results, output_dir)
    log.info("4. Creating error by range plot...")
    plot_mae_per_range(results, output_dir)
    log.info("5. Creating zoomed error by range plot...")
    plot_mae_per_range_zoomed(results, output_dir)
    log.info("5b. Creating relative absolute error by range plot...")
    plot_RAE_per_range(results, output_dir)
    log.info("6. Creating residual distribution plot...")
    plot_residual_distribution(results, output_dir)
    log.info("7. Creating zero vs non-zero comparison...")
    plot_zero_vs_nonzero_comparison(results, output_dir)
    log.info("8. Creating summary table...")
    plot_summary_table(results, output_dir)
    log.info("9. Creating latent vector difference distribution plot...")
    plot_latent_difference_distribution(results, output_dir)
    log.info("10. Creating latent factor value distribution plots...")
    plot_latent_factor_distributions(results, output_dir)

def plot_latent_factor_distributions(results: Dict[str, Any], output_dir: str):
    """
    Plot the distribution of latent factor values for each K, with one subplot per K.
    """
    k_types = [k for k in results.keys() if isinstance(results[k], dict) and "latent_vector" in results[k]]
    if not k_types:
        print("No latent vectors found in results. Skipping latent factor distribution plot.")
        return
    n_k = len(k_types)
    fig, axes = plt.subplots(1, n_k, figsize=(6 * n_k, 5))
    if n_k == 1:
        axes = [axes]
    k_colors = get_dynamic_k_colors(k_types)
    for ax, kt in zip(axes, k_types):
        latent = results[kt]["latent_vector"]
        if latent is None:
            ax.set_title(f"No latent vector for {get_k_label(kt)}")
            continue
        sns.histplot(latent, bins=40, kde=True, color=k_colors[kt], edgecolor=None, alpha=0.8, ax=ax)
        ax.set_title(f"Latent Factor Distribution: {get_k_label(kt)}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Latent Factor Value")
        ax.set_ylabel("Count")
    plt.suptitle("Distribution of Latent Factor Values", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_factor_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()


def load_results(results_path: str) -> Dict[str, Any]:
    with open(results_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K Comparison Visualization: K=13 vs K=197 (or any optimal K)")
    parser.add_argument("--results_path", type=str, default="results/K_comparison_results.pkl",
                        help="Path to K_comparison_results.pkl")
    parser.add_argument("--output_dir", type=str, default="figures", help="Output directory for figures")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = args.results_path
    if not os.path.isabs(results_path):
        results_path = os.path.join(script_dir, results_path)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)

    set_style()
    os.makedirs(output_dir, exist_ok=True)
    results = load_results(results_path)
    create_all_visualizations(results, output_dir)
    log.info(f"All K comparison visualizations saved to {output_dir}")
