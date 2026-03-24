#!/usr/bin/env python
"""
Visualization module for read count preprocessing comparison.
Creates clean, presentation-ready plots comparing preprocessing methods.

This script is separate from read_count_preprocessing.py to allow re-generating visualizations
without re-running the expensive training.

Usage:
    python preprocessing_visualization.py --results_path results/preprocessing_results.pkl --output_dir figures
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mc
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.patches import Patch


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

# Consistent preprocessing method colors
PREPROCESSING_COLORS = {
    "original": "#ff7f0e",      # Orange - Original (normalize + log)
    "normalized": "#1f77b4",    # Blue - Normalized only
    "logarithm": "#2ca02c",     # Green - Logarithm only
}

PREPROCESSING_LABELS = {
    "original": "Original (raw counts)",
    "normalized": "Normalized Only",
    "logarithm": "Logarithm Only",
}

# Color shades for zero/non-zero comparisons
PREPROCESSING_COLOR_SHADES = {
    "original": {"zero": "#ffcc99", "nonzero": "#ff7f0e"},
    "normalized": {"zero": "#a8d4ff", "nonzero": "#1f77b4"},
    "logarithm": {"zero": "#a8e6a8", "nonzero": "#2ca02c"},
}


def get_preprocessing_label(method: str) -> str:
    """Get display label for a preprocessing method."""
    return PREPROCESSING_LABELS.get(method, method)


def get_preprocessing_color(method: str) -> str:
    """Get consistent color for a preprocessing method."""
    return PREPROCESSING_COLORS.get(method, "#9467bd")


def get_preprocessing_color_shades(method: str) -> Dict[str, str]:
    """Get color shades for zero/non-zero comparison."""
    return PREPROCESSING_COLOR_SHADES.get(method, {"zero": "#cccccc", "nonzero": "#666666"})


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
    """Compute comprehensive metrics with proper per-sample handling."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse_macro: Optional[float] = None
    mae_macro: Optional[float] = None
    kl_divergence_macro: Optional[float] = None
    eps = 1e-10
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
            rt_norm = (rt + eps) / (rt + eps).sum()
            rp_norm = (rp + eps) / (rp + eps).sum()
            kl_per.append(float(np.sum(rt_norm * np.log(rt_norm / rp_norm))))
        if rmse_per:
            rmse_macro = float(np.mean(rmse_per))
            mae_macro = float(np.mean(mae_per))
            kl_divergence_macro = float(np.mean(kl_per))
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
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    valid = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true = y_true_flat[valid]
    y_pred = np.clip(y_pred_flat[valid], 0, 1)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse_micro = float(np.sqrt(mse))
    mae_micro = float(np.mean(np.abs(y_true - y_pred)))
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
    """Create bar plots comparing key metrics between preprocessing methods."""
    set_style()
    
    methods = list(results.keys())
    n_methods = len(methods)
    
    # Compute extended metrics for all methods
    extended_metrics = {}
    for method in methods:
        y_true = results[method]["targets"]
        y_pred = results[method]["predictions"]
        extended_metrics[method] = compute_extended_metrics(y_true, y_pred)
    
    metrics_to_plot = ['RMSE_micro', 'MAE_micro', 'Absolute Relative Error', 'KL Divergence', 'MAE (zeros)', 'MAE (non-zeros)', 'Correlation']
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 4))
    axes = axes.flatten()

    metric_cis: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        y_true = results[method]["targets"].flatten()
        y_pred = results[method]["predictions"].flatten()
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        if len(y_true) < 2:
            metric_cis[method] = {k: None for k in metrics_to_plot}
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

        metric_cis[method] = {
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
        values = [extended_metrics[m][metric] if not np.isnan(extended_metrics[m][metric]) and not np.isinf(extended_metrics[m][metric]) else 0.0 for m in methods]
        ci_info = [metric_cis[m].get(metric, None) for m in methods]
        colors = [get_preprocessing_color(m) for m in methods]
        labels = [get_preprocessing_label(m) for m in methods]
        if metric in ['KL Divergence', 'Correlation']:
            yerr = None
        else:
            ci_lower = [ci[0] if isinstance(ci, tuple) else 0.0 for ci in ci_info]
            ci_upper = [ci[1] if isinstance(ci, tuple) else 0.0 for ci in ci_info]
            lower_err = np.abs(np.array(values) - np.array(ci_lower))
            upper_err = np.abs(np.array(ci_upper) - np.array(values))
            yerr = [lower_err, upper_err]
        bars = ax.bar(range(len(methods)), values, color=colors, edgecolor='white', linewidth=1.5,
                      yerr=yerr, capsize=4, error_kw={'elinewidth': 1.5} if yerr is not None else None)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        # Avoid NaN/Inf in ylim
        max_val = max([v for v in values if not np.isnan(v) and not np.isinf(v)] + [1e-6])
        ax.set_ylim(0, max_val * 1.25)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        sns.despine(ax=ax)
    plt.suptitle('Preprocessing Method Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'metrics_comparison.png')}")


def plot_scatter_actual_vs_predicted(results: Dict[str, Any], output_dir: str):
    """Create scatter plots of actual vs predicted values for all preprocessing methods."""
    set_style()
    
    methods = list(results.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    # Compute global min/max for axis limits and densities for color normalization
    all_densities = []
    scatter_data = []
    global_min = float('inf')
    global_max = float('-inf')
    for method in methods:
        y_true = results[method]["targets"].flatten()
        y_pred = results[method]["predictions"].flatten()
        # Remove NaNs/Infs
        valid = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (~np.isinf(y_true)) & (~np.isinf(y_pred))
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        global_min = min(global_min, y_true.min(), y_pred.min())
        global_max = max(global_max, y_true.max(), y_pred.max())
        try:
            xy = np.vstack([y_true, y_pred])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception:
            density = np.ones(len(y_pred))
        all_densities.extend(density)
        scatter_data.append((method, y_pred, y_true, density))
    vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sc = None
    for ax, (method, y_pred, y_true, density) in zip(axes, scatter_data):
        idx = density.argsort()
        y_pred_sorted, y_true_sorted, density_sorted = y_pred[idx], y_true[idx], density[idx]
        sc = ax.scatter(y_true_sorted, y_pred_sorted, c=density_sorted, cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        ax.plot([global_min, global_max], [global_min, global_max], 'r--', lw=1.5, alpha=0.7, label='Perfect')
        corr = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else 0.0
        ax.set_xlabel('Actual Relative Abundance', fontsize=11)
        ax.set_ylabel('Predicted Relative Abundance', fontsize=11)
        ax.set_title(f'{get_preprocessing_label(method)}\n(Pearson r = {corr:.3f})', fontsize=12, fontweight='bold')
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        sns.despine(ax=ax)
    plt.tight_layout()
    fig = plt.gcf()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point Density', fontsize=10)
    plt.savefig(os.path.join(output_dir, "scatter_actual_vs_predicted.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'scatter_actual_vs_predicted.png')}")


def plot_residual_distribution(results: Dict[str, Any], output_dir: str):
    """Plot residual distributions for all preprocessing methods."""
    set_style()
    methods = list(results.keys())
    all_res_flat = []
    method_residuals = {}
    for method in methods:
        y_true = results[method]["targets"].flatten()
        y_pred = results[method]["predictions"].flatten()
        valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        residuals = (y_true - y_pred)[valid]
        method_residuals[method] = residuals
        all_res_flat.extend(residuals.tolist())
    if len(all_res_flat) == 0:
        return
    x_kde = np.linspace(min(all_res_flat), max(all_res_flat), 300)
    fig, ax = plt.subplots(figsize=(10, 6))
    legend_handles = []
    density_values = []
    for method, residuals in method_residuals.items():
        color = get_preprocessing_color(method)
        label = get_preprocessing_label(method)
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax.hist(residuals, bins=50, color=color, alpha=0.3, density=True, edgecolor='none')
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
    ax.set_title('Residual Distribution Comparison', fontweight='bold', fontsize=14)
    ax.legend(handles=legend_handles, frameon=False, fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "residual_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_zero_vs_nonzero_comparison(results: Dict[str, Any], output_dir: str):
    """Compare performance on zero vs non-zero ground truth values (MAE only)."""
    set_style()
    methods = list(results.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.35
    def _blend_with_white(color, alpha=0.6):
        r, g, b = mc.to_rgb(color)
        return (r * (1 - alpha) + alpha, g * (1 - alpha) + alpha, b * (1 - alpha) + alpha)
    for i, method in enumerate(methods):
        y_true = results[method]["targets"].flatten()
        y_pred = results[method]["predictions"].flatten()
        zero_mask = y_true == 0
        nonzero_mask = y_true > 0
        errs_z = np.abs(y_true[zero_mask] - y_pred[zero_mask]) if zero_mask.sum() > 0 else np.array([0.0])
        errs_nz = np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]) if nonzero_mask.sum() > 0 else np.array([0.0])
        mae_z = np.mean(errs_z)
        mae_nz = np.mean(errs_nz)
        base_color = get_preprocessing_color(method)
        zero_color = _blend_with_white(base_color, alpha=0.6)
        nonzero_color = base_color
        ci_z = _ci_tuple_to_errorbar(mae_z, compute_95ci_bootstrap(errs_z))
        ci_nz = _ci_tuple_to_errorbar(mae_nz, compute_95ci_bootstrap(errs_nz))
        yerr_z = np.array([[ci_z[0]], [ci_z[1]]])
        yerr_nz = np.array([[ci_nz[0]], [ci_nz[1]]])
        ax.bar(x[i] - width/2, mae_z, width, color=zero_color, edgecolor='white',
            yerr=yerr_z, capsize=4, error_kw={'elinewidth': 1.5})
        ax.bar(x[i] + width/2, mae_nz, width, color=nonzero_color, edgecolor='white',
            yerr=yerr_nz, capsize=4, error_kw={'elinewidth': 1.5})
    legend_elements = [
        Patch(facecolor='#bbbbbb', edgecolor='white', label='Zero GT (lighter)'),
        Patch(facecolor='#444444', edgecolor='white', label='Non-zero GT (darker)')
    ]
    ax.legend(handles=legend_elements, frameon=False, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels([get_preprocessing_label(m) for m in methods], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE: Zero vs Non-Zero Values', fontsize=13, fontweight='bold')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zero_vs_nonzero_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'zero_vs_nonzero_comparison.png')}")


def plot_summary_table(results: Dict[str, Any], output_dir: str):
    """Create a summary table of all metrics."""
    set_style()
    
    methods = list(results.keys())
    
    # Compute metrics
    metrics_data = []
    for method in methods:
        y_true = results[method]["targets"].flatten()
        y_pred = results[method]["predictions"].flatten()
        metrics = compute_extended_metrics(y_true, y_pred)
        metrics['Method'] = get_preprocessing_label(method)
        metrics_data.append(metrics)
    df = pd.DataFrame(metrics_data)
    col_order = ['Method', 'RMSE_micro', 'MAE_micro', 'R²', 'Correlation', 
                'KL Divergence', 'Absolute Relative Error',
                'RMSE (zeros)', 'MAE (zeros)', 'RMSE (non-zeros)', 'MAE (non-zeros)']
    df = df[[c for c in col_order if c in df.columns]]
    # Format numeric columns and handle NaN/Inf
    formatted_data = []
    for _, row in df.iterrows():
        formatted_row = [row['Method']]
        for col in df.columns[1:]:
            val = row[col]
            if isinstance(val, (int, float)):
                if np.isnan(val) or np.isinf(val):
                    formatted_row.append('0.000000')
                else:
                    formatted_row.append(f"{val:.6f}")
            else:
                formatted_row.append(str(val))
        formatted_data.append(formatted_row)
    # Find best scores for each metric (min for error metrics, max for correlation/R²)
    best_indices = {}
    for col in df.columns[1:]:
        if col in ['Correlation', 'R²']:
            best_value = df[col].max()
            best_indices[col] = df[df[col] == best_value].index.tolist()
        else:
            best_value = df[col].min()
            best_indices[col] = df[df[col] == best_value].index.tolist()
    fig, ax = plt.subplots(figsize=(16, 1.5 + 0.5 * len(methods)))
    ax.axis('off')
    table = ax.table(cellText=formatted_data, colLabels=df.columns, 
                    cellLoc='center', loc='center',
                    colColours=['#f0f0f0'] * len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(fontweight='bold')
        table[(0, i)].set_facecolor('#d0d0d0')
    method_col_idx = list(df.columns).index('Method')
    for row_idx, method in enumerate(methods):
        hex_color = get_preprocessing_color(method)
        cell = table[(row_idx + 1, method_col_idx)]
        cell.set_facecolor(hex_color)
        cell.set_text_props(fontweight='bold', color='white')
    for col_idx, col in enumerate(df.columns):
        if col == 'Method':
            continue
        if col in best_indices:
            # Highlight all rows that have the best value for this metric
            for best_row_idx in best_indices[col]:
                best_row = best_row_idx + 1  # +1 for header
                table[(best_row, col_idx)].set_facecolor('#d5f5e3')
                table[(best_row, col_idx)].set_text_props(fontweight='bold', color='#1a7a40')
    plt.title('Comprehensive Metrics Comparison', fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "summary_table.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ===================== Additional Plots for Ablation-Style =====================
def plot_loglog_scatter_actual_vs_predicted(results: Dict[str, Any], output_dir: str):
    """Create log-log scatter plots of actual vs predicted values."""
    set_style()
    methods = list(results.keys())
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    all_densities = []
    scatter_data = []
    global_min = float('inf')
    global_max = float('-inf')

    for method in methods:
        y_pred = results[method]["predictions"]
        y_true = results[method]["targets"]
        epsilon = 10**(-3.5)  # 1e-3.5
        y_pred_log = np.log10(y_pred + epsilon)
        y_true_log = np.log10(y_true + epsilon)
        # Remove any NaNs or infs
        valid = (~np.isnan(y_pred_log)) & (~np.isnan(y_true_log)) & (~np.isinf(y_pred_log)) & (~np.isinf(y_true_log))
        y_pred_log = y_pred_log[valid]
        y_true_log = y_true_log[valid]
        try:
            xy = np.vstack([y_true_log, y_pred_log])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception:
            density = np.ones(len(y_pred_log))
        # All arrays must be the same length
        all_densities.extend(density)
        scatter_data.append((method, y_pred_log, y_true_log, density))
        if len(y_true_log) > 0 and len(y_pred_log) > 0:
            global_min = min(global_min, np.min(y_true_log), np.min(y_pred_log))
            global_max = max(global_max, np.max(y_true_log), np.max(y_pred_log))

    vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)

    sc = None
    for ax, (method, y_pred_log, y_true_log, density) in zip(axes, scatter_data):
        idx = density.argsort()
        y_pred_sorted, y_true_sorted, density_sorted = y_pred_log[idx], y_true_log[idx], density[idx]
        sc = ax.scatter(y_true_sorted, y_pred_sorted, c=density_sorted, cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        ax.plot([global_min, global_max], [global_min, global_max], 'r--', lw=1.5, alpha=0.7, label='Perfect')
        corr = np.corrcoef(y_true_log, y_pred_log)[0, 1]
        ax.set_xlabel("Log10 Actual", fontsize=11)
        ax.set_ylabel("Log10 Predicted", fontsize=11)
        ax.set_title(f"{get_preprocessing_label(method)}\n(Log-Log Pearson r = {corr:.3f})", fontsize=12, fontweight='bold')
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point Density', fontsize=10)

    plt.savefig(os.path.join(output_dir, "scatter_loglog_predicted_vs_actual.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'scatter_loglog_predicted_vs_actual.png')}")


def plot_scatter_zoomed(results: Dict[str, Any], output_dir: str, max_actual: float = 0.01):
    """Create scatter plots zoomed on ground truth <1% range."""
    set_style()
    methods = list(results.keys())
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    all_densities = []
    scatter_data = []
    global_max_target = 0
    global_max_pred = 0

    for method in methods:
        y_true = results[method]["targets"]
        y_pred = results[method]["predictions"]
        zoom_mask = y_true < max_actual
        y_true_zoomed = y_true[zoom_mask]
        y_pred_zoomed = y_pred[zoom_mask]
        global_max_target = max(global_max_target, y_true_zoomed.max() if len(y_true_zoomed) > 0 else 0)
        global_max_pred = max(global_max_pred, y_pred_zoomed.max() if len(y_pred_zoomed) > 0 else 0)
        try:
            xy = np.vstack([y_true_zoomed, y_pred_zoomed])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception:
            density = np.ones(len(y_pred_zoomed))
        all_densities.extend(density)
        scatter_data.append((method, y_pred_zoomed, y_true_zoomed, density))

    vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)

    sc = None
    for ax, (method, y_pred_zoomed, y_true_zoomed, density) in zip(axes, scatter_data):
        idx = density.argsort()
        y_pred_sorted, y_true_sorted, density_sorted = y_pred_zoomed[idx], y_true_zoomed[idx], density[idx]
        sc = ax.scatter(y_true_sorted, y_pred_sorted, c=density_sorted, cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        axis_max = max(global_max_target, global_max_pred)
        ax.plot([0, axis_max], [0, axis_max], 'r--', lw=1.5, alpha=0.7)
        corr = np.corrcoef(y_true_zoomed, y_pred_zoomed)[0, 1] if len(y_true_zoomed) > 1 else 0
        ax.set_xlabel("Actual", fontsize=11)
        ax.set_ylabel("Predicted", fontsize=11)
        ax.set_title(f"{get_preprocessing_label(method)}\n(Pearson r = {corr:.3f})", fontsize=12, fontweight='bold')
        ax.set_xlim(0, global_max_target)
        ax.set_ylim(0, global_max_pred)
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point Density', fontsize=10)

    plt.suptitle('Predicted vs Actual (Ground Truth <1%)', fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_dir, "scatter_zoomed.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'scatter_zoomed.png')}")


def plot_mae_per_range(results: Dict[str, Any], output_dir: str):
    """Create bar plot of MAE per target value range."""
    set_style()
    methods = list(results.keys())
    bins = [
        ('zero', 'Zero'),
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%'),
    ]
    error_data = []
    for method in methods:
        y_pred = results[method]["predictions"]
        y_true = results[method]["targets"]
        for bin_def in bins:
            if bin_def[0] == 'zero':
                mask = y_true == 0
                range_label = bin_def[1]
            else:
                low, high, range_label = bin_def
                mask = (y_true > low) & (y_true <= high)
            if mask.sum() > 0:
                mae_arr = np.abs(y_true[mask] - y_pred[mask])
                mae = np.mean(mae_arr)
                ci_lower, ci_upper = compute_95ci_bootstrap(mae_arr) if mask.sum() > 1 else (0.0, 0.0)
                error_data.append({
                    'Method': get_preprocessing_label(method),
                    'Range': range_label,
                    'MAE': mae,
                    'MAE_CI_Lower': ci_lower,
                    'MAE_CI_Upper': ci_upper,
                    'Count': int(mask.sum()),
                    'method_key': method
                })
    error_df = pd.DataFrame(error_data)
    if len(error_df) == 0:
        print("No data for MAE per range plot")
        return
    count_df = error_df[error_df['Method'] == get_preprocessing_label(methods[0])][['Range', 'Count']].set_index('Range')
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = error_df.pivot(index='Range', columns='Method', values='MAE')
    pivot_ci_lower = error_df.pivot(index='Range', columns='Method', values='MAE_CI_Lower')
    pivot_ci_upper = error_df.pivot(index='Range', columns='Method', values='MAE_CI_Upper')
    pivot_count_df = error_df.pivot(index='Range', columns='Method', values='Count')
    range_order = ['Zero', '>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_ci_lower = pivot_ci_lower.reindex(range_order)
    pivot_ci_upper = pivot_ci_upper.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    method_labels_ordered = [get_preprocessing_label(m) for m in methods]
    pivot_df = pivot_df[[col for col in method_labels_ordered if col in pivot_df.columns]]
    pivot_ci_lower = pivot_ci_lower[[col for col in method_labels_ordered if col in pivot_ci_lower.columns]]
    pivot_ci_upper = pivot_ci_upper[[col for col in method_labels_ordered if col in pivot_ci_upper.columns]]
    pivot_count_df = pivot_count_df[[col for col in method_labels_ordered if col in pivot_count_df.columns]]

    mean_vals = pivot_df.values
    pivot_ci_lower = pivot_ci_lower.fillna(0)
    pivot_ci_upper = pivot_ci_upper.fillna(0)
    lower_err = np.abs(mean_vals - pivot_ci_lower.values)
    upper_err = np.abs(pivot_ci_upper.values - mean_vals)

    colors = [get_preprocessing_color(m) for m in methods]
    x = np.arange(len(range_order))
    width = 0.7 / len(pivot_df.columns)
    for i, col in enumerate(pivot_df.columns):
        offset = (i - len(pivot_df.columns) / 2 + 0.5) * width
        ax.bar(x + offset, pivot_df[col], width, label=col, color=colors[i],
               edgecolor='white', yerr=[lower_err[:, i], upper_err[:, i]],
               capsize=5, error_kw={'elinewidth': 2})
    range_labels_with_counts = [f"{r}\n(n={int(count_df.loc[r, 'Count']):,})" if r in count_df.index else r
                                 for r in range_order]
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Abundance Range', fontsize=14, fontweight='bold')
    ax.legend(title='Method', frameon=False, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_range.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'error_by_range.png')}")


def plot_RAE_per_range(results: Dict[str, Any], output_dir: str):
    """Create bar plot of Relative Absolute Error per target value range."""
    set_style()
    methods = list(results.keys())
    bins = [
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%'),
    ]
    error_data = []
    for method in methods:
        y_pred = results[method]["predictions"]
        y_true = results[method]["targets"]
        for low, high, range_label in bins:
            mask = (y_true > low) & (y_true <= high) & (y_true != 0)
            if mask.sum() > 0:
                rel_error = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
                rae = np.mean(rel_error)
                ci_lower, ci_upper = compute_95ci_bootstrap(rel_error) if mask.sum() > 1 else (0.0, 0.0)
                error_data.append({
                    'Method': get_preprocessing_label(method),
                    'Range': range_label,
                    'RAE': rae,
                    'RAE_CI_Lower': ci_lower,
                    'RAE_CI_Upper': ci_upper,
                    'Count': int(mask.sum()),
                    'method_key': method
                })
    error_df = pd.DataFrame(error_data)
    if len(error_df) == 0:
        print("No data for RAE per range plot")
        return
    count_df = error_df[error_df['Method'] == get_preprocessing_label(methods[0])][['Range', 'Count']].set_index('Range')
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = error_df.pivot(index='Range', columns='Method', values='RAE')
    pivot_ci_lower = error_df.pivot(index='Range', columns='Method', values='RAE_CI_Lower')
    pivot_ci_upper = error_df.pivot(index='Range', columns='Method', values='RAE_CI_Upper')
    pivot_count_df = error_df.pivot(index='Range', columns='Method', values='Count')
    range_order = ['>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_ci_lower = pivot_ci_lower.reindex(range_order)
    pivot_ci_upper = pivot_ci_upper.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    method_labels_ordered = [get_preprocessing_label(m) for m in methods]
    pivot_df = pivot_df[[col for col in method_labels_ordered if col in pivot_df.columns]]
    pivot_ci_lower = pivot_ci_lower[[col for col in method_labels_ordered if col in pivot_ci_lower.columns]]
    pivot_ci_upper = pivot_ci_upper[[col for col in method_labels_ordered if col in pivot_ci_upper.columns]]
    pivot_count_df = pivot_count_df[[col for col in method_labels_ordered if col in pivot_count_df.columns]]

    mean_vals = pivot_df.values
    pivot_ci_lower = pivot_ci_lower.fillna(0)
    pivot_ci_upper = pivot_ci_upper.fillna(0)
    lower_err = np.abs(mean_vals - pivot_ci_lower.values)
    upper_err = np.abs(pivot_ci_upper.values - mean_vals)

    colors = [get_preprocessing_color(m) for m in methods]
    x = np.arange(len(range_order))
    width = 0.7 / len(pivot_df.columns)
    for i, col in enumerate(pivot_df.columns):
        offset = (i - len(pivot_df.columns) / 2 + 0.5) * width
        ax.bar(x + offset, pivot_df[col], width, label=col, color=colors[i],
               edgecolor='white', yerr=[lower_err[:, i], upper_err[:, i]],
               capsize=5, error_kw={'elinewidth': 2})
    range_labels_with_counts = [f"{r}\n(n={int(count_df.loc[r, 'Count']):,})" if r in count_df.index else r
                                 for r in range_order]
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Relative Absolute Error', fontsize=12)
    ax.set_title('Relative Absolute Error by Abundance Range', fontsize=14, fontweight='bold')
    ax.legend(title='Method', frameon=False, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "relative_err_by_range.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'relative_err_by_range.png')}")


def plot_mae_per_range_zoomed(results: Dict[str, Any], output_dir: str):
    """Create bar plot of MAE per range zoomed with quantile bins."""
    set_style()
    methods = list(results.keys())
    bins = [
        ('zero', 'Zero'),
        (0, 0.0011, '0-0.11%'),
        (0.0011, 0.0015, '0.11-0.15%'),
        (0.0015, 0.0022, '0.15-0.22%'),
        (0.0022, 0.01, '0.22-1%'),
    ]
    error_data = []
    for method in methods:
        y_pred = results[method]["predictions"]
        y_true = results[method]["targets"]
        for bin_def in bins:
            if bin_def[0] == 'zero':
                mask = y_true == 0
                range_label = bin_def[1]
            else:
                low, high, range_label = bin_def
                mask = (y_true > low) & (y_true <= high)
            if mask.sum() > 0:
                mae_arr = np.abs(y_true[mask] - y_pred[mask])
                mae = np.mean(mae_arr)
                ci_lower, ci_upper = compute_95ci_bootstrap(mae_arr) if mask.sum() > 1 else (0.0, 0.0)
                error_data.append({
                    'Method': get_preprocessing_label(method),
                    'Range': range_label,
                    'MAE': mae,
                    'MAE_CI_Lower': ci_lower,
                    'MAE_CI_Upper': ci_upper,
                    'Count': int(mask.sum()),
                    'method_key': method
                })
    error_df = pd.DataFrame(error_data)
    if len(error_df) == 0:
        print("No data for zoomed MAE per range plot")
        return
    count_df = error_df[error_df['Method'] == get_preprocessing_label(methods[0])][['Range', 'Count']].set_index('Range')
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = error_df.pivot(index='Range', columns='Method', values='MAE')
    pivot_ci_lower = error_df.pivot(index='Range', columns='Method', values='MAE_CI_Lower')
    pivot_ci_upper = error_df.pivot(index='Range', columns='Method', values='MAE_CI_Upper')
    range_order = ['Zero', '0-0.11%', '0.11-0.15%', '0.15-0.22%', '0.22-1%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_ci_lower = pivot_ci_lower.reindex(range_order)
    pivot_ci_upper = pivot_ci_upper.reindex(range_order)
    method_labels_ordered = [get_preprocessing_label(m) for m in methods]
    pivot_df = pivot_df[[col for col in method_labels_ordered if col in pivot_df.columns]]
    pivot_ci_lower = pivot_ci_lower[[col for col in method_labels_ordered if col in pivot_ci_lower.columns]]
    pivot_ci_upper = pivot_ci_upper[[col for col in method_labels_ordered if col in pivot_ci_upper.columns]]
    colors = [get_preprocessing_color(m) for m in methods]
    mean_vals = pivot_df.values
    pivot_ci_lower = pivot_ci_lower.fillna(0)
    pivot_ci_upper = pivot_ci_upper.fillna(0)
    lower_err = np.abs(mean_vals - pivot_ci_lower.values)
    upper_err = np.abs(pivot_ci_upper.values - mean_vals)

    x = np.arange(len(range_order))
    width = 0.7 / len(pivot_df.columns)
    for i, col in enumerate(pivot_df.columns):
        offset = (i - len(pivot_df.columns) / 2 + 0.5) * width
        ax.bar(x + offset, pivot_df[col], width, label=col, color=colors[i],
               edgecolor='white', yerr=[lower_err[:, i], upper_err[:, i]],
               capsize=5, error_kw={'elinewidth': 2})
    range_labels_with_counts = []
    for r in range_order:
        if r in count_df.index:
            count = int(count_df.loc[r, 'Count'])
            range_labels_with_counts.append(f"{r}\n(n={count})")
        else:
            range_labels_with_counts.append(r)
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Abundance Range (Zoomed: <1%)', fontsize=14, fontweight='bold')
    ax.legend(title='Method', frameon=False, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_range_zoomed.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'error_by_range_zoomed.png')}")


def create_all_visualizations(results: Dict[str, Any], output_dir: str):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plot_metrics_comparison(results, output_dir)
    plot_scatter_actual_vs_predicted(results, output_dir)
    # Add log-log scatter plot
    plot_loglog_scatter_actual_vs_predicted(results, output_dir)
    # Add zoomed scatter plot for low abundance
    plot_scatter_zoomed(results, output_dir)
    # Add MAE per abundance range
    plot_mae_per_range(results, output_dir)
    # Add RAE per abundance range
    plot_RAE_per_range(results, output_dir)
    # Add zoomed MAE per range
    plot_mae_per_range_zoomed(results, output_dir)
    plot_residual_distribution(results, output_dir)
    plot_zero_vs_nonzero_comparison(results, output_dir)
    plot_summary_table(results, output_dir)

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"All figures saved to: {output_dir}")


def print_comparison(results: Dict[str, Any]):
    """Print text comparison of results."""
    methods = list(results.keys())
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPARISON SUMMARY")
    print("="*70)
    
    for method in methods:
        y_true = results[method]["targets"]
        y_pred = results[method]["predictions"]
        metrics = compute_extended_metrics(y_true, y_pred)
        
        print(f"\n{get_preprocessing_label(method)}:")
        print(f"  RMSE (micro): {metrics['RMSE_micro']:.4f}")
        print(f"  MAE (micro):  {metrics['MAE_micro']:.4f}")
        print(f"  R²:           {metrics['R²']:.4f}")
        print(f"  Correlation:  {metrics['Correlation']:.4f}")
        print(f"  KL Div:       {metrics['KL Divergence']:.4f}")
        print(f"  Rel Error:    {metrics['Absolute Relative Error']:.4f}")
    
    print("\n" + "="*70)


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from pickle file."""
    with open(results_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize read count preprocessing comparison results"
    )
    parser.add_argument("--results_path", type=str, default="results/preprocessing_results.pkl",
                       help="Path to results pickle file")
    parser.add_argument("--output_dir", type=str, default="figures",
                       help="Output directory for figures")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = args.results_path
    if not os.path.isabs(results_path):
        results_path = os.path.join(script_dir, results_path)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)

    # Load results
    results = load_results(results_path)
    
    # Print comparison
    print_comparison(results)
    
    # Create visualizations
    create_all_visualizations(results, output_dir)
