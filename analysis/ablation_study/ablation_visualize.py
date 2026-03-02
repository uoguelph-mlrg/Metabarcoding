#!/usr/bin/env python
"""
Visualization module for ablation study results.
Creates clean, presentation-ready plots comparing model variants.

This script is separate from ablation_study.py to allow re-generating visualizations
without re-running the expensive training.

Usage:
    python ablation_visualize.py --results_path figures/ablation_results.pkl --output_dir figures
"""
from __future__ import annotations

import argparse
import os
import pickle
import logging as log
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.patches import Patch


# ============================================================================
# Style Configuration
# ============================================================================

# Consistent model colors - support up to 5 models
MODEL_COLORS = {
    "mlp_latent": "#ff7f0e",        # Orange - MLP + Latent
    "mlp_no_taxonomy": "#1f77b4",   # Blue - MLP without taxonomy
    "mlp_with_taxonomy": "#2ca02c", # Green - MLP with taxonomy
    # Fallbacks for other model names
    "mlp_taxonomy": "#1f77b4",      # Blue (legacy)
    "original": "#ff7f0e",          # Orange (legacy)
}

MODEL_LABELS = {
    "mlp_latent": "MLP + Latent",
    "mlp_no_taxonomy": "MLP (no taxonomy)",
    "mlp_with_taxonomy": "MLP (with taxonomy)",
    # Legacy names
    "mlp_taxonomy": "MLP only",
    "original": "MLP + Latent"
}

# Color shades for zero/non-zero comparisons (lighter/darker for each model)
MODEL_COLOR_SHADES = {
    "mlp_latent": {"zero": "#ffcc99", "nonzero": "#ff7f0e"},        # light/dark orange
    "mlp_no_taxonomy": {"zero": "#a8d4ff", "nonzero": "#1f77b4"},   # light/dark blue
    "mlp_with_taxonomy": {"zero": "#a8e6a8", "nonzero": "#2ca02c"}, # light/dark green
    # Legacy
    "mlp_taxonomy": {"zero": "#a8d4ff", "nonzero": "#1f77b4"},
    "original": {"zero": "#ffcc99", "nonzero": "#ff7f0e"},
}


def get_model_label(model_name: str) -> str:
    """Get display label for a model."""
    # Try to get from predefined labels, otherwise use the name from results
    if model_name in MODEL_LABELS:
        return MODEL_LABELS[model_name]
    return model_name


def get_model_color(model_name: str) -> str:
    """Get consistent color for a model."""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    # Generate a color for unknown models
    colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    return colors[hash(model_name) % len(colors)]


def get_model_color_shades(model_name: str) -> Dict[str, str]:
    """Get color shades for zero/non-zero comparison."""
    if model_name in MODEL_COLOR_SHADES:
        return MODEL_COLOR_SHADES[model_name]
    # Generate shades for unknown models
    base_color = get_model_color(model_name)
    return {"zero": base_color + "66", "nonzero": base_color}


def compute_95ci_bootstrap(errors: np.ndarray, n_bootstrap: int = 1000) -> float:
    """Compute 95% confidence interval using bootstrap resampling."""
    n = len(errors)
    if n < 2:
        return 0.0
    
    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(errors, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # 95% CI from bootstrap distribution
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    mean_estimate = np.mean(errors)
    
    # Return the half-width (error bar)
    return max(ci_upper - mean_estimate, mean_estimate - ci_lower)


def _contrasting_text_color(hex_color: str) -> str:
    """Return 'white' or 'black' depending on background luminance."""
    import matplotlib.colors as mc
    try:
        r, g, b = mc.to_rgb(hex_color)
    except ValueError:
        return 'black'
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if luminance < 0.5 else 'black'


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

def _filter_finite_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask], int(mask.sum())

def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics.
    """
    y_true, y_pred, n_valid = _filter_finite_pairs(y_true, y_pred)
    if n_valid == 0:
        return {
            'MSE': float('nan'),
            'RMSE_micro': float('nan'),
            'RMSE_macro': float('nan'),
            'MAE_micro': float('nan'),
            'MAE_macro': float('nan'),
            'Absolute Relative Error': float('nan'),
            'R²': float('nan'),
            'RMSE (zeros)': float('nan'),
            'MAE (zeros)': float('nan'),
            'RMSE (non-zeros)': float('nan'),
            'MAE (non-zeros)': float('nan'),
            'KL Divergence': float('nan'),
            'Correlation': float('nan'),
            'n_zeros': 0,
            'n_nonzeros': 0,
        }

    y_pred = np.clip(y_pred, 0, 1)
    
    # Overall metrics (micro average)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse_micro = np.sqrt(mse)
    mae_micro = np.mean(np.abs(y_true - y_pred))

    # Macro average by sample (group by sample if available)
    rmse_macro = None
    mae_macro = None
    if hasattr(y_true, 'shape') and len(y_true.shape) == 2:
        # If y_true is (n_samples, n_bins)
        rmse_per_sample = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
        mae_per_sample = np.mean(np.abs(y_true - y_pred), axis=1)
        rmse_macro = np.mean(rmse_per_sample)
        mae_macro = np.mean(mae_per_sample)
    else:
        rmse_macro = rmse_micro
        mae_macro = mae_micro
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Split by zero/non-zero
    zero_mask = y_true == 0
    nonzero_mask = y_true > 0
    
    # Zero metrics
    if zero_mask.sum() > 0:
        rmse_zeros = np.sqrt(np.mean((y_true[zero_mask] - y_pred[zero_mask]) ** 2))
        mae_zeros = np.mean(np.abs(y_true[zero_mask] - y_pred[zero_mask]))
    else:
        rmse_zeros = 0.0
        mae_zeros = 0.0
    
    # Non-zero metrics
    if nonzero_mask.sum() > 0:
        rmse_nonzeros = np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask]) ** 2))
        mae_nonzeros = np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]))
    else:
        rmse_nonzeros = 0.0
        mae_nonzeros = 0.0
    
    # KL Divergence (with smoothing to avoid log(0))
    epsilon = 1e-10
    y_true_smooth = y_true + epsilon
    y_pred_smooth = y_pred + epsilon
    y_true_norm = y_true_smooth / y_true_smooth.sum()
    y_pred_norm = y_pred_smooth / y_pred_smooth.sum()
    kl_divergence = np.sum(y_true_norm * np.log(y_true_norm / y_pred_norm))
    
    # Correlation
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
        
    # Ablation Relative Error: |prediction - ground_truth| / ground_truth
    # Avoid division by zero by masking ground_truth == 0
    rel_error = np.zeros_like(y_true, dtype=float)
    nonzero_mask = y_true != 0
    rel_error[nonzero_mask] = np.abs(y_pred[nonzero_mask] - y_true[nonzero_mask]) / np.abs(y_true[nonzero_mask])
    absolute_relative_error = np.mean(rel_error[nonzero_mask]) if np.any(nonzero_mask) else 0.0
    
    return {
        'MSE': mse,
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


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_metrics_comparison(results: Dict[str, Any], output_dir: str):
    """Create bar plots comparing key metrics between models."""
    set_style()
    
    models = list(results.keys())
    n_models = len(models)
    
    # Compute extended metrics for all models
    extended_metrics = {}
    for model in models:
        preds = results[model]["predictions"]
        targets = results[model]["targets"]
        extended_metrics[model] = compute_extended_metrics(targets, preds)
    
    # Metrics to plot
    metrics_to_plot = ['RMSE_micro', 'MAE_micro', 'Absolute Relative Error', 'KL Divergence', 'MAE (zeros)', 'MAE (non-zeros)', 'Correlation']
    n_metrics = len(metrics_to_plot)

    # Compute 95% CIs per metric per model
    metric_cis: Dict[str, Dict[str, float]] = {}
    for m in models:
        y_true_m, y_pred_m, n_valid = _filter_finite_pairs(results[m]["targets"], results[m]["predictions"])
        if n_valid < 2:
            metric_cis[m] = {k: 0.0 for k in metrics_to_plot}
            continue
        y_pred_m = np.clip(y_pred_m, 0, 1)
        abs_err = np.abs(y_true_m - y_pred_m)
        sq_err = (y_true_m - y_pred_m) ** 2
        rmse_val = float(np.sqrt(np.mean(sq_err)))
        zero_m = y_true_m == 0
        nz_m = y_true_m > 0
        nonzero_rel = y_true_m != 0
        rel_err = np.zeros_like(y_true_m)
        rel_err[nonzero_rel] = abs_err[nonzero_rel] / np.abs(y_true_m[nonzero_rel])
        metric_cis[m] = {
            'RMSE_micro': compute_95ci_bootstrap(sq_err) / (2 * max(rmse_val, 1e-10)),
            'MAE_micro': compute_95ci_bootstrap(abs_err),
            'Absolute Relative Error': compute_95ci_bootstrap(rel_err[nonzero_rel]) if nonzero_rel.sum() > 1 else 0.0,
            'KL Divergence': 0.0,  # Not applicable for bootstrap CI
            'MAE (zeros)': compute_95ci_bootstrap(abs_err[zero_m]) if zero_m.sum() > 1 else 0.0,
            'MAE (non-zeros)': compute_95ci_bootstrap(abs_err[nz_m]) if nz_m.sum() > 1 else 0.0,
            'Correlation': 0.0,  # Not applicable for bootstrap CI
        }

    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 4))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [extended_metrics[m][metric] for m in models]
        plot_values = [v if np.isfinite(v) else 0.0 for v in values]
        ci_vals = [metric_cis[m].get(metric, 0.0) for m in models]
        ci_vals = [v if np.isfinite(v) else 0.0 for v in ci_vals]
        colors = [get_model_color(m) for m in models]
        labels = [get_model_label(m) for m in models]
        
        bars = ax.bar(range(len(models)), plot_values, color=colors, edgecolor='white', linewidth=1.5,
                      yerr=ci_vals, capsize=4, error_kw={'elinewidth': 1.5})
        
        # Add value labels on bars
        for bar, val, ci in zip(bars, values, ci_vals):
            label = f'{val:.4f}' if np.isfinite(val) else 'nan'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + max(plot_values) * 0.02,
                    label, ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_title(metric, fontsize=11, fontweight='bold')
        max_val = max(plot_values) if plot_values else 0.0
        top = max_val * 1.25
        if not np.isfinite(top) or top <= 0:
            top = 1.0
        ax.set_ylim(0, top)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        sns.despine(ax=ax)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved metrics comparison plot to {output_dir}/metrics_comparison.png")


def plot_scatter_actual_vs_predicted(results: Dict[str, Any], output_dir: str):
    """Create scatter plots of actual vs predicted values with dynamic axis limits."""
    set_style()
    
    models = list(results.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    # First pass: compute densities and find global max for axis limits
    all_densities = []
    scatter_data = []
    global_max_target = 0
    global_max_pred = 0  # Track max predicted across ALL models
    
    for model in models:
        preds_raw = results[model]["predictions"]
        targets_raw = results[model]["targets"]
        targets, preds, n_valid = _filter_finite_pairs(targets_raw, preds_raw)

        if n_valid == 0:
            log.warning(f"No finite prediction/target pairs for {model}")
            scatter_data.append((model, np.array([]), np.array([]), np.array([])))
            continue

        global_max_target = max(global_max_target, targets.max())
        global_max_pred = max(global_max_pred, preds.max())

        try:
            xy = np.vstack([preds, targets])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {model}: {e}")
            density = np.ones(len(preds))

        all_densities.extend(density)
        scatter_data.append((model, preds, targets, density))
    
    # Create shared normalization
    if len(all_densities) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    sc = None
    if global_max_target <= 0:
        global_max_target = 1.0
    if global_max_pred <= 0:
        global_max_pred = 1.0
    # Set axis limits to accommodate max predicted value across all models
    limit_x = global_max_target * 1.05
    limit_y = global_max_pred * 1.05

    for ax, (model, preds, targets, density) in zip(axes, scatter_data):
        if preds.size == 0:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes,
                    ha='center', va='center', fontsize=11)
            ax.set_xlabel("Actual", fontsize=11)
            ax.set_ylabel("Predicted", fontsize=11)
            ax.set_title(f"{get_model_label(model)}\n(No valid data)", fontsize=12, fontweight='bold')
            ax.set_xlim(0, limit_x)
            ax.set_ylim(0, limit_y)  # All subplots have same y-axis range
            sns.despine(ax=ax)
            continue

        idx = density.argsort()
        preds_sorted, targets_sorted, density_sorted = preds[idx], targets[idx], density[idx]
        
        sc = ax.scatter(targets_sorted, preds_sorted, c=density_sorted, 
                        cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        
        axis_max = max(global_max_target, global_max_pred)
        ax.plot([0, axis_max], [0, axis_max], 'r--', lw=1.5, alpha=0.7, label='Perfect')
        
        corr = np.corrcoef(targets, preds)[0, 1] if len(preds) > 1 else 0.0
        
        ax.set_xlabel("Actual", fontsize=11)
        ax.set_ylabel("Predicted", fontsize=11)
        ax.set_title(f"{get_model_label(model)}\n(Pearson r = {corr:.3f})", fontsize=12, fontweight='bold')
        ax.set_xlim(0, limit_x)
        ax.set_ylim(0, limit_y)  # All subplots have same y-axis range
        sns.despine(ax=ax)
    
    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point Density', fontsize=10)
    
    plt.savefig(os.path.join(output_dir, "scatter_predicted_vs_actual.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved scatter plot to {output_dir}/scatter_predicted_vs_actual.png")


def plot_scatter_zoomed(results: Dict[str, Any], output_dir: str, max_actual: float = 0.01):
    """Create scatter plots zoomed on ground truth <1% range."""
    set_style()
    
    models = list(results.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    all_densities = []
    scatter_data = []
    global_max_target = 0
    global_max_pred = 0
    
    for model in models:
        targets_raw = results[model]["targets"]
        preds_raw = results[model]["predictions"]
        targets, preds, n_valid = _filter_finite_pairs(targets_raw, preds_raw)

        if n_valid == 0:
            log.warning(f"No finite prediction/target pairs for {model}")
            scatter_data.append((model, np.array([]), np.array([]), np.array([])))
            continue

        zoom_mask = targets < max_actual
        targets_zoomed = targets[zoom_mask]
        preds_zoomed = preds[zoom_mask]

        if targets_zoomed.size == 0:
            log.warning(f"No zoomed data for {model} below {max_actual}")
            scatter_data.append((model, np.array([]), np.array([]), np.array([])))
            continue

        global_max_target = max(global_max_target, targets_zoomed.max())
        global_max_pred = max(global_max_pred, preds_zoomed.max())

        try:
            xy = np.vstack([preds_zoomed, targets_zoomed])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {model}: {e}")
            density = np.ones(len(preds_zoomed))

        all_densities.extend(density)
        scatter_data.append((model, preds_zoomed, targets_zoomed, density))
    
    if len(all_densities) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    sc = None
    if global_max_target <= 0:
        global_max_target = 1.0
    if global_max_pred <= 0:
        global_max_pred = 1.0

    for ax, (model, preds_zoomed, targets_zoomed, density) in zip(axes, scatter_data):
        if preds_zoomed.size == 0:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes,
                    ha='center', va='center', fontsize=11)
            ax.set_xlabel("Actual", fontsize=11)
            ax.set_ylabel("Predicted", fontsize=11)
            ax.set_title(f"{get_model_label(model)}\n(No valid data)", fontsize=12, fontweight='bold')
            ax.set_xlim(0, global_max_target)
            ax.set_ylim(0, global_max_pred)
            sns.despine(ax=ax)
            continue

        idx = density.argsort()
        preds_sorted, targets_sorted, density_sorted = preds_zoomed[idx], targets_zoomed[idx], density[idx]
        
        sc = ax.scatter(targets_sorted, preds_sorted, c=density_sorted, 
                        cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        
        axis_max = max(global_max_target, global_max_pred)
        ax.plot([0, axis_max], [0, axis_max], 'r--', lw=1.5, alpha=0.7)
        
        corr = np.corrcoef(targets_zoomed, preds_zoomed)[0, 1] if len(preds_zoomed) > 1 else 0.0
        
        ax.set_xlabel("Actual", fontsize=11)
        ax.set_ylabel("Predicted", fontsize=11)
        ax.set_title(f"{get_model_label(model)}\n(Pearson r = {corr:.3f})", fontsize=12, fontweight='bold')
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
    log.info(f"Saved zoomed scatter plot to {output_dir}/scatter_zoomed.png")
    
    
def plot_loglog_scatter_actual_vs_predicted(results: Dict[str, Any], output_dir: str):
    """Create log-log scatter plots of actual vs predicted values."""
    set_style()
    models = list(results.keys())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    # Compute densities and global min/max for color normalization
    all_densities = []
    scatter_data = []
    global_min = float('inf')
    global_max = float('-inf')

    for model in models:
        preds_raw = results[model]["predictions"]
        targets_raw = results[model]["targets"]
        targets, preds, n_valid = _filter_finite_pairs(targets_raw, preds_raw)
        if n_valid == 0:
            log.warning(f"No finite prediction/target pairs for {model}")
            scatter_data.append((model, np.array([]), np.array([]), np.array([])))
            continue

        preds = np.clip(preds, 0, None)
        targets = np.clip(targets, 0, None)
        epsilon = 10**(-3.5)  # 1e-3.5
        preds_log = np.log10(preds + epsilon)
        targets_log = np.log10(targets + epsilon)
        finite_mask = np.isfinite(preds_log) & np.isfinite(targets_log)
        preds_log = preds_log[finite_mask]
        targets_log = targets_log[finite_mask]
        if preds_log.size == 0:
            log.warning(f"No finite log-log pairs for {model}")
            scatter_data.append((model, np.array([]), np.array([]), np.array([])))
            continue
        try:
            xy = np.vstack([targets_log, preds_log])
            xy = xy + np.random.normal(0, 1e-8, xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception as e:
            log.warning(f"Could not compute density for {model}: {e}")
            density = np.ones(len(preds_log))
        all_densities.extend(density)
        scatter_data.append((model, preds_log, targets_log, density))
        global_min = min(global_min, np.min(targets_log), np.min(preds_log))
        global_max = max(global_max, np.max(targets_log), np.max(preds_log))

    if len(all_densities) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)

    sc = None
    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_min == float('inf'):
        global_min, global_max = -4.0, 0.0

    for ax, (model, preds_log, targets_log, density) in zip(axes, scatter_data):
        if preds_log.size == 0:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes,
                    ha='center', va='center', fontsize=11)
            ax.set_xlabel("Log10 Actual", fontsize=11)
            ax.set_ylabel("Log10 Predicted", fontsize=11)
            ax.set_title(f"{get_model_label(model)}\n(No valid data)", fontsize=12, fontweight='bold')
            sns.despine(ax=ax)
            continue

        idx = density.argsort()
        preds_sorted, targets_sorted, density_sorted = preds_log[idx], targets_log[idx], density[idx]
        sc = ax.scatter(targets_sorted, preds_sorted, c=density_sorted, cmap='viridis', norm=norm, s=8, alpha=0.6, edgecolors='none')
        ax.plot([global_min, global_max], [global_min, global_max], 'r--', lw=1.5, alpha=0.7, label='Perfect')
        corr = np.corrcoef(targets_log, preds_log)[0, 1] if len(preds_log) > 1 else 0.0
        ax.set_xlabel("Log10 Actual", fontsize=11)
        ax.set_ylabel("Log10 Predicted", fontsize=11)
        ax.set_title(f"{get_model_label(model)}\n(Log-Log Pearson r = {corr:.3f})", fontsize=12, fontweight='bold')
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point Density', fontsize=10)

    plt.savefig(os.path.join(output_dir, "scatter_loglog_predicted_vs_actual.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved log-log scatter plot to {output_dir}/scatter_loglog_predicted_vs_actual.png")



def plot_mae_per_range(results: Dict[str, Any], output_dir: str):
    """Create bar plot of MAE per target value range."""
    set_style()
    
    models = list(results.keys())
    
    bins = [
        ('zero', 'Zero'),
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%'),
    ]
    
    error_data = []
    for model_name in models:
        y_true, y_pred, n_valid = _filter_finite_pairs(
            results[model_name]["targets"],
            results[model_name]["predictions"],
        )
        if n_valid == 0:
            log.warning(f"No finite prediction/target pairs for {model_name}")
            continue
        
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
                mae_ci = compute_95ci_bootstrap(abs_errors) if mask.sum() > 1 else 0.0
                error_data.append({
                    'Model': get_model_label(model_name),
                    'Range': range_label,
                    'MAE': mae,
                    'MAE_ci': mae_ci,
                    'Count': int(mask.sum()),
                    'model_key': model_name
                })
    
    error_df = pd.DataFrame(error_data)
    
    if len(error_df) == 0:
        log.warning("No data for MAE per range plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_df = error_df.pivot(index='Range', columns='Model', values='MAE')
    pivot_ci_df = error_df.pivot(index='Range', columns='Model', values='MAE_ci')
    pivot_count_df = error_df.pivot(index='Range', columns='Model', values='Count')
    
    range_order = ['Zero', '>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_ci_df = pivot_ci_df.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    
    model_labels_ordered = [get_model_label(m) for m in models]
    pivot_df = pivot_df[[col for col in model_labels_ordered if col in pivot_df.columns]]
    pivot_ci_df = pivot_ci_df[[col for col in model_labels_ordered if col in pivot_ci_df.columns]]
    pivot_count_df = pivot_count_df[[col for col in model_labels_ordered if col in pivot_count_df.columns]]
    
    pivot_ci_df = pivot_ci_df.fillna(0)
    
    # counts for first model (same for all)
    first_label = get_model_label(models[0])
    count_by_range = error_df[error_df['Model'] == first_label][['Range', 'Count']].set_index('Range')
    
    colors = [get_model_color(m) for m in models]
    pivot_df.plot(kind='bar', ax=ax, color=colors, width=0.7, edgecolor='white',
                  yerr=pivot_ci_df, capsize=5, error_kw={'elinewidth': 2})
    
    range_labels_with_counts = []
    for r in range_order:
        if r in count_by_range.index:
            count = int(count_by_range.loc[r, 'Count'])
            range_labels_with_counts.append(f'{r}\n(n={count:,})')
        else:
            range_labels_with_counts.append(r)
    
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Abundance Range', fontsize=14, fontweight='bold')
    ax.legend(title='Model', frameon=False, loc='upper left')
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_range.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved MAE per range plot to {output_dir}/error_by_range.png")
    
    
def plot_RAE_per_range(results: Dict[str, Any], output_dir: str):
    """Create bar plot of Relative Absolute Error per target value range."""
    set_style()
    
    models = list(results.keys())
    
    bins = [
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%'),
    ]
    
    error_data = []
    for model_name in models:
        y_true, y_pred, n_valid = _filter_finite_pairs(
            results[model_name]["targets"],
            results[model_name]["predictions"],
        )
        if n_valid == 0:
            log.warning(f"No finite prediction/target pairs for {model_name}")
            continue
        for bin_def in bins:
            low, high, range_label = bin_def
            mask = (y_true > low) & (y_true <= high)
            if mask.sum() > 0:
                rel_error = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask] + 1e-10)
                rae = np.mean(rel_error)
                rae_std = np.std(rel_error, ddof=1) if mask.sum() > 1 else 0.0
                error_data.append({
                    'Model': get_model_label(model_name),
                    'Range': range_label,
                    'RAE': rae,
                    'RAE_std': rae_std,
                    'Count': int(mask.sum()),
                    'model_key': model_name
                })
    
    error_df = pd.DataFrame(error_data)
    
    if len(error_df) == 0:
        log.warning("No data for RAE per range plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_df = error_df.pivot(index='Range', columns='Model', values='RAE')
    pivot_std_df = error_df.pivot(index='Range', columns='Model', values='RAE_std')
    pivot_count_df = error_df.pivot(index='Range', columns='Model', values='Count')
    
    range_order = ['>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_std_df = pivot_std_df.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    
    model_labels_ordered = [get_model_label(m) for m in models]
    pivot_df = pivot_df[[col for col in model_labels_ordered if col in pivot_df.columns]]
    pivot_std_df = pivot_std_df[[col for col in model_labels_ordered if col in pivot_std_df.columns]]
    pivot_count_df = pivot_count_df[[col for col in model_labels_ordered if col in pivot_count_df.columns]]
    
    # 95% CI
    pivot_ci_df = pivot_std_df * 1.96 / pivot_count_df.apply(np.sqrt)
    pivot_ci_df = pivot_ci_df.fillna(0)
    
    # counts for first model
    first_label = get_model_label(models[0])
    count_by_range = error_df[error_df['Model'] == first_label][['Range', 'Count']].set_index('Range')
    
    colors = [get_model_color(m) for m in models]
    pivot_df.plot(kind='bar', ax=ax, color=colors, width=0.7, edgecolor='white',
                  yerr=pivot_ci_df, capsize=5, error_kw={'elinewidth': 2})
    
    range_labels_with_counts = []
    for r in range_order:
        if r in count_by_range.index:
            count = int(count_by_range.loc[r, 'Count'])
            range_labels_with_counts.append(f'{r}\n(n={count:,})')
        else:
            range_labels_with_counts.append(r)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Relative Absolute Error', fontsize=12)
    ax.set_title('Relative Absolute Error by Abundance Range', fontsize=14, fontweight='bold')
    ax.legend(title='Model', frameon=False, loc='upper right')
    
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "relative_err_by_range.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved RAE per range plot to {output_dir}/relative_err_by_range.png")


def plot_mae_per_range_zoomed(results: Dict[str, Any], output_dir: str):
    """Create bar plot of MAE per range zoomed with quantile bins."""
    set_style()
    
    models = list(results.keys())
    
    bins = [
        ('zero', 'Zero'),
        (0, 0.0011, '0-0.11%'),
        (0.0011, 0.0015, '0.11-0.15%'),
        (0.0015, 0.0022, '0.15-0.22%'),
        (0.0022, 0.01, '0.22-1%'),
    ]
    
    error_data = []
    for model_name in models:
        y_true, y_pred, n_valid = _filter_finite_pairs(
            results[model_name]["targets"],
            results[model_name]["predictions"],
        )
        if n_valid == 0:
            log.warning(f"No finite prediction/target pairs for {model_name}")
            continue
        
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
                mae_ci = compute_95ci_bootstrap(abs_errors) if mask.sum() > 1 else 0.0
                error_data.append({
                    'Model': get_model_label(model_name),
                    'Range': range_label,
                    'MAE': mae,
                    'MAE_ci': mae_ci,
                    'Count': int(mask.sum()),
                    'model_key': model_name
                })
    
    error_df = pd.DataFrame(error_data)
    
    if len(error_df) == 0:
        log.warning("No data for zoomed MAE per range plot")
        return
    
    count_df = error_df[error_df['Model'] == get_model_label(models[0])][['Range', 'Count']].set_index('Range')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_df = error_df.pivot(index='Range', columns='Model', values='MAE')
    pivot_ci_df = error_df.pivot(index='Range', columns='Model', values='MAE_ci')
    pivot_count_df = error_df.pivot(index='Range', columns='Model', values='Count')
    
    range_order = ['Zero', '0-0.11%', '0.11-0.15%', '0.15-0.22%', '0.22-1%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_ci_df = pivot_ci_df.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    
    model_labels_ordered = [get_model_label(m) for m in models]
    pivot_df = pivot_df[[col for col in model_labels_ordered if col in pivot_df.columns]]
    pivot_ci_df = pivot_ci_df[[col for col in model_labels_ordered if col in pivot_ci_df.columns]]
    pivot_count_df = pivot_count_df[[col for col in model_labels_ordered if col in pivot_count_df.columns]]
    pivot_ci_df = pivot_ci_df.fillna(0)
    
    colors = [get_model_color(m) for m in models]
    pivot_df.plot(kind='bar', ax=ax, color=colors, width=0.7, edgecolor='white',
                  yerr=pivot_ci_df, capsize=5, error_kw={'elinewidth': 2})
    
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
    ax.legend(title='Model', frameon=False, loc='upper left')
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_range_zoomed.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved zoomed MAE per range plot to {output_dir}/error_by_range_zoomed.png")


def plot_residual_distribution(results: Dict[str, Any], output_dir: str):
    """Create overlapping residual distribution histograms with KDE for all models."""
    from scipy.stats import gaussian_kde
    set_style()
    
    models = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    all_residuals_flat = []
    model_residuals = {}
    for model in models:
        targets, preds, n_valid = _filter_finite_pairs(
            results[model]["targets"],
            results[model]["predictions"],
        )
        if n_valid == 0:
            log.warning(f"No finite prediction/target pairs for {model}")
            continue
        residuals = targets - preds
        model_residuals[model] = residuals
        all_residuals_flat.extend(residuals.tolist())
    
    if len(all_residuals_flat) == 0:
        log.warning("No residuals available for plotting")
        return
    
    global_min = min(all_residuals_flat)
    global_max = max(all_residuals_flat)
    x_kde = np.linspace(global_min, global_max, 300)
    bin_count = 60
    bin_width = (global_max - global_min) / bin_count if global_max > global_min else 1.0
    
    legend_handles = []
    for model, residuals in model_residuals.items():
        color = get_model_color(model)
        label = get_model_label(model)
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        
        ax.hist(residuals, bins=bin_count, color=color, alpha=0.35, density=False, edgecolor='none')
        
        if len(residuals) > 1:
            kde = gaussian_kde(residuals)
            kde_vals = kde(x_kde)
            kde_counts = kde_vals * len(residuals) * bin_width
            line, = ax.plot(x_kde, kde_counts, color=color, linewidth=2)
            handle = plt.Line2D([0], [0], color=color, linewidth=2,
                                label=f'{label} (μ={mean_res:.4f}, σ={std_res:.4f})')
            legend_handles.append(handle)
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Residual Distributions', fontsize=13, fontweight='bold')
    ax.legend(handles=legend_handles, frameon=False, fontsize=9)
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.5)
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved residual distribution plot to {output_dir}/residual_distribution.png")


def plot_zero_vs_nonzero_comparison(results: Dict[str, Any], output_dir: str):
    """Create a comparison plot focusing on zero vs non-zero MAE performance."""
    set_style()
    
    models = list(results.keys())
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(n_models)
    width = 0.35
    
    mae_zero = []
    mae_zero_ci = []
    mae_nonzero = []
    mae_nonzero_ci = []
    
    for model in models:
        targets, preds, _ = _filter_finite_pairs(
            results[model]["targets"],
            results[model]["predictions"],
        )
        zero_mask = targets == 0
        nonzero_mask = targets != 0
        
        if zero_mask.sum() > 0:
            errs_z = np.abs(targets[zero_mask] - preds[zero_mask])
            mae_zero.append(np.mean(errs_z))
            mae_zero_ci.append(compute_95ci_bootstrap(errs_z) if zero_mask.sum() > 1 else 0.0)
        else:
            mae_zero.append(0.0)
            mae_zero_ci.append(0.0)
        
        if nonzero_mask.sum() > 0:
            errs_nz = np.abs(targets[nonzero_mask] - preds[nonzero_mask])
            mae_nonzero.append(np.mean(errs_nz))
            mae_nonzero_ci.append(compute_95ci_bootstrap(errs_nz) if nonzero_mask.sum() > 1 else 0.0)
        else:
            mae_nonzero.append(0.0)
            mae_nonzero_ci.append(0.0)
    
    zero_color = '#aaaaaa'
    nonzero_color = '#444444'
    
    ax.bar(x - width/2, mae_zero, width, color=zero_color, edgecolor='white',
           yerr=mae_zero_ci, capsize=5, error_kw={'elinewidth': 2})
    ax.bar(x + width/2, mae_nonzero, width, color=nonzero_color, edgecolor='white',
           yerr=mae_nonzero_ci, capsize=5, error_kw={'elinewidth': 2})
    
    legend_elements = [
        Patch(facecolor=zero_color, edgecolor='white', label='Zero values (lighter)'),
        Patch(facecolor=nonzero_color, edgecolor='white', label='Non-zero values (darker)')
    ]
    ax.legend(handles=legend_elements, frameon=False, loc='upper left')
    
    ax.set_xticks(x)
    ax.set_xticklabels([get_model_label(m) for m in models], rotation=15, ha='right')
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE: Zero vs Non-Zero Values', fontsize=13, fontweight='bold')
    sns.despine(ax=ax)
    
    for i, (val_z, val_nz) in enumerate(zip(mae_zero, mae_nonzero)):
        ax.text(x[i] - width/2, val_z * 1.02, f'{val_z:.4f}', ha='center', va='bottom', fontsize=8)
        ax.text(x[i] + width/2, val_nz * 1.02, f'{val_nz:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zero_vs_nonzero_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved zero vs non-zero comparison plot to {output_dir}/zero_vs_nonzero_comparison.png")


def plot_summary_table(results: Dict[str, Any], output_dir: str):
    """Create a clean summary table as an image."""
    set_style()
    
    models = list(results.keys())
    
    extended_metrics = {}
    for model in models:
        preds = results[model]["predictions"]
        targets = results[model]["targets"]
        extended_metrics[model] = compute_extended_metrics(targets, preds)
    
    metrics = ['RMSE_micro', 'MAE_micro', 'Absolute Relative Error', 'KL Divergence', 'MAE (zeros)', 'MAE (non-zeros)', 'Correlation']
    
    data = []
    for model in models:
        row = {'Model': get_model_label(model)}
        for metric in metrics:
            row[metric] = extended_metrics[model][metric]
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Determine best values, skipping non-finite entries
    best_indices = {}
    for col in metrics:
        series = pd.to_numeric(df[col], errors='coerce')
        if series.notna().any():
            if col in ['Correlation']:
                best_indices[col] = series.idxmax()
            else:
                best_indices[col] = series.idxmin()
    
    display_df = df.copy()
    for col in metrics:
        display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')
    
    n_models = len(models)
    col_list = list(display_df.columns)
    fig, ax = plt.subplots(figsize=(15, 1.2 + 0.5 * n_models))
    ax.axis('off')
    
    table = ax.table(
        cellText=display_df.values,
        colLabels=col_list,
        cellLoc='center',
        loc='center',
        colColours=['#e8e8e8'] * len(col_list)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Header row: light gray bold
    for i in range(len(col_list)):
        table[(0, i)].set_text_props(fontweight='bold')
        table[(0, i)].set_facecolor('#d0d0d0')
    
    # Color first column with each model's color
    model_col_idx = col_list.index('Model')
    for row_idx, model in enumerate(models):
        cell = table[(row_idx + 1, model_col_idx)]
        hex_color = get_model_color(model)
        cell.set_facecolor(hex_color)
        text_color = _contrasting_text_color(hex_color)
        cell.set_text_props(fontweight='bold', color=text_color)
    
    # Best value: bold dark green text + light green background
    for col_idx, col in enumerate(col_list):
        if col == 'Model':
            continue
        if col in best_indices:
            best_row = best_indices[col] + 1
            table[(best_row, col_idx)].set_facecolor('#d5f5e3')
            table[(best_row, col_idx)].set_text_props(fontweight='bold', color='#1a7a40')
    
    plt.title('Model Performance Summary', fontweight='bold', fontsize=14, pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved summary table to {output_dir}/summary_table.png")
    
    df.to_csv(os.path.join(output_dir, "model_results.csv"), index=False)
    log.info(f"Saved model results to {output_dir}/model_results.csv")
    
    return df


def create_all_visualizations(results: Dict[str, Any], output_dir: str):
    """Create all visualization plots."""
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
    
    log.info("4. Creating error by range plot...")
    plot_mae_per_range(results, output_dir)
    
    log.info("5. Creating zoomed error by range plot...")
    plot_mae_per_range_zoomed(results, output_dir)
    
    log.info("   Creating RAE by range plot...")
    plot_RAE_per_range(results, output_dir)
    
    log.info("   Creating log-log scatter plot...")
    plot_loglog_scatter_actual_vs_predicted(results, output_dir)
    
    log.info("6. Creating residual distribution plot...")
    plot_residual_distribution(results, output_dir)
    
    log.info("7. Creating zero vs non-zero comparison...")
    plot_zero_vs_nonzero_comparison(results, output_dir)
    
    log.info("8. Creating summary table...")
    results_df = plot_summary_table(results, output_dir)
    
    log.info(f"\n✅ All visualizations saved to: {output_dir}/")
    
    return results_df


def print_comparison(results: Dict[str, Any]):
    """Print comparison table with extended metrics."""
    log.info("\n" + "="*80)
    log.info("ABLATION STUDY RESULTS - COMPREHENSIVE METRICS")
    log.info("="*80)
    
    extended_metrics = {}
    for name, res in results.items():
        preds = res["predictions"]
        targets = res["targets"]
        extended_metrics[name] = compute_extended_metrics(targets, preds)
    
    headers = ["Model", "Features", "RMSE_micro", "MAE_micro", "Correlation"]
    row_format = "{:<25} {:<10} {:<12} {:<12} {:<12}"
    
    log.info(row_format.format(*headers))
    log.info("-" * 95)
    
    for name in results.keys():
        m = extended_metrics[name]
        n_feat = results[name].get("n_features", "N/A")
        log.info(row_format.format(
            get_model_label(name),
            str(n_feat),
            f"{m['RMSE_micro']:.6f}",
            f"{m['MAE_micro']:.6f}",
            f"{m['Correlation']:.4f}",
        ))
    
    log.info("\n" + "-" * 95)
    log.info("Zero vs Non-Zero Breakdown:")
    headers2 = ["Model", "MAE(zero)", "MAE(non-zero)", "RMSE(zero)", "RMSE(non-zero)"]
    row_format2 = "{:<25} {:<15} {:<18} {:<15} {:<18}"
    log.info(row_format2.format(*headers2))
    log.info("-" * 95)
    
    for name in results.keys():
        m = extended_metrics[name]
        log.info(row_format2.format(
            get_model_label(name),
            f"{m['MAE (zeros)']:.6f}",
            f"{m['MAE (non-zeros)']:.6f}",
            f"{m['RMSE (zeros)']:.6f}",
            f"{m['RMSE (non-zeros)']:.6f}",
        ))
    
    log.info("\n" + "-" * 95)
    
    valid_mse = [(k, v) for k, v in extended_metrics.items() if np.isfinite(v['MSE'])]
    valid_mae = [(k, v) for k, v in extended_metrics.items() if np.isfinite(v['MAE_micro'])]
    valid_corr = [(k, v) for k, v in extended_metrics.items() if np.isfinite(v['Correlation'])]

    if valid_mse:
        best_mse = min(valid_mse, key=lambda x: x[1]['MSE'])
        log.info(f"Best model by MSE: {get_model_label(best_mse[0])} ({best_mse[1]['MSE']:.6f})")
    else:
        log.warning("No finite MSE values to determine best model")

    if valid_mae:
        best_mae = min(valid_mae, key=lambda x: x[1]['MAE_micro'])
        log.info(f"Best model by MAE: {get_model_label(best_mae[0])} ({best_mae[1]['MAE_micro']:.6f})")
    else:
        log.warning("No finite MAE values to determine best model")

    if valid_corr:
        best_corr = max(valid_corr, key=lambda x: x[1]['Correlation'])
        log.info(f"Best model by Correlation: {get_model_label(best_corr[0])} ({best_corr[1]['Correlation']:.4f})")
    else:
        log.warning("No finite Correlation values to determine best model")


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from pickle file."""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Ablation Study Results")
    parser.add_argument("--results_path", type=str, default="results/ablation_results.pkl", 
                        help="Path to ablation_results.pkl file")
    parser.add_argument("--output_dir", type=str, default="figures", 
                        help="Output directory for plots")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = args.results_path
    if not os.path.isabs(results_path):
        results_path = os.path.join(script_dir, results_path)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)

    log.info(f"Loading results from {results_path}")
    results = load_results(results_path)
    
    print_comparison(results)
    
    create_all_visualizations(results, output_dir)
