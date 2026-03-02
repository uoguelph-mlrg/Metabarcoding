#!/usr/bin/env python
"""
Centralized visualization module for all analysis results.
Provides reusable, parameterized plotting functions that adapt to any number of comparisons.

This module is designed to replace individual visualization scripts while maintaining
flexibility for different analysis contexts (ablation studies, loss comparison, etc).

Key features:
- Adaptive layouts based on number of comparisons (1 row for <=3, 2 rows for 4-8, etc)
- Parameterizable colors, labels, and titles
- Automatic metric computation
- Consistent styling across all analyses
"""
from __future__ import annotations

import os
import math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.patches import Patch


# ============================================================================
# Style & Utility Functions
# ============================================================================

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


def _contrasting_text_color(hex_color: str) -> str:
    """Return 'white' or 'black' for a hex background color based on luminance."""
    try:
        r, g, b = mc.to_rgb(hex_color)
    except ValueError:
        return 'black'
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if luminance < 0.45 else 'black'


def compute_95ci(errors: np.ndarray) -> float:
    """Return 95% CI half-width for an array of per-sample errors."""
    n = len(errors)
    if n < 2:
        return 0.0
    return 1.96 * np.std(errors, ddof=1) / np.sqrt(n)


def get_layout_dims(n_comparisons: int) -> Tuple[int, int]:
    """
    Determine subplot layout dimensions based on number of comparisons.
    
    - <= 3: 1 row
    - 4-8: 2 rows
    - 9-15: 3 rows
    - >= 16: Use horizontal bars (returns special indicator)
    
    Returns:
        (n_rows, n_cols) tuple. If n_comparisons >= 16, returns (-1, -1) to signal horizontal layout.
    """
    if n_comparisons >= 16:
        return (-1, -1)  # Signal to use horizontal bars
    
    n_rows = max(1, math.floor(math.sqrt(n_comparisons)))
    n_cols = math.ceil(n_comparisons / n_rows)
    return (n_rows, n_cols)


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics for model evaluation.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics including RMSE, MAE, correlation, KL divergence, etc.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Filter out NaN/Inf values
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        return {k: 0.0 for k in ['RMSE_micro', 'MAE_micro', 'Correlation', 'KL Divergence',
                                  'Absolute Relative Error', 'MAE (zeros)', 'MAE (non-zeros)','R²']}
    
    # Clip predictions to [0, 1]
    y_pred = np.clip(y_pred, 0, 1)
    
    # Overall metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse_micro = np.sqrt(mse)
    mae_micro = np.mean(np.abs(y_true - y_pred))
    
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
    
    # KL Divergence
    epsilon = 1e-10
    y_true_smooth = y_true + epsilon
    y_pred_smooth = y_pred + epsilon
    y_true_norm = y_true_smooth / y_true_smooth.sum()
    y_pred_norm = y_pred_smooth / y_pred_smooth.sum()
    kl_divergence = np.sum(y_true_norm * np.log(y_true_norm / y_pred_norm))
    
    # Correlation
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        correlation = 0.0
    if np.isnan(correlation):
        correlation = 0.0
    
    # Absolute Relative Error
    rel_error = np.zeros_like(y_true, dtype=float)
    nonzero_mask_rel = y_true != 0
    rel_error[nonzero_mask_rel] = np.abs(y_pred[nonzero_mask_rel] - y_true[nonzero_mask_rel]) / np.abs(y_true[nonzero_mask_rel])
    absolute_relative_error = np.mean(rel_error[nonzero_mask_rel]) if np.any(nonzero_mask_rel) else 0.0
    
    return {
        'RMSE_micro': float(rmse_micro),
        'MAE_micro': float(mae_micro),
        'Absolute Relative Error': float(absolute_relative_error),
        'R²': float(r2),
        'RMSE (zeros)': float(rmse_zeros),
        'MAE (zeros)': float(mae_zeros),
        'RMSE (non-zeros)': float(rmse_nonzeros),
        'MAE (non-zeros)': float(mae_nonzeros),
        'KL Divergence': float(kl_divergence),
        'Correlation': float(correlation),
        'n_zeros': int(zero_mask.sum()),
        'n_nonzeros': int(nonzero_mask.sum()),
    }


# ============================================================================
# Plotting Functions - Adaptive Layout
# ============================================================================

def plot_metrics_comparison(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    # title argument removed
    metrics_to_plot: Optional[List[str]] = None,
    filename: str = "metrics_comparison.png"
) -> None:
    """
    Create bar plots comparing metrics across comparisons with adaptive layout.
    
    For >= 16 comparisons, creates horizontal bars instead of vertical.
    
    Args:
        results: Dict with comparison names as keys, each containing 'predictions' and 'targets'
        output_dir: Directory to save the plot
        comparison_names: Dict mapping result keys to display names
        comparison_colors: Dict mapping result keys to hex colors
        title: Plot title
        metrics_to_plot: List of metrics to display. If None, uses standard set
        filename: Output filename
    """
    set_style()
    
    comparison_keys = list(results.keys())
    n_comparisons = len(comparison_keys)
    
    if metrics_to_plot is None:
        metrics_to_plot = ['RMSE_micro', 'MAE_micro', 'Absolute Relative Error', 
                          'KL Divergence', 'MAE (zeros)', 'MAE (non-zeros)', 'Correlation']
    
    # Compute metrics for all comparisons
    extended_metrics = {}
    for key in comparison_keys:
        preds = results[key]["predictions"]
        targets = results[key]["targets"]
        extended_metrics[key] = compute_extended_metrics(targets, preds)
    
    # Determine layout
    n_rows, n_cols = get_layout_dims(n_comparisons)
    
    if n_rows == -1:  # Horizontal bars for >= 16 comparisons
        # Create single figure with horizontal bars
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 3 * len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = [extended_metrics[key][metric] for key in comparison_keys]
            uncertainties = [v * 0.05 for v in values]
            
            labels = [comparison_names.get(key, key) if comparison_names else key 
                     for key in comparison_keys]
            colors = [comparison_colors.get(key, '#3498db') if comparison_colors else '#3498db'
                     for key in comparison_keys]
            
            # Horizontal bars
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, values, xerr=uncertainties, color=colors, edgecolor='white', 
                   linewidth=1.5, capsize=4, error_kw={'elinewidth': 1.5})
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel(metric, fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            sns.despine(ax=ax)
        
        plt.suptitle("Performance Metrics Comparison", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
    else:  # Vertical bars for < 16 comparisons
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1 or n_cols == 1:
            axes = [[ax] for ax in axes.flatten()]
        else:
            axes = axes.reshape(n_rows, n_cols)
        
        axes = [ax for row in axes for ax in (row if isinstance(row, list) else [row])]
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break
            ax = axes[idx]
            
            values = [extended_metrics[key][metric] for key in comparison_keys]
            uncertainties = [v * 0.05 for v in values]
            
            labels = [comparison_names.get(key, key) if comparison_names else key 
                     for key in comparison_keys]
            colors = [comparison_colors.get(key, '#3498db') if comparison_colors else '#3498db'
                     for key in comparison_keys]
            
            bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5,
                         yerr=uncertainties, capsize=4, error_kw={'elinewidth': 1.5})
            
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylim(0, max(values) * 1.25)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            sns.despine(ax=ax)
        
        # Hide unused subplots
        for idx in range(len(metrics_to_plot), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle("Performance Metrics Comparison", fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter_predicted_vs_actual(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    # title argument removed
    filename: str = "scatter_predicted_vs_actual.png"
) -> None:
    """
    Create scatter plots with adaptive layout based on number of comparisons.
    
    Args:
        results: Dict with comparison names as keys
        output_dir: Output directory
        comparison_names: Dict mapping keys to display names
        comparison_colors: Dict mapping keys to hex colors
        title: Plot title
        filename: Output filename
    """
    set_style()
    
    comparison_keys = list(results.keys())
    n_comparisons = len(comparison_keys)
    
    # Determine layout
    n_rows, n_cols = get_layout_dims(n_comparisons)
    if n_rows == -1:
        n_rows, n_cols = (int(math.ceil(n_comparisons / 8)), 8)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_comparisons == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Compute global limits across all comparisons
    global_max_target = 0
    global_max_pred = 0
    
    for key in comparison_keys:
        targets = results[key]["targets"].flatten()
        preds = results[key]["predictions"].flatten()
        valid_mask = np.isfinite(targets) & np.isfinite(preds)
        targets = targets[valid_mask]
        preds = preds[valid_mask]
        global_max_target = max(global_max_target, targets.max())
        global_max_pred = max(global_max_pred, preds.max())
    
    # Create plots
    sc = None
    for idx, key in enumerate(comparison_keys):
        ax = axes_flat[idx]
        
        targets = results[key]["targets"].flatten()
        preds = results[key]["predictions"].flatten()
        valid_mask = np.isfinite(targets) & np.isfinite(preds)
        targets = targets[valid_mask]
        preds = preds[valid_mask]
        
        # Compute density
        try:
            xy = np.vstack([preds, targets]) + np.random.normal(0, 1e-8, (2, len(preds)))
            density = gaussian_kde(xy)(xy)
        except:
            density = np.ones(len(preds))
        
        # Sort by density
        sort_idx = density.argsort()
        preds_s, targets_s, density_s = preds[sort_idx], targets[sort_idx], density[sort_idx]
        
        sc = ax.scatter(targets_s, preds_s, c=density_s, cmap='viridis', s=8, alpha=0.6, edgecolors='none')
        
        # Diagonal line
        axis_max = max(global_max_target, global_max_pred)
        ax.plot([0, axis_max], [0, axis_max], 'r--', lw=1.5, alpha=0.7, label='Perfect')
        
        # Correlation
        corr = np.corrcoef(targets, preds)[0, 1] if np.std(targets) > 0 and np.std(preds) > 0 else 0.0
        if np.isnan(corr):
            corr = 0.0
        
        label = comparison_names.get(key, key) if comparison_names else key
        ax.set_xlabel("Actual", fontsize=11)
        ax.set_ylabel("Predicted", fontsize=11)
        ax.set_title(f"{label}\n(r = {corr:.3f})", fontsize=12, fontweight='bold')
        ax.set_xlim(0, global_max_target)
        ax.set_ylim(0, global_max_pred)
        sns.despine(ax=ax)
    
    # Hide unused subplots
    for idx in range(n_comparisons, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle("Predicted vs Actual", fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Add colorbar
    if sc is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('Density', fontsize=10)
    
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter_zoomed(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    max_actual: float = 0.01,
    # title argument removed
    filename: str = "scatter_zoomed.png"
) -> None:
    """
    Create zoomed scatter plots for ground truth < max_actual.
    """
    set_style()
    
    comparison_keys = list(results.keys())
    n_comparisons = len(comparison_keys)
    
    n_rows, n_cols = get_layout_dims(n_comparisons)
    if n_rows == -1:
        n_rows, n_cols = (int(math.ceil(n_comparisons / 8)), 8)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_comparisons == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Compute global limits
    global_max_target = 0
    global_max_pred = 0
    
    for key in comparison_keys:
        targets = results[key]["targets"].flatten()
        preds = results[key]["predictions"].flatten()
        valid_mask = np.isfinite(targets) & np.isfinite(preds)
        targets = targets[valid_mask]
        preds = preds[valid_mask]
        zoom_mask = targets < max_actual
        if zoom_mask.sum() > 0:
            global_max_target = max(global_max_target, targets[zoom_mask].max())
            global_max_pred = max(global_max_pred, preds[zoom_mask].max())
    
    # Create plots
    sc = None
    for idx, key in enumerate(comparison_keys):
        ax = axes_flat[idx]
        
        targets = results[key]["targets"].flatten()
        preds = results[key]["predictions"].flatten()
        valid_mask = np.isfinite(targets) & np.isfinite(preds)
        targets = targets[valid_mask]
        preds = preds[valid_mask]
        
        zoom_mask = targets < max_actual
        targets_z = targets[zoom_mask]
        preds_z = preds[zoom_mask]
        
        if len(targets_z) > 0:
            try:
                xy = np.vstack([preds_z, targets_z]) + np.random.normal(0, 1e-8, (2, len(preds_z)))
                density = gaussian_kde(xy)(xy)
            except:
                density = np.ones(len(preds_z))
            
            sort_idx = density.argsort()
            preds_s, targets_s, density_s = preds_z[sort_idx], targets_z[sort_idx], density[sort_idx]
            
            sc = ax.scatter(targets_s, preds_s, c=density_s, cmap='viridis', s=8, alpha=0.6, edgecolors='none')
            
            ax.plot([0, global_max_target], [0, global_max_target], 'r--', lw=1.5, alpha=0.7)
            
            corr = np.corrcoef(targets_z, preds_z)[0, 1] if np.std(targets_z) > 0 and np.std(preds_z) > 0 else 0.0
            if np.isnan(corr):
                corr = 0.0
            
            ax.set_xlim(0, global_max_target)
            ax.set_ylim(0, global_max_pred)
        
        label = comparison_names.get(key, key) if comparison_names else key
        ax.set_xlabel("Actual", fontsize=11)
        ax.set_ylabel("Predicted", fontsize=11)
        ax.set_title(f"{label}", fontsize=12, fontweight='bold')
        sns.despine(ax=ax)
    
    # Hide unused
    for idx in range(n_comparisons, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle("Predicted vs Actual (Zoomed <1%)", fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if sc is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('Density', fontsize=10)
    
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter_loglog_predicted_vs_actual(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    # title argument removed
    filename: str = "scatter_loglog_predicted_vs_actual.png"
) -> None:
    """
    Create log-log scatter plots with adaptive layout.
    """
    set_style()
    
    comparison_keys = list(results.keys())
    n_comparisons = len(comparison_keys)
    
    n_rows, n_cols = get_layout_dims(n_comparisons)
    if n_rows == -1:
        n_rows, n_cols = (int(math.ceil(n_comparisons / 8)), 8)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_comparisons == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for idx, key in enumerate(comparison_keys):
        ax = axes_flat[idx]
        
        targets = results[key]["targets"].flatten()
        preds = results[key]["predictions"].flatten()
        valid_mask = np.isfinite(targets) & np.isfinite(preds)
        targets = targets[valid_mask]
        preds = preds[valid_mask]
        
        # Filter to positive values
        pos_mask = (targets > 0) & (preds > 0)
        targets_pos = targets[pos_mask]
        preds_pos = preds[pos_mask]
        
        if len(targets_pos) > 1:
            try:
                xy = np.vstack([np.log10(preds_pos), np.log10(targets_pos)]) + np.random.normal(0, 1e-2, (2, len(preds_pos)))
                density = gaussian_kde(xy)(xy)
            except:
                density = np.ones(len(preds_pos))
            
            sort_idx = density.argsort()
            preds_s = preds_pos[sort_idx]
            targets_s = targets_pos[sort_idx]
            density_s = density[sort_idx]
            
            ax.scatter(targets_s, preds_s, c=density_s, cmap='viridis', s=8, alpha=0.6, edgecolors='none')
            
            # Diagonal
            axis_min = min(targets_s.min(), preds_s.min())
            axis_max = max(targets_s.max(), preds_s.max())
            ax.loglog([axis_min, axis_max], [axis_min, axis_max], 'r--', lw=1.5, alpha=0.7)
            
            corr = np.corrcoef(np.log10(targets_s), np.log10(preds_s))[0, 1] if len(targets_s) > 1 else 0.0
            if np.isnan(corr):
                corr = 0.0
        
        label = comparison_names.get(key, key) if comparison_names else key
        ax.set_xlabel("Actual (log scale)", fontsize=11)
        ax.set_ylabel("Predicted (log scale)", fontsize=11)
        ax.set_title(f"{label}", fontsize=12, fontweight='bold')
        sns.despine(ax=ax)
    
    # Hide unused
    for idx in range(n_comparisons, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle("Log-Log: Predicted vs Actual", fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_by_range(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    # title argument removed
    filename: str = "error_by_range.png"
) -> None:
    """
    Create grouped bar plot showing MAE by abundance range.
    """
    set_style()
    
    comparison_keys = list(results.keys())
    bins = [
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%')
    ]
    
    error_data = []
    for key in comparison_keys:
        targets = results[key]["targets"].flatten()
        preds = results[key]["predictions"].flatten()
        valid_mask = np.isfinite(targets) & np.isfinite(preds)
        targets = targets[valid_mask]
        preds = preds[valid_mask]
        
        for lo, hi, label in bins:
            mask = (targets >= lo) & (targets < hi)
            if mask.sum() > 0:
                mae = np.mean(np.abs(targets[mask] - preds[mask]))
                mae_std = np.std(np.abs(targets[mask] - preds[mask]), ddof=1) if mask.sum() > 1 else 0
                error_data.append({
                    'Comparison': comparison_names.get(key, key) if comparison_names else key,
                    'Range': label,
                    'MAE': mae,
                    'MAE_std': mae_std,
                    'Count': mask.sum()
                })
    
    df = pd.DataFrame(error_data)
    
    fig, ax = plt.subplots(figsize=(13, 6))
    pivot_df = df.pivot(index='Range', columns='Comparison', values='MAE')
    pivot_std = df.pivot(index='Range', columns='Comparison', values='MAE_std')
    pivot_count = df.pivot(index='Range', columns='Comparison', values='Count')
    
    range_order = ['>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_std = pivot_std.reindex(range_order)
    pivot_count = pivot_count.reindex(range_order)
    
    # Confidence intervals
    pivot_ci = pivot_std * 1.96 / pivot_count.apply(np.sqrt)
    pivot_ci = pivot_ci.fillna(0)
    
    colors = [comparison_colors.get(key, '#3498db') if comparison_colors else '#3498db'
             for key in pivot_df.columns]
    
    pivot_df.plot(kind='bar', ax=ax, color=colors, width=0.8, edgecolor='white',
                 yerr=pivot_ci, capsize=5, error_kw={'elinewidth': 2})
    
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title("MAE by Abundance Range", fontsize=14, fontweight='bold')
    ax.legend(title='Comparison', frameon=False, fontsize=9, loc='upper right')
    ax.set_xticklabels(range_order, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_by_range_zoomed(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    # title argument removed
    filename: str = "error_by_range_zoomed.png"
) -> None:
    """
    Create grouped bar plot for MAE by range with zoomed view (≤1%).
    """
    set_style()
    
    comparison_keys = list(results.keys())
    bins = [
        (0, 0.0001, '>0% to 0.01%'),
        (0.0001, 0.001, '0.01-0.1%'),
        (0.001, 0.01, '0.1-1%'),
    ]
    
    error_data = []
    for key in comparison_keys:
        targets = results[key]["targets"].flatten()
        preds = results[key]["predictions"].flatten()
        valid_mask = np.isfinite(targets) & np.isfinite(preds)
        targets = targets[valid_mask]
        preds = preds[valid_mask]
        
        for lo, hi, label in bins:
            mask = (targets >= lo) & (targets < hi)
            if mask.sum() > 0:
                mae = np.mean(np.abs(targets[mask] - preds[mask]))
                mae_std = np.std(np.abs(targets[mask] - preds[mask]), ddof=1) if mask.sum() > 1 else 0
                error_data.append({
                    'Comparison': comparison_names.get(key, key) if comparison_names else key,
                    'Range': label,
                    'MAE': mae,
                    'MAE_std': mae_std,
                    'Count': mask.sum()
                })
    
    df = pd.DataFrame(error_data)
    
    fig, ax = plt.subplots(figsize=(13, 6))
    pivot_df = df.pivot(index='Range', columns='Comparison', values='MAE')
    pivot_std = df.pivot(index='Range', columns='Comparison', values='MAE_std')
    pivot_count = df.pivot(index='Range', columns='Comparison', values='Count')
    
    range_order = ['>0% to 0.01%', '0.01-0.1%', '0.1-1%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_std = pivot_std.reindex(range_order)
    pivot_count = pivot_count.reindex(range_order)
    
    pivot_ci = pivot_std * 1.96 / pivot_count.apply(np.sqrt)
    pivot_ci = pivot_ci.fillna(0)
    
    colors = [comparison_colors.get(key, '#3498db') if comparison_colors else '#3498db'
             for key in pivot_df.columns]
    
    pivot_df.plot(kind='bar', ax=ax, color=colors, width=0.8, edgecolor='white',
                 yerr=pivot_ci, capsize=5, error_kw={'elinewidth': 2})
    
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title("MAE by Range (≤1%)", fontsize=14, fontweight='bold')
    ax.legend(title='Comparison', frameon=False, fontsize=9, loc='upper right')
    ax.set_xticklabels(range_order, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_relative_err_by_range(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    # title argument removed
    filename: str = "relative_err_by_range.png"
) -> None:
    """
    Create grouped bar plot for Relative Absolute Error by range (excluding zero GT).
    """
    set_style()
    
    comparison_keys = list(results.keys())
    bins = [
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%')
    ]
    
    error_data = []
    for key in comparison_keys:
        targets = results[key]["targets"].flatten()
        preds = results[key]["predictions"].flatten()
        valid_mask = np.isfinite(targets) & np.isfinite(preds)
        targets = targets[valid_mask]
        preds = preds[valid_mask]
        
        for lo, hi, label in bins:
            mask = (targets > lo) & (targets < hi)  # Exclude zero GT
            if mask.sum() > 0:
                rel_err = np.abs(preds[mask] - targets[mask]) / np.abs(targets[mask])
                rae = np.mean(rel_err)
                rae_std = np.std(rel_err, ddof=1) if mask.sum() > 1 else 0
                error_data.append({
                    'Comparison': comparison_names.get(key, key) if comparison_names else key,
                    'Range': label,
                    'RAE': rae,
                    'RAE_std': rae_std,
                    'Count': mask.sum()
                })
    
    df = pd.DataFrame(error_data)
    
    fig, ax = plt.subplots(figsize=(13, 6))
    pivot_df = df.pivot(index='Range', columns='Comparison', values='RAE')
    pivot_std = df.pivot(index='Range', columns='Comparison', values='RAE_std')
    pivot_count = df.pivot(index='Range', columns='Comparison', values='Count')
    
    range_order = ['>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_std = pivot_std.reindex(range_order)
    pivot_count = pivot_count.reindex(range_order)
    
    pivot_ci = pivot_std * 1.96 / pivot_count.apply(np.sqrt)
    pivot_ci = pivot_ci.fillna(0)
    
    colors = [comparison_colors.get(key, '#3498db') if comparison_colors else '#3498db'
             for key in pivot_df.columns]
    
    pivot_df.plot(kind='bar', ax=ax, color=colors, width=0.8, edgecolor='white',
                 yerr=pivot_ci, capsize=5, error_kw={'elinewidth': 2})
    
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Relative Absolute Error', fontsize=12)
    ax.set_title("Relative Absolute Error by Range", fontsize=14, fontweight='bold')
    ax.legend(title='Comparison', frameon=False, fontsize=9, loc='upper right')
    ax.set_xticklabels(range_order, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_residual_distribution(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    # title argument removed
    filename: str = "residual_distribution.png"
) -> None:
    """
    Create residual distribution plots with adaptive layout.
    
    For <= 3 comparisons: overlapping histograms and KDE on single subplot
    For > 3 comparisons: individual subplots (with adaptive layout)
    """
    set_style()
    
    comparison_keys = list(results.keys())
    n_comparisons = len(comparison_keys)
    
    if n_comparisons <= 3:
        # Single subplot with overlapping distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for key in comparison_keys:
            targets = results[key]["targets"].flatten()
            preds = results[key]["predictions"].flatten()
            valid_mask = np.isfinite(targets) & np.isfinite(preds)
            targets = targets[valid_mask]
            preds = preds[valid_mask]
            
            residuals = targets - preds
            
            label = comparison_names.get(key, key) if comparison_names else key
            color = comparison_colors.get(key, '#3498db') if comparison_colors else '#3498db'
            
            ax.hist(residuals, bins=50, alpha=0.5, label=label, color=color, density=True, edgecolor='black', linewidth=0.5)
            
            # KDE overlay
            try:
                kde = gaussian_kde(residuals)
                x_range = np.linspace(residuals.min(), residuals.max(), 200)
                ax.plot(x_range, kde(x_range), linewidth=2, color=color)
            except:
                pass
        
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect')
        ax.set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title("Zero vs Non-Zero Comparison", fontsize=14, fontweight='bold')
        ax.legend(frameon=False, fontsize=10)
        sns.despine(ax=ax)
        
    else:
        # Multiple subplots
        n_rows, n_cols = get_layout_dims(n_comparisons)
        if n_rows == -1:
            n_rows, n_cols = (int(math.ceil(n_comparisons / 8)), 8)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_comparisons == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for idx, key in enumerate(comparison_keys):
            ax = axes_flat[idx]
            
            targets = results[key]["targets"].flatten()
            preds = results[key]["predictions"].flatten()
            valid_mask = np.isfinite(targets) & np.isfinite(preds)
            targets = targets[valid_mask]
            preds = preds[valid_mask]
            
            residuals = targets - preds
            
            label = comparison_names.get(key, key) if comparison_names else key
            color = comparison_colors.get(key, '#3498db') if comparison_colors else '#3498db'
            
            ax.hist(residuals, bins=50, alpha=0.7, color=color, density=True, edgecolor='black', linewidth=0.5)
            
            try:
                kde = gaussian_kde(residuals)
                x_range = np.linspace(residuals.min(), residuals.max(), 200)
                ax.plot(x_range, kde(x_range), linewidth=2, color='darkred')
            except:
                pass
            
            ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.set_xlabel('Residuals', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(label, fontsize=11, fontweight='bold')
            sns.despine(ax=ax)
        
        # Hide unused
        for idx in range(n_comparisons, len(axes_flat)):
            axes_flat[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_zero_vs_nonzero_comparison(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    # title argument removed
    filename: str = "zero_vs_nonzero_comparison.png"
) -> None:
    """
    Create grouped comparison of metrics for zero vs non-zero ground truth.
    """
    set_style()
    
    comparison_keys = list(results.keys())
    comparison_data = []
    
    for key in comparison_keys:
        metrics = compute_extended_metrics(results[key]["targets"], results[key]["predictions"])
        label = comparison_names.get(key, key) if comparison_names else key
        
        comparison_data.append({
            'Comparison': label,
            'Type': 'Zero GT',
            'MAE': metrics['MAE (zeros)'],
            'Count': metrics['n_zeros']
        })
        comparison_data.append({
            'Comparison': label,
            'Type': 'Non-Zero GT',
            'MAE': metrics['MAE (non-zeros)'],
            'Count': metrics['n_nonzeros']
        })
    
    df = pd.DataFrame(comparison_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by comparison
    comparisons = df['Comparison'].unique()
    x = np.arange(len(comparisons))
    width = 0.35
    
    zero_vals = [df[(df['Comparison'] == c) & (df['Type'] == 'Zero GT')]['MAE'].values[0] 
                if len(df[(df['Comparison'] == c) & (df['Type'] == 'Zero GT')]) > 0 else 0
                for c in comparisons]
    nonzero_vals = [df[(df['Comparison'] == c) & (df['Type'] == 'Non-Zero GT')]['MAE'].values[0]
                   if len(df[(df['Comparison'] == c) & (df['Type'] == 'Non-Zero GT')]) > 0 else 0
                   for c in comparisons]
    
    colors_zero = [comparison_colors.get(key, '#a8d4ff') if comparison_colors else '#a8d4ff'
                  for key in comparison_keys]
    colors_nonzero = [comparison_colors.get(key, '#3498db') if comparison_colors else '#3498db'
                     for key in comparison_keys]
    
    ax.bar(x - width/2, zero_vals, width, label='Zero GT', color=colors_zero, edgecolor='white', linewidth=1.5)
    ax.bar(x + width/2, nonzero_vals, width, label='Non-Zero GT', color=colors_nonzero, edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Comparison', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title("Residual Distribution", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparisons, rotation=45, ha='right')
    ax.legend(frameon=False, fontsize=10)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_table(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    # title argument removed
    metrics_to_show: Optional[List[str]] = None,
    filename: str = "summary_table.png"
) -> None:
    """
    Create a summary table with key metrics for all comparisons.
    """
    set_style()
    
    comparison_keys = list(results.keys())
    
    if metrics_to_show is None:
        metrics_to_show = ['RMSE_micro', 'MAE_micro', 'Correlation', 'KL Divergence']
    
    table_data = []
    for key in comparison_keys:
        metrics = compute_extended_metrics(results[key]["targets"], results[key]["predictions"])
        label = comparison_names.get(key, key) if comparison_names else key
        
        row = [label]
        for metric in metrics_to_show:
            val = metrics.get(metric, 0.0)
            row.append(f"{val:.4f}")
        table_data.append(row)
    
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.7 + 1))
    ax.axis('off')
    
    columns = ['Comparison'] + metrics_to_show
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center',
                    colWidths=[0.3] + [0.7 / len(metrics_to_show)] * len(metrics_to_show))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)
    
    plt.suptitle("Summary Metrics Table", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_all_visualizations(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    title_prefix: str = "Analysis"
) -> None:
    """
    Create all visualizations in one call.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        comparison_names: Display names for comparisons
        comparison_colors: Colors for comparisons
        title_prefix: Prefix for all titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_metrics_comparison(results, output_dir, comparison_names, comparison_colors)
    plot_scatter_predicted_vs_actual(results, output_dir, comparison_names, comparison_colors)
    plot_scatter_zoomed(results, output_dir, comparison_names, comparison_colors)
    plot_scatter_loglog_predicted_vs_actual(results, output_dir, comparison_names, comparison_colors)
    plot_error_by_range(results, output_dir, comparison_names, comparison_colors)
    plot_error_by_range_zoomed(results, output_dir, comparison_names, comparison_colors)
    plot_relative_err_by_range(results, output_dir, comparison_names, comparison_colors)
    plot_residual_distribution(results, output_dir, comparison_names, comparison_colors)
    plot_zero_vs_nonzero_comparison(results, output_dir, comparison_names, comparison_colors)
    plot_summary_table(results, output_dir, comparison_names, comparison_colors)
    plot_latent_difference_distribution(results, output_dir, comparison_names)
    plot_latent_factor_distributions(results, output_dir, comparison_names, comparison_colors)


def plot_latent_difference_distribution(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    filename: str = "latent_difference_distribution.png"
) -> None:
    """
    Plot the distribution of differences in latent vectors between two comparisons.
    
    Expected results format for each comparison:
    {
        'latent_vector': np.ndarray  # Latent vectors for samples
        ... other keys ...
    }
    
    Args:
        results: Dict where values may contain 'latent_vector' key
        output_dir: Output directory
        comparison_names: Optional dict for display names
        filename: Output filename
    """
    set_style()
    
    # Find comparisons with latent vectors
    latent_keys = [k for k in results.keys() 
                   if isinstance(results[k], dict) and "latent_vector" in results[k]]
    
    if len(latent_keys) != 2:
        print(f"Expected exactly 2 comparisons with latent vectors, found {len(latent_keys)}. Skipping latent difference plot.")
        return
    
    # Get the two latent vectors
    key1, key2 = latent_keys[0], latent_keys[1]
    latent_1 = results[key1].get("latent_vector")
    latent_2 = results[key2].get("latent_vector")
    
    if latent_1 is None or latent_2 is None:
        print("Latent vectors not found. Skipping latent difference plot.")
        return
    
    if latent_1.shape != latent_2.shape:
        print(f"Latent vectors have different shapes: {latent_1.shape} vs {latent_2.shape}. Skipping plot.")
        return
    
    # Compute difference
    diff = latent_2 - latent_1
    
    # Get display names
    label1 = comparison_names.get(key1, key1) if comparison_names else key1
    label2 = comparison_names.get(key2, key2) if comparison_names else key2
    
    # Create histogram with KDE
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(data=None, x=diff, bins=40, kde=True, color="#4a90d9", 
                edgecolor=None, alpha=0.8, ax=ax, stat='count')
    
    # Add zero line
    ax.axvline(0, color="black", linestyle="--", lw=2, alpha=0.7, label='Zero difference')
    
    ax.set_xlabel(f"Latent Value Difference ({label2} - {label1})", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax.set_title(f"Distribution of Latent Vector Differences ({label2} - {label1})", 
                fontsize=14, fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_factor_distributions(
    results: Dict[str, Any],
    output_dir: str,
    comparison_names: Optional[Dict[str, str]] = None,
    comparison_colors: Optional[Dict[str, str]] = None,
    # title argument removed
    filename: str = "latent_factor_distributions.png"
) -> None:
    """
    Plot the distribution of latent factor values for each comparison with latent vectors.
    
    Creates one subplot per comparison with latent vectors.
    
    Expected results format for each comparison:
    {
        'latent_vector': np.ndarray  # Latent vectors for samples
        ... other keys ...
    }
    
    Args:
        results: Dict where values may contain 'latent_vector' key
        output_dir: Output directory
        comparison_names: Optional dict for display names
        comparison_colors: Optional dict for colors
        title: Plot title
        filename: Output filename
    """
    set_style()
    
    # Find comparisons with latent vectors
    latent_keys = [k for k in results.keys() 
                   if isinstance(results[k], dict) and "latent_vector" in results[k]]
    
    if not latent_keys:
        print("No latent vectors found in results. Skipping latent factor distribution plot.")
        return
    
    n_comparisons = len(latent_keys)
    
    # Determine layout
    n_rows, n_cols = get_layout_dims(n_comparisons)
    if n_rows == -1:
        n_rows, n_cols = (int(math.ceil(n_comparisons / 8)), 8)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_comparisons == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Get colors
    colors = {}
    if comparison_colors:
        for key in latent_keys:
            colors[key] = comparison_colors.get(key, '#3498db')
    else:
        base_colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c', '#9b59b6', 
                      '#1abc9c', '#f39c12', '#34495e']
        colors = {key: base_colors[i % len(base_colors)] for i, key in enumerate(latent_keys)}
    
    # Create plots
    for idx, key in enumerate(latent_keys):
        ax = axes_flat[idx]
        
        latent_vector = results[key].get("latent_vector")
        
        if latent_vector is None:
            ax.text(0.5, 0.5, f"No latent vector", ha='center', va='center', fontsize=12)
            ax.set_title(comparison_names.get(key, key) if comparison_names else key, 
                        fontsize=12, fontweight='bold')
            continue
        
        # Create histogram with KDE
        sns.histplot(data=None, x=latent_vector.flatten(), bins=40, kde=True, 
                    color=colors[key], edgecolor=None, alpha=0.8, ax=ax, stat='count')
        
        label = comparison_names.get(key, key) if comparison_names else key
        ax.set_title(f"Latent Factor Distribution: {label}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Latent Factor Value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        sns.despine(ax=ax)
    
    # Hide unused subplots
    for idx in range(n_comparisons, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle("Distribution of Latent Factor Values", fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
