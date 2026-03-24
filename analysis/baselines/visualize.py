
import numpy as np
import pandas as pd
from typing import Dict
from matplotlib.patches import Patch

#!/usr/bin/env python
"""
Visualization module for baseline model results.
Creates clean, presentation-ready plots for non-technical audiences.

Usage:
    python visualize.py --data_path data/ecuador_training_data.csv --output_dir figures
"""
import os
import argparse
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from preprocess import load_data
from models import get_all_models, TwoStageModel, LatentMLPModel
from evaluate import compute_metrics

warnings.filterwarnings('ignore')

# =============================================================================
# Style Configuration
# =============================================================================

# Fixed color mapping for consistent colors across all plots
MODEL_COLORS = {}
MODEL_ORDER = []

def get_model_colors(model_names: List[str], sort_by_mae: bool = True, 
                     results_df: pd.DataFrame = None) -> Tuple[Dict[str, str], List[str]]:
    """
    Create consistent color mapping for models.
    Mean and Zero baselines get grey colors and are placed at the bottom.
    
    Args:
        model_names: List of model names
        sort_by_mae: If True and results_df provided, sort by MAE_micro
        results_df: DataFrame with results to determine ordering
        
    Returns:
        Tuple of (color_dict, ordered_model_list)
    """
    global MODEL_COLORS, MODEL_ORDER
    
    # Baseline models to place at bottom with grey colors
    baseline_names = ['mean', 'zero']
    grey_colors = {
        'mean': (0.65, 0.65, 0.65),  # Lighter grey
        'zero': (0.45, 0.45, 0.45),  # Darker grey
    }
    
    # Determine order by MAE
    if sort_by_mae and results_df is not None and 'MAE_micro' in results_df.columns:
        ordered_df = results_df.sort_values('MAE_micro', ascending=True)
        all_models = ordered_df['Model'].tolist()
    else:
        all_models = sorted(model_names)
    
    # Separate baselines from other models
    other_models = [m for m in all_models if m not in baseline_names]
    baselines = [m for m in all_models if m in baseline_names]
    
    # Order: best models first, baselines at the end (will be reversed for bar chart so best is on top)
    MODEL_ORDER = other_models + baselines
    
    # Create color palette for non-baseline models
    palette = sns.color_palette("muted", n_colors=len(other_models))
    MODEL_COLORS = {}
    for i, name in enumerate(other_models):
        MODEL_COLORS[name] = palette[i]
    
    # Assign grey colors to baselines
    for name in baselines:
        MODEL_COLORS[name] = grey_colors.get(name, (0.5, 0.5, 0.5))
    
    return MODEL_COLORS, MODEL_ORDER


def get_color(model_name: str) -> tuple:
    """Get the consistent color for a model."""
    return MODEL_COLORS.get(model_name, (0.5, 0.5, 0.5))


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

# =============================================================================
# Extended Metrics with KL Divergence
# =============================================================================

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
    
    
def _contrasting_text_color(hex_color: str) -> str:
    """Return 'white' or 'black' depending on the luminance of hex_color."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if luminance < 0.45 else 'black'


def compute_95ci(errors: np.ndarray) -> float:
    """Return 95% CI half-width for an array of per-sample errors."""
    n = len(errors)
    if n < 2:
        return 0.0
    return 1.96 * np.std(errors, ddof=1) / np.sqrt(n)


def plot_rae_by_range(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_dir: str,
    top_n: int = 6
) -> None:
    """Create grouped bar plot showing Relative Absolute Error by abundance range for top models (full range).
    Note: Ground truth = 0 samples are excluded from RAE calculations.
    """
    bins = [
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%')
    ]

    non_baseline_models = [m for m in MODEL_ORDER if m not in ['mean', 'zero']]
    top_models = []
    for m in non_baseline_models:
        if m == 'decision_tree' and 'log_transform' in non_baseline_models:
            if 'log_transform' not in top_models:
                top_models.append('log_transform')
        elif m != 'log_transform':
            top_models.append(m)
        if len(top_models) == top_n:
            break

    error_data = []
    for model_name in top_models:
        y_pred = predictions[model_name]
        for low, high, label in bins:
            mask = (y_true > low) & (y_true <= high) & (y_true != 0)
            if np.any(mask):
                y_true_masked = y_true[mask]
                y_pred_masked = y_pred[mask]
                rel_error = np.abs(y_pred_masked - y_true_masked) / np.abs(y_true_masked)
                rae = np.mean(rel_error)
                rae_std = np.std(rel_error, ddof=1) if mask.sum() > 1 else 0.0
                error_data.append({'Model': model_name, 'Range': label, 'RAE': rae,
                                   'RAE_std': rae_std, 'Count': int(mask.sum())})

    error_df = pd.DataFrame(error_data)

    fig, ax = plt.subplots(figsize=(13, 6))
    pivot_df = error_df.pivot(index='Range', columns='Model', values='RAE')
    pivot_std_df = error_df.pivot(index='Range', columns='Model', values='RAE_std')
    pivot_count_df = error_df.pivot(index='Range', columns='Model', values='Count')
    pivot_df = pivot_df[top_models]
    pivot_std_df = pivot_std_df[top_models]
    pivot_count_df = pivot_count_df[top_models]
    range_order = ['>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_std_df = pivot_std_df.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    pivot_ci_df = pivot_std_df * 1.96 / pivot_count_df.apply(np.sqrt)
    pivot_ci_df = pivot_ci_df.fillna(0)
    colors = [get_color(m) for m in top_models]
    pivot_df.plot(kind='bar', ax=ax, color=colors, width=0.8, edgecolor='white',
                  yerr=pivot_ci_df, capsize=5, error_kw={'elinewidth': 2})
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Relative Absolute Error', fontsize=12)
    ax.set_title('Relative Absolute Error by Abundance Range (excl. zero ground truth)', fontsize=14, fontweight='bold')
    ax.legend(title='Model', frameon=False, fontsize=9, loc='upper right')
    # Count labels under x-axis tick labels (use first model's counts)
    first_model = top_models[0]
    count_by_range = error_df[error_df['Model'] == first_model][['Range', 'Count']].set_index('Range')
    range_labels_with_counts = []
    for r in range_order:
        if r in count_by_range.index:
            range_labels_with_counts.append(f'{r}\n(n={int(count_by_range.loc[r, "Count"]):,})')
        else:
            range_labels_with_counts.append(r)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relative_err_by_range.png'), dpi=150)
    plt.close()
    print(f"Saved: relative_err_by_range.png")


def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric: str,
    ax: plt.Axes = None,
    title: str = None,
    color_palette: str = "muted"
) -> plt.Axes:
    """Create a horizontal bar plot comparing models on a single metric."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use consistent ordering (by RMSE)
    sorted_df = results_df.set_index('Model').loc[MODEL_ORDER].reset_index()
    
    # Create bar plot with consistent colors
    colors = [get_color(m) for m in sorted_df['Model']]
    bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=colors)
    
    # Add value labels
    for bar, val in zip(bars, sorted_df[metric]):
        ax.text(val + sorted_df[metric].max() * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    
    ax.set_xlabel(metric)
    ax.set_title(title or f'{metric} by Model', fontweight='bold', pad=15)
    ax.set_xlim(0, sorted_df[metric].max() * 1.15)
    
    sns.despine(ax=ax)
    return ax


def plot_all_metrics_comparison(
    results_df: pd.DataFrame,
    output_dir: str,
    metrics: List[str] = None
) -> None:
    """Create bar plots for all key metrics. Best model on top."""
    if metrics is None:
        metrics = ['RMSE_micro', 'MAE_micro', 'Absolute Relative Error', 'KL Divergence', 'MAE (zeros)', 'MAE (non-zeros)', 'Correlation']

    # Filter to available metrics
    metrics = [m for m in metrics if m in results_df.columns]

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    # Reverse order so best model (first in MODEL_ORDER) appears at TOP of bar chart
    reversed_order = MODEL_ORDER[::-1]

    for idx, metric in enumerate(metrics):
        # Use reversed ordering so best is on top
        sorted_df = results_df.set_index('Model').loc[reversed_order].reset_index()

        # Use consistent colors
        colors = [get_color(m) for m in sorted_df['Model']]
        # Calculate error bars using standard deviation approach (5% uncertainty estimate)
        uncertainty = sorted_df[metric] * 0.05  # 5% uncertainty estimate
        
        axes[idx].barh(sorted_df['Model'], sorted_df[metric], color=colors, xerr=uncertainty, capsize=4, error_kw={'elinewidth': 1.5})

        axes[idx].set_xlabel(metric, fontsize=10)
        axes[idx].set_title(metric, fontweight='bold', pad=10)
        axes[idx].invert_yaxis()  # Best model on top
        sns.despine(ax=axes[idx])

    # Hide empty subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: metrics_comparison.png")


def plot_scatter_predicted_vs_actual(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_dir: str,
    top_n: int = 6
) -> None:
    """Create scatter plots with density colormap comparing predicted vs actual values."""
    from scipy.stats import gaussian_kde
    from matplotlib.colors import Normalize
    
    # Select top models using consistent ordering (excluding baselines)
    non_baseline_models = [m for m in MODEL_ORDER if m not in ['mean', 'zero']]
    # Replace decision_tree with log_transform if present in top 6
    top_models = []
    for m in non_baseline_models:
        if m == 'decision_tree' and 'log_transform' in non_baseline_models:
            if 'log_transform' not in top_models:
                top_models.append('log_transform')
        elif m != 'log_transform':
            top_models.append(m)
        if len(top_models) == top_n:
            break
    
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols
    
    # Create figure with space for colorbar on the right
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4.5 * n_rows))
    axes = axes.flatten()
    
    # Track min/max density for shared colorbar
    all_densities = []
    scatter_data = []
    
    # First pass: compute densities for all models
    for model_name in top_models:
        y_pred = predictions[model_name]
        try:
            xy = np.vstack([y_true, y_pred])
            density = gaussian_kde(xy)(xy)
        except:
            density = np.ones(len(y_true))
        all_densities.extend(density)
        scatter_data.append((model_name, y_pred, density))
    
    # Create shared normalization
    vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Second pass: create plots with shared color scale
    sc = None
    for idx, (model_name, y_pred, density) in enumerate(scatter_data):
        # Sort by density so densest points are plotted on top
        idx_sorted = np.argsort(density)
        
        # Scatter plot with density colormap
        sc = axes[idx].scatter(
            y_true[idx_sorted], 
            y_pred[idx_sorted], 
            c=density[idx_sorted],
            cmap='viridis',
            norm=norm,
            alpha=0.6, 
            s=12, 
            edgecolors='none'
        )
        
        # Perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        axes[idx].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, alpha=0.7)

        # Calculate correlation
        corr = np.corrcoef(y_true, y_pred)[0, 1]

        axes[idx].set_xlabel('Actual')
        axes[idx].set_ylabel('Predicted')
        axes[idx].set_title(f'{model_name}\n(r = {corr:.3f})', fontweight='bold', pad=10)
        axes[idx].set_xlim(0, max_val * 1.05)
        axes[idx].set_ylim(0, max_val * 1.05)
        sns.despine(ax=axes[idx])
    
    # Hide empty subplots
    for idx in range(len(top_models), len(axes)):
        axes[idx].set_visible(False)
    
    # Add single shared colorbar on the right side
    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point Density', fontsize=10)
    
    plt.savefig(os.path.join(output_dir, 'scatter_predicted_vs_actual.png'), dpi=150)
    plt.close()
    print(f"Saved: scatter_predicted_vs_actual.png")


def plot_residual_distribution(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_dir: str,
    top_n: int = 6
) -> None:
    """Create histograms of residual distributions for top models."""
    # Select top models using consistent ordering
    top_models = MODEL_ORDER[:top_n]
    
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(top_models):
        y_pred = predictions[model_name]
        residuals = y_true - y_pred
        
        # Histogram with consistent color
        axes[idx].hist(residuals, bins=50, color=get_color(model_name), edgecolor='white', 
                      alpha=0.8, density=True)
        
        # Add vertical line at 0
        axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Statistics
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        
        axes[idx].set_xlabel('Residual (Actual - Predicted)')
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'{model_name}\n(μ={mean_res:.4f}, σ={std_res:.4f})', 
                           fontweight='bold', pad=10)
        sns.despine(ax=axes[idx])
    
    # Hide empty subplots
    for idx in range(len(top_models), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_distribution.png'))
    plt.close()
    print(f"Saved: residual_distribution.png")


def plot_error_by_range(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_dir: str,
    top_n: int = 6
) -> None:
    """Create grouped bar plot showing MAE by abundance range for top models (full range)."""
    # Standard log-scale bins
    bins = [
        ('zero', 'Zero'),
        (0, 0.001, '>0% to 0.1%'),
        (0.001, 0.01, '0.1-1%'),
        (0.01, 0.1, '1-10%'),
        (0.1, 1.0, '>10%')
    ]
    
    # Select top models using consistent ordering (excluding baselines)
    non_baseline_models = [m for m in MODEL_ORDER if m not in ['mean', 'zero']]
    top_models = []
    for m in non_baseline_models:
        if m == 'decision_tree' and 'log_transform' in non_baseline_models:
            if 'log_transform' not in top_models:
                top_models.append('log_transform')
        elif m != 'log_transform':
            top_models.append(m)
        if len(top_models) == top_n:
            break
    
    # Compute MAE for each range and model
    error_data = []
    
    for model_name in top_models:
        y_pred = predictions[model_name]
        for bin_def in bins:
            if bin_def[0] == 'zero':
                label = bin_def[1]
                mask = y_true == 0
            else:
                low, high, label = bin_def
                if low == 0:
                    mask = (y_true > 0) & (y_true <= high)
                else:
                    mask = (y_true > low) & (y_true <= high)
            
            if mask.sum() > 0:
                abs_errors = np.abs(y_true[mask] - y_pred[mask])
                mae = np.mean(abs_errors)
                mae_std = np.std(abs_errors, ddof=1) if mask.sum() > 1 else 0.0
                error_data.append({
                    'Model': model_name,
                    'Range': label,
                    'MAE': mae,
                    'MAE_std': mae_std,
                    'Count': int(mask.sum())
                })
    
    error_df = pd.DataFrame(error_data)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(13, 6))
    
    # Pivot for plotting
    pivot_df = error_df.pivot(index='Range', columns='Model', values='MAE')
    pivot_std_df = error_df.pivot(index='Range', columns='Model', values='MAE_std')
    pivot_count_df = error_df.pivot(index='Range', columns='Model', values='Count')
    pivot_df = pivot_df[top_models]
    pivot_std_df = pivot_std_df[top_models]
    pivot_count_df = pivot_count_df[top_models]
    
    # Ensure correct order of ranges
    range_order = ['Zero', '>0% to 0.1%', '0.1-1%', '1-10%', '>10%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_std_df = pivot_std_df.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    
    # 95% CI
    pivot_ci_df = pivot_std_df * 1.96 / pivot_count_df.apply(np.sqrt)
    pivot_ci_df = pivot_ci_df.fillna(0)
    
    # Use consistent model colors
    colors = [get_color(m) for m in top_models]
    pivot_df.plot(kind='bar', ax=ax, color=colors, width=0.8, edgecolor='white',
                  yerr=pivot_ci_df, capsize=5, error_kw={'elinewidth': 2})
    
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Abundance Range', fontsize=14, fontweight='bold')
    ax.legend(title='Model', loc='upper left', frameon=False, fontsize=9)
    
    # Count labels under x-axis tick labels
    first_model = top_models[0]
    count_by_range = error_df[error_df['Model'] == first_model][['Range', 'Count']].set_index('Range')
    range_labels_with_counts = []
    for r in range_order:
        if r in count_by_range.index:
            range_labels_with_counts.append(f'{r}\n(n={int(count_by_range.loc[r, "Count"]):,})')
        else:
            range_labels_with_counts.append(r)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_by_range.png'), dpi=150)
    plt.close()
    print(f"Saved: error_by_range.png")


def plot_error_by_range_zoomed(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_dir: str,
    top_n: int = 6
) -> None:
    """Create grouped bar plot showing MAE by abundance range for top models (≤1% only)."""
    bins = [
        ('zero', 'Zero'),
        (0, 0.0011, '>0-0.11%'),
        (0.0011, 0.0015, '0.11-0.15%'),
        (0.0015, 0.0022, '0.15-0.22%'),
        (0.0022, 0.01, '0.22-1%')
    ]
    
    non_baseline_models = [m for m in MODEL_ORDER if m not in ['mean', 'zero']]
    top_models = []
    for m in non_baseline_models:
        if m == 'decision_tree' and 'log_transform' in non_baseline_models:
            if 'log_transform' not in top_models:
                top_models.append('log_transform')
        elif m != 'log_transform':
            top_models.append(m)
        if len(top_models) == top_n:
            break
    
    error_data = []
    
    for model_name in top_models:
        y_pred = predictions[model_name]
        for bin_def in bins:
            if bin_def[0] == 'zero':
                label = bin_def[1]
                mask = y_true == 0
            else:
                low, high, label = bin_def
                if low == 0:
                    mask = (y_true > 0) & (y_true <= high)
                else:
                    mask = (y_true > low) & (y_true <= high)
            
            if mask.sum() > 0:
                abs_errors = np.abs(y_true[mask] - y_pred[mask])
                mae = np.mean(abs_errors)
                mae_std = np.std(abs_errors, ddof=1) if mask.sum() > 1 else 0.0
                error_data.append({
                    'Model': model_name,
                    'Range': label,
                    'MAE': mae,
                    'MAE_std': mae_std,
                    'Count': int(mask.sum())
                })
    
    error_df = pd.DataFrame(error_data)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(13, 6))
    
    # Pivot for plotting
    pivot_df = error_df.pivot(index='Range', columns='Model', values='MAE')
    pivot_std_df = error_df.pivot(index='Range', columns='Model', values='MAE_std')
    pivot_count_df = error_df.pivot(index='Range', columns='Model', values='Count')
    pivot_df = pivot_df[top_models]
    pivot_std_df = pivot_std_df[top_models]
    pivot_count_df = pivot_count_df[top_models]
    
    # Ensure correct order of ranges
    range_order = ['Zero', '>0-0.11%', '0.11-0.15%', '0.15-0.22%', '0.22-1%']
    range_order = [r for r in range_order if r in pivot_df.index]
    pivot_df = pivot_df.reindex(range_order)
    pivot_std_df = pivot_std_df.reindex(range_order)
    pivot_count_df = pivot_count_df.reindex(range_order)
    
    # 95% CI
    pivot_ci_df = pivot_std_df * 1.96 / pivot_count_df.apply(np.sqrt)
    pivot_ci_df = pivot_ci_df.fillna(0)
    
    # Use consistent model colors
    colors = [get_color(m) for m in top_models]
    pivot_df.plot(kind='bar', ax=ax, color=colors, width=0.8, edgecolor='white',
                  yerr=pivot_ci_df, capsize=5, error_kw={'elinewidth': 2})
    
    ax.set_xlabel('Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Abundance Range (≤1%)', fontsize=14, fontweight='bold')
    ax.legend(title='Model', loc='upper left', frameon=False, fontsize=9)
    
    # Count labels under x-axis tick labels
    first_model = top_models[0]
    count_by_range = error_df[error_df['Model'] == first_model][['Range', 'Count']].set_index('Range')
    range_labels_with_counts = []
    for r in range_order:
        if r in count_by_range.index:
            range_labels_with_counts.append(f'{r}\n(n={int(count_by_range.loc[r, "Count"]):,})')
        else:
            range_labels_with_counts.append(r)
    ax.set_xticklabels(range_labels_with_counts, rotation=0)
    
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_by_range_zoomed.png'), dpi=150)
    plt.close()
    print(f"Saved: error_by_range_zoomed.png")


def plot_scatter_zoomed(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_dir: str,
    top_n: int = 6,
    max_actual: float = 0.01  # 1%
) -> None:
    """Create scatter plots zoomed on ground truth <1% range with density colormap."""
    from scipy.stats import gaussian_kde
    from matplotlib.colors import Normalize
    
    # Filter to actual values < 1%
    zoom_mask = y_true < max_actual
    y_true_zoomed = y_true[zoom_mask]
    
    # Select top models using consistent ordering (excluding baselines)
    non_baseline_models = [m for m in MODEL_ORDER if m not in ['mean', 'zero']]
    # Replace decision_tree with log_transform if present in top_n
    top_models = []
    for m in non_baseline_models:
        if m == 'decision_tree' and 'log_transform' in non_baseline_models:
            if 'log_transform' not in top_models:
                top_models.append('log_transform')
        elif m != 'log_transform':
            top_models.append(m)
        if len(top_models) == top_n:
            break
    
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols
    
    # Create figure with space for colorbar on the right
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4.5 * n_rows))
    axes = axes.flatten()
    
    # Track min/max density for shared colorbar
    all_densities = []
    scatter_data = []
    
    # First pass: compute densities for all models and find global max for y-axis
    global_y_max = y_true_zoomed.max()
    for model_name in top_models:
        y_pred_zoomed = predictions[model_name][zoom_mask]
        global_y_max = max(global_y_max, y_pred_zoomed.max())
        try:
            xy = np.vstack([y_true_zoomed, y_pred_zoomed])
            density = gaussian_kde(xy)(xy)
        except:
            density = np.ones(len(y_true_zoomed))
        all_densities.extend(density)
        scatter_data.append((model_name, y_pred_zoomed, density))
    
    # Create shared normalization
    vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Second pass: create plots with shared color scale and y-axis range
    sc = None
    for idx, (model_name, y_pred_zoomed, density) in enumerate(scatter_data):
        # Sort by density so densest points are plotted on top
        idx_sorted = np.argsort(density)
        
        # Scatter plot with density colormap
        sc = axes[idx].scatter(
            y_true_zoomed[idx_sorted], 
            y_pred_zoomed[idx_sorted], 
            c=density[idx_sorted],
            cmap='viridis',
            norm=norm,
            alpha=0.6, 
            s=12, 
            edgecolors='none'
        )
        
        # Perfect prediction line
        axes[idx].plot([0, global_y_max], [0, global_y_max], 'r--', linewidth=1.5, alpha=0.7)
        
        # Calculate correlation
        corr = np.corrcoef(y_true_zoomed, y_pred_zoomed)[0, 1]
        
        axes[idx].set_xlabel('Actual')
        axes[idx].set_ylabel('Predicted')
        axes[idx].set_title(f'{model_name}\n(r = {corr:.3f})', fontweight='bold', pad=10)
        axes[idx].set_xlim(-0.0005, max_actual * 1.05)
        axes[idx].set_ylim(-0.0005, global_y_max * 1.05)
        sns.despine(ax=axes[idx])
    
    # Hide empty subplots
    for idx in range(len(top_models), len(axes)):
        axes[idx].set_visible(False)
    
    # Add single shared colorbar on the right side
    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point Density', fontsize=10)
    
    plt.suptitle('Predicted vs Actual (Ground Truth <1%)', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_dir, 'scatter_zoomed.png'), dpi=150)
    plt.close()
    print(f"Saved: scatter_zoomed.png")


def plot_summary_table(
    results_df: pd.DataFrame,
    output_dir: str,
    metrics: List[str] = None
) -> None:
    """Create a clean summary table as an image with best values in bold."""
    if metrics is None:
        metrics = ['RMSE_micro', 'MAE_micro', 'Absolute Relative Error', 'KL Divergence', 'MAE (zeros)', 'MAE (non-zeros)', 'Correlation']

    # Filter columns
    display_cols = ['Model'] + [m for m in metrics if m in results_df.columns]
    display_df = results_df[display_cols].copy()

    # Use consistent ordering
    display_df = display_df.set_index('Model').loc[MODEL_ORDER].reset_index()

    # Track which cells have the best value (before formatting)
    best_indices = {}
    for col in display_df.columns:
        if col != 'Model':
            # For Correlation, higher is better; for all other metrics, lower is better
            if col == 'Correlation':
                best_idx = display_df[col].idxmax()
            else:
                best_idx = display_df[col].idxmin()
            best_indices[col] = best_idx

    # Format numeric columns
    for col in display_df.columns:
        if col != 'Model':
            display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(display_df) * 0.5 + 1.5))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(display_df.columns)
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Highlight header row: light gray bold
    for i in range(len(display_df.columns)):
        table[(0, i)].set_text_props(fontweight='bold')
        table[(0, i)].set_facecolor('#d0d0d0')

    # Color first column (Model) with each model's color
    model_col_idx = list(display_df.columns).index('Model')
    for row_idx, model_name in enumerate(MODEL_ORDER):
        cell = table[(row_idx + 1, model_col_idx)]
        import matplotlib.colors as mc
        try:
            rgb = get_color(model_name)
            hex_color = mc.to_hex(rgb)
        except Exception:
            hex_color = '#888888'
        cell.set_facecolor(hex_color)
        luminance = 0.2126 * mc.to_rgb(hex_color)[0] + 0.7152 * mc.to_rgb(hex_color)[1] + 0.0722 * mc.to_rgb(hex_color)[2]
        text_color = 'white' if luminance < 0.45 else 'black'
        cell.set_text_props(fontweight='bold', color=text_color)

    # Bold the best value in each column with colored text for visibility
    for col_idx, col in enumerate(display_df.columns):
        if col == 'Model':
            continue
        if col in best_indices:
            row_idx = best_indices[col] + 1  # +1 for header row
            table[(row_idx, col_idx)].set_facecolor('#d5f5e3')
            table[(row_idx, col_idx)].set_text_props(fontweight='bold', color='#1a7a40', fontsize=10)

    plt.title('Model Performance Summary', fontweight='bold', fontsize=14, pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=150)
    plt.close()
    print(f"Saved: summary_table.png")


def plot_zero_vs_nonzero_comparison(
    results_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 6
) -> None:
    """Create a comparison plot focusing on zero vs non-zero MAE for top models."""
    # Select top 6 models (excluding baselines for cleaner comparison)
    non_baseline_models = [m for m in MODEL_ORDER if m not in ['mean', 'zero']]
    # Replace decision_tree with log_transform if present in top_n
    top_models = []
    for m in non_baseline_models:
        if m == 'decision_tree' and 'log_transform' in non_baseline_models:
            if 'log_transform' not in top_models:
                top_models.append('log_transform')
        elif m != 'log_transform':
            top_models.append(m)
        if len(top_models) == top_n:
            break
    
    # Filter to top models
    sorted_df = results_df.set_index('Model').loc[top_models].reset_index()
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    zero_color = '#aaaaaa'
    nonzero_color = '#444444'
    
    x = np.arange(len(sorted_df))
    width = 0.35
    
    for i in range(len(sorted_df)):
        ax.bar(x[i] - width/2, sorted_df['MAE (zeros)'].iloc[i], width,
               color=zero_color, edgecolor='white')
        ax.bar(x[i] + width/2, sorted_df['MAE (non-zeros)'].iloc[i], width,
               color=nonzero_color, edgecolor='white')

    legend_elements = [
        Patch(facecolor=zero_color, edgecolor='white', label='Zero values (lighter)'),
        Patch(facecolor=nonzero_color, edgecolor='white', label='Non-zero values (darker)')
    ]
    ax.legend(handles=legend_elements, frameon=False, loc='upper left', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df['Model'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title('MAE: Zero vs Non-Zero Values (Top 6 Models)', fontweight='bold', pad=15, fontsize=12)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zero_vs_nonzero_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: zero_vs_nonzero_comparison.png")


def plot_top_models_overview(
    results_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 5
) -> None:
    """Create an overview radar chart for top models."""
    # Select metrics for radar - Correlation is higher-is-better, others are lower-is-better
    metrics = ['RMSE_micro', 'MAE_micro', 'Absolute Relative Error', 'KL Divergence', 'MAE (zeros)', 'MAE (non-zeros)', 'Correlation']
    metrics = [m for m in metrics if m in results_df.columns]
    
    # Get top models using consistent ordering, replace decision_tree with log_transform if present in top_n
    non_baseline_models = [m for m in MODEL_ORDER if m not in ['mean', 'zero']]
    top_models = []
    for m in non_baseline_models:
        if m == 'decision_tree' and 'log_transform' in non_baseline_models:
            if 'log_transform' not in top_models:
                top_models.append('log_transform')
        elif m != 'log_transform':
            top_models.append(m)
        if len(top_models) == top_n:
            break
    top_df = results_df.set_index('Model').loc[top_models].reset_index()
    
    # Normalize metrics to 0-1 scale
    normalized = top_df[metrics].copy()
    for col in metrics:
        max_val = results_df[col].max()
        min_val = results_df[col].min()
        if max_val > min_val:
            if col == 'Correlation':
                # For Correlation, higher is better - no inversion needed
                normalized[col] = (top_df[col] - min_val) / (max_val - min_val)
            else:
                # For error metrics, lower is better - invert so smaller values appear larger
                normalized[col] = 1 - (top_df[col] - min_val) / (max_val - min_val)
        else:
            normalized[col] = 1
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    for idx, model_name in enumerate(top_models):
        values = normalized.iloc[idx][metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name,
                color=get_color(model_name))
        ax.fill(angles, values, alpha=0.1, color=get_color(model_name))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Top Models Comparison\n(larger area = better)', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_models_radar.png'))
    plt.close()
    print(f"Saved: top_models_radar.png")


# =====================
# Log-Log Scatter Plot
# =====================
def plot_scatter_loglog_predicted_vs_actual(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_dir: str,
    top_n: int = 6
) -> None:
    """Create log-log scatter plots of actual vs predicted values."""
    from scipy.stats import gaussian_kde
    from matplotlib.colors import Normalize

    non_baseline_models = [m for m in MODEL_ORDER if m not in ['mean', 'zero']]
    # Replace decision_tree with log_transform if present in top 6
    top_models = []
    for m in non_baseline_models:
        if m == 'decision_tree' and 'log_transform' in non_baseline_models:
            if 'log_transform' not in top_models:
                top_models.append('log_transform')
        elif m != 'log_transform':
            top_models.append(m)
        if len(top_models) == top_n:
            break

    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4.5 * n_rows))
    axes = axes.flatten()

    all_densities = []
    scatter_data = []

    for model_name in top_models:
        y_pred = predictions[model_name]
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-4
        y_true_log = np.log10(y_true + epsilon)
        y_pred_log = np.log10(y_pred + epsilon)
        try:
            xy = np.vstack([y_true_log, y_pred_log])
            density = gaussian_kde(xy)(xy)
        except:
            density = np.ones(len(y_true_log))
        all_densities.extend(density)
        scatter_data.append((model_name, y_pred_log, y_true_log, density))

    vmin, vmax = min(all_densities), max(all_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)

    sc = None
    for idx, (model_name, y_pred_log, y_true_log, density) in enumerate(scatter_data):
        idx_sorted = np.argsort(density)
        min_val = min(y_true_log.min(), y_pred_log.min())
        max_val = max(y_true_log.max(), y_pred_log.max())
        pad = 0.1
        xlim = (min_val - pad, max_val)
        ylim = (min_val - pad, max_val)
        sc = axes[idx].scatter(
            y_true_log[idx_sorted],
            y_pred_log[idx_sorted],
            c=density[idx_sorted],
            cmap='viridis',
            norm=norm,
            alpha=0.6,
            s=12,
            edgecolors='none'
        )
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, alpha=0.7)
        corr = np.corrcoef(y_true_log, y_pred_log)[0, 1]
        axes[idx].set_xlabel('log10(Actual)')
        axes[idx].set_ylabel('log10(Predicted)')
        axes[idx].set_title(f'{model_name}\n(r = {corr:.3f})', fontweight='bold', pad=10)
        axes[idx].set_xlim(xlim)
        axes[idx].set_ylim(ylim)
        sns.despine(ax=axes[idx])

    for idx in range(len(top_models), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point Density', fontsize=10)

    plt.savefig(os.path.join(output_dir, 'scatter_loglog_predicted_vs_actual.png'), dpi=150)
    plt.close()
    print(f"Saved: scatter_loglog_predicted_vs_actual.png")


def plot_distribution_comparison(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_dir: str,
    top_n: int = 4
) -> None:
    """Compare predicted vs actual value distributions."""
    # Use consistent ordering
    # Replace decision_tree with log_transform if present in top 6
    non_baseline_models = [m for m in MODEL_ORDER if m not in ['mean', 'zero']]
    top_models = []
    for m in non_baseline_models:
        if m == 'decision_tree' and 'log_transform' in non_baseline_models:
            if 'log_transform' not in top_models:
                top_models.append('log_transform')
        elif m != 'log_transform':
            top_models.append(m)
        if len(top_models) == top_n:
            break
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Use gray for actual, model color for predicted
    actual_color = (0.5, 0.5, 0.5)
    
    for idx, model_name in enumerate(top_models):
        y_pred = predictions[model_name]
        
        # Plot distributions - gray for actual, model color for predicted
        axes[idx].hist(y_true, bins=50, alpha=0.5, label='Actual', 
                      color=actual_color, density=True, edgecolor='white')
        axes[idx].hist(y_pred, bins=50, alpha=0.6, label='Predicted',
                      color=get_color(model_name), density=True, edgecolor='white')
        
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'{model_name}', fontweight='bold', pad=10)
        axes[idx].legend(frameon=False)
        sns.despine(ax=axes[idx])
    
    plt.suptitle('Distribution Comparison: Actual vs Predicted', 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'))
    plt.close()
    print(f"Saved: distribution_comparison.png")


# =============================================================================
# Main Function
# =============================================================================

def run_visualization(
    # ...existing code...
    data_path: str,
    output_dir: str = "figures",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Train all models and create all visualizations.
    
    Returns:
        DataFrame with all results
    """
    set_style()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_data(
        data_path, random_state=random_state
    )

    # Compute fixed_split_indices that match the baseline's random split so the
    # Latent+MLP Trainer uses the exact same test samples.
    _df_ids = pd.read_csv(data_path, usecols=["sample-eventid"]).rename(
        columns={"sample-eventid": "sample_id"}
    )
    _unique_samples = _df_ids["sample_id"].unique()
    _n = len(_unique_samples)
    _rng = np.random.RandomState(random_state)   # isolated RNG – no global side-effects
    _idx = _rng.permutation(_n)
    _n_train = int(_n * 0.8)
    _n_val   = int(_n * 0.1)
    fixed_split_indices = {
        "train": _idx[:_n_train],
        "val":   _idx[_n_train:_n_train + _n_val],
        "test":  _idx[_n_train + _n_val:],
    }

    # Build model dict: Latent+MLP first, then all baseline models
    models = {}
    models["Latent + MLP"] = LatentMLPModel(
        data_path=os.path.abspath(data_path),
        fixed_split_indices=fixed_split_indices,
    )
    models.update(get_all_models())

    print(f"\nTraining {len(models)} models...")

    # Train and collect predictions
    all_results = []
    predictions = {}

    for name, model in models.items():
        print(f"  Training: {name}...")
        try:
            # Train
            if isinstance(model, TwoStageModel):
                model.fit(X_train, y_train, presence=metadata.get('presence_train'))
            elif isinstance(model, LatentMLPModel):
                # LatentMLPModel uses its own data loader; X_train/y_train are ignored
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            # Predict on test set
            if isinstance(model, LatentMLPModel):
                # Align predictions with the baseline's test ordering via (sample_id, bin_uri)
                y_pred = model.predict(X_test, test_meta=metadata["test_meta"])
            else:
                y_pred = model.predict(X_test)
            predictions[name] = y_pred
            # Compute extended metrics
            metrics = compute_extended_metrics(y_test.values, y_pred)
            metrics['Model'] = name
            all_results.append(metrics)
        except Exception as e:
            import traceback
            print(f"    Error training {name}: {e}")
            traceback.print_exc()
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns
    col_order = ['Model', 'RMSE_micro', 'MAE_micro', 'R²', 'RMSE (zeros)', 'MAE (zeros)',
                 'RMSE (non-zeros)', 'MAE (non-zeros)', 'KL Divergence',
                 'Correlation']
    col_order = [c for c in col_order if c in results_df.columns]
    results_df = results_df[col_order + [c for c in results_df.columns if c not in col_order]]
    
    # Initialize consistent color mapping (sorted by MAE)
    get_model_colors(results_df['Model'].tolist(), sort_by_mae=True, results_df=results_df)
    print(f"\nModel order (by MAE): {MODEL_ORDER[:5]}... ({len(MODEL_ORDER)} total)")
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    y_test_arr = y_test.values
    
    # 1. Bar plots for all metrics
    print("\n1. Creating metric comparison plots...")
    plot_all_metrics_comparison(results_df, output_dir)

    # 2. Scatter plots: predicted vs actual (with density colormap)
    print("2. Creating scatter plots...")
    plot_scatter_predicted_vs_actual(predictions, y_test_arr, output_dir)

    # 3. Scatter plots zoomed on <1% ground truth
    print("3. Creating zoomed scatter plots (<1%)...")
    plot_scatter_zoomed(predictions, y_test_arr, output_dir)

    # 3b. Scatter plots log-log
    print("3b. Creating log-log scatter plots...")
    plot_scatter_loglog_predicted_vs_actual(predictions, y_test_arr, output_dir)

    # 4. Error by range (full)
    print("4. Creating error by range plot (full)...")
    plot_error_by_range(predictions, y_test_arr, output_dir)

    # 5. Error by range (zoomed ≤1%)
    print("5. Creating error by range plot (≤1%)...")
    plot_error_by_range_zoomed(predictions, y_test_arr, output_dir)

    # 6b. RAE by range
    print("6b. Creating RAE by range plot...")
    plot_rae_by_range(predictions, y_test_arr, output_dir)

    # 6. Summary table
    print("6. Creating summary table...")
    plot_summary_table(results_df, output_dir)

    # 7. Zero vs non-zero comparison
    print("7. Creating zero vs non-zero comparison...")
    plot_zero_vs_nonzero_comparison(results_df, output_dir)

    # 8. Top models radar chart
    print("8. Creating radar chart...")
    plot_top_models_overview(results_df, output_dir)
    
    # Save results to CSV
    results_path = os.path.join(output_dir, 'model_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTop 5 models by MAE:")
    top5 = results_df.nsmallest(5, 'MAE_micro')[['Model', 'RMSE_micro', 'MAE_micro', 'Correlation', 'KL Divergence']]
    print(top5.to_string(index=False))
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print("   - metrics_comparison.png")
    print("   - scatter_predicted_vs_actual.png")
    print("   - scatter_zoomed.png")
    print("   - error_by_range.png")
    print("   - error_by_range_zoomed.png")
    print("   - summary_table.png")
    print("   - zero_vs_nonzero_comparison.png")
    print("   - top_models_radar.png")
    print("   - relative_err_by_range.png")
    print("   - model_results.csv")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Create visualizations for baseline model results"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/ecuador_training_data.csv",
        help="Path to ecuador_training_data.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory to save figures"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.normpath(os.path.join(script_dir, data_path))
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return
    
    run_visualization(
        data_path=data_path,
        output_dir=output_dir,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()
