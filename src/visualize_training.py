#!/usr/bin/env python
"""
Visualization module for MLP+Latent model training results.
Creates clean, publication-ready plots for non-technical audiences.

Usage:
    python visualize_training.py --data_path data/ecuador_training_data.csv --output_dir figures
"""
import os
import argparse
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')


# =============================================================================
# Style Configuration
# =============================================================================

def set_style():
    """Set clean, publication-ready style for all plots."""
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
    })


# Color palette - consistent across all plots
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'success': '#2ECC71',      # Green
    'warning': '#F39C12',      # Orange
    'danger': '#E74C3C',       # Red
    'neutral': '#7F8C8D',      # Gray
    'light': '#BDC3C7',        # Light gray
}


# =============================================================================
# Metrics Computation
# =============================================================================

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

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_dir: str,
    title: str = "Training Progress",
    phase_info: Optional[List[str]] = None
) -> None:
    """
    Plot training and validation loss curves with phase coloring.
    
    Args:
        phase_info: List of phase labels ('mlp' or 'latent') for each epoch, same length as losses
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Define phase colors
    PHASE_COLORS = {
        'mlp': '#2E86AB',      # Blue for MLP training
        'latent': '#A23B72',   # Magenta for latent solver
    }
    
    if phase_info is None or len(phase_info) != len(train_losses):
        # No phase info: plot as-is
        ax.plot(epochs, train_losses, color=PHASE_COLORS['mlp'], linewidth=2.5, 
                label='Training Loss (MLP)', alpha=0.9)
        ax.plot(epochs, val_losses, color=PHASE_COLORS['latent'], linewidth=2.5, 
                label='Validation Loss', alpha=0.9)
    else:
        # Segment by phase for better visualization
        current_phase = phase_info[0]
        phase_start = 0
        
        for i in range(1, len(phase_info)):
            if phase_info[i] != current_phase:
                # Plot segment
                segment_epochs = epochs[phase_start:i]
                segment_train = train_losses[phase_start:i]
                segment_val = val_losses[phase_start:i]
                
                label_train = f'Training Loss ({current_phase.upper()})'
                label_val = f'Validation Loss ({current_phase.upper()})'
                if phase_start > 0:
                    label_train = None
                    label_val = None
                
                ax.plot(segment_epochs, segment_train, color=PHASE_COLORS[current_phase], 
                       linewidth=2.5, label=label_train, alpha=0.9, linestyle='-')
                ax.plot(segment_epochs, segment_val, color=PHASE_COLORS[current_phase], 
                       linewidth=2.5, label=label_val, alpha=0.6, linestyle='--')
                
                current_phase = phase_info[i]
                phase_start = i
        
        # Plot final segment
        segment_epochs = epochs[phase_start:]
        segment_train = train_losses[phase_start:]
        segment_val = val_losses[phase_start:]
        
        ax.plot(segment_epochs, segment_train, color=PHASE_COLORS[current_phase], 
               linewidth=2.5, alpha=0.9, linestyle='-')
        ax.plot(segment_epochs, segment_val, color=PHASE_COLORS[current_phase], 
               linewidth=2.5, alpha=0.6, linestyle='--')
    
    # Mark minimum validation loss
    min_idx = np.argmin(val_losses)
    min_val = val_losses[min_idx]
    ax.scatter([min_idx + 1], [min_val], color=COLORS['success'], s=150, zorder=5,
               marker='*', edgecolors='white', linewidths=1.5)
    ax.annotate(f'Best: {min_val:.3f}', xy=(min_idx + 1, min_val), 
                xytext=(min_idx + 20, min_val + 0.1),
                fontsize=11, fontweight='bold', color=COLORS['success'],
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))
    
    # Add light grid for readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cross-Entropy Loss', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=False, fontsize=11, loc='upper right')
    
    # Add final values annotation
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    textstr = f'Final Train: {final_train:.4f}\nFinal Val: {final_val:.4f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: training_curves.png")


def plot_scatter_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    title: str = "MLP + Latent Model"
) -> None:
    """Create scatter plot with density colormap - publication quality."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ===== Left: Density scatter plot =====
    ax1 = axes[0]
    
    # Compute density
    try:
        xy = np.vstack([y_true, y_pred])
        density = gaussian_kde(xy)(xy)
        idx_sorted = np.argsort(density)
        
        scatter = ax1.scatter(
            y_true[idx_sorted], 
            y_pred[idx_sorted], 
            c=density[idx_sorted],
            cmap='viridis',
            alpha=0.7, 
            s=15, 
            edgecolors='none'
        )
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
        cbar.set_label('Point Density', fontsize=12)
    except:
        ax1.scatter(y_true, y_pred, alpha=0.3, s=15, color=COLORS['primary'])
    
    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2.5, alpha=0.8, 
             label='Perfect Prediction')
    
    # Calculate and display correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    stats_text = f'Correlation: r = {corr:.3f}\nR² = {r2:.3f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=13,
             verticalalignment='top', bbox=props, fontweight='bold')
    
    ax1.set_xlabel('Actual Relative Abundance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Predicted Relative Abundance', fontsize=14, fontweight='bold')
    ax1.set_title(f'{title}\nPredicted vs Actual', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlim(-0.005, max_val)
    ax1.set_ylim(-0.005, max_val)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ===== Right: Hexbin plot for better density visualization =====
    ax2 = axes[1]
    
    hb = ax2.hexbin(y_true, y_pred, gridsize=50, cmap='viridis', mincnt=1,
                    extent=[0, max_val, 0, max_val])
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2.5, alpha=0.8)
    
    cbar2 = plt.colorbar(hb, ax=ax2, shrink=0.8)
    cbar2.set_label('Count', fontsize=12)
    
    ax2.set_xlabel('Actual Relative Abundance', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Predicted Relative Abundance', fontsize=14, fontweight='bold')
    ax2.set_title('Hexbin Density Plot', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlim(-0.005, max_val)
    ax2.set_ylim(-0.005, max_val)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_predicted_vs_actual.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved: scatter_predicted_vs_actual.png")


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str
) -> None:
    """Create publication-quality residual distribution plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    residuals = y_true - y_pred
    
    # ===== Left: Overall residual distribution =====
    ax1 = axes[0]
    
    sns.histplot(residuals, bins=60, color=COLORS['primary'], edgecolor='white', 
                 alpha=0.8, ax=ax1, stat='density')
    ax1.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=2.5, 
                label='Zero Error')
    ax1.axvline(x=np.mean(residuals), color=COLORS['warning'], linestyle='-', 
                linewidth=2.5, label=f'Mean: {np.mean(residuals):.4f}')
    
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    median_res = np.median(residuals)
    
    stats_text = f'Mean: {mean_res:.5f}\nStd: {std_res:.5f}\nMedian: {median_res:.5f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax1.set_xlabel('Residual (Actual - Predicted)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax1.set_title('Overall Residual Distribution', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ===== Middle: Split by zero/non-zero =====
    ax2 = axes[1]
    
    zero_mask = y_true == 0
    nonzero_mask = y_true > 0
    
    if zero_mask.sum() > 0:
        sns.histplot(residuals[zero_mask], bins=40, color=COLORS['warning'], 
                    edgecolor='white', alpha=0.6, ax=ax2, stat='density',
                    label=f'Zeros (n={zero_mask.sum():,})')
    if nonzero_mask.sum() > 0:
        sns.histplot(residuals[nonzero_mask], bins=40, color=COLORS['primary'], 
                    edgecolor='white', alpha=0.6, ax=ax2, stat='density',
                    label=f'Non-zeros (n={nonzero_mask.sum():,})')
    
    ax2.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual (Actual - Predicted)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax2.set_title('Residuals by Actual Value Type', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # ===== Right: Q-Q plot =====
    ax3 = axes[2]
    
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.get_lines()[0].set_markerfacecolor(COLORS['primary'])
    ax3.get_lines()[0].set_markeredgecolor('white')
    ax3.get_lines()[0].set_markersize(4)
    ax3.get_lines()[1].set_color(COLORS['danger'])
    ax3.get_lines()[1].set_linewidth(2)
    
    ax3.set_xlabel('Theoretical Quantiles', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Sample Quantiles', fontsize=13, fontweight='bold')
    ax3.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_distribution.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved: residual_distribution.png")


def plot_metrics_summary(
    metrics: Dict[str, float],
    output_dir: str
) -> None:
    """Create a clean summary bar chart of key metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== Left: Error metrics (lower is better) =====
    ax1 = axes[0]
    
    error_metrics = ['RMSE', 'MAE', 'RMSE (zeros)', 'RMSE (non-zeros)', 'Bray-Curtis']
    error_values = [metrics.get(m, 0) for m in error_metrics]
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.7, len(error_metrics)))
    bars = ax1.barh(error_metrics, error_values, color=colors, edgecolor='white', 
                    linewidth=1.5, height=0.6)
    
    # Add value labels
    max_val = max(error_values) * 1.15
    for bar, val in zip(bars, error_values):
        ax1.text(val + max_val * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Value (lower is better)', fontsize=13, fontweight='bold')
    ax1.set_title('Error Metrics', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlim(0, max_val)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # ===== Right: Quality metrics =====
    ax2 = axes[1]
    
    quality_metrics = ['R²', 'Correlation']
    quality_values = [metrics.get(m, 0) for m in quality_metrics]
    
    # Color based on whether value is good or bad
    colors = [COLORS['success'] if v > 0 else COLORS['danger'] for v in quality_values]
    bars = ax2.barh(quality_metrics, quality_values, color=colors, edgecolor='white', 
                    linewidth=1.5, height=0.5)
    
    # Add value labels
    for bar, val in zip(bars, quality_values):
        x_pos = val + 0.05 if val >= 0 else val - 0.1
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=12, fontweight='bold')
    
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Value (higher is better)', fontsize=13, fontweight='bold')
    ax2.set_title('Quality Metrics', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xlim(min(min(quality_values) - 0.5, -1), max(max(quality_values) + 0.3, 1.1))
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved: metrics_summary.png")


def plot_sample_predictions(
    y_true_by_sample: Dict[int, np.ndarray],
    y_pred_by_sample: Dict[int, np.ndarray],
    output_dir: str,
    n_samples: int = 6
) -> None:
    """Plot predicted vs actual distributions for individual samples."""
    sample_ids = list(y_true_by_sample.keys())[:n_samples]
    
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, sample_id in enumerate(sample_ids):
        y_true = y_true_by_sample[sample_id]
        y_pred = y_pred_by_sample[sample_id]
        
        # Sort by actual abundance
        sort_idx = np.argsort(y_true)[::-1]
        y_true_sorted = y_true[sort_idx][:30]  # Top 30 bins
        y_pred_sorted = y_pred[sort_idx][:30]
        
        x = np.arange(len(y_true_sorted))
        width = 0.35
        
        axes[idx].bar(x - width/2, y_true_sorted, width, label='Actual', 
                     alpha=0.8, color=COLORS['primary'], edgecolor='white')
        axes[idx].bar(x + width/2, y_pred_sorted, width, label='Predicted', 
                     alpha=0.8, color=COLORS['secondary'], edgecolor='white')
        
        # Sample-level correlation
        corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        
        axes[idx].set_xlabel('BIN (top 30 by abundance)', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Relative Abundance', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'Sample {sample_id}\n(r = {corr:.3f}, n_bins = {len(y_true)})', 
                           fontsize=12, fontweight='bold', pad=10)
        axes[idx].legend(fontsize=9, loc='upper right')
        axes[idx].set_xticks([])
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide empty subplots
    for idx in range(len(sample_ids), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved: sample_predictions.png")


def plot_latent_distribution(
    latent_vec: np.ndarray,
    output_dir: str
) -> None:
    """Plot the distribution of latent variables with clear visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Left: Histogram =====
    ax1 = axes[0]
    
    sns.histplot(latent_vec, bins=60, color=COLORS['primary'], edgecolor='white', 
                 alpha=0.8, ax=ax1)
    ax1.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=2.5, label='Zero')
    ax1.axvline(x=np.mean(latent_vec), color=COLORS['warning'], linestyle='-', 
                linewidth=2.5, label=f'Mean: {np.mean(latent_vec):.2f}')
    
    stats_text = f'Mean: {latent_vec.mean():.2f}\nStd: {latent_vec.std():.2f}\nMin: {latent_vec.min():.2f}\nMax: {latent_vec.max():.2f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax1.set_xlabel('Latent Value (D)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax1.set_title('Latent Vector Distribution', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ===== Right: Sorted latent values =====
    ax2 = axes[1]
    
    sorted_latent = np.sort(latent_vec)
    x = np.arange(len(sorted_latent))
    
    # Fill positive and negative regions
    ax2.fill_between(x, sorted_latent, 0, where=(sorted_latent > 0), 
                     alpha=0.4, color=COLORS['success'], label='Positive (abundant)')
    ax2.fill_between(x, sorted_latent, 0, where=(sorted_latent < 0), 
                     alpha=0.4, color=COLORS['danger'], label='Negative (rare)')
    ax2.plot(x, sorted_latent, color=COLORS['primary'], linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Count positive/negative
    n_pos = (latent_vec > 0).sum()
    n_neg = (latent_vec < 0).sum()
    ax2.text(0.05, 0.95, f'Positive: {n_pos:,} BINs\nNegative: {n_neg:,} BINs', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    ax2.set_xlabel('BIN (sorted by latent value)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Latent Value (D)', fontsize=13, fontweight='bold')
    ax2.set_title('Sorted Latent Values per BIN', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_distribution.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved: latent_distribution.png")


def plot_model_overview(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latent_vec: np.ndarray,
    metrics: Dict[str, float],
    train_losses: List[float],
    val_losses: List[float],
    output_dir: str
) -> None:
    """Create a single infographic-style overview of the model."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Title
    fig.suptitle('MLP + Latent Model - Results Overview', fontsize=22, fontweight='bold', y=0.98)
    
    # ===== 1. Training Curves (top-left) =====
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, color=COLORS['primary'], linewidth=2, label='Train')
    ax1.plot(epochs, val_losses, color=COLORS['secondary'], linewidth=2, label='Val')
    min_idx = np.argmin(val_losses)
    ax1.scatter([min_idx + 1], [val_losses[min_idx]], color=COLORS['success'], s=100, 
                marker='*', zorder=5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ===== 2. Scatter Plot (top-middle) =====
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        xy = np.vstack([y_true, y_pred])
        density = gaussian_kde(xy)(xy)
        idx_sorted = np.argsort(density)
        sc = ax2.scatter(y_true[idx_sorted], y_pred[idx_sorted], c=density[idx_sorted],
                        cmap='viridis', alpha=0.6, s=10)
    except:
        ax2.scatter(y_true, y_pred, alpha=0.3, s=10, color=COLORS['primary'])
    
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'Pred vs Actual (r={metrics["Correlation"]:.3f})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # ===== 3. Residuals (top-right) =====
    ax3 = fig.add_subplot(gs[0, 2])
    residuals = y_true - y_pred
    sns.histplot(residuals, bins=50, color=COLORS['primary'], ax=ax3, alpha=0.7)
    ax3.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=2)
    ax3.set_xlabel('Residual')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Residuals (μ={np.mean(residuals):.4f})', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ===== 4. Latent Distribution (middle-left) =====
    ax4 = fig.add_subplot(gs[1, 0])
    sns.histplot(latent_vec, bins=50, color=COLORS['secondary'], ax=ax4, alpha=0.7)
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Latent Value')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Latent Vector (μ={latent_vec.mean():.2f})', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ===== 5. MAE by Type (middle-middle) =====
    ax5 = fig.add_subplot(gs[1, 1])
    categories = ['All', 'Zeros', 'Non-zeros']
    mae_values = [metrics['MAE'], metrics.get('MAE (zeros)', 0), metrics.get('MAE (non-zeros)', 0)]
    colors = [COLORS['primary'], COLORS['warning'], COLORS['success']]
    bars = ax5.bar(categories, mae_values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, mae_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax5.set_ylabel('MAE')
    ax5.set_title('MAE by Data Type', fontsize=14, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # ===== 6. RMSE by Type (middle-right) =====
    ax6 = fig.add_subplot(gs[1, 2])
    rmse_values = [metrics['RMSE'], metrics.get('RMSE (zeros)', 0), metrics.get('RMSE (non-zeros)', 0)]
    bars = ax6.bar(categories, rmse_values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, rmse_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax6.set_ylabel('RMSE')
    ax6.set_title('RMSE by Data Type', fontsize=14, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # ===== 7. Summary Statistics (bottom - spanning all columns) =====
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                    MODEL PERFORMANCE SUMMARY                              ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                           ║
    ║   Overall Metrics                          │   Split Metrics                              ║
    ║   ─────────────────────                    │   ─────────────────────                      ║
    ║   • RMSE:          {metrics['RMSE']:.5f}                │   • Zeros:     n = {metrics['n_zeros']:,}                     ║
    ║   • MAE:           {metrics['MAE']:.5f}                │   • Non-zeros: n = {metrics['n_nonzeros']:,}                     ║
    ║   • R²:            {metrics['R²']:.4f}                 │                                              ║
    ║   • Correlation:   {metrics['Correlation']:.4f}                 │   Latent Vector                              ║
    ║   • Bray-Curtis:   {metrics['Bray-Curtis']:.4f}                 │   ─────────────────────                      ║
    ║   • KL Divergence: {metrics['KL Divergence']:.4f}                 │   • Mean: {latent_vec.mean():.2f}  Std: {latent_vec.std():.2f}               ║
    ║                                                                                           ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round,pad=1', 
             facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
    
    plt.savefig(os.path.join(output_dir, 'model_overview.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved: model_overview.png")


# =============================================================================
# Main Prediction & Visualization
# =============================================================================

def get_predictions(trainer, split: str = "test") -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """Get predictions and organize by sample."""
    data_loader = (
        trainer.train_loader_ordered if split == "train" else
        trainer.val_loader if split == "val" else
        trainer.test_loader
    )
    
    trainer.model.eval()
    
    all_y_true = []
    all_y_pred = []
    y_true_by_sample = {}
    y_pred_by_sample = {}
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets, bin_idx, sample_idx, mask = trainer._to_device(batch)
            
            if trainer.loss_mode == "sample":
                B, max_bins, n_feat = inputs.shape
                inputs_flat = inputs.view(B * max_bins, n_feat)
                bin_idx_flat = bin_idx.view(B * max_bins)
                
                outputs_flat = trainer.model(inputs_flat, bin_idx_flat)
                outputs = outputs_flat.view(B, max_bins)
                outputs = outputs.masked_fill(mask == 0, float('-inf'))
                
                probs = F.softmax(outputs, dim=-1)
                
                for b in range(B):
                    valid_mask = mask[b].bool()
                    y_true_sample = targets[b, valid_mask].cpu().numpy()
                    y_pred_sample = probs[b, valid_mask].cpu().numpy()
                    
                    sid = sample_idx[b].item()
                    y_true_by_sample[sid] = y_true_sample
                    y_pred_by_sample[sid] = y_pred_sample
                    
                    all_y_true.extend(y_true_sample)
                    all_y_pred.extend(y_pred_sample)
            else:
                outputs = trainer.model(inputs, bin_idx)
                probs = torch.sigmoid(outputs)
                
                all_y_true.extend(targets.cpu().numpy().flatten())
                all_y_pred.extend(probs.cpu().numpy().flatten())
    
    return np.array(all_y_true), np.array(all_y_pred), y_true_by_sample, y_pred_by_sample


def visualize_results(
    trainer,
    train_losses: List[float],
    val_losses: List[float],
    output_dir: str,
    phase_info: Optional[List[str]] = None
) -> Dict[str, float]:
    """Generate all visualizations and return metrics."""
    os.makedirs(output_dir, exist_ok=True)
    set_style()
    
    print("\n" + "="*60)
    print("           GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Get test predictions
    print("  Getting predictions...")
    y_true, y_pred, y_true_by_sample, y_pred_by_sample = get_predictions(trainer, split="test")
    
    # Compute metrics
    metrics = compute_extended_metrics(y_true, y_pred)
    
    print("\n  Test Set Metrics:")
    print("  " + "-"*40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.5f}")
        else:
            print(f"    {k}: {v}")
    print()
    
    # Get latent vector
    latent_vec = trainer.model.latent_vec.detach().cpu().numpy()
    
    # Generate all plots
    print("  Generating plots...")
    plot_training_curves(train_losses, val_losses, output_dir, phase_info=phase_info)
    plot_scatter_predicted_vs_actual(y_true, y_pred, output_dir)
    plot_residual_distribution(y_true, y_pred, output_dir)
    plot_metrics_summary(metrics, output_dir)
    
    if y_true_by_sample:
        plot_sample_predictions(y_true_by_sample, y_pred_by_sample, output_dir)
    
    plot_latent_distribution(latent_vec, output_dir)
    plot_model_overview(y_true, y_pred, latent_vec, metrics, train_losses, val_losses, output_dir)
    
    print(f"\n  All visualizations saved to: {output_dir}")
    print("="*60 + "\n")
    
    return metrics


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description="Train and visualize MLP+Latent model")
    parser.add_argument("--data_path", type=str, default="../data/ecuador_training_data.csv",
                        help="Path to training data")
    # Always use the project root for figures
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DEFAULT_FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_FIGURES_DIR,
                        help="Directory to save figures")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    os.environ['WANDB_MODE'] = 'disabled' if args.no_wandb else 'online'

    from train import Trainer
    from config import Config, set_seed

    set_seed()
    cfg = Config()

    # Use absolute output_dir
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("         MLP + LATENT MODEL TRAINING")
    print("="*60)
    print(f"\n  Configuration:")
    print(f"    - Latent warmup fraction: {cfg.latent_warmup_frac}")
    print(f"    - Epochs per cycle: {cfg.epochs}")
    print(f"    - Max cycles: {cfg.max_cycles}")
    print(f"    - Learning rate: {cfg.lr}")
    print(f"    - Latent L2 reg: {cfg.latent_l2_reg}")
    print(f"    - Latent smooth reg: {cfg.latent_smooth_reg}")
    print()
    
    print("  Creating trainer...")
    trainer = Trainer(cfg, args.data_path, loss_type="cross_entropy")
    
    print("\n  Training model...")
    train_losses = []
    val_losses = []
    phase_info = []  # Track which phase each epoch is (MLP or latent)
    
    # Alternating training with latent warmup
    warmup_cycles = max(1, int(cfg.latent_warmup_frac * cfg.max_cycles))
    print(f"\n  Alternating Training ({cfg.max_cycles} max cycles, latent warmup over first {warmup_cycles} cycles)")
    print("  " + "-"*50)
    
    best_val_loss = float('inf')
    no_improve = 0
    
    for cycle in range(cfg.max_cycles):
        # Compute latent update weight: linearly ramp from 0 → 1 over warmup_cycles
        alpha = min(1.0, cycle / warmup_cycles)

        # Solve latent, then blend if still in warmup
        old_latent = trainer.model.latent_vec.detach().cpu().numpy().copy()
        solved_latent = trainer.solve_latent()
        if alpha < 1.0:
            blended = (1.0 - alpha) * old_latent + alpha * solved_latent
            trainer.model.set_latent(blended)
        
        # Train MLP
        cycle_train_loss = 0
        for epoch in range(cfg.epochs):
            train_loss = trainer.train_epoch()
            val_loss = trainer.validate("val")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            phase_info.append('mlp')  # In alternating phase, still training MLP
            cycle_train_loss = train_loss
        
        print(f"    Cycle {cycle+1:2d}/{cfg.max_cycles} (α={alpha:.2f}): train={cycle_train_loss:.4f}, val={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve = 0
            trainer.model.save_model(trainer.save_path)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"    Early stopping at cycle {cycle+1}")
                break
    
    # Generate visualizations
    metrics = visualize_results(trainer, train_losses, val_losses, output_dir, phase_info=phase_info)
