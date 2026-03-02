"""
Visualization utilities for Two-Stage Random Forest Model.
Generates clear, publication-ready figures for non-technical audiences.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import argparse

from two_stage_model import TwoStageRandomForest, TwoStageConfig


def load_data(data_dir: str):
    """Load train, validation, and test data from CSV files."""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    y_val = pd.read_csv(os.path.join(data_dir, "y_val.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def set_style():
    """Set publication-ready style for plots."""
    plt.style.use('seaborn-v0_8-white')  # Clean style without grid
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['axes.grid'] = False  # Disable grid globally


def plot_classifier_confusion_matrix(model, X, y, save_path=None):
    """
    Plot confusion matrix for the classifier stage.
    Shows how well the model distinguishes zero vs non-zero abundance.
    """
    X_proc = model.preprocess_features(X)
    y_arr = y.values.ravel()
    y_binary_true = (y_arr > 0).astype(int)
    y_binary_pred = model.classifier.predict(X_proc)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_binary_true, y_binary_pred)
    
    # Create custom labels
    labels = ['Zero\nAbundance', 'Non-Zero\nAbundance']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 16})
    
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title('Classification Results:\nZero vs Non-Zero Abundance', fontsize=16, fontweight='bold')
    
    # Add accuracy annotation
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.1%}', 
            transform=ax.transAxes, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_classifier_metrics(model, X, y, save_path=None):
    """
    Bar chart showing classifier performance metrics.
    """
    X_proc = model.preprocess_features(X)
    y_arr = y.values.ravel()
    y_binary_true = (y_arr > 0).astype(int)
    y_binary_pred = model.classifier.predict(X_proc)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Accuracy': accuracy_score(y_binary_true, y_binary_pred),
        'Precision': precision_score(y_binary_true, y_binary_pred),
        'Recall': recall_score(y_binary_true, y_binary_pred),
        'F1-Score': f1_score(y_binary_true, y_binary_pred)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use seaborn color palette
    colors = sns.color_palette("deep", n_colors=4)
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Stage 1: Classifier Performance\n(Predicting Zero vs Non-Zero Abundance)', 
                 fontsize=16, fontweight='bold')
    
    # Remove spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_regressor_predictions(model, X, y, save_path=None):
    """
    Scatter plot comparing predicted vs actual abundance values.
    Only shows non-zero samples.
    """
    X_proc = model.preprocess_features(X)
    y_arr = y.values.ravel()
    
    # Filter to non-zero
    nonzero_mask = y_arr > 0
    X_nonzero = X_proc[nonzero_mask]
    y_true = y_arr[nonzero_mask]
    y_pred = model.regressor.predict(X_nonzero)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Scatter plot with viridis colormap (density-based)
    ax1 = axes[0]
    
    # Create 2D histogram for density coloring
    from scipy.stats import gaussian_kde
    xy = np.vstack([y_true, y_pred])
    try:
        density = gaussian_kde(xy)(xy)
        idx = density.argsort()
        y_true_sorted, y_pred_sorted, density_sorted = y_true[idx], y_pred[idx], density[idx]
        scatter = ax1.scatter(y_true_sorted, y_pred_sorted, c=density_sorted, s=20, 
                             cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label='Density')
    except:
        # Fallback if KDE fails
        ax1.scatter(y_true, y_pred, alpha=0.3, s=20, c=sns.color_palette("viridis", as_cmap=True)(0.5))
    
    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Relative Abundance', fontsize=12)
    ax1.set_ylabel('Predicted Relative Abundance', fontsize=12)
    ax1.set_title('Stage 2: Predicted vs Actual Abundance\n(Non-Zero Samples Only)', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add R² annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right: Residual distribution
    ax2 = axes[1]
    residuals = y_pred - y_true
    sns.histplot(residuals, bins=50, color=sns.color_palette("deep")[0], ax=ax2, alpha=0.7)
    ax2.axvline(x=0, color=sns.color_palette("deep")[3], linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(x=residuals.mean(), color=sns.color_palette("deep")[1], linestyle='-', linewidth=2, 
                label=f'Mean Error: {residuals.mean():.4f}')
    
    ax2.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_nonzero_actual_predictions(model, X, y, save_path=None):
    """
    Scatter plot of predicted vs actual values for samples with non-zero ACTUAL values.
    Uses the full two-stage pipeline prediction.
    """
    y_arr = y.values.ravel()
    
    # Get full pipeline predictions
    y_pred_full = model.predict(X)
    
    # Filter to non-zero ACTUAL values
    nonzero_actual_mask = y_arr > 0
    y_true_nonzero = y_arr[nonzero_actual_mask]
    y_pred_nonzero = y_pred_full[nonzero_actual_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Scatter plot with viridis colormap
    ax1 = axes[0]
    
    from scipy.stats import gaussian_kde
    xy = np.vstack([y_true_nonzero, y_pred_nonzero])
    try:
        density = gaussian_kde(xy)(xy)
        idx = density.argsort()
        y_t_sorted, y_p_sorted, d_sorted = y_true_nonzero[idx], y_pred_nonzero[idx], density[idx]
        scatter = ax1.scatter(y_t_sorted, y_p_sorted, c=d_sorted, s=20, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label='Density')
    except:
        ax1.scatter(y_true_nonzero, y_pred_nonzero, alpha=0.3, s=20, cmap='viridis')
    
    # Perfect prediction line
    max_val = max(y_true_nonzero.max(), y_pred_nonzero.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Relative Abundance', fontsize=12)
    ax1.set_ylabel('Predicted Relative Abundance', fontsize=12)
    ax1.set_title('Full Pipeline: Predicted vs Actual\n(Non-Zero Actual Values Only)', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_true_nonzero, y_pred_nonzero)
    mae = mean_absolute_error(y_true_nonzero, y_pred_nonzero)
    
    # Count how many were predicted as zero
    pred_zero_count = (y_pred_nonzero == 0).sum()
    pred_zero_pct = pred_zero_count / len(y_pred_nonzero) * 100
    
    stats_text = f'R² = {r2:.3f}\nMAE = {mae:.5f}\nMissed (pred=0): {pred_zero_pct:.1f}%'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right: Breakdown by prediction type
    ax2 = axes[1]
    
    # Categorize predictions
    correctly_nonzero = (y_pred_nonzero > 0).sum()
    incorrectly_zero = (y_pred_nonzero == 0).sum()
    
    categories = ['Correctly Predicted\nas Non-Zero', 'Incorrectly Predicted\nas Zero']
    counts = [correctly_nonzero, incorrectly_zero]
    colors = [sns.color_palette("deep")[2], sns.color_palette("deep")[3]]
    
    bars = ax2.bar(categories, counts, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        pct = count / len(y_pred_nonzero) * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Classification of True Non-Zero Samples', fontsize=14, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_permutation_importance(model, X, y, n_repeats=10, save_path=None):
    """
    Permutation importance analysis for both classifier and regressor.
    Shows how much each feature contributes to model performance.
    """
    X_proc = model.preprocess_features(X)
    y_arr = y.values.ravel()
    y_binary = (y_arr > 0).astype(int)
    
    n_features = len(model.feature_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(8, n_features * 0.4)))
    
    # Classifier permutation importance
    ax1 = axes[0]
    print("  Computing classifier permutation importance...")
    clf_perm = permutation_importance(
        model.classifier, X_proc, y_binary, 
        n_repeats=n_repeats, random_state=42, n_jobs=-1,
        scoring='f1'
    )
    
    clf_importance = pd.DataFrame({
        'feature': model.feature_names,
        'importance': clf_perm.importances_mean,
        'std': clf_perm.importances_std
    }).sort_values('importance', ascending=True)
    
    colors1 = plt.cm.viridis(np.linspace(0.3, 0.8, n_features))
    ax1.barh(clf_importance['feature'], clf_importance['importance'], 
             xerr=clf_importance['std'], color=colors1, edgecolor='white', capsize=3)
    ax1.set_xlabel('Mean Decrease in F1-Score', fontsize=12)
    ax1.set_title('Classifier Permutation Importance\n(Impact on Zero/Non-Zero Detection)', 
                  fontsize=14, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Regressor permutation importance (on non-zero samples)
    ax2 = axes[1]
    nonzero_mask = y_arr > 0
    X_nonzero = X_proc[nonzero_mask]
    y_nonzero = y_arr[nonzero_mask]
    
    print("  Computing regressor permutation importance...")
    reg_perm = permutation_importance(
        model.regressor, X_nonzero, y_nonzero,
        n_repeats=n_repeats, random_state=42, n_jobs=-1,
        scoring='neg_root_mean_squared_error'
    )
    
    # Convert negative scores to positive RMSE increase
    reg_importance = pd.DataFrame({
        'feature': model.feature_names,
        'importance': -reg_perm.importances_mean,  # Negate to get RMSE increase
        'std': reg_perm.importances_std
    }).sort_values('importance', ascending=False)  # Largest increase at top
    
    colors2 = plt.cm.viridis(np.linspace(0.3, 0.8, n_features))
    ax2.barh(reg_importance['feature'], reg_importance['importance'],
             xerr=reg_importance['std'], color=colors2, edgecolor='white', capsize=3)
    ax2.set_xlabel('Mean Increase in RMSE', fontsize=12)
    ax2.set_title('Regressor Permutation Importance\n(Impact on Abundance Prediction)', 
                  fontsize=14, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_mae_per_range(model, X, y, save_path=None):
    """
    Bar chart showing MAE for different abundance ranges.
    Excludes bins with abundance > 1%.
    Uses quartile-based bins for balanced non-zero sample counts.
    """
    from sklearn.metrics import mean_absolute_error
    
    y_arr = y.values.ravel()
    y_pred = model.predict(X)
    
    # Define abundance ranges (excluding >1%)
    # Quartile-based thresholds for ~equal non-zero bin sizes
    ranges = [
        (0, 0, 'Zero'),
        (0, 0.00106, '>0-0.11%'),
        (0.00106, 0.00145, '0.11-0.15%'),
        (0.00145, 0.0022, '0.15-0.22%'),
        (0.0022, 0.01, '0.22-1%'),
    ]
    
    mae_values = []
    range_labels = []
    counts = []
    
    for low, high, label in ranges:
        if low == 0 and high == 0:
            # Zero abundance
            mask = y_arr == 0
        elif low == 0:
            # >0 to high (exclusive of zero)
            mask = (y_arr > 0) & (y_arr <= high)
        else:
            mask = (y_arr > low) & (y_arr <= high)
        
        if mask.sum() > 0:
            mae = mean_absolute_error(y_arr[mask], y_pred[mask])
            mae_values.append(mae)
            range_labels.append(label)
            counts.append(mask.sum())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(range_labels)))
    bars = ax.bar(range_labels, mae_values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add count labels on bars
    for bar, count, mae in zip(bars, counts, mae_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'n={count:,}\nMAE={mae:.5f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Actual Abundance Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('MAE by Abundance Range\n(Excluding >1% Abundance)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_predictions_zoomed(model, X, y, save_path=None):
    """
    Scatter plot of ground truth vs prediction zoomed to <1% abundance range.
    """
    from sklearn.metrics import mean_absolute_error
    from scipy.stats import gaussian_kde, pearsonr
    
    y_arr = y.values.ravel()
    y_pred = model.predict(X)
    
    # Filter to ground truth < 1% (0.01)
    mask = y_arr < 0.01
    y_true_filtered = y_arr[mask]
    y_pred_filtered = y_pred[mask]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Density-based coloring
    xy = np.vstack([y_true_filtered, y_pred_filtered])
    try:
        density = gaussian_kde(xy)(xy)
        idx = density.argsort()
        y_t_sorted = y_true_filtered[idx]
        y_p_sorted = y_pred_filtered[idx]
        d_sorted = density[idx]
        scatter = ax.scatter(y_t_sorted, y_p_sorted, c=d_sorted, s=20, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Density')
    except:
        ax.scatter(y_true_filtered, y_pred_filtered, alpha=0.3, s=20, color=sns.color_palette("deep")[0])
    
    # Perfect prediction line
    max_val = 0.01
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Relative Abundance', fontsize=12)
    ax.set_ylabel('Predicted Relative Abundance', fontsize=12)
    ax.set_title('Predicted vs Actual Abundance\n(Ground Truth < 1%)', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.0005, 0.01)
    ax.set_ylim(-0.0005, max(0.01, y_pred_filtered.max() * 1.05))
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add metrics - use Pearson correlation instead of R² for low-variance data
    mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
    corr, _ = pearsonr(y_true_filtered, y_pred_filtered)
    n_samples = len(y_true_filtered)
    
    stats_text = f'n = {n_samples:,}\nPearson r = {corr:.3f}\nMAE = {mae:.5f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_feature_importance(model, save_path=None):
    """
    Horizontal bar chart showing all feature importances.
    """
    importance = model.get_feature_importance()
    all_features = importance.iloc[::-1]  # Reverse for horizontal bar (lowest at top)
    n_features = len(all_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(8, n_features * 0.4)))
    
    # Classifier importance
    ax1 = axes[0]
    colors1 = plt.cm.viridis(np.linspace(0.3, 0.8, n_features))
    ax1.barh(all_features['feature'], all_features['classifier_importance'], 
             color=colors1, edgecolor='white')
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title('Classifier Feature Importance\n(Zero vs Non-Zero)', fontsize=14, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Regressor importance
    ax2 = axes[1]
    colors2 = plt.cm.viridis(np.linspace(0.3, 0.8, n_features))
    ax2.barh(all_features['feature'], all_features['regressor_importance'],
             color=colors2, edgecolor='white')
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title('Regressor Feature Importance\n(Abundance Prediction)', fontsize=14, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_combined_performance(model, X, y, save_path=None):
    """
    Combined visualization showing end-to-end model performance.
    """
    y_arr = y.values.ravel()
    y_pred = model.predict(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Distribution comparison
    ax1 = axes[0, 0]
    sns.histplot(y_arr, bins=50, alpha=0.6, label='Actual', color=sns.color_palette("deep")[0], ax=ax1)
    sns.histplot(y_pred, bins=50, alpha=0.6, label='Predicted', color=sns.color_palette("deep")[3], ax=ax1)
    ax1.set_xlabel('Relative Abundance', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Distribution: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Zero classification bar chart
    ax2 = axes[0, 1]
    true_zeros = (y_arr == 0).sum()
    true_nonzeros = (y_arr > 0).sum()
    pred_zeros = (y_pred == 0).sum()
    pred_nonzeros = (y_pred > 0).sum()
    
    # Stacked bar for comparison
    x = ['Actual', 'Predicted']
    zeros = [true_zeros, pred_zeros]
    nonzeros = [true_nonzeros, pred_nonzeros]
    
    ax2.bar(x, zeros, label='Zero Abundance', color=sns.color_palette("deep")[7], edgecolor='white')
    ax2.bar(x, nonzeros, bottom=zeros, label='Non-Zero Abundance', color=sns.color_palette("deep")[2], edgecolor='white')
    
    # Add count labels
    for i, (z, nz) in enumerate(zip(zeros, nonzeros)):
        ax2.text(i, z/2, f'{z:,}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax2.text(i, z + nz/2, f'{nz:,}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Zero vs Non-Zero: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Error by abundance range
    ax3 = axes[1, 0]
    
    # Create bins
    bins = [(0, 0, 'Zero'), (0.0001, 0.001, '0.01-0.1%'), (0.001, 0.01, '0.1-1%'), 
            (0.01, 0.1, '1-10%'), (0.1, 1.0, '>10%')]
    
    mae_by_bin = []
    labels = []
    counts = []
    
    for low, high, label in bins:
        if low == 0 and high == 0:
            mask = y_arr == 0
        else:
            mask = (y_arr >= low) & (y_arr < high)
        
        if mask.sum() > 0:
            mae = np.mean(np.abs(y_arr[mask] - y_pred[mask]))
            mae_by_bin.append(mae)
            labels.append(label)
            counts.append(mask.sum())
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    bars = ax3.bar(labels, mae_by_bin, color=colors, edgecolor='white')
    
    ax3.set_xlabel('Abundance Range', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error', fontsize=12)
    ax3.set_title('Prediction Error by Abundance Range', fontsize=14, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add sample counts
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'n={count:,}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    # 4. Summary metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    
    overall_mae = mean_absolute_error(y_arr, y_pred)
    overall_r2 = r2_score(y_arr, y_pred)
    
    # Zero detection accuracy
    true_zero_mask = y_arr == 0
    pred_zero_correct = (y_pred[true_zero_mask] == 0).mean()
    
    true_nonzero_mask = y_arr > 0
    pred_nonzero_correct = (y_pred[true_nonzero_mask] > 0).mean()
    
    summary_text = f"""
    MODEL PERFORMANCE SUMMARY
    {'='*40}
    
    Overall Metrics:
    - Mean Absolute Error: {overall_mae:.5f}
    - R² Score: {overall_r2:.3f}
    
    Zero Detection:
    - True zeros correctly predicted: {pred_zero_correct:.1%}
    - True non-zeros correctly detected: {pred_nonzero_correct:.1%}
    
    Data Summary:
    - Total samples: {len(y_arr):,}
    - Zero abundance: {true_zero_mask.sum():,} ({true_zero_mask.mean():.1%})
    - Non-zero abundance: {true_nonzero_mask.sum():,} ({true_nonzero_mask.mean():.1%})
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_model_overview(model, X, y, save_path=None):
    """
    Single infographic-style overview of the two-stage model.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Two-Stage Random Forest Model Results', fontsize=20, fontweight='bold', y=0.98)
    
    # Subtitle explaining the approach
    fig.text(0.5, 0.93, 'Stage 1: Classify Zero vs Non-Zero  →  Stage 2: Predict Abundance for Non-Zero', 
             ha='center', fontsize=12, style='italic', color='gray')
    
    y_arr = y.values.ravel()
    y_pred = model.predict(X)
    X_proc = model.preprocess_features(X)
    
    # 1. Classifier Confusion Matrix (top-left)
    ax1 = fig.add_subplot(gs[0, 0:2])
    y_binary_true = (y_arr > 0).astype(int)
    y_binary_pred = model.classifier.predict(X_proc)
    cm = confusion_matrix(y_binary_true, y_binary_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax1,
                xticklabels=['Pred Zero', 'Pred Non-Zero'],
                yticklabels=['True Zero', 'True Non-Zero'],
                annot_kws={'size': 12})
    ax1.set_title('Stage 1: Classification', fontsize=12, fontweight='bold')
    
    # 2. Classifier metrics (top-right)
    ax2 = fig.add_subplot(gs[0, 2:4])
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        'Accuracy': accuracy_score(y_binary_true, y_binary_pred),
        'Precision': precision_score(y_binary_true, y_binary_pred),
        'Recall': recall_score(y_binary_true, y_binary_pred),
        'F1': f1_score(y_binary_true, y_binary_pred)
    }
    colors = sns.color_palette("deep", n_colors=4)
    bars = ax2.bar(metrics.keys(), metrics.values(), color=colors, edgecolor='white')
    for bar, v in zip(bars, metrics.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Classifier Metrics', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Regressor scatter (middle-left)
    ax3 = fig.add_subplot(gs[1, 0:2])
    nonzero_mask = y_arr > 0
    y_true_nz = y_arr[nonzero_mask]
    y_pred_nz = model.regressor.predict(X_proc[nonzero_mask])
    
    from scipy.stats import gaussian_kde
    try:
        xy = np.vstack([y_true_nz, y_pred_nz])
        density = gaussian_kde(xy)(xy)
        idx = density.argsort()
        scatter = ax3.scatter(y_true_nz[idx], y_pred_nz[idx], c=density[idx], s=10, cmap='viridis', alpha=0.6)
    except:
        ax3.scatter(y_true_nz, y_pred_nz, alpha=0.2, s=10, c=sns.color_palette("deep")[0])
    
    max_val = max(y_true_nz.max(), y_pred_nz.max())
    ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    from sklearn.metrics import r2_score as r2_score_fn
    r2 = r2_score_fn(y_true_nz, y_pred_nz)
    ax3.text(0.05, 0.95, f'R² = {r2:.2f}', transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='wheat', alpha=0.5))
    ax3.set_xlabel('Actual')
    ax3.set_ylabel('Predicted')
    ax3.set_title('Stage 2: Abundance Prediction', fontsize=12, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Feature importance (middle-right)
    ax4 = fig.add_subplot(gs[1, 2:4])
    importance = model.get_feature_importance().head(8).iloc[::-1]
    ax4.barh(importance['feature'], importance['avg_importance'], 
             color=plt.cm.viridis(np.linspace(0.3, 0.8, 8)), edgecolor='white')
    ax4.set_xlabel('Importance')
    ax4.set_title('Top Features', fontsize=12, fontweight='bold')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # 5. Distribution comparison (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0:2])
    sns.histplot(y_arr[y_arr > 0], bins=40, alpha=0.6, label='Actual', color=sns.color_palette("deep")[0], ax=ax5)
    sns.histplot(y_pred[y_pred > 0], bins=40, alpha=0.6, label='Predicted', color=sns.color_palette("deep")[3], ax=ax5)
    ax5.set_xlabel('Relative Abundance (non-zero)')
    ax5.set_ylabel('Count')
    ax5.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # 6. Summary box (bottom-right)
    ax6 = fig.add_subplot(gs[2, 2:4])
    ax6.axis('off')
    
    from sklearn.metrics import mean_absolute_error
    overall_mae = mean_absolute_error(y_arr, y_pred)
    zero_accuracy = (y_pred[y_arr == 0] == 0).mean()
    
    summary = f"""
    KEY RESULTS
    
    - F1-Score: {metrics['F1']:.1%}
    - Zero Detection: {zero_accuracy:.1%}
    - Overall MAE: {overall_mae:.5f}
    - R² (non-zero): {r2:.2f}
    
    The model correctly identifies
    {zero_accuracy:.0%} of zero-abundance cases
    and predicts non-zero values with
    R² = {r2:.2f}
    """
    
    ax6.text(0.1, 0.9, summary, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_all_visualizations(model_path: str, data_dir: str, output_dir: str):
    """Generate all visualizations and save to output directory."""
    set_style()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model and data...")
    model = TwoStageRandomForest.load(model_path)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_dir)
    
    print("\nGenerating visualizations for TEST set...\n")
    
    # Generate all plots
    print("1. Confusion Matrix")
    plot_classifier_confusion_matrix(model, X_test, y_test,
                                     save_path=os.path.join(output_dir, "1_confusion_matrix.png"))
    
    print("2. Classifier Metrics")
    plot_classifier_metrics(model, X_test, y_test,
                           save_path=os.path.join(output_dir, "2_classifier_metrics.png"))
    
    print("3. Regressor Predictions (Stage 2 only)")
    plot_regressor_predictions(model, X_test, y_test,
                              save_path=os.path.join(output_dir, "3_regressor_predictions.png"))
    
    print("4. MAE per Abundance Range")
    plot_mae_per_range(model, X_test, y_test,
                       save_path=os.path.join(output_dir, "4_mae_per_range.png"))
    
    print("5. Predictions Zoomed (<1%)")
    plot_predictions_zoomed(model, X_test, y_test,
                            save_path=os.path.join(output_dir, "5_predictions_zoomed.png"))
    
    print("6. Feature Importance (Gini)")
    plot_feature_importance(model,
                           save_path=os.path.join(output_dir, "6_feature_importance.png"))
    
    print("7. Permutation Importance")
    plot_permutation_importance(model, X_test, y_test, n_repeats=10,
                               save_path=os.path.join(output_dir, "7_permutation_importance.png"))
    
    print("8. Combined Performance")
    plot_combined_performance(model, X_test, y_test,
                             save_path=os.path.join(output_dir, "8_combined_performance.png"))
    
    print("9. Model Overview (Infographic)")
    plot_model_overview(model, X_test, y_test,
                       save_path=os.path.join(output_dir, "9_model_overview.png"))
    
    print(f"\nAll visualizations saved to: {output_dir}/")
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for Two-Stage RF Model")
    parser.add_argument("--model_path", type=str, default="models/rf_model.pkl", help="Path to saved model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="figures", help="Output directory for figures")
    
    args = parser.parse_args()
    
    generate_all_visualizations(args.model_path, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
