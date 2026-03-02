"""
Evaluation metrics and utilities for baseline models.
Includes metrics specifically suited for zero-inflated data.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metric name -> value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Ensure predictions are clipped to valid range
    y_pred = np.clip(y_pred, 0, 1)
    
    # Standard regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # R² score (can be negative for poor models)
    r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    
    # Relative metrics
    mape = _mean_absolute_percentage_error(y_true, y_pred)
    
    # Zero-specific metrics
    zero_mask = y_true == 0
    nonzero_mask = y_true > 0
    
    # MSE on zeros only
    mse_zeros = mean_squared_error(y_true[zero_mask], y_pred[zero_mask]) if zero_mask.sum() > 0 else 0.0
    
    # MSE on non-zeros only  
    mse_nonzeros = mean_squared_error(y_true[nonzero_mask], y_pred[nonzero_mask]) if nonzero_mask.sum() > 0 else 0.0
    
    # MAE on non-zeros only
    mae_nonzeros = mean_absolute_error(y_true[nonzero_mask], y_pred[nonzero_mask]) if nonzero_mask.sum() > 0 else 0.0
    
    # Zero prediction accuracy (how well we predict zeros)
    pred_zero_mask = y_pred < 1e-6
    zero_recall = (zero_mask & pred_zero_mask).sum() / zero_mask.sum() if zero_mask.sum() > 0 else 0.0
    zero_precision = (zero_mask & pred_zero_mask).sum() / pred_zero_mask.sum() if pred_zero_mask.sum() > 0 else 0.0
    
    # Correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    if np.isnan(correlation):
        correlation = 0.0
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "mse_zeros": mse_zeros,
        "mse_nonzeros": mse_nonzeros,
        "mae_nonzeros": mae_nonzeros,
        "zero_recall": zero_recall,
        "zero_precision": zero_precision,
        "correlation": correlation,
        "n_zeros": int(zero_mask.sum()),
        "n_nonzeros": int(nonzero_mask.sum()),
        "frac_zeros": zero_mask.sum() / len(y_true),
    }


def _mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, 
                                     epsilon: float = 1e-10) -> float:
    """Compute MAPE, handling zeros in y_true."""
    mask = y_true > epsilon
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100


def compute_sample_level_metrics(
    y_true: pd.Series, 
    y_pred: np.ndarray, 
    sample_ids: pd.Series
) -> Dict[str, float]:
    """
    Compute metrics aggregated at sample level.
    
    For each sample, compare the predicted vs true distributions.
    """
    df = pd.DataFrame({
        'sample_id': sample_ids.values,
        'y_true': y_true.values,
        'y_pred': y_pred
    })
    
    sample_metrics = []
    for sample_id, group in df.groupby('sample_id'):
        if len(group) > 1:
            sample_r2 = r2_score(group['y_true'], group['y_pred'])
            sample_rmse = np.sqrt(mean_squared_error(group['y_true'], group['y_pred']))
            sample_mae = mean_absolute_error(group['y_true'], group['y_pred'])

            sample_metrics.append({
                'sample_id': sample_id,
                'r2': sample_r2,
                'rmse': sample_rmse,
                'mae': sample_mae,
            })
    
    if not sample_metrics:
        return {
            "sample_r2_mean": 0.0,
            "sample_r2_std": 0.0,
            "sample_rmse_mean": 0.0,
            "sample_rmse_std": 0.0,
            "sample_mae_mean": 0.0,
            "sample_mae_std": 0.0,
        }
    
    metrics_df = pd.DataFrame(sample_metrics)
    
    return {
        "sample_r2_mean": metrics_df['r2'].mean(),
        "sample_r2_std": metrics_df['r2'].std(),
        "sample_rmse_mean": metrics_df['rmse'].mean(),
        "sample_rmse_std": metrics_df['rmse'].std(),
        "sample_mae_mean": metrics_df['mae'].mean(),
        "sample_mae_std": metrics_df['mae'].std(),
    }


def print_metrics_table(results: Dict[str, Dict[str, float]], 
                        metrics_to_show: list = None) -> None:
    """
    Print a formatted table of results for multiple models.
    
    Args:
        results: Dict of model_name -> metrics dict
        metrics_to_show: List of metric names to display
    """
    if metrics_to_show is None:
        metrics_to_show = ['rmse', 'mae', 'r2', 'mse_zeros', 'mse_nonzeros', 
                          'zero_recall']
    
    # Build dataframe
    rows = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        for m in metrics_to_show:
            if m in metrics:
                row[m] = metrics[m]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index('Model')
    
    # Sort by RMSE
    if 'rmse' in df.columns:
        df = df.sort_values('rmse')
    
    # Format for display
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n" + "="*80)
    print("BASELINE MODEL RESULTS")
    print("="*80)
    print(df.to_string())
    print("="*80 + "\n")


def save_results_to_csv(results: Dict[str, Dict[str, float]], 
                        filepath: str) -> None:
    """Save results to CSV file."""
    rows = []
    for model_name, metrics in results.items():
        row = {'model': model_name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")
