from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _shannon_diversity(values: np.ndarray, eps: float = 1e-10) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.nan
    probs = (arr + eps) / np.sum(arr + eps)
    return float(-np.sum(probs * np.log(probs + eps)))


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    rank_x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    rank_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        return None
    rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return rho if np.isfinite(rho) else None


def _r2_and_intercept(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10):
    if len(y_true) < 2:
        return np.nan, np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + eps))
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    intercept = float(intercept) if np.isfinite(intercept) else np.nan
    return r2, intercept


def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sample_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive prediction metrics (micro and macro-averaged).

    Micro metrics pool all observations; macro metrics average per sample to
    avoid domination by samples with many observed BINs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = np.clip(y_pred[valid], 0, 1)
    eps = 1e-10

    rmse_macro = np.nan
    mae_macro = np.nan
    kl_divergence = np.nan
    shannon_r2 = np.nan
    shannon_intercept = np.nan
    spearman_macro = np.nan

    if sample_labels is not None:
        sample_labels_v = np.asarray(sample_labels)[valid]
        rmse_per, mae_per, kl_per = [], [], []
        shannon_true_per, shannon_pred_per = [], []
        spearman_per = []
        for sample in np.unique(sample_labels_v):
            mask = sample_labels_v == sample
            true_s = y_true[mask]
            pred_s = y_pred[mask]
            if len(true_s) == 0:
                continue
            rmse_per.append(float(np.sqrt(np.mean((true_s - pred_s) ** 2))))
            mae_per.append(float(np.mean(np.abs(true_s - pred_s))))
            true_s_norm = (true_s + eps) / (true_s + eps).sum()
            pred_s_norm = (pred_s + eps) / (pred_s + eps).sum()
            kl_per.append(float(np.sum(true_s_norm * np.log(true_s_norm / pred_s_norm))))

            s_true = _shannon_diversity(true_s, eps)
            s_pred = _shannon_diversity(pred_s, eps)
            if np.isfinite(s_true) and np.isfinite(s_pred):
                shannon_true_per.append(s_true)
                shannon_pred_per.append(s_pred)

            if len(true_s) > 1:
                rho = _spearman_rho(true_s, pred_s)
                if rho is not None:
                    spearman_per.append(rho)

        if rmse_per:
            rmse_macro = float(np.mean(rmse_per))
            mae_macro = float(np.mean(mae_per))
            kl_divergence = float(np.mean(kl_per))

        if len(shannon_true_per) > 1:
            shannon_r2, shannon_intercept = _r2_and_intercept(
                np.array(shannon_true_per), np.array(shannon_pred_per), eps
            )

        if spearman_per:
            spearman_macro = float(np.mean(spearman_per))

    rmse_micro = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae_micro = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + eps))

    y_true_log = np.log(y_true + 1)
    y_pred_log = np.log(y_pred + 1)
    ss_res_log = np.sum((y_true_log - y_pred_log) ** 2)
    ss_tot_log = np.sum((y_true_log - np.mean(y_true_log)) ** 2)
    r2_log = float(1 - ss_res_log / (ss_tot_log + eps))

    zero_mask = y_true == 0
    nonzero_mask = y_true > 0
    rmse_zeros = float(np.sqrt(np.mean((y_true[zero_mask] - y_pred[zero_mask]) ** 2))) if zero_mask.sum() > 0 else np.nan
    mae_zeros = float(np.mean(np.abs(y_true[zero_mask] - y_pred[zero_mask]))) if zero_mask.sum() > 0 else np.nan
    rmse_nonzeros = float(np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask]) ** 2))) if nonzero_mask.sum() > 0 else np.nan
    mae_nonzeros = float(np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]))) if nonzero_mask.sum() > 0 else np.nan

    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
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
        "n_zeros": float(zero_mask.sum()),
        "n_nonzeros": float(nonzero_mask.sum()),
    }


def metric_key(metric_name: str) -> str:
    """Normalize a metric name to a valid wandb/logging key."""
    return (
        metric_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("²", "2")
        .replace("+", "plus")
        .replace("-", "_")
        .replace("/", "_")
    )
