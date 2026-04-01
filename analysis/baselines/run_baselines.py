#!/usr/bin/env python
"""
Main training and evaluation script for baseline models.
Trains all baseline models and compares their performance.

Usage:
    python run_baselines.py --data_path data/ecuador_training_data.csv
    python run_baselines.py --data_path data/ecuador_training_data.csv --models two_stage ridge random_forest
    python run_baselines.py --data_path data/ecuador_training_data.csv --zero_inflated_only
"""
import os
import sys
import argparse
import warnings
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from preprocess import DataPreprocessor, load_data
from models import get_all_models, get_zero_inflated_models, BaselineModel, TwoStageModel
from evaluate import compute_metrics, compute_sample_level_metrics, print_metrics_table, save_results_to_csv

warnings.filterwarnings('ignore')


def train_and_evaluate_model(
    model: BaselineModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metadata: Dict,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a single model and evaluate on validation and test sets.
    
    Returns:
        Dictionary with 'val' and 'test' metric dicts
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {model.name}")
        print(f"{'='*60}")
    
    # Special handling for two-stage model
    if isinstance(model, TwoStageModel):
        presence_train = metadata['presence_train']
        model.fit(
            X_train, y_train,
            presence=presence_train
        )
    else:
        model.fit(X_train, y_train)
    
    # Predict on validation set
    val_preds = model.predict(X_val)
    val_metrics = compute_metrics(y_val.to_numpy(), val_preds)
    
    # Sample-level metrics for validation
    val_sample_metrics = compute_sample_level_metrics(
        y_val, val_preds, 
        metadata['val_meta']['sample_id']
    )
    val_metrics.update(val_sample_metrics)
    
    # Predict on test set
    test_preds = model.predict(X_test)
    test_metrics = compute_metrics(y_test.to_numpy(), test_preds)
    
    # Sample-level metrics for test
    test_sample_metrics = compute_sample_level_metrics(
        y_test, test_preds,
        metadata['test_meta']['sample_id']
    )
    test_metrics.update(test_sample_metrics)
    
    if verbose:
        print(f"  Validation - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
        print(f"  Test       - RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")
        print(f"  Zero Recall (test): {test_metrics['zero_recall']:.2%}")
    
    test_sample_labels = metadata['test_meta']['sample_id'].to_numpy()
    test_bin_labels = metadata['test_meta']['bin_uri'].to_numpy()
    unified_payload = {
        "model": model.name,
        "run_id": "baseline",
        "best_val_loss": float('nan'),
        "test_loss": float(test_metrics.get('rmse', np.nan)),
        "predictions": np.asarray(test_preds, dtype=np.float32),
        "targets": np.asarray(y_test.to_numpy(), dtype=np.float32),
        "sample_labels": np.asarray(test_sample_labels),
        "bin_labels": np.asarray(test_bin_labels),
        "latent_vector": np.array([np.nan], dtype=np.float32),
        "train_losses": [],
        "val_losses": [],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    return {
        'val': val_metrics,
        'test': test_metrics,
        'payload': unified_payload,
    }


def run_all_baselines(
    data_path: str,
    model_names: Optional[List[str]] = None,
    zero_inflated_only: bool = False,
    output_dir: str = "results",
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Run all baseline models and return comparison results.
    
    Args:
        data_path: Path to ecuador_training_data.csv
        model_names: List of model names to run (None = all)
        zero_inflated_only: Only run zero-inflated models
        output_dir: Directory to save results
        random_state: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        Dictionary of model_name -> test metrics
    """
    # Load and preprocess data
    print("\n" + "="*60)
    print("LOADING AND PREPROCESSING DATA")
    print("="*60)
    
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_data(
        data_path, random_state=random_state
    )
    
    # Get models to run
    if zero_inflated_only:
        all_models = get_zero_inflated_models()
    else:
        all_models = get_all_models()
    
    if model_names is not None:
        models = {name: all_models[name] for name in model_names if name in all_models}
        if len(models) < len(model_names):
            missing = set(model_names) - set(models.keys())
            print(f"Warning: Models not found: {missing}")
    else:
        models = all_models
    
    print(f"\nRunning {len(models)} baseline models:")
    for name in models.keys():
        print(f"  - {name}")
    
    # Train and evaluate each model
    all_results = {}
    test_results = {}
    unified_results = {}
    
    for name, model in models.items():
        try:
            results = train_and_evaluate_model(
                model,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                metadata,
                verbose=verbose
            )
            all_results[name] = results
            test_results[name] = results['test']
            unified_results[name] = results['payload']
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    # =============================
    # Add Latent+MLP (Cross-Entropy)
    # =============================
    try:
        import sys
        import importlib
        sys.path.insert(0, '../../src')
        config_mod = importlib.import_module('config')
        train_mod = importlib.import_module('train')
        set_seed = getattr(config_mod, 'set_seed')
        Config = getattr(config_mod, 'Config')
        Trainer = getattr(train_mod, 'Trainer')
        set_seed(14)
        cfg = Config()
        # Use same data_path as baselines
        latent_trainer = Trainer(cfg, data_path=data_path)
        latent_results = latent_trainer.run(use_wandb=False)
        # Compute metrics using baselines' compute_metrics
        from evaluate import compute_metrics, compute_sample_level_metrics
        y_pred = latent_results["predictions"]
        y_true = latent_results["targets"]
        val_metrics = compute_metrics(y_true, y_pred)  # Use test split for fair comparison
        # No sample-level metrics for now (unless needed)
        test_results["latent_mlp"] = val_metrics
        all_results["latent_mlp"] = {"val": val_metrics, "test": val_metrics}
        unified_results["latent_mlp"] = latent_results
        print("\nLatent+MLP (cross-entropy) model evaluated and added to results.")
    except Exception as e:
        print(f"Error training latent+MLP model: {e}")

    # Print comparison table
    print_metrics_table(test_results)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save test results
    test_filepath = os.path.join(output_dir, f"baseline_results_test_{timestamp}.csv")
    save_results_to_csv(test_results, test_filepath)

    # Save unified model payloads for centralized visualization.
    unified_filepath = os.path.join(output_dir, f"baseline_model_comparison_results_{timestamp}.pkl")
    with open(unified_filepath, "wb") as f:
        pickle.dump(unified_results, f)
    print(f"Unified visualization payload saved to: {unified_filepath}")
    
    # Save validation results
    val_results = {name: res['val'] for name, res in all_results.items()}
    val_filepath = os.path.join(output_dir, f"baseline_results_val_{timestamp}.csv")
    save_results_to_csv(val_results, val_filepath)
    
    # Create summary report
    create_summary_report(test_results, output_dir, timestamp)
    
    return test_results


def create_summary_report(results: Dict[str, Dict[str, float]], 
                          output_dir: str, timestamp: str) -> None:
    """Create a text summary report of model performance."""
    
    report_path = os.path.join(output_dir, f"baseline_summary_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE MODELS SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Sort by RMSE
        sorted_models = sorted(results.items(), key=lambda x: x[1].get('rmse', float('inf')))
        
        f.write("RANKING BY RMSE (lower is better):\n")
        f.write("-"*60 + "\n")
        for rank, (name, metrics) in enumerate(sorted_models, 1):
            f.write(f"{rank:2d}. {name:30s} RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}\n")
        
        f.write("\n\nDETAILED METRICS FOR TOP 5 MODELS:\n")
        f.write("-"*60 + "\n")
        
        for name, metrics in sorted_models[:5]:
            f.write(f"\n{name}:\n")
            f.write(f"  Standard Metrics:\n")
            f.write(f"    RMSE:        {metrics['rmse']:.4f}\n")
            f.write(f"    MAE:         {metrics['mae']:.4f}\n")
            f.write(f"    R²:          {metrics['r2']:.4f}\n")
            f.write(f"  Zero-Inflation Metrics:\n")
            f.write(f"    MSE (zeros):    {metrics['mse_zeros']:.6f}\n")
            f.write(f"    MSE (non-zeros):{metrics['mse_nonzeros']:.6f}\n")
            f.write(f"    Zero Recall:    {metrics['zero_recall']:.2%}\n")
            f.write(f"  Sample-Level Metrics:\n")
            f.write(f"    Sample R² (mean):   {metrics.get('sample_r2_mean', 0):.4f} ± {metrics.get('sample_r2_std', 0):.4f}\n")
            f.write(f"    Sample RMSE (mean): {metrics.get('sample_rmse_mean', 0):.4f} ± {metrics.get('sample_rmse_std', 0):.4f}\n")
            f.write(f"    Sample MAE (mean):  {metrics.get('sample_mae_mean', 0):.4f} ± {metrics.get('sample_mae_std', 0):.4f}\n")
        
        f.write("\n\nZERO-INFLATION ANALYSIS:\n")
        f.write("-"*60 + "\n")
        # Find best model for zero handling
        best_zero = min(results.items(), key=lambda x: x[1].get('mse_zeros', float('inf')))
        f.write(f"Best for predicting zeros: {best_zero[0]} (MSE_zeros={best_zero[1]['mse_zeros']:.6f})\n")
        
        best_nonzero = min(results.items(), key=lambda x: x[1].get('mse_nonzeros', float('inf')))
        f.write(f"Best for non-zeros: {best_nonzero[0]} (MSE_nonzeros={best_nonzero[1]['mse_nonzeros']:.6f})\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline models for metabarcoding abundance prediction"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="../../data/ecuador_training_data.csv",
        help="Path to the ecuador_training_data.csv file"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of model names to run (default: all models)"
    )
    parser.add_argument(
        "--zero_inflated_only",
        action="store_true",
        help="Only run zero-inflated models"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        print("Please provide the correct path to ecuador_training_data.csv")
        sys.exit(1)
    
    # Run baselines
    results = run_all_baselines(
        data_path=args.data_path,
        model_names=args.models,
        zero_inflated_only=args.zero_inflated_only,
        output_dir=args.output_dir,
        random_state=args.seed,
        verbose=not args.quiet
    )
    
    # Print best model
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    print(f"\n🏆 Best model by RMSE: {best_model[0]} (RMSE={best_model[1]['rmse']:.4f})")
    
    return results


if __name__ == "__main__":
    main()
