"""
Evaluate a trained Random Forest model and generate detailed analysis.
"""
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import RandomForestConfig


def load_model(model_path: str):
    """Load a saved model from pickle file."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['feature_names'], data.get('config')


def load_data(data_dir: str):
    """Load test data."""
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))
    return X_test, y_test


def preprocess_features(X: pd.DataFrame, feature_names: list) -> np.ndarray:
    """Extract only the features used during training."""
    return X[feature_names].values


def evaluate_model(model, X: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    # Correlation coefficient
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Spearman correlation (rank-based)
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_corr': corr,
        'spearman_corr': spearman_corr,
        'y_pred': y_pred,
        'y_true': y_true
    }


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """Create prediction vs actual plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.1, s=1)
    ax1.plot([0, max(y_true.max(), y_pred.max())], 
             [0, max(y_true.max(), y_pred.max())], 
             'r--', label='Perfect prediction')
    ax1.set_xlabel('Actual Relative Abundance')
    ax1.set_ylabel('Predicted Relative Abundance')
    ax1.set_title('Predictions vs Actual')
    ax1.legend()
    
    # Residuals
    ax2 = axes[1]
    residuals = y_pred - y_true
    ax2.scatter(y_true, residuals, alpha=0.1, s=1)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Actual Relative Abundance')
    ax2.set_ylabel('Residual (Predicted - Actual)')
    ax2.set_title('Residuals vs Actual')
    
    # Histogram of residuals
    ax3 = axes[2]
    ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--')
    ax3.set_xlabel('Residual')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15, save_path: str = None):
    """Plot feature importance."""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance['feature'], importance['importance'])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return importance


def evaluate_by_abundance_bins(y_true: np.ndarray, y_pred: np.ndarray):
    """Evaluate model performance across different abundance ranges."""
    # Create abundance bins
    bins = [0, 0.001, 0.01, 0.1, 1.0]
    bin_labels = ['0-0.001', '0.001-0.01', '0.01-0.1', '0.1-1.0']
    
    results = []
    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if mask.sum() > 0:
            y_t = y_true[mask]
            y_p = y_pred[mask]
            results.append({
                'bin': bin_labels[i],
                'n_samples': mask.sum(),
                'mae': mean_absolute_error(y_t, y_p),
                'rmse': np.sqrt(mean_squared_error(y_t, y_p)),
                'mean_actual': y_t.mean(),
                'mean_predicted': y_p.mean()
            })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Random Forest model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for plots")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, feature_names, config = load_model(args.model_path)
    print(f"  Loaded model with {len(feature_names)} features")
    
    # Load data
    print("\nLoading test data...")
    X_test, y_test = load_data(args.data_dir)
    print(f"  Test size: {len(X_test)}")
    
    # Preprocess
    X_test_proc = preprocess_features(X_test, feature_names)
    y_test_arr = y_test.values.ravel()
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, X_test_proc, y_test_arr)
    
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"MSE:              {results['mse']:.6f}")
    print(f"RMSE:             {results['rmse']:.6f}")
    print(f"MAE:              {results['mae']:.6f}")
    print(f"R²:               {results['r2']:.6f}")
    print(f"Pearson Corr:     {results['pearson_corr']:.6f}")
    print(f"Spearman Corr:    {results['spearman_corr']:.6f}")
    
    # Performance by abundance bins
    print("\n" + "="*50)
    print("PERFORMANCE BY ABUNDANCE RANGE")
    print("="*50)
    bin_results = evaluate_by_abundance_bins(results['y_true'], results['y_pred'])
    print(bin_results.to_string(index=False))
    
    # Generate plots
    if not args.no_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        
        print("\nGenerating plots...")
        plot_predictions(
            results['y_true'], 
            results['y_pred'],
            save_path=os.path.join(args.output_dir, "predictions.png")
        )
        
        plot_feature_importance(
            model, 
            feature_names,
            save_path=os.path.join(args.output_dir, "feature_importance.png")
        )
    
    return results


if __name__ == "__main__":
    main()
