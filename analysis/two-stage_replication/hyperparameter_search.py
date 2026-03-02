"""
Hyperparameter search for Random Forest Regressor using cross-validation.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import argparse
from datetime import datetime

from config import RandomForestConfig, set_seed
from train import load_data, preprocess_features


def rmse_scorer(y_true, y_pred):
    """RMSE scorer for cross-validation."""
    return -np.sqrt(mean_squared_error(y_true, y_pred))


def grid_search(X_train: np.ndarray, y_train: np.ndarray, config: RandomForestConfig):
    """
    Perform grid search over hyperparameters.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    
    rf = RandomForestRegressor(
        n_jobs=config.n_jobs,
        random_state=config.random_state
    )
    
    scorer = make_scorer(rmse_scorer, greater_is_better=True)
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        scoring=scorer,
        cv=3,
        n_jobs=1,  # Already parallelizing RF internally
        verbose=2,
        return_train_score=True
    )
    
    print("Starting Grid Search...")
    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    grid_search.fit(X_train, y_train)
    
    return grid_search


def randomized_search(X_train: np.ndarray, y_train: np.ndarray, config: RandomForestConfig, n_iter: int = 50):
    """
    Perform randomized search over hyperparameters.
    """
    param_distributions = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [5, 10, 15, 20, 30, 50, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]
    }
    
    rf = RandomForestRegressor(
        n_jobs=config.n_jobs,
        random_state=config.random_state
    )
    
    scorer = make_scorer(rmse_scorer, greater_is_better=True)
    
    random_search = RandomizedSearchCV(
        rf,
        param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=3,
        n_jobs=1,
        verbose=2,
        random_state=config.random_state,
        return_train_score=True
    )
    
    print(f"Starting Randomized Search with {n_iter} iterations...")
    
    random_search.fit(X_train, y_train)
    
    return random_search


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for Random Forest")
    parser.add_argument("--method", type=str, choices=['grid', 'random'], default='random',
                        help="Search method (grid or random)")
    parser.add_argument("--n_iter", type=int, default=30, help="Number of iterations for random search")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    config = RandomForestConfig(data_dir=args.data_dir, random_state=args.seed)
    set_seed(args.seed)
    
    # Load and preprocess data
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(config)
    
    # Combine train and val for cross-validation
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    
    X_trainval_proc, feature_names = preprocess_features(X_trainval, config)
    y_trainval_arr = y_trainval.values.ravel()
    
    print(f"Training data shape: {X_trainval_proc.shape}")
    
    # Run search
    if args.method == 'grid':
        search = grid_search(X_trainval_proc, y_trainval_arr, config)
    else:
        search = randomized_search(X_trainval_proc, y_trainval_arr, config, n_iter=args.n_iter)
    
    # Print results
    print("\n" + "="*60)
    print("SEARCH RESULTS")
    print("="*60)
    print(f"\nBest parameters: {search.best_params_}")
    print(f"Best RMSE: {-search.best_score_:.6f}")
    
    # Show top 10 results
    results_df = pd.DataFrame(search.cv_results_)
    results_df['rmse'] = -results_df['mean_test_score']
    results_df = results_df.sort_values('rmse')
    
    print("\nTop 10 configurations:")
    cols_to_show = ['params', 'rmse', 'std_test_score', 'mean_train_score']
    print(results_df[cols_to_show].head(10).to_string())
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    results_path = os.path.join(args.output_dir, f"hyperparam_search_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nFull results saved to: {results_path}")
    
    # Save best params
    best_params_path = os.path.join(args.output_dir, f"best_params_{timestamp}.txt")
    with open(best_params_path, 'w') as f:
        f.write(f"Best RMSE: {-search.best_score_:.6f}\n\n")
        f.write("Best parameters:\n")
        for k, v in search.best_params_.items():
            f.write(f"  {k}: {v}\n")
    print(f"Best parameters saved to: {best_params_path}")
    
    return search


if __name__ == "__main__":
    main()
