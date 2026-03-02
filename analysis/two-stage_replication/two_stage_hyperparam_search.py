"""
Hyperparameter search for Two-Stage Random Forest Model.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, mean_squared_error
import argparse
from datetime import datetime
from itertools import product

from config import set_seed
from two_stage_model import TwoStageConfig, TwoStageRandomForest, load_data


def search_classifier_params(X_train: np.ndarray, y_binary: np.ndarray, 
                             random_state: int = 42, n_jobs: int = -1):
    """
    Search for best classifier hyperparameters.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    results = []
    total = np.prod([len(v) for v in param_grid.values()])
    
    print(f"Searching {total} classifier configurations...")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    
    for i, (n_est, max_d, min_split, cw) in enumerate(product(*param_grid.values())):
        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_split,
            class_weight=cw,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        scores = cross_val_score(clf, X_train, y_binary, cv=cv, scoring='f1')
        
        results.append({
            'n_estimators': n_est,
            'max_depth': max_d,
            'min_samples_split': min_split,
            'class_weight': cw,
            'mean_f1': scores.mean(),
            'std_f1': scores.std()
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{total}")
    
    results_df = pd.DataFrame(results).sort_values('mean_f1', ascending=False)
    return results_df


def search_regressor_params(X_train: np.ndarray, y_train: np.ndarray,
                            random_state: int = 42, n_jobs: int = -1):
    """
    Search for best regressor hyperparameters (on non-zero samples).
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    results = []
    total = np.prod([len(v) for v in param_grid.values()])
    
    print(f"Searching {total} regressor configurations...")
    
    def neg_rmse(y_true, y_pred):
        return -np.sqrt(mean_squared_error(y_true, y_pred))
    
    scorer = make_scorer(neg_rmse)
    
    for i, (n_est, max_d, min_split, min_leaf) in enumerate(product(*param_grid.values())):
        reg = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        scores = cross_val_score(reg, X_train, y_train, cv=3, scoring=scorer)
        
        results.append({
            'n_estimators': n_est,
            'max_depth': max_d,
            'min_samples_split': min_split,
            'min_samples_leaf': min_leaf,
            'mean_rmse': -scores.mean(),
            'std_rmse': scores.std()
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{total}")
    
    results_df = pd.DataFrame(results).sort_values('mean_rmse', ascending=True)
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for Two-Stage RF")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--stage", type=str, choices=['classifier', 'regressor', 'both'],
                        default='both', help="Which stage to search")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load data
    print("Loading data...")
    config = TwoStageConfig(data_dir=args.data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(config.data_dir)
    
    # Combine train and val
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    
    # Preprocess
    model = TwoStageRandomForest(config)
    X_trainval_proc = model.preprocess_features(X_trainval)
    y_trainval_arr = y_trainval.values.ravel()
    y_binary = (y_trainval_arr > 0).astype(int)
    
    # Non-zero subset for regressor
    nonzero_mask = y_trainval_arr > 0
    X_nonzero = X_trainval_proc[nonzero_mask]
    y_nonzero = y_trainval_arr[nonzero_mask]
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Search classifier params
    if args.stage in ['classifier', 'both']:
        print("\n" + "="*60)
        print("CLASSIFIER HYPERPARAMETER SEARCH")
        print("="*60)
        clf_results = search_classifier_params(X_trainval_proc, y_binary, args.seed)
        
        print("\nTop 5 Classifier Configurations:")
        print(clf_results.head(5).to_string(index=False))
        
        clf_path = os.path.join(args.output_dir, f"clf_hyperparam_{timestamp}.csv")
        clf_results.to_csv(clf_path, index=False)
        print(f"\nResults saved to: {clf_path}")
    
    # Search regressor params
    if args.stage in ['regressor', 'both']:
        print("\n" + "="*60)
        print("REGRESSOR HYPERPARAMETER SEARCH")
        print("="*60)
        reg_results = search_regressor_params(X_nonzero, y_nonzero, args.seed)
        
        print("\nTop 5 Regressor Configurations:")
        print(reg_results.head(5).to_string(index=False))
        
        reg_path = os.path.join(args.output_dir, f"reg_hyperparam_{timestamp}.csv")
        reg_results.to_csv(reg_path, index=False)
        print(f"\nResults saved to: {reg_path}")


if __name__ == "__main__":
    main()
