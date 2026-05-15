#!/usr/bin/env python
"""
Train and evaluate all baseline models, then save a unified pickle for visualize_results.py.

Usage:
    python run_baselines.py
    python run_baselines.py --data_path ../../data/metabarcoding_dataset.csv
    python run_baselines.py --models two_stage ridge random_forest
    python run_baselines.py --output_dir results
"""
import os
import sys
import argparse
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Make src/ importable
_SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import Config, set_seed
from metrics import compute_metrics
from utils import load

from models import get_all_models, TwoStageModel

warnings.filterwarnings("ignore")


def _flat_split(data: Dict[str, Any], split: str) -> tuple:
    """Return (X, y_series, sample_ids, bin_uris) for a given split."""
    X = data[split]["X"]                   # DataFrame indexed by (sample_id, bin_uri)
    y = data[split]["y_prob"]              # Series of rel_abundance, same MultiIndex
    idx = X.index.to_frame(index=False)   # columns: sample_id, bin_uri
    # Reset to RangeIndex so boolean masks from y align correctly with X in all models
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y, idx["sample_id"].to_numpy(), idx["bin_uri"].to_numpy()


def train_and_evaluate(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_sample_labels: np.ndarray,
    test_bin_labels: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit a model and return its unified payload dict."""
    if verbose:
        print(f"  {model.name} ...", end=" ", flush=True)

    if isinstance(model, TwoStageModel):
        # presence can be derived from y — TwoStageModel.fit() already handles None
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    test_preds = np.asarray(model.predict(X_test), dtype=np.float32)
    y_true = np.asarray(y_test.to_numpy(), dtype=np.float32)

    metrics = compute_metrics(test_preds, y_true, sample_labels=test_sample_labels)

    if verbose:
        print(
            f"MAE(micro)={metrics['MAE (micro)']:.4f}  "
            f"KL={metrics['KL Divergence']:.4f}  "
            f"R²(Shannon)={metrics['R² (Shannon diversity)']:.3f}"
        )

    return {
        "predictions": test_preds,
        "targets": y_true,
        "sample_labels": test_sample_labels,
        "bin_labels": test_bin_labels,
        "test_metrics": metrics,
        "train_losses": [],
        "val_losses": [],
    }


def run_all_baselines(
    data_path: str,
    model_names: Optional[List[str]] = None,
    output_dir: str = "results",
    random_state: int = 14,
    verbose: bool = True,
) -> str:
    """
    Train all baseline models and save a unified pkl for visualize_results.py.

    Returns the path to the saved pkl file.
    """
    set_seed(random_state)

    # --- Load data via src/utils.py ---
    cfg = Config(data_path=os.path.abspath(data_path))
    cfg.use_embedding = False   # baselines don't use DNA embeddings
    cfg.use_taxonomy = False    # taxonomy is label-encoded inside models.py directly

    print(f"Loading data from: {cfg.data_path}")
    data, _taxonomy_df, _emb, _bwe, _bin_idx, _sample_idx, split_indices, _state = load(cfg)

    X_train, y_train, train_sample_ids, _ = _flat_split(data, "train")
    X_test,  y_test,  test_sample_ids,  test_bin_ids  = _flat_split(data, "test")

    print(
        f"  train={len(X_train):,} obs  "
        f"test={len(X_test):,} obs  "
        f"zero%={100*(y_test == 0).mean():.1f}%"
    )

    # --- Select models ---
    all_models = get_all_models()
    if model_names:
        missing = set(model_names) - set(all_models)
        if missing:
            print(f"Warning: unknown model keys: {missing}")
        models = {k: all_models[k] for k in model_names if k in all_models}
    else:
        models = all_models

    print(f"\nRunning {len(models)} baseline models:")

    # --- Train & evaluate ---
    unified: Dict[str, Dict] = {}
    for name, model in models.items():
        try:
            payload = train_and_evaluate(
                model,
                X_train, y_train,
                X_test,  y_test,
                test_sample_ids, test_bin_ids,
                verbose=verbose,
            )
            unified[name] = payload
        except Exception as exc:
            import traceback
            print(f"\n  ERROR in {name}: {exc}")
            traceback.print_exc()

    # --- Save unified pkl ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    pkl_path = os.path.join(out_dir, f"baselines_{timestamp}.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(unified, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved {len(unified)} model payloads → {pkl_path}")
    return pkl_path


def run_one_variant(
    variant_name: str,
    data_path: str,
    output_dir: str,
    run_id: str,
    random_state: int = 14,
    verbose: bool = True,
) -> None:
    """Train a single named baseline model and save its per-variant pickle."""
    all_models = get_all_models()
    if variant_name not in all_models:
        print(f"Error: unknown model key '{variant_name}'. Valid: {sorted(all_models)}")
        sys.exit(1)

    set_seed(random_state)

    cfg = Config(data_path=os.path.abspath(data_path))
    cfg.use_embedding = False
    cfg.use_taxonomy = False

    print(f"Loading data from: {cfg.data_path}")
    data, _taxonomy_df, _emb, _bwe, _bin_idx, _sample_idx, _split_indices, _state = load(cfg)

    X_train, y_train, _, _ = _flat_split(data, "train")
    X_test,  y_test,  test_sample_ids, test_bin_ids = _flat_split(data, "test")

    model = all_models[variant_name]
    payload = train_and_evaluate(
        model,
        X_train, y_train,
        X_test,  y_test,
        test_sample_ids, test_bin_ids,
        verbose=verbose,
    )

    analysis_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(analysis_root, output_dir, "baselines_comparison", run_id)
    os.makedirs(out_dir, exist_ok=True)

    pkl_path = os.path.join(out_dir, f"baselines_comparison_{variant_name}.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump({variant_name: payload}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {variant_name} → {pkl_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models for metabarcoding abundance prediction"
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "metabarcoding_dataset.csv"
        ),
        help="Path to data CSV (default: data/metabarcoding_dataset.csv)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model keys to run (default: all).",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Output directory base (relative to analysis/)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-model output")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Ignored (sklearn models have no epoch count; accepted for CLI consistency)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Ignored (baselines don't use wandb; accepted for CLI consistency)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Train only this single model key (used by SLURM per-variant dispatch)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Shared run ID for output directory (default: current timestamp)")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: data file not found: {args.data_path}")
        sys.exit(1)

    if args.variant:
        run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_one_variant(
            variant_name=args.variant,
            data_path=args.data_path,
            output_dir=args.output_dir,
            run_id=run_id,
            verbose=not args.quiet,
        )
    else:
        run_all_baselines(
            data_path=args.data_path,
            model_names=args.models,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
