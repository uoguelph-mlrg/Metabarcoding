"""
Model Comparison: Baseline vs Interpolated-Latent variant.

This script trains two models using the shared trainer implementation from src/train.py:
1. Baseline model from src/ using the standard src/model.py and src/latent_solver.py
2. Interpolated-latent variant using the local model.py and latent_solver.py in this folder

Results are saved to pickle for visualization by interpolated_latent_visualisation.py.

Usage:
	python interpolated_latent.py --data_path ../../data/ecuador_training_data.csv
	python interpolated_latent.py --data_dir ../../data --no_wandb
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
import time
import logging as log
from typing import Dict, Any


# Try to import wandb, but make it optional
try:
	import wandb
	WANDB_AVAILABLE = True
except ImportError:
	WANDB_AVAILABLE = False


def load_baseline_trainer(src_path: str):
	"""Load baseline Trainer and Config from src/ with isolated imports."""
	return _load_trainer_from_src(src_path, variant_dir=None)


def _load_module(module_name: str, file_path: str):
	"""Load a Python module directly from a file path."""
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load spec for {module_name} from {file_path}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


def _load_trainer_from_src(src_path: str, variant_dir: str | None):
	"""Load src/train.py, optionally overriding model and latent_solver with local variant files."""
	local_dir = os.path.dirname(os.path.abspath(__file__))
	original_sys_path = list(sys.path)
	tracked_modules = [
		"config",
		"train",
		"model",
		"latent_solver",
		"neighbor_graph",
		"dataset",
		"loss",
		"mlp",
		"utils",
	]
	original_modules = {name: sys.modules.get(name) for name in tracked_modules}

	try:
		if local_dir in sys.path:
			sys.path.remove(local_dir)
		if src_path not in sys.path:
			sys.path.insert(0, src_path)

		for mod in tracked_modules:
			sys.modules.pop(mod, None)

		src_config = _load_module("config", os.path.join(src_path, "config.py"))

		if variant_dir is None:
			_load_module("latent_solver", os.path.join(src_path, "latent_solver.py"))
			_load_module("model", os.path.join(src_path, "model.py"))
			trainer_module = _load_module("train", os.path.join(src_path, "train.py"))
		else:
			_load_module("latent_solver", os.path.join(variant_dir, "latent_solver.py"))
			_load_module("model", os.path.join(variant_dir, "model.py"))
			trainer_module = _load_module("train", os.path.join(variant_dir, "train.py"))

		return trainer_module.Trainer, src_config.Config, src_config.set_seed
	finally:
		sys.path[:] = original_sys_path
		for name, module in original_modules.items():
			if module is None:
				sys.modules.pop(name, None)
			else:
				sys.modules[name] = module


def load_variant_trainer(local_dir: str, src_path: str):
	"""Load src Trainer using local interpolated-latent model and solver overrides."""
	return _load_trainer_from_src(src_path, variant_dir=local_dir)


def run_comparison(
	data_path: str | None,
	data_dir: str | None,
	use_wandb: bool = True,
) -> Dict[str, Any]:
	"""Run baseline vs interpolated-latent comparison and return results."""
	results: Dict[str, Any] = {}

	root_dir = os.path.dirname(os.path.abspath(__file__))
	src_path = os.path.abspath(os.path.join(root_dir, "..", "..", "..", "src"))

	# ------------------ Baseline (src) ------------------
	log.info("\n" + "=" * 70)
	log.info("TRAINING BASELINE MODEL (SRC)")
	log.info("=" * 70)

	BaselineTrainer, BaselineConfig, baseline_set_seed = load_baseline_trainer(src_path)
	baseline_set_seed(14)
	baseline_cfg = BaselineConfig()
	baseline_trainer = BaselineTrainer(
		baseline_cfg,
		data_path=data_path,
		data_dir=data_dir,
		loss_type="cross_entropy",
	)
	baseline_results = baseline_trainer.run(use_wandb=use_wandb)
	results["baseline"] = baseline_results

	# ---------------- Interpolated latent (local overrides) ----------------
	log.info("\n" + "=" * 70)
	log.info("TRAINING INTERPOLATED-LATENT VARIANT")
	log.info("=" * 70)

	LocalTrainer, LocalConfig, local_set_seed = load_variant_trainer(root_dir, src_path)
	local_set_seed(14)
	local_cfg = LocalConfig()
	local_trainer = LocalTrainer(
		local_cfg,
		data_path=data_path,
		data_dir=data_dir,
		loss_type="cross_entropy",
	)
	local_results = local_trainer.run(use_wandb=use_wandb)
	results["interpolated_latent"] = local_results

	return results


def save_results(results: Dict[str, Any], output_path: str) -> None:
	"""Save results to pickle file."""
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "wb") as f:
		pickle.dump(results, f)
	log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Model Comparison: Baseline vs Interpolated-Latent"
	)
	group = parser.add_mutually_exclusive_group(required=False)
	group.add_argument(
		"--data_path",
		type=str,
		default=None,
		help="Path to raw data CSV file (e.g. ../../data/ecuador_training_data.csv)",
	)
	group.add_argument(
		"--data_dir",
		type=str,
		default=None,
		help="Path to directory containing processed CSV files (X_*.csv, y_*.csv, taxonomic_data.csv)",
	)
	parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
	parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
	parser.add_argument(
		"--output_dir",
		type=str,
		default="results",
		help="Output directory for results pickle",
	)
	args = parser.parse_args()

	# Setup logging
	log_level = log.DEBUG if args.verbose else log.INFO
	log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

	# Decide data source
	data_path = args.data_path
	data_dir = args.data_dir
	if data_path is None and data_dir is None:
		data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "data_merged.csv"))
		log.info(f"No data_path or data_dir provided. Using default: {data_path}")

	use_wandb = WANDB_AVAILABLE and not args.no_wandb
	if use_wandb:
		wandb.init(
			project="metabarcoding-model-comparison",
			name=f"interpolated_latent_comparison_{time.strftime('%Y-%m-%d_%H-%M')}",
			reinit=True,
		)

	# Run comparison
	results = run_comparison(data_path=data_path, data_dir=data_dir, use_wandb=use_wandb)

	# Save results

	script_dir = os.path.dirname(os.path.abspath(__file__))
	output_dir = os.path.join(script_dir, args.output_dir)
	results_path = os.path.join(output_dir, "model_comparison_results.pkl")
	print(f"[INFO] Saving results to: {os.path.abspath(results_path)}")
	save_results(results, results_path)

	log.info(f"\n{'='*70}")
	log.info("COMPARISON COMPLETE")
	log.info(f"{'='*70}")
	log.info(f"Results saved to: {results_path}")
	log.info(f"Run visualization: python interpolated_latent_visualisation.py --results_path {results_path}")

	if use_wandb:
		wandb.finish()
