"""
Interpolated-Latent Variant Runner.

This script trains only the interpolated-latent variant
(no baseline retraining in this analysis script).
Results are saved as one variant pickle for later comparison.

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
import logging as log
from typing import Dict, Any

sys.path.insert(0, '../..')
from variant_helpers import (
	make_output_dir,
	make_run_group,
	save_variant_result,
	variant_wandb_run,
)


# Try to import wandb, but make it optional
try:
	import wandb
	WANDB_AVAILABLE = True
except ImportError:
	wandb = None
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
		else:
			_load_module("latent_solver", os.path.join(variant_dir, "latent_solver.py"))
			_load_module("model", os.path.join(variant_dir, "model.py"))

		src_train = _load_module("train", os.path.join(src_path, "train.py"))
		return src_train.Trainer, src_config.Config, src_config.set_seed
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
	run_group: str | None = None,
) -> Dict[str, Any]:
	"""Train interpolated-latent variant and return result dict keyed by variant name."""
	results: Dict[str, Any] = {}

	root_dir = os.path.dirname(os.path.abspath(__file__))
	src_path = os.path.abspath(os.path.join(root_dir, "..", "..", "..", "src"))

	# ---------------- Interpolated latent (local overrides) ----------------
	log.info("\n" + "=" * 70)
	log.info("TRAINING INTERPOLATED-LATENT VARIANT")
	log.info("=" * 70)

	LocalTrainer, LocalConfig, local_set_seed = load_variant_trainer(root_dir, src_path)
	local_set_seed(14)
	local_cfg = LocalConfig()
	with variant_wandb_run(
		use_wandb=use_wandb,
		wandb_module=wandb,
		project="metabarcoding-model-comparison",
		analysis_name="interpolated_latent_v1",
		variant_name="interpolated_latent",
		run_group=run_group,
		tags=["interpolated_latent_v1", "variant_only"],
	):
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
		data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "ecuador_training_data.csv"))
		log.info(f"No data_path or data_dir provided. Using default: {data_path}")

	use_wandb = WANDB_AVAILABLE and not args.no_wandb
	run_group = make_run_group("interpolated_latent_v1_comparison")

	# Run comparison
	results = run_comparison(data_path=data_path, data_dir=data_dir, use_wandb=use_wandb, run_group=run_group)

	# Save results

	output_dir = make_output_dir(__file__, args.output_dir)
	results_path = save_variant_result(output_dir, "interpolated_latent_v1", "interpolated_latent", results["interpolated_latent"])
	print(f"[INFO] Saving results to: {os.path.abspath(results_path)}")

	log.info(f"\n{'='*70}")
	log.info("VARIANT TRAINING COMPLETE")
	log.info(f"{'='*70}")
	log.info(f"Results saved to: {results_path}")
	log.info(f"Run visualization: python ../visualize_results.py --results_path {results_path}")
