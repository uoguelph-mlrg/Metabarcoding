"""
Model Comparison: Baseline (additive latent) vs Latent-as-Input variant

This script trains two models using the existing Trainer classes:
1. Baseline model from src/ (latent added to output)
2. Latent-as-input variant from this folder

Results are saved to pickle for visualization by latent_as_input_visualisation.py

Usage:
	python latent_as_input.py --data_path ../../data/ecuador_training_data.csv
	python latent_as_input.py --data_dir ../../data --no_wandb
"""
from __future__ import annotations

import argparse
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
	# Remove local latent_as_input from sys.path if present to avoid shadowing src
	local_dir = os.path.dirname(os.path.abspath(__file__))
	if local_dir in sys.path:
		sys.path.remove(local_dir)
	# Insert src_path at the front
	if src_path not in sys.path:
		sys.path.insert(0, src_path)
	# Remove cached modules to force reload from src
	for mod in ["config", "train", "model", "latent_solver"]:
		if mod in sys.modules:
			del sys.modules[mod]
	import train as src_train
	import config as src_config
	return src_train.Trainer, src_config.Config, src_config.set_seed


def load_variant_trainer(local_dir: str, src_path: str):
	"""Load latent-as-input Trainer and Config from local folder with clean imports."""
	# Ensure local dir takes precedence over src for config.py
	if local_dir not in sys.path:
		sys.path.insert(0, local_dir)
	if src_path not in sys.path:
		sys.path.append(src_path)

	# Clear cached modules that could shadow local config
	for mod in ["config", "train", "model", "latent_solver"]:
		if mod in sys.modules:
			del sys.modules[mod]

	import train as local_train
	import config as local_config
	return local_train.Trainer, local_config.Config, local_config.set_seed


def run_comparison(
	data_path: str | None,
	data_dir: str | None,
	use_wandb: bool = True,
) -> Dict[str, Any]:
	"""Run baseline vs latent-as-input comparison and return results."""
	results: Dict[str, Any] = {}

	root_dir = os.path.dirname(os.path.abspath(__file__))
	src_path = os.path.abspath(os.path.join(root_dir, "..", "..", "src"))

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

	# ---------------- Latent-as-input (local) ----------------
	log.info("\n" + "=" * 70)
	log.info("TRAINING LATENT-AS-INPUT VARIANT")
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
	results["latent_as_input"] = local_results

	return results


def save_results(results: Dict[str, Any], output_path: str) -> None:
	"""Save results to pickle file."""
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "wb") as f:
		pickle.dump(results, f)
	log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Model Comparison: Baseline vs Latent-as-Input"
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
		data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "ecuador_training_data.csv"))
		log.info(f"No data_path or data_dir provided. Using default: {data_path}")

	use_wandb = WANDB_AVAILABLE and not args.no_wandb
	if use_wandb:
		wandb.init(
			project="metabarcoding-model-comparison",
			name=f"model_comparison_{time.strftime('%Y-%m-%d_%H-%M')}",
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
	log.info(f"Run visualization: python latent_as_input_visualisation.py --results_path {results_path}")

	if use_wandb:
		wandb.finish()
