"""
Latent-as-Input Variant Runner

This script trains only the latent-as-input variant
(no baseline retraining in this analysis script).
Results are saved as one variant pickle for later comparison.

Usage:
	python latent_as_input.py --data_path ../../data/data_merged.csv
	python latent_as_input.py --data_dir ../../data --no_wandb
"""
from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
import logging as log
from typing import Dict, Any

ANALYSIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ANALYSIS_DIR not in sys.path:
	sys.path.insert(0, ANALYSIS_DIR)
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


def load_variant_trainer(local_dir: str, src_path: str):
	"""Load latent-as-input Trainer and Config from local folder with clean imports."""
	# Ensure deterministic import order: local analysis dir first, src second.
	if local_dir in sys.path:
		sys.path.remove(local_dir)
	sys.path.insert(0, local_dir)
	if src_path in sys.path:
		sys.path.remove(src_path)
	sys.path.insert(1, src_path)
	importlib.invalidate_caches()

	# Clear cached modules that could shadow local config
	for mod in ["config", "train", "model", "latent_solver"]:
		if mod in sys.modules:
			del sys.modules[mod]

	local_model = importlib.import_module("model")
	importlib.import_module("latent_solver")
	local_train = importlib.import_module("train")
	local_config = importlib.import_module("config")

	model_file = os.path.abspath(getattr(local_model, "__file__", ""))
	if not model_file.startswith(os.path.abspath(local_dir) + os.sep):
		raise ImportError(
			f"Resolved wrong model module: {model_file}. Expected under {local_dir}"
		)
	return local_train.Trainer, local_config.Config, local_config.set_seed


def run_comparison(
	data_path: str | None,
	data_dir: str | None,
	use_wandb: bool = True,
	run_group: str | None = None,
) -> Dict[str, Any]:
	"""Train latent-as-input variant and return result dict keyed by variant name."""
	results: Dict[str, Any] = {}

	root_dir = os.path.dirname(os.path.abspath(__file__))
	src_path = os.path.abspath(os.path.join(root_dir, "..", "..", "src"))

	# ---------------- Latent-as-input (local) ----------------
	log.info("\n" + "=" * 70)
	log.info("TRAINING LATENT-AS-INPUT VARIANT")
	log.info("=" * 70)

	LocalTrainer, LocalConfig, local_set_seed = load_variant_trainer(root_dir, src_path)
	local_set_seed(14)
	local_cfg = LocalConfig()
	with variant_wandb_run(
		use_wandb=use_wandb,
		wandb_module=wandb,
		analysis_name="latent_as_input_v2",
		variant_name="latent_as_input",
		run_group=run_group,
		tags=["latent_as_input_v2", "variant_only"],
	):
		local_trainer = LocalTrainer(
			local_cfg,
			data_path=data_path,
			data_dir=data_dir,
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
		help="Path to raw data CSV file (e.g. ../../data/data_merged.csv)",
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
		data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "data_merged.csv"))
		log.info(f"No data_path or data_dir provided. Using default: {data_path}")

	use_wandb = WANDB_AVAILABLE and not args.no_wandb
	run_group = make_run_group("latent_as_input_v2_comparison")

	# Run comparison
	results = run_comparison(data_path=data_path, data_dir=data_dir, use_wandb=use_wandb, run_group=run_group)

	# Save results

	output_dir = make_output_dir(__file__, args.output_dir)
	results_path = save_variant_result(output_dir, "latent_as_input_v2", "latent_as_input", results["latent_as_input"])
	print(f"[INFO] Saving results to: {os.path.abspath(results_path)}")

	log.info(f"\n{'='*70}")
	log.info("VARIANT TRAINING COMPLETE")
	log.info(f"{'='*70}")
	log.info(f"Results saved to: {results_path}")
	log.info(f"Run visualization: python ../visualize_results.py --results_path {results_path}")
