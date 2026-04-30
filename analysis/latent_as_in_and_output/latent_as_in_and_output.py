"""
Latent Variant Runner

This script trains two variants:
1) latent as input and output with both dimensions set to 5
2) latent as input only with input dimension set to 10

No baseline retraining is performed in this analysis script.
Results are saved as one variant pickle per variant for later comparison.

Usage:
	python latent_as_in_and_output.py --no_wandb
"""
from __future__ import annotations

import argparse
import importlib
import os
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
	use_wandb: bool = True,
	run_group: str | None = None,
	output_dir: str | None = None,
) -> Dict[str, Any]:
	"""Train requested latent variants and return result dict keyed by variant name."""
	results: Dict[str, Any] = {}
	analysis_name = "latent_as_in_and_output"

	root_dir = os.path.dirname(os.path.abspath(__file__))
	src_path = os.path.abspath(os.path.join(root_dir, "..", "..", "src"))

	LocalTrainer, LocalConfig, local_set_seed = load_variant_trainer(root_dir, src_path)

	variant_specs = [
		{
			"variant_name": "both_dim_5",
			"latent_input_dim": 5,
			"embed_dim": 5,
			"tags": ["latent_as_in_and_output", "variant_only", "input_and_output", "dim_5"],
		},
		{
			"variant_name": "input_only_dim_10",
			"latent_input_dim": 10,
			"embed_dim": 0,
			"tags": ["latent_as_in_and_output", "variant_only", "input_only", "dim_10"],
		},
	]

	for variant in variant_specs:
		variant_name = variant["variant_name"]
		latent_input_dim = variant["latent_input_dim"]
		embed_dim = variant["embed_dim"]

		log.info("\n" + "=" * 70)
		log.info(
			"TRAINING VARIANT %s (latent_input_dim=%s, embed_dim=%s)",
			variant_name,
			latent_input_dim,
			embed_dim,
		)
		log.info("=" * 70)

		local_set_seed()
		local_cfg = LocalConfig()
		local_cfg.latent_input_dim = int(latent_input_dim)
		local_cfg.embed_dim = int(embed_dim)

		with variant_wandb_run(
			use_wandb=use_wandb,
			wandb_module=wandb,
			analysis_name=analysis_name,
			variant_name=variant_name,
			run_group=run_group,
			tags=variant["tags"],
			config={
				"latent_input_dim": int(latent_input_dim),
				"embed_dim": int(embed_dim),
			},
		):
			local_trainer = LocalTrainer(local_cfg, model_name=variant_name, results_dir=output_dir)
			local_results = local_trainer.run(use_wandb=use_wandb)
			results[variant_name] = local_results

	return results


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Latent variants: in-and-output dim=5 and input-only dim=10"
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

	use_wandb = WANDB_AVAILABLE and not args.no_wandb
	run_group = make_run_group("latent_as_in_and_output_comparison")
 
	# Create output dir before training so Trainer artifacts land inside it
    output_dir = make_output_dir(__file__, args.output_dir)

	# Run comparison
	results = run_comparison(use_wandb=use_wandb, run_group=run_group, output_dir=output_dir)

	# Save one pickle per variant
	analysis_name = "latent_as_in_and_output"
	for variant_name, variant_result in results.items():
		results_path = save_variant_result(output_dir, analysis_name, variant_name, variant_result)
		print(f"[INFO] Saved {variant_name} results to: {os.path.abspath(results_path)}")

	log.info(f"\n{'='*70}")
	log.info("VARIANT TRAINING COMPLETE")
	log.info(f"{'='*70}")
	log.info(f"Results saved to directory: {output_dir}")