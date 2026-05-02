"""
Generic analysis entry point — trains one variant per invocation.

Usage (single variant, called by submit_subanalysis.sh):
    python run_analysis.py --analysis loss_comparison --variant logistic --run_id 20240501_120000 --no_wandb

Usage (manual multi-variant run, for local testing):
    python run_analysis.py --analysis optimal_K --variant K_5 --run_id $(date +%Y%m%d_%H%M%S) --no_wandb
    python run_analysis.py --analysis interpolated_latent --no_wandb  # runs all variants sequentially (dev only)
"""
from __future__ import annotations

import argparse
import logging as log
import os
import sys
import time

ANALYSIS_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(ANALYSIS_ROOT, "..", "src"))
if ANALYSIS_ROOT not in sys.path:
    sys.path.insert(0, ANALYSIS_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(1, SRC_ROOT)

from analyses import REGISTRY
from variant_helpers import run_one_variant


def _build_base_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one variant of a registered analysis",
        add_help=add_help,
    )
    parser.add_argument(
        "--analysis", required=True,
        choices=sorted(REGISTRY),
        help="Analysis to run",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Variant name to train. If omitted, runs all variants sequentially (local dev only).",
    )
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Shared run ID (timestamp) so all variants land in the same output directory. "
             "Auto-generated if not provided.",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Base output directory name (default: results)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count for quick dry-runs")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override data CSV path (e.g. for small test dataset)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser


if __name__ == "__main__":
    # Pass 1: get --analysis so we can load analysis-specific CLI args
    pre_parser = _build_base_parser(add_help=False)
    pre_args, _ = pre_parser.parse_known_args()

    analysis_def = REGISTRY[pre_args.analysis]

    # Pass 2: full parser with analysis-specific args merged in
    parser = _build_base_parser(add_help=True)
    if analysis_def.add_cli_args is not None:
        analysis_def.add_cli_args(parser)
    args = parser.parse_args()

    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    if args.variant is not None:
        run_one_variant(analysis_def, args.variant, args, ANALYSIS_ROOT, run_id)
    else:
        # Sequential fallback for local dev — mirrors what the SLURM jobs do in parallel
        variants = (
            analysis_def.make_variants(args)
            if analysis_def.make_variants is not None
            else analysis_def.DEFAULT_VARIANTS
        )
        for v in variants:
            run_one_variant(analysis_def, v["name"], args, ANALYSIS_ROOT, run_id)
