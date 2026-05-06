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
import copy
import logging as log
import os
import pickle
import re
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

ANALYSIS_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(ANALYSIS_ROOT, "..", "src"))
if ANALYSIS_ROOT not in sys.path:
    sys.path.insert(0, ANALYSIS_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(1, SRC_ROOT)

from analyses import REGISTRY


def _sanitize_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    token = token.strip("_")
    return token or "variant"


def make_run_group(analysis_name: str, timestamp: Optional[str] = None) -> str:
    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    return f"{_sanitize_token(analysis_name)}_{ts}"


def make_variant_run_name(
    analysis_name: str,
    variant_name: str,
    timestamp: Optional[str] = None,
) -> str:
    ts = timestamp or time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{_sanitize_token(analysis_name)}_{_sanitize_token(variant_name)}_{ts}"


def make_variant_filename(analysis_name: str, variant_name: str) -> str:
    return f"{_sanitize_token(analysis_name)}_{_sanitize_token(variant_name)}.pkl"


def make_output_dir(script_file: str, output_dir_arg: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(script_file))
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, output_dir_arg, ts)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_variant_result(
    output_dir: str,
    analysis_name: str,
    variant_name: str,
    result: Dict[str, Any],
) -> str:
    output_path = os.path.join(output_dir, make_variant_filename(analysis_name, variant_name))
    with open(output_path, "wb") as f:
        pickle.dump({variant_name: result}, f)
    return output_path


def make_output_dir_for_analysis(
    analysis_name: str,
    analysis_root: str,
    output_dir_arg: str = "results",
) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(analysis_root, output_dir_arg, analysis_name, ts)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_one_variant(
    analysis_def: Any,
    variant_name: str,
    args: Any,
    analysis_root: str,
    run_id: str,
) -> None:
    """Train a single named variant. Caller must have src/ and analysis root on sys.path.

    All variants of the same analysis share the same *run_id* so their result
    pickles land in the same timestamped directory:
      results/{analysis_name}/{run_id}/{analysis_name}_{variant_name}.pkl
    """
    import logging as log
    from config import Config, set_seed
    from train import Trainer

    try:
        import wandb as _wandb
        _WANDB_AVAILABLE = True
    except ImportError:
        _wandb = None  # type: ignore[assignment]
        _WANDB_AVAILABLE = False

    analysis_name: str = analysis_def.ANALYSIS_NAME

    # Resolve the variant dict from the analysis definition
    all_variants: List[Dict[str, Any]]
    if analysis_def.make_variants is not None:
        all_variants = analysis_def.make_variants(args)
    else:
        all_variants = list(analysis_def.DEFAULT_VARIANTS)

    matches = [v for v in all_variants if v["name"] == variant_name]
    if not matches:
        valid = [v["name"] for v in all_variants]
        raise ValueError(f"Unknown variant '{variant_name}' for '{analysis_name}'. Valid: {valid}")
    variant = matches[0]

    output_dir = os.path.join(
        analysis_root,
        getattr(args, "output_dir", "results"),
        analysis_name,
        run_id,
    )
    os.makedirs(output_dir, exist_ok=True)

    use_wandb: bool = _WANDB_AVAILABLE and not getattr(args, "no_wandb", True)
    run_group = f"{_sanitize_token(analysis_name)}_{run_id}"

    base_cfg = Config()
    if getattr(args, "epochs", None) is not None:
        base_cfg.epochs = args.epochs
    if getattr(args, "data_path", None) is not None:
        base_cfg.data_path = os.path.abspath(args.data_path)
        base_cfg.preprocessed_dir = None

    cfg = copy.deepcopy(base_cfg)
    for key, val in variant["config"].items():
        setattr(cfg, key, val)
    if analysis_def.apply_args_to_cfg is not None:
        analysis_def.apply_args_to_cfg(cfg, args)
    set_seed()

    log.info("\n" + "=" * 70)
    log.info("TRAINING VARIANT: %s / %s  (run_id=%s)", analysis_name, variant_name, run_id)
    log.info("Config overrides: %s", variant["config"])
    log.info("=" * 70)

    with variant_wandb_run(
        use_wandb=use_wandb,
        wandb_module=_wandb,
        analysis_name=analysis_name,
        variant_name=variant_name,
        run_group=run_group,
        config={**cfg.__dict__, "variant": variant_name},
    ):
        trainer = Trainer(cfg, model_name=variant_name, results_dir=output_dir)
        result = trainer.run(use_wandb=use_wandb)

    save_variant_result(output_dir, analysis_name, variant_name, result)
    log.info("Saved %s to %s", variant_name, output_dir)
    log.info("=" * 70)


@contextmanager
def variant_wandb_run(
    *,
    use_wandb: bool,
    wandb_module: Any,
    analysis_name: str,
    variant_name: str,
    run_group: Optional[str],
    tags: Optional[list[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    if use_wandb:
        wandb_module.init(
            project="metabarcoding",
            name=make_variant_run_name(analysis_name, variant_name),
            group=run_group,
            tags=tags or [analysis_name, variant_name, "variant_only"],
            config=config,
            reinit=True,
        )
    try:
        yield
    finally:
        if use_wandb:
            wandb_module.finish()


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
