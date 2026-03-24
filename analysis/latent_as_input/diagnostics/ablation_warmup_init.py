"""
Ablation study: latent init strategy vs warmup

Tests four training recipes for the latent-as-input model to determine
whether the latent pre-warmup phase is beneficial, and whether simply
initialising the embedding with small non-zero noise is sufficient.

Variants
--------
  A  latent_init_std=0.0,  latent_warmup_epochs=20  (current default)
  B  latent_init_std=0.0,  latent_warmup_epochs=0   (no warmup, zero init)
  C  latent_init_std=0.01, latent_warmup_epochs=0   (non-zero init, no warmup)
  D  latent_init_std=0.05, latent_warmup_epochs=0   (larger init, no warmup)

All variants use the same reduced regularization (1e-4) and the same
fixed data split (loaded from data_dir pre-processed files).

Usage
-----
  python ablation_warmup_init.py --data_dir ../../data --no_wandb
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import logging as log
from typing import Any, Dict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "A_warmup20_std0": dict(
        label="A: warmup=20, init_std=0.0",
        latent_init_std=0.0,
        latent_warmup_epochs=20,
    ),
    "B_no_warmup_std0": dict(
        label="B: warmup=0, init_std=0.0",
        latent_init_std=0.0,
        latent_warmup_epochs=0,
    ),
    "C_no_warmup_std001": dict(
        label="C: warmup=0, init_std=0.01",
        latent_init_std=0.01,
        latent_warmup_epochs=0,
    ),
    "D_no_warmup_std005": dict(
        label="D: warmup=0, init_std=0.05",
        latent_init_std=0.05,
        latent_warmup_epochs=0,
    ),
}


def load_local_trainer(local_dir: str, src_path: str):
    """Load latent-as-input Trainer and Config with clean import isolation."""
    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)
    if src_path not in sys.path:
        sys.path.append(src_path)
    for mod in ["config", "train", "model", "latent_solver"]:
        if mod in sys.modules:
            del sys.modules[mod]
    import train as local_train
    import config as local_config
    return local_train.Trainer, local_config.Config, local_config.set_seed


def run_ablation(
    data_dir: str,
    use_wandb: bool = False,
) -> Dict[str, Any]:
    root_dir   = os.path.dirname(os.path.abspath(__file__))
    src_path   = os.path.abspath(os.path.join(root_dir, "..", "..", "src"))
    results: Dict[str, Any] = {}

    for variant_key, variant_cfg in ABLATION_VARIANTS.items():
        log.info("\n" + "=" * 70)
        log.info(f"TRAINING VARIANT {variant_cfg['label']}")
        log.info("=" * 70)

        LocalTrainer, LocalConfig, set_seed = load_local_trainer(root_dir, src_path)
        set_seed(14)

        cfg = LocalConfig(
            latent_init_std=variant_cfg["latent_init_std"],
            latent_warmup_epochs=variant_cfg["latent_warmup_epochs"],
        )

        trainer = LocalTrainer(
            cfg,
            data_dir=data_dir,
            loss_type="cross_entropy",
        )
        variant_results = trainer.run(use_wandb=use_wandb)

        # Store variant metadata alongside results
        variant_results["variant_label"] = variant_cfg["label"]
        variant_results["variant_init_std"] = variant_cfg["latent_init_std"]
        variant_results["variant_warmup_epochs"] = variant_cfg["latent_warmup_epochs"]
        results[variant_key] = variant_results

        log.info(
            f"  ✓ {variant_cfg['label']} done — "
            f"best_val={variant_results['best_val_loss']:.6f}"
        )

    return results


def save_results(results: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation: warmup vs init strategy")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with pre-processed train/val/test CSV files")
    parser.add_argument("--output_path", type=str,
                        default="results/ablation_warmup_init.pkl",
                        help="Output path for results pickle")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log.basicConfig(
        level=log.DEBUG if args.verbose else log.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    use_wandb = WANDB_AVAILABLE and not args.no_wandb

    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output_path
    if not os.path.isabs(output_path):
        output_path = os.path.join(root_dir, output_path)

    results = run_ablation(data_dir=args.data_dir, use_wandb=use_wandb)
    save_results(results, output_path)
