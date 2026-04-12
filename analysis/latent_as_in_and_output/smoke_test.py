"""
Smoke Test for Latent-as-Input-and-Output Variant

Tests both:
1. With output latent D (embed_dim > 0)
2. Without output latent D (embed_dim = 0)

Usage:
    python smoke_test.py --no_wandb
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import logging as log
from typing import Dict, Any, Optional

ANALYSIS_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(ANALYSIS_DIR, "..", "..", "src"))

# Setup logging
log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = log.getLogger(__name__)

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


def run_smoke_test(
    use_wandb: bool = False,
    epochs_per_test: int = 3,
) -> Dict[str, Any]:
    """Run smoke test for both embed_dim=10 (with D) and embed_dim=0 (no D)."""
    results: Dict[str, Any] = {
        "with_d": {"status": "not_started", "error": None, "output": None},
        "without_d": {"status": "not_started", "error": None, "output": None},
    }

    LocalTrainer, LocalConfig, local_set_seed = load_variant_trainer(ANALYSIS_DIR, SRC_DIR)

    # ============================================================================
    # TEST 1: WITH OUTPUT LATENT D (embed_dim = 10)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: WITH OUTPUT LATENT D (embed_dim = 10)")
    logger.info("=" * 80)
    
    try:
        local_set_seed(42)
        cfg1 = LocalConfig()
        cfg1.epochs = epochs_per_test
        cfg1.embed_dim = 10  # With output latent D
        cfg1.diag_ablation_interval = 1  # Enable diagnostics every epoch for testing
        
        logger.info(f"Config: embed_dim={cfg1.embed_dim}, epochs={cfg1.epochs}")
        logger.info("Instantiating Trainer...")
        trainer1 = LocalTrainer(cfg1)
        logger.info("Trainer instantiated successfully")
        
        logger.info("Running training...")
        output1 = trainer1.run(use_wandb=False)
        
        results["with_d"]["status"] = "passed"
        results["with_d"]["output"] = {
            "num_epochs": len(output1.get("epoch_metrics", [])),
            "final_val_auc": output1.get("final_val_auc"),
            "has_latent_d": "latent_d_mean" in output1.get("final_diagnostics", {}),
            "has_latent_vector": "latent_vector" in output1.get("latent_vector", {}),
        }
        logger.info(f"✓ TEST 1 PASSED")
        logger.info(f"  - Final val AUC: {output1.get('final_val_auc'):.4f}")
        logger.info(f"  - Has D diagnostics: {results['with_d']['output']['has_latent_d']}")
        logger.info(f"  - Output payload keys: {list(output1.keys())}")
        
    except Exception as e:
        results["with_d"]["status"] = "failed"
        results["with_d"]["error"] = str(e)
        logger.error(f"✗ TEST 1 FAILED with error:", exc_info=True)
        import traceback
        logger.error(traceback.format_exc())

    # ============================================================================
    # TEST 2: WITHOUT OUTPUT LATENT D (embed_dim = 0)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: WITHOUT OUTPUT LATENT D (embed_dim = 0)")
    logger.info("=" * 80)
    
    try:
        # Reload trainer to clear state
        for mod in ["config", "train", "model", "latent_solver"]:
            if mod in sys.modules:
                del sys.modules[mod]
        
        LocalTrainer, LocalConfig, local_set_seed = load_variant_trainer(ANALYSIS_DIR, SRC_DIR)
        
        local_set_seed(42)
        cfg2 = LocalConfig()
        cfg2.epochs = epochs_per_test
        cfg2.embed_dim = 0  # WITHOUT output latent D
        cfg2.diag_ablation_interval = 1  # Enable diagnostics every epoch for testing
        
        logger.info(f"Config: embed_dim={cfg2.embed_dim}, epochs={cfg2.epochs}")
        logger.info("Instantiating Trainer...")
        trainer2 = LocalTrainer(cfg2)
        logger.info("Trainer instantiated successfully")
        
        logger.info("Running training...")
        output2 = trainer2.run(use_wandb=False)
        
        results["without_d"]["status"] = "passed"
        results["without_d"]["output"] = {
            "num_epochs": len(output2.get("epoch_metrics", [])),
            "final_val_auc": output2.get("final_val_auc"),
            "has_latent_z": "latent_z_vector" in output2,
            "latent_d_missing": "latent_d_vector" not in output2,  # Should be True (removed)
            "latent_vector_exists": "latent_vector" in output2,  # Should be True (now points to Z)
        }
        logger.info(f"✓ TEST 2 PASSED")
        logger.info(f"  - Final val AUC: {output2.get('final_val_auc'):.4f}")
        logger.info(f"  - Has Z vector: {results['without_d']['output']['has_latent_z']}")
        logger.info(f"  - D vector absent: {results['without_d']['output']['latent_d_missing']}")
        logger.info(f"  - latent_vector exists: {results['without_d']['output']['latent_vector_exists']}")
        logger.info(f"  - Output payload keys: {list(output2.keys())}")
        
    except Exception as e:
        results["without_d"]["status"] = "failed"
        results["without_d"]["error"] = str(e)
        logger.error(f"✗ TEST 2 FAILED with error:", exc_info=True)
        import traceback
        logger.error(traceback.format_exc())

    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print test summary."""
    logger.info("\n" + "=" * 80)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 80)
    
    for test_name, result in results.items():
        status = result["status"]
        status_symbol = "✓ PASSED" if status == "passed" else "✗ FAILED" if status == "failed" else "⊘ SKIPPED"
        logger.info(f"\n{test_name.upper()}: {status_symbol}")
        
        if result["error"]:
            logger.error(f"  Error: {result['error']}")
        
        if result["output"]:
            for key, value in result["output"].items():
                logger.info(f"  - {key}: {value}")
    
    all_passed = all(r["status"] == "passed" for r in results.values())
    if all_passed:
        logger.info("\n✓ ALL TESTS PASSED")
    else:
        logger.info("\n✗ SOME TESTS FAILED")
        failed = [name for name, result in results.items() if result["status"] == "failed"]
        logger.error(f"Failed tests: {', '.join(failed)}")
    
    logger.info("=" * 80 + "\n")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke Test for Latent-as-Input-and-Output Variant"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging (default: disabled for tests)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs per test (default: 3)")
    args = parser.parse_args()

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    results = run_smoke_test(use_wandb=use_wandb, epochs_per_test=args.epochs)
    all_passed = print_summary(results)
    
    sys.exit(0 if all_passed else 1)
