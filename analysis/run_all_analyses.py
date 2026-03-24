#!/usr/bin/env python
"""Run all analysis/visualization pipelines sequentially from one entrypoint.

Usage:
    python run_all_analyses.py --data_path ../../data/ecuador_training_data.csv
    python run_all_analyses.py --data_path ../../data/ecuador_training_data.csv --no_wandb
    python run_all_analyses.py --data_path ../../data/ecuador_training_data.csv --analyses ablation_study loss_comparison
"""
from __future__ import annotations

import os
import sys
import argparse
import subprocess
import logging
import time
import shlex
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("run_all_analyses")

ANALYSIS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"


# ============================================================================
# Analysis Definitions
# ============================================================================

ANALYSES = {
    "baselines": "Baseline Models (TwoStage, Ridge, Random Forest, etc.)",
    "ablation_study": "Ablation Study (MLP variants)",
    "loss_comparison": "Loss Function Comparison (Cross-Entropy vs Logistic)",
    "k_comparison": "K Neighbour Comparison",
    "preprocessing": "Read Count Preprocessing Methods",
    "latent_as_input": "Latent-as-Input Model Variant",
    "barcodebert": "Taxonomy vs BarcodeBERT Embeddings",
    "dimensionality_gating": "Dimensionality Analysis: Gating Functions",
    "dimensionality_vector": "Dimensionality Analysis: Vector Sizes",
}


# ============================================================================
# Helper Functions
# ============================================================================

def run_command(
    cmd: List[str],
    description: str,
    cwd: Path,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Execute a command and return success status and output.
    
    Args:
        cmd: Command to execute
        description: Description of the command
        cwd: Working directory
        verbose: Print command output
        
    Returns:
        Tuple of (success, output_message)
    """
    try:
        log.info(f"  Running in {cwd}: {shlex.join(cmd)}")
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        pythonpath_entries = [str(SRC_ROOT)]
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=6 * 3600,
        )

        if verbose and result.stdout:
            print(result.stdout)
        if verbose and result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode == 0:
            return True, "Success"

        error_msg = (result.stderr or result.stdout or "Unknown subprocess error").strip()
        return False, error_msg
    except subprocess.TimeoutExpired:
        return False, "Command timed out (>6 hours)"
    except Exception as e:
        return False, str(e)


def python_cmd(script_path: Path, args: List[str]) -> List[str]:
    return [sys.executable, str(script_path), *args]


def build_pipeline_commands(
    analysis_key: str,
    analysis_dir: Path,
    data_path: str,
    no_wandb: bool,
    verbose: bool,
) -> List[Tuple[str, List[str], Path]]:
    """Return ordered commands for one analysis: [(label, cmd, cwd), ...]."""
    cmds: List[Tuple[str, List[str], Path]] = []

    if analysis_key == "baselines":
        train_script = analysis_dir / "baselines/run_baselines.py"
        viz_script = analysis_dir / "baselines/visualize.py"
        train_args = ["--data_path", data_path, "--output_dir", "results"]
        viz_args = ["--data_path", data_path, "--output_dir", "figures"]
        if verbose:
            train_args.append("--verbose")
        cmds.append(("analysis", python_cmd(train_script, train_args), train_script.parent))
        cmds.append(("visualization", python_cmd(viz_script, viz_args), viz_script.parent))
        return cmds

    if analysis_key == "ablation_study":
        train_script = analysis_dir / "ablation_study/ablation_study.py"
        viz_script = analysis_dir / "ablation_study/ablation_visualize.py"
        train_args = ["--data_path", data_path, "--output_dir", "results"]
        if no_wandb:
            train_args.append("--no_wandb")
        if verbose:
            train_args.append("--verbose")
        viz_args = ["--results_path", "results/ablation_results.pkl", "--output_dir", "figures"]
        cmds.append(("analysis", python_cmd(train_script, train_args), train_script.parent))
        cmds.append(("visualization", python_cmd(viz_script, viz_args), viz_script.parent))
        return cmds

    if analysis_key == "loss_comparison":
        train_script = analysis_dir / "loss_comparison/loss_comparison.py"
        viz_script = analysis_dir / "loss_comparison/loss_comparison_visualize.py"
        train_args = ["--data_path", data_path, "--output_dir", "results"]
        if no_wandb:
            train_args.append("--no_wandb")
        if verbose:
            train_args.append("--verbose")
        viz_args = ["--results_path", "results/loss_comparison_results.pkl", "--output_dir", "figures"]
        cmds.append(("analysis", python_cmd(train_script, train_args), train_script.parent))
        cmds.append(("visualization", python_cmd(viz_script, viz_args), viz_script.parent))
        return cmds

    if analysis_key == "k_comparison":
        train_script = analysis_dir / "optimal_K/K_comparison.py"
        viz_script = analysis_dir / "optimal_K/K_comparison_visualize.py"
        train_args = ["--data_path", data_path, "--output_dir", "results"]
        if no_wandb:
            train_args.append("--no_wandb")
        if verbose:
            train_args.append("--verbose")
        viz_args = ["--results_path", "results/K_comparison_results.pkl", "--output_dir", "figures"]
        cmds.append(("analysis", python_cmd(train_script, train_args), train_script.parent))
        cmds.append(("visualization", python_cmd(viz_script, viz_args), viz_script.parent))
        return cmds

    if analysis_key == "preprocessing":
        prep_script = analysis_dir / "preprocessing/utils_test.py"
        train_script = analysis_dir / "preprocessing/read_count_preprocessing.py"
        viz_script = analysis_dir / "preprocessing/preprocessing_visualization.py"
        prep_args = ["--data_path", data_path]
        train_args = ["--output_dir", "results"]
        if no_wandb:
            train_args.append("--no_wandb")
        if verbose:
            train_args.append("--verbose")
        viz_args = ["--results_path", "results/preprocessing_results.pkl", "--output_dir", "figures"]
        cmds.append(("dataset_prep", python_cmd(prep_script, prep_args), prep_script.parent))
        cmds.append(("analysis", python_cmd(train_script, train_args), train_script.parent))
        cmds.append(("visualization", python_cmd(viz_script, viz_args), viz_script.parent))
        return cmds

    if analysis_key == "latent_as_input":
        train_script = analysis_dir / "latent_as_input/latent_as_input.py"
        viz_script = analysis_dir / "latent_as_input/latent_as_input_visualisation.py"
        train_args = ["--data_path", data_path, "--output_dir", "results"]
        if no_wandb:
            train_args.append("--no_wandb")
        if verbose:
            train_args.append("--verbose")
        viz_args = ["--results_path", "results/model_comparison_results.pkl", "--output_dir", "figures"]
        cmds.append(("analysis", python_cmd(train_script, train_args), train_script.parent))
        cmds.append(("visualization", python_cmd(viz_script, viz_args), viz_script.parent))
        return cmds

    if analysis_key == "barcodebert":
        taxonomy_script = analysis_dir.parent / "src/train.py"
        barcodebert_script = analysis_dir / "BarcodeBERT/train.py"
        viz_script = analysis_dir / "BarcodeBERT/barcodebert_visualize.py"

        taxonomy_args = ["--data_path", data_path]
        barcodebert_args = ["--data_path", data_path]
        viz_args = [
            "--taxonomy_results", "results/taxonomy_results.pkl",
            "--barcodebert_results", "results/barcodebert_results.pkl",
            "--output_dir", "figures",
        ]
        if verbose:
            taxonomy_args.append("--verbose")
            barcodebert_args.append("--verbose")
            viz_args.append("--verbose")

        cmds.append(("analysis_taxonomy", python_cmd(taxonomy_script, taxonomy_args), taxonomy_script.parent))
        cmds.append(("analysis_barcodebert", python_cmd(barcodebert_script, barcodebert_args), barcodebert_script.parent))
        cmds.append(("visualization", python_cmd(viz_script, viz_args), viz_script.parent))
        return cmds

    if analysis_key == "dimensionality_gating":
        train_script = analysis_dir / "dimensionality_increase/gating_function/dimensionality_increase.py"
        viz_script = analysis_dir / "dimensionality_increase/gating_function/dimensionality_increase_visualize.py"
        train_args = ["--data_path", data_path, "--output_dir", "results"]
        if no_wandb:
            train_args.append("--no_wandb")
        if verbose:
            train_args.append("--verbose")
        viz_args = ["--results_path", "results/gating_comparison_results.pkl", "--output_dir", "figures"]
        cmds.append(("analysis", python_cmd(train_script, train_args), train_script.parent))
        cmds.append(("visualization", python_cmd(viz_script, viz_args), viz_script.parent))
        return cmds

    if analysis_key == "dimensionality_vector":
        train_script = analysis_dir / "dimensionality_increase/vector_size/dimensionality_increase.py"
        viz_script = analysis_dir / "dimensionality_increase/vector_size/dimensionality_increase_visualize.py"
        train_args = ["--data_path", data_path, "--output_dir", "results"]
        if no_wandb:
            train_args.append("--no_wandb")
        if verbose:
            train_args.append("--verbose")
        viz_args = ["--results_path", "results/dimensionality_analysis_results.pkl", "--output_dir", "figures"]
        cmds.append(("analysis", python_cmd(train_script, train_args), train_script.parent))
        cmds.append(("visualization", python_cmd(viz_script, viz_args), viz_script.parent))
        return cmds

    raise ValueError(f"Unhandled analysis key: {analysis_key}")


def run_analysis(
    analysis_key: str,
    analysis_dir: Path,
    data_path: str,
    no_wandb: bool = False,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """Run one analysis pipeline: analysis script(s) then visualization script."""
    label = ANALYSES[analysis_key]
    log.info(f"\n{'=' * 70}")
    log.info(f"ANALYSIS: {label}")
    log.info(f"{'=' * 70}")

    pipeline = build_pipeline_commands(
        analysis_key=analysis_key,
        analysis_dir=analysis_dir,
        data_path=data_path,
        no_wandb=no_wandb,
        verbose=verbose,
    )

    for step_name, cmd, cwd in pipeline:
        script_path = Path(cmd[1])
        if not script_path.exists():
            msg = f"Missing script for step '{step_name}': {script_path}"
            log.error(f"✗ {msg}")
            return False, msg

        log.info(f"Step ({step_name}): {script_path.name}")
        success, output = run_command(cmd, f"{analysis_key}:{step_name}", cwd=cwd, verbose=verbose)
        if not success:
            msg = f"{step_name} failed: {output}"
            log.error(f"✗ {msg}")
            return False, msg
        log.info(f"✓ {step_name} complete")

    return True, f"{label} - Complete"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all analyses and visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_analyses.py --data_path ../../data/ecuador_training_data.csv
  python run_all_analyses.py --data_path ../../data/ecuador_training_data.csv --no_wandb
  python run_all_analyses.py --data_path ../../data/ecuador_training_data.csv --analyses baselines ablation_study
        """
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--analyses",
        type=str,
        nargs="+",
        default=list(ANALYSES.keys()),
        help="Specific analyses to run (space-separated). Default: all"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate data path
    data_path = os.path.abspath(args.data_path)
    if not os.path.exists(data_path):
        log.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    log.info(f"Data path: {data_path}")
    
    # Get analysis directory
    analysis_dir = Path(__file__).parent.absolute()
    
    log.info(f"Analysis directory: {analysis_dir}")
    log.info(f"Number of analyses to run: {len(args.analyses)}")
    
    # Validate analysis keys
    invalid_keys = [k for k in args.analyses if k not in ANALYSES]
    if invalid_keys:
        log.error(f"Unknown analyses: {invalid_keys}")
        log.error(f"Available: {list(ANALYSES.keys())}")
        sys.exit(1)
    
    # Track results
    results: Dict[str, Tuple[bool, str]] = {}
    start_time = time.time()
    
    # Run each analysis
    for i, analysis_key in enumerate(args.analyses, 1):
        try:
            success, message = run_analysis(
                analysis_key,
                analysis_dir,
                data_path,
                no_wandb=args.no_wandb,
                verbose=args.verbose
            )
            results[analysis_key] = (success, message)
            
            # Log progress
            status = "✓" if success else "✗"
            log.info(f"{status} [{i}/{len(args.analyses)}] {message}")
            
        except Exception as e:
            log.error(f"Exception in analysis {analysis_key}: {e}")
            results[analysis_key] = (False, str(e))
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Print summary
    elapsed = time.time() - start_time
    
    log.info(f"\n{'='*70}")
    log.info("SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    log.info(f"Total analyses: {len(results)}")
    
    successful = sum(1 for success, _ in results.values() if success)
    failed = len(results) - successful
    
    log.info(f"Successful: {successful}")
    log.info(f"Failed: {failed}")
    
    if failed > 0:
        log.info(f"\n{'='*70}")
        log.info("FAILED ANALYSES:")
        log.info(f"{'='*70}")
        for analysis_key, (success, message) in results.items():
            if not success:
                log.error(f"✗ {analysis_key}: {message}")
    
    log.info(f"\n{'='*70}")
    log.info("COMPLETED")
    log.info(f"{'='*70}\n")
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
