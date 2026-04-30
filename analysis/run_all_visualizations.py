#!/usr/bin/env python3
"""
Run visualize_results.py for all analyses.

For each analysis, all result pkl files are discovered under its results/
directory. If none exist the analysis is skipped. Labels, colors, key
remapping, latent concatenation, and subset_top_n are configured per
analysis below.

Usage
-----
    python analysis/run_all_visualizations.py            # all analyses
    python analysis/run_all_visualizations.py ablation_study optimal_K
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pickle

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loader helper (mirrors the logic used in the ad-hoc runner scripts)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from visualize_results import (  # noqa: E402
    _is_model_payload,
    _validate_model_payload,
    create_all_visualizations,
    print_comparison,
)


def _load_merged(
    file_specs: List[Tuple[str, Optional[str]]],
    latent_concat_keys: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load and merge pkl files into a single results dict.

    *file_specs* is a list of (glob_pattern_or_path, key_override) pairs.
    *latent_concat_keys* lists model keys whose latent_vector should be
    replaced by concatenating their latent_z_vector and latent_d_vector
    (used for latent_as_in_and_output/both_dim_5).

    Returns None if no files matched.
    """
    merged: Dict[str, Any] = {}

    for pattern, key_override in file_specs:
        paths = sorted(glob.glob(pattern))
        if not paths:
            log.debug("No files matched: %s", pattern)
            continue
        for fpath in paths:
            with open(fpath, "rb") as f:
                raw = pickle.load(f)
            if not isinstance(raw, dict):
                log.warning("Skipping non-dict pickle: %s", fpath)
                continue

            if _is_model_payload(raw):
                # Single model payload (not wrapped in a model key)
                key = key_override or os.path.splitext(os.path.basename(fpath))[0]
                payload = dict(raw)
            else:
                if key_override is not None and len(raw) == 1:
                    key = key_override
                    payload = dict(list(raw.values())[0])
                elif len(raw) == 1:
                    key = list(raw.keys())[0]
                    payload = dict(raw[key])
                else:
                    # Multi-key dict (e.g. baselines comparison combined file)
                    for k, v in raw.items():
                        _validate_model_payload(k, v)
                        merged[k] = v
                    continue

            # Optionally replace latent_vector with concat(z, d)
            if latent_concat_keys and key in latent_concat_keys:
                z_raw = payload.get("latent_z_vector")
                d_raw = payload.get("latent_d_vector")
                z = np.asarray(z_raw) if z_raw is not None else None
                d = np.asarray(d_raw) if d_raw is not None else None
                if z is not None and z.ndim >= 1 and d is not None and d.ndim >= 1:
                    payload["latent_vector"] = np.concatenate([z, d], axis=1)
                    log.info("  Concatenated z+d latent for '%s': shape %s", key, payload["latent_vector"].shape)

            _validate_model_payload(key, payload)
            if key in merged:
                log.warning("Duplicate model key '%s' from %s — skipping", key, fpath)
                continue
            merged[key] = payload

    return merged if merged else None


# ---------------------------------------------------------------------------
# Per-analysis configuration
# ---------------------------------------------------------------------------
# Each entry is a dict with:
#   name         – display title
#   output_dir   – relative to SCRIPT_DIR
#   file_specs   – list of (glob_pattern, key_override)  (key_override=None → use pkl key/stem)
#   labels       – {model_key: display_label}
#   colors       – {model_key: hex_color}
#   subset_top_n – int or None  (restrict detail plots to top-N by KL divergence)
#   latent_concat_keys – list of model keys to apply z+d concat to, or None

_R = os.path.join(SCRIPT_DIR, "{subdir}", "results")  # helper

ANALYSES: List[Dict[str, Any]] = [

    # ── BarcodeBERT ──────────────────────────────────────────────────────────
    dict(
        name="BarcodeBERT",
        output_dir="figures/BarcodeBERT",
        file_specs=[
            (os.path.join(SCRIPT_DIR, "BarcodeBERT/results/*/barcodebert*.pkl"),   "baseline"),
            (os.path.join(SCRIPT_DIR, "BarcodeBERT/results/*/taxonomy_taxonomy.pkl"), None),
        ],
        labels={
            "baseline": "BarcodeBERT",
            "taxonomy": "Taxonomy",
        },
        colors={
            "baseline": "#2ecc71",
            "taxonomy": "#9b59b6",
        },
        subset_top_n=None,
        latent_concat_keys=None,
    ),

    # ── baselines_comparison ─────────────────────────────────────────────────
    dict(
        name="Baselines Comparison",
        output_dir="figures/baselines_comparison",
        file_specs=[
            # combined file holds all sklearn baselines
            (os.path.join(SCRIPT_DIR, "baselines/results/*/baseline_model_comparison_results_*.pkl"), None),
            # MLP + Latent baseline
            (os.path.join(SCRIPT_DIR, "baselines/results/*/baseline.pkl"), "baseline"),
        ],
        labels={
            "mean":               "Mean",
            "zero":               "Zero",
            "linear_regression":  "Linear Regression",
            "ridge":              "Ridge",
            "elasticnet":         "ElasticNet",
            "decision_tree":      "Decision Tree",
            "random_forest":      "Random Forest",
            "gradient_boosting":  "Gradient Boosting",
            "knn":                "KNN",
            "two_stage":          "Two-Stage",
            "zero_inflated_ridge":"Zero-Inflated Ridge",
            "tweedie":            "Tweedie",
            "log_transform":      "Log-Transform",
            "quantile_rf":        "Quantile RF",
            "baseline":           "MLP + Latent (Baseline)",
        },
        colors={
            "mean":               "#808080",
            "zero":               "#4d4d4d",
            "linear_regression":  "#f28e2b",
            "ridge":              "#4e79a7",
            "elasticnet":         "#e15759",
            "decision_tree":      "#76b7b2",
            "random_forest":      "#59a14f",
            "gradient_boosting":  "#edc948",
            "knn":                "#b07aa1",
            "two_stage":          "#9c755f",
            "zero_inflated_ridge":"#bab0ab",
            "tweedie":            "#ff9da7",
            "log_transform":      "#8cd17d",
            "quantile_rf":        "#af7aa1",
            "baseline":           "#e74c3c",
        },
        subset_top_n=6,
        latent_concat_keys=None,
    ),

    # ── interpolated_latent ──────────────────────────────────────────────────
    dict(
        name="Interpolated Latent",
        output_dir="figures/interpolated_latent",
        file_specs=[
            (os.path.join(SCRIPT_DIR, "interpolated_latent/results/*/baseline.pkl"), "baseline"),
            (os.path.join(SCRIPT_DIR, "interpolated_latent/results/*/interpolated_latent_*.pkl"), None),
        ],
        labels={
            "baseline":                    "Baseline",
            "default_with_interpolation":  "Interpolation (20%)",
            "include_self_false":          "Interpolation (20%, no self latent)",
            "inference_true":              "Interpolation (20%, at inference)",
            "train_mlp_false":             "Interpolation (20%, no MLP interpolation)",
            "fraction_0p1":                "Interpolation (10%)",
            "fraction_0p5":                "Interpolation (50%)",
            "fraction_1p0":                "Interpolation (100%)",
        },
        colors={
            "baseline":                    "#95a5a6",
            "default_with_interpolation":  "#e74c3c",
            "include_self_false":          "#e67e22",
            "inference_true":              "#f39c12",
            "train_mlp_false":             "#2ecc71",
            "fraction_0p1":                "#3498db",
            "fraction_0p5":                "#9b59b6",
            "fraction_1p0":                "#1abc9c",
        },
        subset_top_n=None,
        latent_concat_keys=None,
    ),

    # ── location_embedding ───────────────────────────────────────────────────
    dict(
        name="Location Embedding",
        output_dir="figures/location_embedding",
        file_specs=[
            (os.path.join(SCRIPT_DIR, "location_embedding/results/*/baseline.pkl"), "baseline"),
            (os.path.join(SCRIPT_DIR, "location_embedding/results/*/location_embedding_*.pkl"), None),
        ],
        labels={
            "baseline":    "Baseline (No Location Embedding)",
            "satclip":     "SatCLIP (256D)",
            "range":       "RANGE (1280D)",
            "geoclip":     "GeoCLIP (512D)",
            "alphaearth":  "AlphaEarth (64D)",
        },
        colors={
            "baseline":    "#95a5a6",
            "satclip":     "#e74c3c",
            "range":       "#3498db",
            "geoclip":     "#2ecc71",
            "alphaearth":  "#f39c12",
        },
        subset_top_n=None,
        latent_concat_keys=None,
    ),

    # ── latent_as_in_and_output ──────────────────────────────────────────────
    dict(
        name="Latent As Input and Output",
        output_dir="figures/latent_as_in_and_output",
        file_specs=[
            (os.path.join(SCRIPT_DIR, "latent_as_in_and_output/results/*/baseline.pkl"), "baseline"),
            (os.path.join(SCRIPT_DIR, "latent_as_in_and_output/results/*/latent_as_in_and_output_*.pkl"), None),
        ],
        labels={
            "baseline":          "Baseline",
            "both_dim_5":        "Latent In+Out (dim=5)",
            "input_only_dim_10": "Latent Input Only (dim=10)",
        },
        colors={
            "baseline":          "#95a5a6",
            "both_dim_5":        "#e74c3c",
            "input_only_dim_10": "#3498db",
        },
        subset_top_n=None,
        # Concatenate z+d vectors so both_dim_5 matches baseline latent size
        latent_concat_keys=["both_dim_5"],
    ),

    # ── ablation_study ───────────────────────────────────────────────────────
    dict(
        name="Ablation Study",
        output_dir="figures/ablation_study",
        file_specs=[
            (os.path.join(SCRIPT_DIR, "ablation_study/results/*/baseline.pkl"), "baseline"),
            (os.path.join(SCRIPT_DIR, "ablation_study/results/*/ablation_study_*.pkl"), None),
        ],
        labels={
            "baseline":          "MLP + Latent",
            "mlp_no_taxonomy":   "MLP (no taxonomy)",
            "mlp_with_taxonomy": "MLP (with taxonomy)",
        },
        colors={
            "baseline":          "#ff7f0e",
            "mlp_no_taxonomy":   "#1f77b4",
            "mlp_with_taxonomy": "#2ca02c",
        },
        subset_top_n=None,
        latent_concat_keys=None,
    ),

    # ── loss_comparison ──────────────────────────────────────────────────────
    dict(
        name="Loss Comparison",
        output_dir="figures/loss_comparison",
        file_specs=[
            (os.path.join(SCRIPT_DIR, "loss_comparison/results/*/baseline.pkl"), "baseline"),
            (os.path.join(SCRIPT_DIR, "loss_comparison/results/*/loss_comparison_*.pkl"), None),
        ],
        labels={
            "baseline":  "Cross-Entropy",
            "logistic":  "Logistic (BCE)",
        },
        colors={
            "baseline":  "#2ecc71",
            "logistic":  "#9b59b6",
        },
        subset_top_n=None,
        latent_concat_keys=None,
    ),

    # ── optimal_K ────────────────────────────────────────────────────────────
    dict(
        name="Optimal K",
        output_dir="figures/optimal_K",
        file_specs=[
            (os.path.join(SCRIPT_DIR, "optimal_K/results/*/baseline.pkl"), "baseline"),
            (os.path.join(SCRIPT_DIR, "optimal_K/results/*/K_comparison_*.pkl"), None),
        ],
        labels={
            "K_5":      "K=5",
            "baseline": "K=25",
            "K_100":    "K=100",
            "K_500":    "K=500",
        },
        colors={
            # Brown → amber → orange → yellow gradient; K=25 sits mid-range
            "K_5":      "#5c2f00",
            "baseline": "#bf7a1a",
            "K_100":    "#e89f2e",
            "K_500":    "#f5c842",
        },
        subset_top_n=None,
        latent_concat_keys=None,
    ),

    # ── preprocessing ────────────────────────────────────────────────────────
    dict(
        name="Preprocessing",
        output_dir="figures/preprocessing",
        file_specs=[
            (os.path.join(SCRIPT_DIR, "preprocessing/results/*/preprocessing_*.pkl"), None),
            (os.path.join(SCRIPT_DIR, "preprocessing/results/*/baseline.pkl"), "baseline"),
        ],
        labels={
            "baseline":    "Original (raw counts)",
            "normalized":  "Normalized Only",
            "logarithm":   "Logarithm Only",
        },
        colors={
            "baseline":    "#ff7f0e",
            "normalized":  "#1f77b4",
            "logarithm":   "#2ca02c",
        },
        subset_top_n=None,
        latent_concat_keys=None,
    ),

    # ── dimensionality_increase/gating_function ──────────────────────────────
    dict(
        name="Dimensionality Increase: Gating Function",
        output_dir="figures/dimensionality_gating",
        file_specs=[
            (os.path.join(SCRIPT_DIR, "dimensionality_increase/gating_function/results/*/gating_comparison_*.pkl"), None),
        ],
        labels={
            "exp":          "Exponential",
            "scaled_exp":   "Scaled Exponential",
            "additive":     "Additive (1+h)",
            "softplus":     "Softplus",
            "tanh":         "Tanh",
            "sigmoid":      "Sigmoid",
            "dot_product":  "Dot Product",
        },
        colors={
            "exp":          "#e74c3c",
            "scaled_exp":   "#e67e22",
            "additive":     "#f39c12",
            "softplus":     "#2ecc71",
            "tanh":         "#3498db",
            "sigmoid":      "#9b59b6",
            "dot_product":  "#1abc9c",
        },
        subset_top_n=None,
        latent_concat_keys=None,
    ),

    # ── dimensionality_increase/vector_size ──────────────────────────────────
    dict(
        name="Dimensionality Increase: Vector Size",
        output_dir="figures/dimensionality_vector",
        file_specs=[
            (os.path.join(SCRIPT_DIR, "dimensionality_increase/vector_size/results/*/dimensionality_analysis_*.pkl"), None),
        ],
        labels={
            "dim_1":  "Dim=1",
            "dim_2":  "Dim=2",
            "dim_5":  "Dim=5",
            "dim_10": "Dim=10",
            "dim_20": "Dim=20",
            "dim_50": "Dim=50",
        },
        colors={
            "dim_1":  "#95a5a6",
            "dim_2":  "#824e05",
            "dim_5":  "#e74c3c",
            "dim_10": "#e67e22",
            "dim_20": "#f39c12",
            "dim_50": "#f1c40f",
        },
        subset_top_n=None,
        latent_concat_keys=None,
    ),
]

# Map short name → config for CLI filtering
_ANALYSIS_MAP = {cfg["output_dir"].split("/")[-1]: cfg for cfg in ANALYSES}
_ANALYSIS_MAP.update({cfg["name"].lower().replace(" ", "_"): cfg for cfg in ANALYSES})


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_analysis(cfg: Dict[str, Any]) -> bool:
    name = cfg["name"]
    output_dir = os.path.join(SCRIPT_DIR, cfg["output_dir"])
    file_specs = cfg["file_specs"]
    labels = cfg["labels"]
    colors = cfg["colors"]
    subset_top_n = cfg.get("subset_top_n")
    latent_concat_keys = cfg.get("latent_concat_keys")

    log.info("\n%s\n%s\n%s", "=" * 60, name, "=" * 60)

    merged = _load_merged(file_specs, latent_concat_keys=latent_concat_keys)
    if merged is None:
        log.info("  (Skipping — no result files found)")
        return False

    log.info("  Loaded %d model(s): %s", len(merged), list(merged.keys()))

    print_comparison(merged, labels=labels, title=name.upper())
    create_all_visualizations(
        merged,
        output_dir,
        colors=colors,
        labels=labels,
        title=name,
        subset_top_n=subset_top_n,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analyses", nargs="*",
        help="Names of analyses to run (default: all). "
             "Use the output_dir suffix, e.g. 'optimal_K', 'dimensionality_gating'.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.analyses:
        targets = []
        for a in args.analyses:
            cfg = _ANALYSIS_MAP.get(a)
            if cfg is None:
                log.error("Unknown analysis: %s. Valid names: %s", a, list(_ANALYSIS_MAP.keys()))
                sys.exit(1)
            if cfg not in targets:
                targets.append(cfg)
    else:
        targets = ANALYSES

    n_ok = n_skip = 0
    for cfg in targets:
        ok = run_analysis(cfg)
        if ok:
            n_ok += 1
        else:
            n_skip += 1

    log.info("\n✅ Done. Ran %d analysis(es), skipped %d (no results).", n_ok, n_skip)


if __name__ == "__main__":
    main()
