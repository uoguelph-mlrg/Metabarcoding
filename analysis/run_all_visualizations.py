#!/usr/bin/env python3
"""
Run visualizations for all analyses (or a specified subset).

Result pkl files are discovered from the registry:
  - Generic analyses (run_script=None):  results/{analysis_name}/*/
  - Legacy analyses (run_script set):    {script_dir}/results/*/

The baseline is auto-discovered from results/baseline/baseline/*/results_baseline_*.pkl
(the latest run is used) and included in all analyses where include_baseline=True.

Labels, colors, subset_top_n, latent_concat_keys, and baseline display config
all come from analyses.py (single source of truth).

Usage
-----
    python run_all_visualizations.py               # all analyses
    python run_all_visualizations.py optimal_K barcodebert
    python run_all_visualizations.py --baseline /path/to/results_baseline.pkl optimal_K
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
for _p in (SCRIPT_DIR, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyses import REGISTRY, Analysis  # noqa: E402
from visualize_results import (  # noqa: E402
    _is_model_payload,
    _validate_model_payload,
    create_all_visualizations,
    print_comparison,
)


# ---------------------------------------------------------------------------
# Baseline discovery
# ---------------------------------------------------------------------------

def _find_latest_baseline_pkl() -> Optional[str]:
    """Return path to the most recent results_baseline_*.pkl, or None."""
    baseline_root = os.path.join(SCRIPT_DIR, "results", "baseline", "baseline")
    dirs = sorted(glob.glob(os.path.join(baseline_root, "*/")), reverse=True)
    for d in dirs:
        pkls = sorted(glob.glob(os.path.join(d, "results_baseline_*.pkl")), reverse=True)
        if pkls:
            return pkls[0]
    return None


# ---------------------------------------------------------------------------
# Loader helper
# ---------------------------------------------------------------------------

def _load_merged(
    file_specs: List[Tuple[str, Optional[str]]],
    latent_concat_keys: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
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
                    for k, v in raw.items():
                        _validate_model_payload(k, v)
                        merged[k] = v
                    continue

            if latent_concat_keys and key in latent_concat_keys:
                z_raw = payload.get("latent_z_vector")
                d_raw = payload.get("latent_d_vector")
                z = np.asarray(z_raw) if z_raw is not None else None
                d = np.asarray(d_raw) if d_raw is not None else None
                if z is not None and z.ndim >= 1 and d is not None and d.ndim >= 1:
                    payload["latent_vector"] = np.concatenate([z, d], axis=1)

            _validate_model_payload(key, payload)
            if key in merged:
                log.warning("Duplicate model key '%s' from %s — skipping", key, fpath)
                continue
            merged[key] = payload

    return merged if merged else None


# ---------------------------------------------------------------------------
# File spec resolution
# ---------------------------------------------------------------------------

def _file_specs_for(analysis: Analysis) -> List[Tuple[str, Optional[str]]]:
    """Return (glob_pattern, key_override) pairs for an analysis's result pkls."""
    name = analysis.name
    # All analyses (generic and legacy) save to results/{name}/{run_id}/{name}_*.pkl
    pattern = os.path.join(SCRIPT_DIR, "results", name, "*", f"{name}_*.pkl")
    return [(pattern, None)]


def _output_dir_for(analysis: Analysis) -> str:
    """Return the figures output directory for an analysis."""
    # barcodebert is stored under BarcodeBERT (uppercase) for historical reasons
    if analysis.name == "barcodebert":
        return os.path.join(SCRIPT_DIR, "figures", "BarcodeBERT")
    return os.path.join(SCRIPT_DIR, "figures", analysis.name)


def _labels_and_colors(analysis: Analysis) -> Tuple[Dict[str, str], Dict[str, str]]:
    labels = {v.name: v.label for v in analysis.variants}
    colors = {v.name: v.color for v in analysis.variants}
    return labels, colors


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(analysis_key: str, baseline_pkl: Optional[str] = None) -> bool:
    if analysis_key not in REGISTRY:
        log.error("Unknown analysis key: %s", analysis_key)
        return False

    analysis = REGISTRY[analysis_key]
    variant_labels, variant_colors = _labels_and_colors(analysis)
    file_specs = _file_specs_for(analysis)

    if baseline_pkl and analysis.include_baseline:
        # Baseline first so it appears first in all plots and legends
        labels = {"baseline": analysis.baseline_label, **variant_labels}
        colors = {"baseline": analysis.baseline_color, **variant_colors}
        file_specs = [(baseline_pkl, "baseline")] + file_specs
    else:
        labels = variant_labels
        colors = variant_colors

    output_dir = _output_dir_for(analysis)
    name = analysis.name.replace("_", " ").title()

    log.info("\n%s\n%s\n%s", "=" * 60, name, "=" * 60)
    merged = _load_merged(file_specs, latent_concat_keys=analysis.latent_concat_keys)
    if merged is None:
        log.info("  (Skipping — no result files found)")
        return False

    log.info("  Loaded %d model(s): %s", len(merged), list(merged.keys()))
    print_comparison(merged, labels=labels, title=name.upper())
    create_all_visualizations(
        merged, output_dir,
        colors=colors, labels=labels, title=name,
        subset_top_n=analysis.subset_top_n,
    )
    return True


def main() -> None:
    all_keys = sorted(REGISTRY)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analyses", nargs="*",
        help=f"Analysis keys to visualize (default: all). Available: {all_keys}",
    )
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to a baseline pkl to include in analyses")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    targets = args.analyses if args.analyses else all_keys

    unknown = [a for a in targets if a not in REGISTRY]
    if unknown:
        log.error("Unknown analysis key(s): %s. Valid: %s", unknown, all_keys)
        sys.exit(1)

    # Auto-discover baseline if not provided
    baseline_pkl = args.baseline
    if baseline_pkl is None:
        baseline_pkl = _find_latest_baseline_pkl()
        if baseline_pkl:
            log.info("Auto-discovered baseline: %s", baseline_pkl)
        else:
            log.info("No baseline found in results/baseline/baseline/; running without baseline")

    n_ok = n_skip = 0
    for key in targets:
        ok = run_one(key, baseline_pkl=baseline_pkl)
        if ok:
            n_ok += 1
        else:
            n_skip += 1

    log.info("\nDone. Ran %d analysis(es), skipped %d (no results).", n_ok, n_skip)


if __name__ == "__main__":
    main()
