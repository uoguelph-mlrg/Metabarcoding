from __future__ import annotations

import argparse
import logging as log
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_curation_pipeline import DataCurationPipeline
from visualizations import make_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full data curation analyses, diagnostics, and visualizations."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="../../data/data_merged.csv",
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results",
        help="Root directory where profile run folders are created.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="moderate",
        choices=["conservative", "moderate", "strict"],
        help="Curation strictness profile.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional path to criteria profile JSON.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    run_name = f"{args.profile}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(args.output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    pipeline = DataCurationPipeline(
        input_path=args.input_path,
        output_dir=output_dir,
        profile_name=args.profile,
        config_path=args.config_path,
    )
    pipeline.run()
    make_plots(output_dir)

    log.info("Full curation run completed: %s", output_dir)


if __name__ == "__main__":
    main()
