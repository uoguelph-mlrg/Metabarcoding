from __future__ import annotations

import argparse
import json
import logging as log
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _set_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _load_thresholds(results_dir: str) -> Dict[str, Any]:
    with open(os.path.join(results_dir, "rule_thresholds.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def make_plots(results_dir: str) -> None:
    _set_style()

    sample_metrics = pd.read_csv(os.path.join(results_dir, "sample_qc_metrics.csv"))
    audit = pd.read_csv(os.path.join(results_dir, "sample_rule_audit.csv"))
    summary = pd.read_json(os.path.join(results_dir, "curation_summary.json"), typ="series")
    thresholds = _load_thresholds(results_dir)

    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    # 1) Total reads distribution with thresholds
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(sample_metrics, x="total_reads_per_sample", bins=80, ax=ax)
    ax.set_xscale("log")
    ax.set_title("Total Reads Per Sample (Log Scale)")
    ax.set_xlabel("Total Reads Per Sample")
    ax.axvline(thresholds["low_reads_flag"], color="orange", linestyle="--", label="Flag")
    ax.axvline(thresholds["low_reads_fail"], color="red", linestyle="--", label="Fail")
    ax.legend()
    fig.savefig(os.path.join(results_dir, "figures", "reads_distribution.png"))
    plt.close(fig)

    # 2) Replicate fraction distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(sample_metrics, x="replicate_fraction", bins=30, ax=ax)
    ax.set_title("Replicate Coverage Distribution")
    ax.set_xlabel("Replicate Fraction")
    ax.axvline(
        thresholds["min_repl_fraction_flag"],
        color="orange",
        linestyle="--",
        label="Flag",
    )
    ax.axvline(
        thresholds["min_repl_fraction_fail"],
        color="red",
        linestyle="--",
        label="Fail",
    )
    ax.legend()
    fig.savefig(os.path.join(results_dir, "figures", "replicate_fraction_distribution.png"))
    plt.close(fig)

    # 3) Taxonomy and metadata missingness
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(sample_metrics, x="taxonomy_missing_fraction", bins=40, ax=axes[0])
    axes[0].set_title("Taxonomy Missing Fraction")
    axes[0].axvline(thresholds["max_tax_missing_flag"], color="orange", linestyle="--")
    axes[0].axvline(thresholds["max_tax_missing_fail"], color="red", linestyle="--")

    sns.histplot(sample_metrics, x="metadata_missing_fraction", bins=40, ax=axes[1])
    axes[1].set_title("Metadata Missing Fraction")
    axes[1].axvline(thresholds["max_meta_missing_flag"], color="orange", linestyle="--")
    axes[1].axvline(thresholds["max_meta_missing_fail"], color="red", linestyle="--")
    fig.savefig(os.path.join(results_dir, "figures", "missingness_distributions.png"))
    plt.close(fig)

    # 4) Decision counts
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["PASS", "FLAG", "FAIL"]
    counts = sample_metrics["overall_decision"].value_counts().reindex(order, fill_value=0)
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=["#2ca02c", "#ff7f0e", "#d62728"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Sample-Level Curation Decisions")
    ax.set_ylabel("Count")
    fig.savefig(os.path.join(results_dir, "figures", "decision_counts.png"))
    plt.close(fig)

    # 5) Rule-level flag/fail rates
    rule_rates = (
        audit.assign(flag_or_fail=audit["decision"].isin(["FLAG", "FAIL"]).astype(int))
        .groupby("rule")["flag_or_fail"]
        .mean()
        .reset_index(name="flag_or_fail")
        .sort_values(by="flag_or_fail", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=rule_rates, y="rule", x="flag_or_fail", color="#4c78a8", ax=ax)
    ax.set_title("Rule-Level Flag/Fail Rate")
    ax.set_xlabel("Rate")
    ax.set_ylabel("Rule")
    fig.savefig(os.path.join(results_dir, "figures", "rule_flag_fail_rates.png"))
    plt.close(fig)

    # 6) Shannon vs total reads scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    decision_palette = {"PASS": "#2ca02c", "FLAG": "#ff7f0e", "FAIL": "#d62728"}
    sns.scatterplot(
        data=sample_metrics,
        x="total_reads_per_sample",
        y="shannon_occurrences",
        hue="overall_decision",
        palette=decision_palette,
        alpha=0.8,
        s=40,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_title("Shannon Diversity vs Total Reads")
    fig.savefig(os.path.join(results_dir, "figures", "shannon_vs_reads.png"))
    plt.close(fig)

    # 7) Contamination burden (if available)
    if "contamination_burden" in sample_metrics.columns and sample_metrics[
        "contamination_burden"
    ].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(sample_metrics, x="contamination_burden", bins=40, ax=ax)
        ax.set_title("Contamination Burden Distribution")
        ax.axvline(
            thresholds["max_contam_burden_flag"],
            color="orange",
            linestyle="--",
            label="Flag",
        )
        ax.axvline(
            thresholds["max_contam_burden_fail"],
            color="red",
            linestyle="--",
            label="Fail",
        )
        ax.legend()
        fig.savefig(os.path.join(results_dir, "figures", "contamination_burden.png"))
        plt.close(fig)

    # 8) Compact one-page diagnostics panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(sample_metrics, x="total_reads_per_sample", bins=60, ax=axes[0, 0])
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Reads (log scale)")

    sns.histplot(sample_metrics, x="bins_per_sample", bins=60, ax=axes[0, 1])
    axes[0, 1].set_title("BINs per Sample")

    sns.histplot(sample_metrics, x="zero_fraction_occurrences", bins=40, ax=axes[1, 0])
    axes[1, 0].set_title("Zero Fraction (Occurrences)")

    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=["#2ca02c", "#ff7f0e", "#d62728"],
        legend=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Decision Counts")
    fig.suptitle(
        f"Data Curation Diagnostics | Profile={summary['profile']} | n={summary['n_samples']}",
        fontsize=14,
    )
    fig.savefig(os.path.join(results_dir, "figures", "diagnostics_panel.png"))
    plt.close(fig)

    log.info("Visualization files written under: %s", os.path.join(results_dir, "figures"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate curation diagnostics figures.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path containing pipeline output CSV/JSON files.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    make_plots(args.results_dir)


if __name__ == "__main__":
    main()
