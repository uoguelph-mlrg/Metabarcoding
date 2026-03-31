from __future__ import annotations

import argparse
import hashlib
import json
import logging as log
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def _robust_z(values: pd.Series) -> pd.Series:
    arr = _safe_float(values).to_numpy(dtype=float)
    med = np.nanmedian(arr)
    mad = _mad(arr)
    if mad == 0:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    z = 0.6745 * (arr - med) / mad
    z = np.where(np.isfinite(z), z, 0.0)
    return pd.Series(z, index=values.index)


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _normalized_decision(decisions: Iterable[str]) -> str:
    vals = set(decisions)
    if "FAIL" in vals:
        return "FAIL"
    if "FLAG" in vals:
        return "FLAG"
    return "PASS"


@dataclass
class RuleThresholds:
    profile_name: str
    low_reads_flag: float
    low_reads_fail: float
    low_bins_flag: float
    low_bins_fail: float
    low_shannon_flag: float
    high_shannon_flag: float
    low_shannon_fail: float
    high_shannon_fail: float
    min_repl_fraction_flag: float
    min_repl_fraction_fail: float
    max_tax_missing_flag: float
    max_tax_missing_fail: float
    max_meta_missing_flag: float
    max_meta_missing_fail: float
    max_invalid_coord_flag: float
    max_invalid_coord_fail: float
    max_duplicate_rows_flag: int
    max_duplicate_rows_fail: int
    max_contam_burden_flag: float
    max_contam_burden_fail: float
    outlier_z_flag: float
    outlier_z_fail: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "low_reads_flag": self.low_reads_flag,
            "low_reads_fail": self.low_reads_fail,
            "low_bins_flag": self.low_bins_flag,
            "low_bins_fail": self.low_bins_fail,
            "low_shannon_flag": self.low_shannon_flag,
            "high_shannon_flag": self.high_shannon_flag,
            "low_shannon_fail": self.low_shannon_fail,
            "high_shannon_fail": self.high_shannon_fail,
            "min_repl_fraction_flag": self.min_repl_fraction_flag,
            "min_repl_fraction_fail": self.min_repl_fraction_fail,
            "max_tax_missing_flag": self.max_tax_missing_flag,
            "max_tax_missing_fail": self.max_tax_missing_fail,
            "max_meta_missing_flag": self.max_meta_missing_flag,
            "max_meta_missing_fail": self.max_meta_missing_fail,
            "max_invalid_coord_flag": self.max_invalid_coord_flag,
            "max_invalid_coord_fail": self.max_invalid_coord_fail,
            "max_duplicate_rows_flag": self.max_duplicate_rows_flag,
            "max_duplicate_rows_fail": self.max_duplicate_rows_fail,
            "max_contam_burden_flag": self.max_contam_burden_flag,
            "max_contam_burden_fail": self.max_contam_burden_fail,
            "outlier_z_flag": self.outlier_z_flag,
            "outlier_z_fail": self.outlier_z_fail,
        }


class DataCurationPipeline:
    def __init__(
        self,
        input_path: str,
        output_dir: str,
        profile_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.input_path = input_path
        self.output_dir = output_dir
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "criteria_profiles.json"
        )

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = json.load(f)

        default_profile = self.config["default_profile"]
        self.profile_name = profile_name or default_profile
        if self.profile_name not in self.config["profiles"]:
            raise ValueError(
                f"Unknown profile '{self.profile_name}'. Available: "
                f"{list(self.config['profiles'].keys())}"
            )
        self.profile = self.config["profiles"][self.profile_name]
        self.colmap = self.config["required_columns"]
        self.taxonomy_cols = self.config["taxonomy_columns"]
        self.metadata_cols = self.config["metadata_columns"]

        os.makedirs(self.output_dir, exist_ok=True)

    def _resolve_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        resolved: Dict[str, Optional[str]] = {}
        for key, col_name in self.colmap.items():
            resolved[key] = col_name if col_name in df.columns else None
        return resolved

    def _parse_dates(self, s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, errors="coerce")

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(df)

        reads_col = cols["reads"]
        occ_col = cols["occurrences"]
        lat_col = cols["latitude"]
        lon_col = cols["longitude"]
        start_col = cols["collection_start"]
        end_col = cols["collection_end"]

        if reads_col:
            df[reads_col] = _safe_float(df[reads_col]).fillna(0.0)
        if occ_col:
            df[occ_col] = _safe_float(df[occ_col]).fillna(0.0)
        if lat_col:
            df[lat_col] = _safe_float(df[lat_col])
        if lon_col:
            df[lon_col] = _safe_float(df[lon_col])
        if start_col:
            df[start_col] = self._parse_dates(df[start_col])
        if end_col:
            df[end_col] = self._parse_dates(df[end_col])

        return df

    def _sample_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(df)
        sample_col = cols["sample"]
        bin_col = cols["bin"]

        if not sample_col or not bin_col:
            raise ValueError("Input data requires sample and bin columns.")

        reads_col = cols["reads"]
        occ_col = cols["occurrences"]
        repl_frac_col = cols["replicate_fraction"]
        repl_w_reads_col = cols["replicates_with_reads"]
        repl_tot_col = cols["replicates_total"]
        lat_col = cols["latitude"]
        lon_col = cols["longitude"]
        start_col = cols["collection_start"]
        end_col = cols["collection_end"]

        world = self.config["world_coordinate_bounds"]

        group = df.groupby(sample_col, sort=False)
        sample_df = pd.DataFrame(index=group.size().index)
        sample_df.index.name = sample_col
        sample_df["records_per_sample"] = group.size()
        sample_df["bins_per_sample"] = group[bin_col].nunique()

        if reads_col:
            sample_df["total_reads_per_sample"] = group[reads_col].sum()
            sample_df["median_reads_per_record"] = group[reads_col].median()
            sample_df["reads_present_fraction"] = group[reads_col].apply(
                lambda x: float((x > 0).mean())
            )
        else:
            sample_df["total_reads_per_sample"] = np.nan
            sample_df["median_reads_per_record"] = np.nan
            sample_df["reads_present_fraction"] = np.nan

        if occ_col:
            sample_df["occurrences_present_fraction"] = group[occ_col].apply(
                lambda x: float((x > 0).mean())
            )
            sample_df["zero_fraction_occurrences"] = 1.0 - sample_df[
                "occurrences_present_fraction"
            ]
        else:
            sample_df["occurrences_present_fraction"] = np.nan
            sample_df["zero_fraction_occurrences"] = np.nan

        if repl_frac_col and repl_frac_col in df.columns:
            sample_df["replicate_fraction"] = group[repl_frac_col].median()
        elif repl_w_reads_col and repl_tot_col:
            repl_w = group[repl_w_reads_col].median()
            repl_t = group[repl_tot_col].median().replace(0, np.nan)
            sample_df["replicate_fraction"] = repl_w / repl_t
        else:
            sample_df["replicate_fraction"] = np.nan

        available_tax_cols = [c for c in self.taxonomy_cols if c in df.columns]
        if available_tax_cols:
            taxonomy_missing = df[available_tax_cols].isna().mean(axis=1)
            sample_df["taxonomy_missing_fraction"] = group.apply(
                lambda x: float(taxonomy_missing.loc[x.index].mean())
            )
        else:
            sample_df["taxonomy_missing_fraction"] = np.nan

        available_meta_cols = [c for c in self.metadata_cols if c in df.columns]
        if available_meta_cols:
            metadata_missing = df[available_meta_cols].isna().mean(axis=1)
            sample_df["metadata_missing_fraction"] = group.apply(
                lambda x: float(metadata_missing.loc[x.index].mean())
            )
        else:
            sample_df["metadata_missing_fraction"] = np.nan

        if lat_col and lon_col:
            invalid_coord = (
                df[lat_col].isna()
                | df[lon_col].isna()
                | (df[lat_col] < world["lat_min"])
                | (df[lat_col] > world["lat_max"])
                | (df[lon_col] < world["lon_min"])
                | (df[lon_col] > world["lon_max"])
            )
            sample_df["invalid_coord_fraction"] = group.apply(
                lambda x: float(invalid_coord.loc[x.index].mean())
            )
        else:
            sample_df["invalid_coord_fraction"] = np.nan

        if start_col and end_col:
            window_days = (df[end_col] - df[start_col]).dt.days
            sample_df["collection_window_days_median"] = group.apply(
                lambda x: float(np.nanmedian(window_days.loc[x.index]))
            )
        else:
            sample_df["collection_window_days_median"] = np.nan

        duplicated = df.duplicated(subset=[sample_col, bin_col], keep=False)
        sample_df["duplicate_bin_rows"] = group.apply(
            lambda x: int(duplicated.loc[x.index].sum())
        )

        if occ_col:
            def _shannon(vals: pd.Series) -> float:
                arr = _safe_float(vals).to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return float("nan")
                arr = np.clip(arr, 0.0, None)
                s = arr.sum()
                if s <= 0:
                    return 0.0
                p = arr / s
                p = p[p > 0]
                if p.size == 0:
                    return 0.0
                return float(-np.sum(p * np.log(p)))

            sample_df["shannon_occurrences"] = group[occ_col].apply(_shannon)
        else:
            sample_df["shannon_occurrences"] = np.nan

        if reads_col:
            def _shannon_reads(vals: pd.Series) -> float:
                arr = _safe_float(vals).to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return float("nan")
                arr = np.clip(arr, 0.0, None)
                s = arr.sum()
                if s <= 0:
                    return 0.0
                p = arr / s
                p = p[p > 0]
                if p.size == 0:
                    return 0.0
                return float(-np.sum(p * np.log(p)))

            sample_df["shannon_reads"] = group[reads_col].apply(_shannon_reads)
        else:
            sample_df["shannon_reads"] = np.nan

        sample_df["total_reads_robust_z"] = _robust_z(sample_df["total_reads_per_sample"])
        sample_df = sample_df.reset_index()
        return sample_df

    def _bin_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(df)
        sample_col = cols["sample"]
        bin_col = cols["bin"]
        reads_col = cols["reads"]
        occ_col = cols["occurrences"]

        if not sample_col or not bin_col:
            raise ValueError("Input data requires sample and bin columns.")

        group = df.groupby(bin_col, sort=False)
        out = pd.DataFrame(index=group.size().index)
        out.index.name = bin_col
        out["records_per_bin"] = group.size()
        out["samples_per_bin"] = group[sample_col].nunique()
        if reads_col:
            out["total_reads"] = group[reads_col].sum()
            out["median_reads"] = group[reads_col].median()
        else:
            out["total_reads"] = np.nan
            out["median_reads"] = np.nan
        if occ_col:
            out["nonzero_occurrence_fraction"] = group[occ_col].apply(
                lambda x: float((x > 0).mean())
            )
        else:
            out["nonzero_occurrence_fraction"] = np.nan
        return out.reset_index()

    def _contamination_stats(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
        cols = self._resolve_columns(df)
        sample_col = cols["sample"]
        bin_col = cols["bin"]
        plate_col = cols["plate"]
        occ_col = cols["occurrences"]

        if occ_col is None:
            return pd.DataFrame(), pd.DataFrame(), False

        control_cfg = self.config["contamination"]
        sample_re = re.compile(control_cfg["control_sample_regex"], re.IGNORECASE)
        plate_re = re.compile(control_cfg["control_plate_regex"], re.IGNORECASE)

        sample_is_ctrl = df[sample_col].astype(str).str.contains(sample_re)
        if plate_col and plate_col in df.columns:
            plate_is_ctrl = df[plate_col].astype(str).str.contains(plate_re)
            is_control = sample_is_ctrl | plate_is_ctrl
        else:
            is_control = sample_is_ctrl

        if is_control.sum() == 0:
            return pd.DataFrame(), pd.DataFrame(), False

        tmp = df[[sample_col, bin_col, occ_col]].copy()
        tmp["is_control"] = is_control.values
        tmp["is_present"] = tmp[occ_col] > 0

        control_n = tmp.loc[tmp["is_control"], sample_col].nunique()
        noncontrol_n = tmp.loc[~tmp["is_control"], sample_col].nunique()

        by_bin_ctrl = (
            tmp[tmp["is_control"]]
            .groupby(bin_col)["is_present"]
            .mean()
            .rename("control_prevalence")
        )
        by_bin_non = (
            tmp[~tmp["is_control"]]
            .groupby(bin_col)["is_present"]
            .mean()
            .rename("noncontrol_prevalence")
        )

        bin_stats = pd.concat([by_bin_ctrl, by_bin_non], axis=1).fillna(0.0)
        eps = 1e-9
        bin_stats["control_to_noncontrol_ratio"] = (
            (bin_stats["control_prevalence"] + eps)
            / (bin_stats["noncontrol_prevalence"] + eps)
        )
        bin_stats["control_sample_count"] = control_n
        bin_stats["noncontrol_sample_count"] = noncontrol_n
        bin_stats = bin_stats.reset_index()

        tmp = tmp.merge(bin_stats, on=bin_col, how="left")
        tmp["contamination_flag_record"] = (
            ~tmp["is_control"]
            & tmp["is_present"]
            & (
                tmp["control_prevalence"]
                >= control_cfg["min_control_prevalence_for_flag"]
            )
            & (
                tmp["control_to_noncontrol_ratio"]
                >= control_cfg["control_to_noncontrol_prevalence_ratio_flag"]
            )
        )
        tmp["contamination_fail_record"] = (
            ~tmp["is_control"]
            & tmp["is_present"]
            & (
                tmp["control_prevalence"]
                >= control_cfg["min_control_prevalence_for_fail"]
            )
            & (
                tmp["control_to_noncontrol_ratio"]
                >= control_cfg["control_to_noncontrol_prevalence_ratio_fail"]
            )
        )

        sample_contam = (
            tmp.groupby(sample_col)
            .agg(
                contamination_burden=("contamination_flag_record", "mean"),
                contamination_fail_burden=("contamination_fail_record", "mean"),
                contamination_flag_records=("contamination_flag_record", "sum"),
                contamination_fail_records=("contamination_fail_record", "sum"),
                is_control=("is_control", "max"),
            )
            .reset_index()
        )

        return bin_stats, sample_contam, True

    def _build_thresholds(self, sample_df: pd.DataFrame) -> RuleThresholds:
        dyn = self.profile["dynamic_quantiles"]
        floor = self.profile["hard_floors"]
        abs_th = self.profile["absolute_thresholds"]

        reads_q = float(sample_df["total_reads_per_sample"].quantile(dyn["min_total_reads_q"]))
        bins_q = float(sample_df["bins_per_sample"].quantile(dyn["min_bins_per_sample_q"]))
        shannon_q_low = float(sample_df["shannon_occurrences"].quantile(dyn["shannon_low_q"]))
        shannon_q_high = float(sample_df["shannon_occurrences"].quantile(dyn["shannon_high_q"]))

        low_reads_flag = max(reads_q, floor["min_total_reads_fail"])
        low_bins_flag = max(bins_q, floor["min_bins_per_sample_fail"])

        return RuleThresholds(
            profile_name=self.profile_name,
            low_reads_flag=float(low_reads_flag),
            low_reads_fail=float(floor["min_total_reads_fail"]),
            low_bins_flag=float(low_bins_flag),
            low_bins_fail=float(floor["min_bins_per_sample_fail"]),
            low_shannon_flag=float(shannon_q_low),
            high_shannon_flag=float(shannon_q_high),
            low_shannon_fail=float(min(0.0, shannon_q_low - 0.25)),
            high_shannon_fail=float(shannon_q_high + 0.25),
            min_repl_fraction_flag=float(abs_th["min_replicate_fraction_flag"]),
            min_repl_fraction_fail=float(abs_th["min_replicate_fraction_fail"]),
            max_tax_missing_flag=float(abs_th["max_taxonomy_missing_fraction_flag"]),
            max_tax_missing_fail=float(abs_th["max_taxonomy_missing_fraction_fail"]),
            max_meta_missing_flag=float(abs_th["max_metadata_missing_fraction_flag"]),
            max_meta_missing_fail=float(abs_th["max_metadata_missing_fraction_fail"]),
            max_invalid_coord_flag=float(abs_th["max_invalid_coord_fraction_flag"]),
            max_invalid_coord_fail=float(abs_th["max_invalid_coord_fraction_fail"]),
            max_duplicate_rows_flag=int(abs_th["max_duplicate_bin_rows_flag"]),
            max_duplicate_rows_fail=int(abs_th["max_duplicate_bin_rows_fail"]),
            max_contam_burden_flag=float(abs_th["max_contamination_burden_flag"]),
            max_contam_burden_fail=float(abs_th["max_contamination_burden_fail"]),
            outlier_z_flag=float(abs_th["outlier_robust_z_flag"]),
            outlier_z_fail=float(abs_th["outlier_robust_z_fail"]),
        )

    @staticmethod
    def _rule_direction(
        value: float,
        flag_threshold: float,
        fail_threshold: float,
        mode: str,
    ) -> str:
        if not np.isfinite(value):
            return "FLAG"
        if mode == "low_bad":
            if value < fail_threshold:
                return "FAIL"
            if value < flag_threshold:
                return "FLAG"
            return "PASS"
        if mode == "high_bad":
            if value > fail_threshold:
                return "FAIL"
            if value > flag_threshold:
                return "FLAG"
            return "PASS"
        if mode == "abs_high_bad":
            av = abs(value)
            if av > fail_threshold:
                return "FAIL"
            if av > flag_threshold:
                return "FLAG"
            return "PASS"
        raise ValueError(f"Unknown rule mode: {mode}")

    def _apply_rules(
        self,
        sample_df: pd.DataFrame,
        thresholds: RuleThresholds,
        contamination_evaluated: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        records: List[Dict[str, Any]] = []

        for _, row in sample_df.iterrows():
            sid = row[self.colmap["sample"]]

            rule_specs = [
                (
                    "low_total_reads",
                    float(row.get("total_reads_per_sample", np.nan)),
                    thresholds.low_reads_flag,
                    thresholds.low_reads_fail,
                    "low_bad",
                    "Low total reads indicate potential PCR or sequencing failure.",
                ),
                (
                    "low_bins_per_sample",
                    float(row.get("bins_per_sample", np.nan)),
                    thresholds.low_bins_flag,
                    thresholds.low_bins_fail,
                    "low_bad",
                    "Very low detected BIN richness can indicate technical dropout.",
                ),
                (
                    "replicate_fraction_low",
                    float(row.get("replicate_fraction", np.nan)),
                    thresholds.min_repl_fraction_flag,
                    thresholds.min_repl_fraction_fail,
                    "low_bad",
                    "Low replicate support reduces confidence in sample abundance.",
                ),
                (
                    "taxonomy_missing_high",
                    float(row.get("taxonomy_missing_fraction", np.nan)),
                    thresholds.max_tax_missing_flag,
                    thresholds.max_tax_missing_fail,
                    "high_bad",
                    "High taxonomy missingness weakens biological interpretability.",
                ),
                (
                    "metadata_missing_high",
                    float(row.get("metadata_missing_fraction", np.nan)),
                    thresholds.max_meta_missing_flag,
                    thresholds.max_meta_missing_fail,
                    "high_bad",
                    "Missing metadata compromises downstream ecological analysis.",
                ),
                (
                    "invalid_coordinates_high",
                    float(row.get("invalid_coord_fraction", np.nan)),
                    thresholds.max_invalid_coord_flag,
                    thresholds.max_invalid_coord_fail,
                    "high_bad",
                    "Invalid geographic coordinates indicate curation or logging issues.",
                ),
                (
                    "duplicate_bin_rows",
                    float(row.get("duplicate_bin_rows", np.nan)),
                    float(thresholds.max_duplicate_rows_flag),
                    float(thresholds.max_duplicate_rows_fail),
                    "high_bad",
                    "Duplicate sample-bin rows suggest merge or preprocessing errors.",
                ),
                (
                    "total_reads_outlier",
                    float(row.get("total_reads_robust_z", np.nan)),
                    thresholds.outlier_z_flag,
                    thresholds.outlier_z_fail,
                    "abs_high_bad",
                    "Extreme depth outliers can distort comparative abundance patterns.",
                ),
            ]

            shannon_val = float(row.get("shannon_occurrences", np.nan))
            if np.isfinite(shannon_val):
                low_dec = self._rule_direction(
                    shannon_val,
                    thresholds.low_shannon_flag,
                    thresholds.low_shannon_fail,
                    "low_bad",
                )
                high_dec = self._rule_direction(
                    shannon_val,
                    thresholds.high_shannon_flag,
                    thresholds.high_shannon_fail,
                    "high_bad",
                )
                shannon_decision = _normalized_decision([low_dec, high_dec])
            else:
                shannon_decision = "FLAG"

            records.append(
                {
                    self.colmap["sample"]: sid,
                    "rule": "shannon_diversity_outlier",
                    "metric": "shannon_occurrences",
                    "value": shannon_val,
                    "flag_threshold": f"[{thresholds.low_shannon_flag:.4f}, {thresholds.high_shannon_flag:.4f}]",
                    "fail_threshold": f"[{thresholds.low_shannon_fail:.4f}, {thresholds.high_shannon_fail:.4f}]",
                    "decision": shannon_decision,
                    "rationale": "Very low or high diversity can indicate technical anomalies or contamination.",
                }
            )

            for rule_name, value, flag_t, fail_t, mode, rationale in rule_specs:
                decision = self._rule_direction(value, flag_t, fail_t, mode)
                records.append(
                    {
                        self.colmap["sample"]: sid,
                        "rule": rule_name,
                        "metric": rule_name,
                        "value": value,
                        "flag_threshold": flag_t,
                        "fail_threshold": fail_t,
                        "decision": decision,
                        "rationale": rationale,
                    }
                )

            if contamination_evaluated:
                c_val = float(row.get("contamination_burden", np.nan))
                c_dec = self._rule_direction(
                    c_val,
                    thresholds.max_contam_burden_flag,
                    thresholds.max_contam_burden_fail,
                    "high_bad",
                )
                records.append(
                    {
                        self.colmap["sample"]: sid,
                        "rule": "contamination_burden",
                        "metric": "contamination_burden",
                        "value": c_val,
                        "flag_threshold": thresholds.max_contam_burden_flag,
                        "fail_threshold": thresholds.max_contam_burden_fail,
                        "decision": c_dec,
                        "rationale": "High overlap with control-enriched BINs suggests contamination risk.",
                    }
                )

        audit_df = pd.DataFrame(records)
        decision_df = audit_df.groupby(self.colmap["sample"], as_index=False).agg(
            overall_decision=("decision", _normalized_decision)
        )
        return audit_df, decision_df

    def _summary(
        self,
        sample_df: pd.DataFrame,
        decision_df: pd.DataFrame,
        thresholds: RuleThresholds,
        contamination_evaluated: bool,
    ) -> Dict[str, Any]:
        sample_col = self.colmap["sample"]
        merged = sample_df.merge(decision_df, on=sample_col, how="left")
        if "overall_decision" not in merged.columns:
            fallback_cols = [c for c in merged.columns if c.startswith("overall_decision")]
            if fallback_cols:
                merged = merged.rename(columns={fallback_cols[0]: "overall_decision"})
        counts = merged["overall_decision"].value_counts(dropna=False).to_dict()

        pass_count = int(counts.get("PASS", 0))
        flag_count = int(counts.get("FLAG", 0))
        fail_count = int(counts.get("FAIL", 0))
        total_count = len(merged)

        return {
            "profile": self.profile_name,
            "input_path": self.input_path,
            "input_sha256": _sha256_file(self.input_path),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_samples": total_count,
            "decision_counts": {
                "PASS": pass_count,
                "FLAG": flag_count,
                "FAIL": fail_count,
            },
            "retention_rates": {
                "pass_only": (pass_count / total_count) if total_count else 0.0,
                "pass_plus_flag": ((pass_count + flag_count) / total_count)
                if total_count
                else 0.0,
            },
            "contamination_evaluated": contamination_evaluated,
            "thresholds": thresholds.to_json(),
        }

    def run(self) -> Dict[str, Any]:
        log.info("Loading data: %s", self.input_path)
        df = pd.read_csv(self.input_path)
        df = self._prepare_data(df)

        sample_df = self._sample_metrics(df)
        bin_df = self._bin_metrics(df)

        contamination_bin_df, contamination_sample_df, contamination_evaluated = (
            self._contamination_stats(df)
        )

        if contamination_evaluated:
            sample_df = sample_df.merge(
                contamination_sample_df,
                on=self.colmap["sample"],
                how="left",
            )
        else:
            sample_df["contamination_burden"] = np.nan
            sample_df["contamination_fail_burden"] = np.nan
            sample_df["contamination_flag_records"] = 0
            sample_df["contamination_fail_records"] = 0
            sample_df["is_control"] = False

        thresholds = self._build_thresholds(sample_df)
        audit_df, decision_df = self._apply_rules(
            sample_df,
            thresholds,
            contamination_evaluated,
        )

        sample_col = self.colmap["sample"]
        sample_df = sample_df.merge(decision_df, on=sample_col, how="left")

        merged_decisions = df[[sample_col]].drop_duplicates().merge(
            decision_df,
            on=sample_col,
            how="left",
        )
        df_with_decisions = df.merge(merged_decisions, on=sample_col, how="left")

        curated_pass = df_with_decisions[df_with_decisions["overall_decision"] == "PASS"]
        curated_pass_flag = df_with_decisions[
            df_with_decisions["overall_decision"].isin(["PASS", "FLAG"])
        ]
        flagged_or_failed = df_with_decisions[
            df_with_decisions["overall_decision"].isin(["FLAG", "FAIL"])
        ]

        summary = self._summary(sample_df, decision_df, thresholds, contamination_evaluated)

        sample_df.to_csv(os.path.join(self.output_dir, "sample_qc_metrics.csv"), index=False)
        bin_df.to_csv(os.path.join(self.output_dir, "bin_qc_metrics.csv"), index=False)
        audit_df.to_csv(os.path.join(self.output_dir, "sample_rule_audit.csv"), index=False)
        decision_df.to_csv(os.path.join(self.output_dir, "sample_decisions.csv"), index=False)
        curated_pass.to_csv(os.path.join(self.output_dir, "curated_pass_only.csv"), index=False)
        curated_pass_flag.to_csv(
            os.path.join(self.output_dir, "curated_pass_plus_flag.csv"), index=False
        )
        flagged_or_failed.to_csv(
            os.path.join(self.output_dir, "flagged_or_failed_records.csv"), index=False
        )

        if contamination_evaluated:
            contamination_bin_df.to_csv(
                os.path.join(self.output_dir, "contamination_bin_stats.csv"),
                index=False,
            )

        with open(
            os.path.join(self.output_dir, "rule_thresholds.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(thresholds.to_json(), f, indent=2)
        with open(
            os.path.join(self.output_dir, "curation_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

        log.info("Data curation outputs written to: %s", self.output_dir)

        return {
            "sample_metrics": sample_df,
            "bin_metrics": bin_df,
            "audit": audit_df,
            "decisions": decision_df,
            "summary": summary,
            "contamination_bin_stats": contamination_bin_df,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run objective data curation diagnostics for metabarcoding data."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="../../data/ecuador_training_data.csv",
        help="Path to input CSV dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/default_run",
        help="Output directory for diagnostics and curated datasets.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["conservative", "moderate", "strict"],
        help="Threshold profile to use.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional custom criteria JSON path.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    pipeline = DataCurationPipeline(
        input_path=args.input_path,
        output_dir=args.output_dir,
        profile_name=args.profile,
        config_path=args.config_path,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
