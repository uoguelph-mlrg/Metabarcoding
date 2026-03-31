# Data Curation Suite

This folder contains a reproducible curation workflow for metabarcoding datasets before touching `src` model code.

## What It Produces

Running the suite generates:

- Sample-level QC metrics (`sample_qc_metrics.csv`)
- Bin-level QC metrics (`bin_qc_metrics.csv`)
- Rule-level audit trail (`sample_rule_audit.csv`)
- Final sample decisions (`sample_decisions.csv`)
- Curated subsets:
  - `curated_pass_only.csv` (strict curated)
  - `curated_pass_plus_flag.csv` (recommended for sensitivity analyses)
  - `flagged_or_failed_records.csv`
- Run metadata:
  - `rule_thresholds.json`
  - `curation_summary.json`
- Figures under `figures/`:
  - reads distribution with thresholds
  - replicate coverage distribution
  - missingness distributions
  - decision counts
  - rule flag/fail rates
  - shannon vs reads
  - contamination burden (if controls exist)
  - diagnostics panel

## Rule Methodology

Rules are objective and profile-driven:

1. Dynamic thresholds from empirical quantiles (conservative/moderate/strict profiles)
2. Hard fail floors for severe technical failures
3. Absolute thresholds for missingness, replicate support, coordinate validity, duplicates
4. Optional contamination diagnostics based on control sample detection

Each rule emits `PASS`, `FLAG`, or `FAIL` with rationale in the audit table.

## Run

From this folder:

```bash
python run_data_curation.py \
  --input_path ../../data/ecuador_training_data.csv \
  --profile moderate \
  --output_root results
```

## Profile Configuration

Edit `criteria_profiles.json` to:

- Adjust quantile thresholds and hard floors
- Tune contamination control patterns
- Change column mappings for other datasets
- Switch metadata/taxonomy fields used in diagnostics

## Recommended Workflow

1. Run `moderate` profile first and inspect `curation_summary.json`.
2. Compare conservative/strict to quantify sensitivity of retention and diagnostics.
3. Use `curated_pass_plus_flag.csv` for initial model sensitivity checks.
4. Use `curated_pass_only.csv` for strict analyses.
5. Promote validated rules into `src` only after retention/performance/ecological sanity checks.
