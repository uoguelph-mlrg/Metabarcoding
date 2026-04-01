# Metabarcoding Analysis Cluster Launching

Use a single control script to submit one independent SLURM job per subanalysis target.

## Main launcher

From `Metabarcoding/analysis`:

- List supported targets:
  - `./submit_subanalysis.sh --list-targets`
- Train baseline model once:
  - `./submit_subanalysis.sh --baseline-train`
- Submit one target:
  - `./submit_subanalysis.sh --target interpolated_latent/V4`
- Submit multiple targets in parallel:
  - `./submit_subanalysis.sh --target interpolated_latent/V4 --target location_embedding --target latent_as_input`
- Dry-run (no submit):
  - `./submit_subanalysis.sh --target location_embedding --dry-run`

## Baseline-once workflow (recommended)

**Option A: Use integrated baseline training**

Submit baseline training directly from launcher:

```bash
# Train baseline model once (from Metabarcoding/)
./submit_subanalysis.sh --baseline-train

# Wait for baseline job to complete, then find the results file
# Result will be in: ../../results/baseline/results_baseline_<run_id>.pkl

# Reuse baseline for every subanalysis visualization
./submit_subanalysis.sh --target interpolated_latent/V4 --baseline-results ../../results/baseline/results_baseline_<run_id>.pkl
./submit_subanalysis.sh --target location_embedding --target latent_as_input --baseline-results ../../results/baseline/results_baseline_<run_id>.pkl
```

**Option B: Manual baseline training and reuse**

Run one baseline model once from `Metabarcoding/`:

- `python src/train.py --model baseline`

Then reuse that single result file for every subanalysis visualization:

- `./submit_subanalysis.sh --target interpolated_latent/V4 --baseline-results ../results/baseline/results_baseline_<timestamp>.pkl`
- `./submit_subanalysis.sh --target location_embedding --target latent_as_input --baseline-results ../results/baseline/results_baseline_<timestamp>.pkl`

Optional baseline controls:

- Change baseline key shown in plots:
  - `./submit_subanalysis.sh --target loss_comparison --baseline-results <path> --baseline-key baseline`
- Disable baseline for a run:
  - `./submit_subanalysis.sh --target optimal_K --no-baseline`

## Useful overrides

- Override resources:
  - `./submit_subanalysis.sh --target location_embedding --time 16:00:00 --mem 48G --cpus 12 --gpu l40s:1`
- Disable W&B (supported targets):
  - `./submit_subanalysis.sh --target latent_as_input --no-wandb`
- Override data path for `--data_path`-based targets:
  - `./submit_subanalysis.sh --target ablation_study --data-path ../../data/ecuador_training_data.csv`
- Override labels/colors JSON passed to unified visualizer:
  - `./submit_subanalysis.sh --target interpolated_latent/V4 --labels-json '{"baseline":"Baseline","interpolated_latent":"Interpolated Latent"}' --colors-json '{"baseline":"#1f6feb","interpolated_latent":"#d97706"}'`

## Legacy wrappers

These now call the unified launcher:

- `./run_interpolated.sh`
- `./run_location.sh`

You can still pass launcher flags through wrappers, for example:

- `./run_interpolated.sh --dry-run`
- `./run_location.sh --time 14:00:00`

## Notes

- The launcher submits one job per target, so selected targets run in parallel.
- Each job runs training and unified visualization in the same SLURM job.
- Visualization auto-discovers result pickle files using target-specific filename rules.
- If `--baseline-results` is provided, the baseline is normalized and merged with target variant files for plotting.
- Generated sbatch scripts are written to `.slurm_jobs/` and logs to `slogs/`.
