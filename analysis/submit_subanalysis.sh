#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SLOG_DIR="$SCRIPT_DIR/slogs"
JOB_DIR="$SCRIPT_DIR/.slurm_jobs"

mkdir -p "$SLOG_DIR" "$JOB_DIR"

GPU="l40s:1"
CPUS="8"
MEM="32G"
QOS="normal"
TIME_OVERRIDE=""
VENV_ACTIVATE='~/barcode/bin/activate'
MODULE_LOAD='python/3.12 cuda/12.6 arrow/21.0.0 opencv/4.12.0'
NO_WANDB="0"
DRY_RUN="0"
ALL_TARGETS="0"
BASELINE_TRAIN="0"

declare -a TARGETS=()
LIST_TARGETS="0"

usage() {
  # List valid targets from the registry at usage time
  local all_keys
  all_keys=$(cd "$SCRIPT_DIR" && python -c "
import sys; sys.path.insert(0, '.')
from analyses import REGISTRY
print('  ' + '  '.join(sorted(REGISTRY)))
" 2>/dev/null || echo "(unavailable)")

  cat <<EOF
Usage:
  ./submit_subanalysis.sh --target interpolated_latent
  ./submit_subanalysis.sh --target interpolated_latent --target location_embedding
  ./submit_subanalysis.sh --all
  ./submit_subanalysis.sh --baseline-train
  ./submit_subanalysis.sh --baseline-train --target optimal_K

Options:
  --target NAME            Analysis key to submit (from analyses.py registry)
  --all                    Submit all registered analyses
  --baseline-train         Train src baseline model once (from src/train.py)
  --list-targets           Print all registered analysis keys and exit
  --no-wandb               Disable Weights & Biases logging
  --gpu SPEC               SLURM --gres gpu spec (default: l40s:1)
  --cpus N                 SLURM cpus-per-task (default: 8)
  --mem SIZE               SLURM memory (default: 32G)
  --time HH:MM:SS          Override walltime for all variant jobs
  --qos NAME               SLURM QoS (default: normal)
  --venv-activate PATH     Venv activate script (default: ~/barcode/bin/activate)
  --module-load "A B C"    Modules for module load (default: python/3.12 cuda/12.6 arrow/21.0.0 opencv/4.12.0)
  --dry-run                Print generated sbatch scripts, do not submit
  -h, --help               Show this help

Registered analyses (targets):
$all_keys
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)         TARGETS+=("$2"); shift 2 ;;
    --all)            ALL_TARGETS="1"; shift ;;
    --baseline-train) BASELINE_TRAIN="1"; shift ;;
    --list-targets)   LIST_TARGETS="1"; shift ;;
    --no-wandb)       NO_WANDB="1"; shift ;;
    --gpu)            GPU="$2"; shift 2 ;;
    --cpus)           CPUS="$2"; shift 2 ;;
    --mem)            MEM="$2"; shift 2 ;;
    --time)           TIME_OVERRIDE="$2"; shift 2 ;;
    --qos)            QOS="$2"; shift 2 ;;
    --venv-activate)  VENV_ACTIVATE="$2"; shift 2 ;;
    --module-load)    MODULE_LOAD="$2"; shift 2 ;;
    --dry-run)        DRY_RUN="1"; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$LIST_TARGETS" == "1" ]]; then
  cd "$SCRIPT_DIR" && python - <<'PY'
import sys; sys.path.insert(0, '.')
from analyses import REGISTRY
for k in sorted(REGISTRY): print(k)
PY
  exit 0
fi

if [[ "$ALL_TARGETS" == "1" && ${#TARGETS[@]} -gt 0 ]]; then
  echo "Use either --all or --target, not both." >&2; exit 1
fi

if [[ "$ALL_TARGETS" == "1" ]]; then
  mapfile -t TARGETS < <(cd "$SCRIPT_DIR" && python - <<'PY'
import sys; sys.path.insert(0, '.')
from analyses import REGISTRY
for k in sorted(REGISTRY): print(k)
PY
  )
fi

if [[ ${#TARGETS[@]} -eq 0 && "$BASELINE_TRAIN" == "0" ]]; then
  echo "No targets specified. Use --target NAME or --all." >&2; usage; exit 1
fi

if ! [[ "$CPUS" =~ ^[0-9]+$ ]] || (( CPUS > 16 )); then
  echo "Invalid --cpus value: $CPUS (must be integer ≤ 16)" >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Shared SLURM header generator
# ---------------------------------------------------------------------------

_slurm_header() {
  local job_name="$1" walltime="$2" out_pattern="$3"
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --gres=gpu:$GPU
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$walltime
#SBATCH --job-name=$job_name
#SBATCH --output=$out_pattern
#SBATCH --qos=$QOS
#SBATCH --open-mode=append

set -euo pipefail

export OMP_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export NUMEXPR_NUM_THREADS=$CPUS

module load $MODULE_LOAD
source ~/.bashrc
source $VENV_ACTIVATE
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:\${PYTHONPATH:-}"
EOF
}

# ---------------------------------------------------------------------------
# Baseline training job
# ---------------------------------------------------------------------------

submit_baseline() {
  local baseline_results_dir="$SCRIPT_DIR/results/baseline"
  mkdir -p "$baseline_results_dir"
  local walltime="${TIME_OVERRIDE:-0:45:00}"
  local job_file="$JOB_DIR/baseline_$(date +%Y%m%d_%H%M%S).sbatch"

  {
    _slurm_header "baseline" "$walltime" "$SLOG_DIR/%x_%A.out"
    echo ""
    echo "cd \"$PROJECT_ROOT\""
    echo "echo \"[\$(date)] Training baseline model\""
    echo "python src/train.py --model baseline --results_dir \"$baseline_results_dir\""
    echo "echo \"[\$(date)] Baseline training completed\""
  } > "$job_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] Generated baseline job: $job_file"
    echo "[DRY-RUN] Would submit: sbatch $job_file"
    return 0
  fi
  sbatch "$job_file"
  echo "Submitted baseline training: $job_file"
}

# ---------------------------------------------------------------------------
# Main target submission: one SLURM job per variant + visualization dependency
#
# For generic analyses (run_script=None):
#   python run_analysis.py --analysis KEY --variant NAME --run_id RUN_ID
# For legacy analyses (run_script set):
#   python RUN_SCRIPT --variant NAME --run_id RUN_ID
# ---------------------------------------------------------------------------

submit_target() {
  local analysis_key="$1"

  # Query the registry for variant names, walltimes, and run_script
  local meta_json
  meta_json=$(
    cd "$SCRIPT_DIR" && python - "$analysis_key" <<'PY'
import sys, json
sys.path.insert(0, '.')
from analyses import REGISTRY
key = sys.argv[1]
if key not in REGISTRY:
    print(json.dumps({"error": f"Unknown analysis key: {key}"}))
    sys.exit(1)
a = REGISTRY[key]
print(json.dumps({
    "variants": [{"name": v.name, "time": v.time} for v in a.variants],
    "run_script": a.run_script,
}))
PY
  )

  local run_script
  run_script="$(echo "$meta_json" | python -c 'import sys,json; d=json.load(sys.stdin); print(d["run_script"] or "")')"

  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)"
  local safe_key="${analysis_key//_/-}"   # for job names (underscores fine, but dashes look cleaner)
  safe_key="${analysis_key}"
  local wandb_arg=""
  [[ "$NO_WANDB" == "1" ]] && wandb_arg="--no_wandb"

  local -a job_ids=()

  # Submit one training job per variant
  while IFS= read -r variant_json; do
    local vname vtime
    vname="$(echo "$variant_json" | python -c 'import sys,json; d=json.load(sys.stdin); print(d["name"])')"
    vtime="$(echo "$variant_json" | python -c 'import sys,json; d=json.load(sys.stdin); print(d["time"])')"
    [[ -n "$TIME_OVERRIDE" ]] && vtime="$TIME_OVERRIDE"

    local job_name="${analysis_key}_${vname}"
    local job_file="$JOB_DIR/${job_name}_$(date +%Y%m%d_%H%M%S%N).sbatch"

    {
      _slurm_header "$job_name" "$vtime" "$SLOG_DIR/%x_%A.out"
      echo ""
      echo "cd \"$SCRIPT_DIR\""
      echo "echo \"[\$(date)] ${analysis_key} / ${vname}\""
      if [[ -z "$run_script" ]]; then
        # Generic runner
        echo "python run_analysis.py --analysis ${analysis_key} --variant ${vname} --run_id ${run_id} ${wandb_arg}"
      else
        # Legacy script — must support --variant and --run_id
        echo "python ${run_script} --variant ${vname} --run_id ${run_id} ${wandb_arg}"
      fi
      echo "echo \"[\$(date)] Done: ${vname}\""
    } > "$job_file"

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[DRY-RUN] Generated: $job_file"
    else
      local jid
      jid="$(sbatch --parsable "$job_file")"
      job_ids+=("$jid")
      echo "Submitted ${analysis_key}/${vname} → job $jid"
    fi

  done < <(echo "$meta_json" | python -c '
import sys, json
d = json.load(sys.stdin)
for v in d["variants"]:
    print(json.dumps(v))
')

  # Submit visualization job (depends on all training jobs)
  local viz_walltime="0:05:00"
  local viz_job_file="$JOB_DIR/${analysis_key}_viz_$(date +%Y%m%d_%H%M%S).sbatch"

  {
    _slurm_header "${analysis_key}_viz" "$viz_walltime" "$SLOG_DIR/%x_%A.out"
    echo ""
    echo "cd \"$SCRIPT_DIR\""
    echo "echo \"[\$(date)] Visualizing ${analysis_key}\""
    echo "python run_all_visualizations.py ${analysis_key}"
    echo "echo \"[\$(date)] Visualization complete: ${analysis_key}\""
  } > "$viz_job_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] Generated viz job: $viz_job_file"
    echo "[DRY-RUN] Would submit with dependency on training jobs"
    return 0
  fi

  if [[ ${#job_ids[@]} -gt 0 ]]; then
    local dep
    dep="afterok:$(IFS=:; echo "${job_ids[*]}")"
    local viz_jid
    viz_jid="$(sbatch --parsable --dependency="$dep" "$viz_job_file")"
    echo "Submitted ${analysis_key} visualization → job $viz_jid (depends on: ${job_ids[*]})"
  fi
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

[[ "$BASELINE_TRAIN" == "1" ]] && submit_baseline

for target in "${TARGETS[@]}"; do
  cd "$SCRIPT_DIR"
  # Validate target exists in registry
  if ! python - "$target" <<'PY' 2>/dev/null; then
import sys; sys.path.insert(0, '.')
from analyses import REGISTRY
if sys.argv[1] not in REGISTRY:
    print(f"Unknown target: {sys.argv[1]}", file=sys.stderr)
    sys.exit(1)
PY
    echo "Unknown target: $target. Run --list-targets for valid keys." >&2
    exit 1
  fi
  submit_target "$target"
done
