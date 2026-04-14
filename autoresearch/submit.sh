#!/usr/bin/env bash
# submit.sh — Submit one autoresearch training experiment to SLURM.
#
# Designed to be called once per experiment in the autoresearch loop:
#
#   ./submit.sh --wait          # submit + block until job completes
#   ./submit.sh                 # fire-and-forget (returns job ID)
#   ./submit.sh --dry-run       # print the generated sbatch script, do not submit
#
# Training output is always written to run.log in this directory.
# After the job finishes, read the primary metric with:
#   grep "^val_kl_divergence:" run.log
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SLOG_DIR="$SCRIPT_DIR/slogs"
JOB_DIR="$SCRIPT_DIR/.slurm_jobs"

mkdir -p "$SLOG_DIR" "$JOB_DIR"

# ── SLURM defaults (same as submit_subanalysis.sh) ───────────────────────────
GPU="l40s:1"
CPUS="8"
MEM="32G"
QOS="normal"
# 20 min per experiment: 5 min TIME_BUDGET + ~2 min overhead (data cache, eval)
# First run is longer due to embedding computation — use --time 60:00 the first time.
WALLTIME="20:00"
VENV_ACTIVATE="~/barcode/bin/activate"
MODULE_LOAD="python/3.12 cuda/12.6 arrow/21.0.0 opencv/4.12.0"

# ── Script flags ─────────────────────────────────────────────────────────────
WAIT="0"
DRY_RUN="0"
USE_UV="0"   # set to 1 if uv is installed on the cluster; 0 to use venv python

usage() {
  cat <<'EOF'
Usage:
  ./submit.sh [options]

Options:
  --wait               Block until the SLURM job finishes (required for the agent loop)
  --dry-run            Print the generated sbatch script; do not submit
  --gpu SPEC           SLURM --gres gpu spec            (default: l40s:1)
  --cpus N             SLURM cpus-per-task              (default: 8)
  --mem SIZE           SLURM memory                     (default: 32G)
  --time HH:MM:SS      Walltime override                (default: 20:00)
  --qos NAME           SLURM QoS                        (default: normal)
  --venv PATH          Path to venv activate script     (default: ~/barcode/bin/activate)
  --module-load "..."  Modules for `module load`        (default: python/3.12 cuda/12.6 ...)
  --use-uv             Use `uv run train.py` instead of `python train.py`
  -h, --help           Show this help

First-run note:
  The first experiment also computes BarcodeBERT embeddings. Use --time 90:00 for
  the first run, then revert to the default 20:00 for all subsequent experiments.

Example agent loop (from login node inside tmux/screen):
  while true; do
    # 1. edit train.py, then:
    git commit -m "description"
    ./submit.sh --wait > submit.log 2>&1
    grep "^val_kl_divergence:" run.log
    # 2. decide keep or discard, update results.tsv
  done
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait)        WAIT="1";             shift ;;
    --dry-run)     DRY_RUN="1";         shift ;;
    --use-uv)      USE_UV="1";          shift ;;
    --gpu)         GPU="$2";            shift 2 ;;
    --cpus)        CPUS="$2";           shift 2 ;;
    --mem)         MEM="$2";            shift 2 ;;
    --time)        WALLTIME="$2";       shift 2 ;;
    --qos)         QOS="$2";            shift 2 ;;
    --venv)        VENV_ACTIVATE="$2";  shift 2 ;;
    --module-load) MODULE_LOAD="$2";    shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# ── Determine the training command ───────────────────────────────────────────
if [[ "$USE_UV" == "1" ]]; then
  TRAIN_CMD="uv run train.py"
else
  TRAIN_CMD="python train.py"
fi

# ── Build the sbatch script ───────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_FILE="$JOB_DIR/autoresearch_${TIMESTAMP}.sbatch"
JOB_NAME="autoresearch"
LOG_FILE="$SCRIPT_DIR/run.log"

cat > "$JOB_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --gres=gpu:$GPU
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$WALLTIME
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$SLOG_DIR/%x_%A.out
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

# autoresearch/train.py imports from prepare.py in the same directory;
# no PYTHONPATH manipulation needed since we cd into the folder.
cd "$SCRIPT_DIR"

echo "[autoresearch] \$(date) — starting $TRAIN_CMD"
$TRAIN_CMD > "$LOG_FILE" 2>&1
echo "[autoresearch] \$(date) — done. Exit code: \$?"
EOF

# ── Submit or dry-run ─────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY-RUN] Generated sbatch script: $JOB_FILE"
  echo "[DRY-RUN] Contents:"
  cat "$JOB_FILE"
  exit 0
fi

SBATCH_ARGS=("$JOB_FILE")
if [[ "$WAIT" == "1" ]]; then
  SBATCH_ARGS=("--wait" "${SBATCH_ARGS[@]}")
fi

SBATCH_OUTPUT="$(sbatch "${SBATCH_ARGS[@]}")"
echo "$SBATCH_OUTPUT"

JOB_ID="$(echo "$SBATCH_OUTPUT" | grep -oP '(?<=Submitted batch job )\d+')"
if [[ -n "$JOB_ID" ]]; then
  echo "Job ID: $JOB_ID"
  echo "SLURM log:   $SLOG_DIR/${JOB_NAME}_${JOB_ID}.out"
  echo "Training log: $LOG_FILE"
fi
