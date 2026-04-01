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
DATA_PATH="$PROJECT_ROOT/data/data_merged.csv"
DATA_PATH_SET="0"
VENV_ACTIVATE='~/barcode/bin/activate'
MODULE_LOAD='python/3.12 cuda/12.6 arrow/21.0.0 opencv/4.12.0'
NO_WANDB="0"
DRY_RUN="0"
ALL_TARGETS="0"
BASELINE_TRAIN="0"
INCLUDE_BASELINE="1"
BASELINE_RESULTS=""
BASELINE_KEY="baseline"
LABELS_JSON_OVERRIDE=""
COLORS_JSON_OVERRIDE=""

declare -a TARGETS=()
declare -a DEFAULT_TARGETS=(
  "BarcodeBERT"
  "interpolated_latent/V1"
  "interpolated_latent/V2"
  "interpolated_latent/V3"
  "interpolated_latent/V4"
  "location_embedding"
  "latent_as_input"
  "latent_as_input_V2"
  "ablation_study"
  "loss_comparison"
  "optimal_K"
  "preprocessing"
  "dimensionality_increase/gating_function"
  "dimensionality_increase/vector_size"
)
LIST_TARGETS="0"

usage() {
  cat <<'EOF'
Usage:
  ./submit_subanalysis.sh --target interpolated_latent/V4
  ./submit_subanalysis.sh --target interpolated_latent/V4 --target location_embedding
  ./submit_subanalysis.sh --all
  ./submit_subanalysis.sh --baseline-train
  ./submit_subanalysis.sh --baseline-train --target interpolated_latent/V4

Options:
  --target PATH            Subanalysis folder path under analysis/
  --all                    Submit all default targets
  --baseline-train         Train baseline model once (from Metabarcoding/)
  --list-targets           Print supported targets and exit
  --data-path PATH         Override Data CSV path (default: PROJECT_ROOT/data/data_merged.csv)
  --baseline-results PATH  Path to one reusable baseline pickle from src/train.py
  --baseline-key KEY       Model key to use for baseline in merged visualization (default: baseline)
  --no-baseline            Do not include baseline in visualization input
  --labels-json JSON       Override labels passed to visualize_results.py
  --colors-json JSON       Override colors passed to visualize_results.py
  --no-wandb               Append --no_wandb to supported training commands
  --gpu SPEC               SLURM --gres gpu spec (default: l40s:1)
  --cpus N                 SLURM cpus-per-task (default: 8)
  --mem SIZE               SLURM memory (default: 32G)
  --time HH:MM:SS          Override walltime for all targets
  --qos NAME               SLURM QoS (default: normal)
  --venv-activate PATH     Venv activate script (default: ~/barcode/bin/activate)
  --module-load "A B C"     Modules for `module load` (default: python/3.12 cuda/12.6 arrow/21.0.0 opencv/4.12.0)
  --dry-run                Print generated sbatch script path and command, do not submit
  -h, --help               Show this help

Supported targets (first batch):
  interpolated_latent/V1
  interpolated_latent/V2
  interpolated_latent/V3
  interpolated_latent/V4
  location_embedding
  latent_as_input
  latent_as_input_V2
  ablation_study
  loss_comparison
  optimal_K
  preprocessing
  dimensionality_increase/gating_function
  dimensionality_increase/vector_size
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGETS+=("$2")
      shift 2
      ;;
    --all)
      ALL_TARGETS="1"
      shift
      ;;
    --baseline-train)
      BASELINE_TRAIN="1"
      shift
      ;;
    --list-targets)
      LIST_TARGETS="1"
      shift
      ;;
    --data-path)
      DATA_PATH="$2"
      DATA_PATH_SET="1"
      shift 2
      ;;
    --baseline-results)
      BASELINE_RESULTS="$2"
      shift 2
      ;;
    --baseline-key)
      BASELINE_KEY="$2"
      shift 2
      ;;
    --no-baseline)
      INCLUDE_BASELINE="0"
      shift
      ;;
    --labels-json)
      LABELS_JSON_OVERRIDE="$2"
      shift 2
      ;;
    --colors-json)
      COLORS_JSON_OVERRIDE="$2"
      shift 2
      ;;
    --no-wandb)
      NO_WANDB="1"
      shift
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --mem)
      MEM="$2"
      shift 2
      ;;
    --time)
      TIME_OVERRIDE="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --venv-activate)
      VENV_ACTIVATE="$2"
      shift 2
      ;;
    --module-load)
      MODULE_LOAD="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$ALL_TARGETS" == "1" && ${#TARGETS[@]} -gt 0 ]]; then
  echo "Use either --all or --target, not both." >&2
  exit 1
fi

if [[ "$LIST_TARGETS" == "1" ]]; then
  cat <<'EOF'
BarcodeBERT
interpolated_latent/V1
interpolated_latent/V2
interpolated_latent/V3
interpolated_latent/V4
location_embedding
latent_as_input
latent_as_input_V2
ablation_study
loss_comparison
optimal_K
preprocessing
dimensionality_increase/gating_function
dimensionality_increase/vector_size
EOF
  exit 0
fi

if [[ "$ALL_TARGETS" == "1" ]]; then
  TARGETS=("${DEFAULT_TARGETS[@]}")
fi

if [[ ${#TARGETS[@]} -eq 0 && "$BASELINE_TRAIN" == "0" ]]; then
  echo "No targets specified. Use --target or --all." >&2
  usage
  exit 1
fi

# Resolve data path once to an absolute path so nested target directories behave consistently.
if [[ "$DATA_PATH" != /* ]]; then
  if [[ -e "$DATA_PATH" ]]; then
    DATA_PATH="$(cd "$(dirname "$DATA_PATH")" && pwd)/$(basename "$DATA_PATH")"
  else
    DATA_PATH="$(cd "$SCRIPT_DIR" && cd "$(dirname "$DATA_PATH")" && pwd)/$(basename "$DATA_PATH")"
  fi
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Data file not found: $DATA_PATH" >&2
  exit 1
fi

if ! [[ "$CPUS" =~ ^[0-9]+$ ]]; then
  echo "Invalid --cpus value: $CPUS (must be an integer)" >&2
  exit 1
fi

if (( CPUS > 16 )); then
  echo "Invalid --cpus value: $CPUS (maximum supported is 16)" >&2
  exit 1
fi

resolve_target() {
  local target="$1"
  TARGET_DIR=""
  TRAIN_CMD_TEMPLATE=""
  RESULTS_PATTERNS=""
  FIGURES_DIR=""
  DEFAULT_LABELS_JSON=""
  DEFAULT_COLORS_JSON=""
  DEFAULT_TIME="8:00:00"
  OPTIONAL_DATA_ARG_TEMPLATE=""

  case "$target" in
    BarcodeBERT)
      TARGET_DIR="BarcodeBERT"
      OPTIONAL_DATA_ARG_TEMPLATE='--data_path "__DATA_PATH__"'
      TRAIN_CMD_TEMPLATE='python barcode_bert.py __OPTIONAL_DATA_ARG__'
      RESULTS_PATTERNS='BarcodeBERT/results/barcode_bert_*.pkl'
      FIGURES_DIR='figures/BarcodeBERT'
      DEFAULT_LABELS_JSON='{"baseline":"Taxonomy","barcode_bert":"BarcodeBERT"}'
      DEFAULT_COLORS_JSON='{"baseline":"#2ecc71","barcode_bert":"#9b59b6"}'
      DEFAULT_TIME="8:00:00"
      ;;
    interpolated_latent/V1)
      TARGET_DIR="interpolated_latent/V1"
      OPTIONAL_DATA_ARG_TEMPLATE='--data_path "__DATA_PATH__"'
      TRAIN_CMD_TEMPLATE='python interpolated_latent.py __OPTIONAL_DATA_ARG__'
      RESULTS_PATTERNS='interpolated_latent/V1/results/interpolated_latent_v1_*.pkl'
      FIGURES_DIR='figures/interpolated_latent/V1'
      DEFAULT_LABELS_JSON='{"baseline":"Baseline","interpolated_latent":"Interpolated Latent"}'
      DEFAULT_COLORS_JSON='{"baseline":"#1f6feb","interpolated_latent":"#d97706"}'
      DEFAULT_TIME="8:00:00"
      ;;
    interpolated_latent/V2)
      TARGET_DIR="interpolated_latent/V2"
      OPTIONAL_DATA_ARG_TEMPLATE='--data_path "__DATA_PATH__"'
      TRAIN_CMD_TEMPLATE='python interpolated_latent.py __OPTIONAL_DATA_ARG__'
      RESULTS_PATTERNS='interpolated_latent/V2/results/interpolated_latent_v2_*.pkl'
      FIGURES_DIR='figures/interpolated_latent/V2'
      DEFAULT_LABELS_JSON='{"baseline":"Baseline","interpolated_latent":"Interpolated Latent"}'
      DEFAULT_COLORS_JSON='{"baseline":"#1f6feb","interpolated_latent":"#d97706"}'
      DEFAULT_TIME="8:00:00"
      ;;
    interpolated_latent/V3)
      TARGET_DIR="interpolated_latent/V3"
      OPTIONAL_DATA_ARG_TEMPLATE='--data_path "__DATA_PATH__"'
      TRAIN_CMD_TEMPLATE='python interpolated_latent.py __OPTIONAL_DATA_ARG__'
      RESULTS_PATTERNS='interpolated_latent/V3/results/interpolated_latent_v3_*.pkl'
      FIGURES_DIR='figures/interpolated_latent/V3'
      DEFAULT_LABELS_JSON='{"baseline":"Baseline","interpolated_latent":"Interpolated Latent"}'
      DEFAULT_COLORS_JSON='{"baseline":"#1f6feb","interpolated_latent":"#d97706"}'
      DEFAULT_TIME="8:00:00"
      ;;
    interpolated_latent/V4)
      TARGET_DIR="interpolated_latent/V4"
      OPTIONAL_DATA_ARG_TEMPLATE='--data_path "__DATA_PATH__"'
      TRAIN_CMD_TEMPLATE='python interpolated_latent.py __OPTIONAL_DATA_ARG__'
      RESULTS_PATTERNS='interpolated_latent/V4/results/interpolated_latent_v4_*.pkl'
      FIGURES_DIR='figures/interpolated_latent/V4'
      DEFAULT_LABELS_JSON='{"baseline":"Baseline","interpolated_latent":"Interpolated Latent"}'
      DEFAULT_COLORS_JSON='{"baseline":"#1f6feb","interpolated_latent":"#d97706"}'
      DEFAULT_TIME="8:00:00"
      ;;
    location_embedding)
      TARGET_DIR="location_embedding"
      OPTIONAL_DATA_ARG_TEMPLATE='--data_path "__DATA_PATH__"'
      TRAIN_CMD_TEMPLATE='python location_embedding.py __OPTIONAL_DATA_ARG__'
      RESULTS_PATTERNS='location_embedding/results/location_embedding_*.pkl'
      FIGURES_DIR='figures/location_embedding'
      DEFAULT_LABELS_JSON='{"baseline":"Baseline (No Location Embedding)","satclip":"SatCLIP (256D)","range":"RANGE (1280D)","geoclip":"GeoCLIP (512D)","alphaearth":"AlphaEarth (64D)"}'
      DEFAULT_COLORS_JSON='{"baseline":"#95a5a6","satclip":"#e74c3c","range":"#3498db","geoclip":"#2ecc71","alphaearth":"#f39c12"}'
      DEFAULT_TIME="12:00:00"
      ;;
    latent_as_input)
      TARGET_DIR='latent_as_input'
      OPTIONAL_DATA_ARG_TEMPLATE='--data_path "__DATA_PATH__"'
      TRAIN_CMD_TEMPLATE='python latent_as_input.py __OPTIONAL_DATA_ARG__ --output_dir results'
      RESULTS_PATTERNS='latent_as_input/results/latent_as_input_*.pkl'
      FIGURES_DIR='figures/latent_as_input'
      DEFAULT_LABELS_JSON='{"baseline":"Baseline (Latent + MLP)","latent_as_input":"Latent as Input"}'
      DEFAULT_COLORS_JSON='{"baseline":"#2ecc71","latent_as_input":"#e67e22"}'
      DEFAULT_TIME="8:00:00"
      ;;
    latent_as_input_V2)
      TARGET_DIR='latent_as_input_V2'
      OPTIONAL_DATA_ARG_TEMPLATE='--data_path "__DATA_PATH__"'
      TRAIN_CMD_TEMPLATE='python latent_as_input.py __OPTIONAL_DATA_ARG__ --output_dir results'
      RESULTS_PATTERNS='latent_as_input_V2/results/latent_as_input_v2_*.pkl'
      FIGURES_DIR='figures/latent_as_input_V2'
      DEFAULT_LABELS_JSON='{"baseline":"Baseline (Latent + MLP)","latent_as_input":"Latent as In-&-Output"}'
      DEFAULT_COLORS_JSON='{"baseline":"#2ecc71","latent_as_input":"#e67e22"}'
      DEFAULT_TIME="8:00:00"
      ;;
    ablation_study)
      TARGET_DIR='ablation_study'
      TRAIN_CMD_TEMPLATE='python ablation_study.py --data_path "__DATA_PATH__" --output_dir results'
      RESULTS_PATTERNS='ablation_study/results/ablation_study_*.pkl'
      FIGURES_DIR='figures/ablation_study'
      DEFAULT_LABELS_JSON='{"baseline":"MLP + Latent","mlp_no_taxonomy":"MLP (no taxonomy)","mlp_with_taxonomy":"MLP (with taxonomy)"}'
      DEFAULT_COLORS_JSON='{"baseline":"#ff7f0e","mlp_no_taxonomy":"#1f77b4","mlp_with_taxonomy":"#2ca02c"}'
      DEFAULT_TIME="8:00:00"
      ;;
    loss_comparison)
      TARGET_DIR='loss_comparison'
      TRAIN_CMD_TEMPLATE='python loss_comparison.py --data_path "__DATA_PATH__" --output_dir results'
      RESULTS_PATTERNS='loss_comparison/results/loss_comparison_*.pkl'
      FIGURES_DIR='figures/loss_comparison'
      DEFAULT_LABELS_JSON='{"baseline":"Baseline","cross_entropy":"Cross-Entropy","logistic":"Logistic (BCE)"}'
      DEFAULT_COLORS_JSON='{"baseline":"#95a5a6","cross_entropy":"#2ecc71","logistic":"#9b59b6"}'
      DEFAULT_TIME="8:00:00"
      ;;
    optimal_K)
      TARGET_DIR='optimal_K'
      TRAIN_CMD_TEMPLATE='python K_comparison.py --data_path "__DATA_PATH__" --output_dir results'
      RESULTS_PATTERNS='optimal_K/results/K_comparison_*.pkl'
      FIGURES_DIR='figures/optimal_K'
      DEFAULT_LABELS_JSON='{"baseline":"Baseline","K_13":"K=13","K_78":"K=78 (Optimal)","K_972":"K=972"}'
      DEFAULT_COLORS_JSON='{"baseline":"#95a5a6","K_13":"#3498db","K_78":"#e67e22","K_972":"#e67e22"}'
      DEFAULT_TIME="8:00:00"
      ;;
    preprocessing)
      TARGET_DIR='preprocessing'
      TRAIN_CMD_TEMPLATE='python utils_test.py --data_path "__DATA_PATH__" && python read_count_preprocessing.py --output_dir results'
      RESULTS_PATTERNS='preprocessing/results/preprocessing_*.pkl'
      FIGURES_DIR='figures/preprocessing'
      DEFAULT_LABELS_JSON='{"baseline":"Baseline","original":"Original (raw counts)","normalized":"Normalized Only","logarithm":"Logarithm Only"}'
      DEFAULT_COLORS_JSON='{"baseline":"#95a5a6","original":"#ff7f0e","normalized":"#1f77b4","logarithm":"#2ca02c"}'
      DEFAULT_TIME="10:00:00"
      ;;
    dimensionality_increase/gating_function)
      TARGET_DIR='dimensionality_increase/gating_function'
      TRAIN_CMD_TEMPLATE='python dimensionality_increase.py --data_path "__DATA_PATH__" --output_dir results'
      RESULTS_PATTERNS='dimensionality_increase/gating_function/results/gating_comparison_*.pkl'
      FIGURES_DIR='figures/dimensionality_gating'
      DEFAULT_LABELS_JSON='{"baseline":"Baseline (Additive)","exp":"Exponential","scaled_exp":"Scaled Exponential","additive":"Additive (1+h)","softplus":"Softplus","tanh":"Tanh","sigmoid":"Sigmoid","dot_product":"Dot Product"}'
      DEFAULT_COLORS_JSON='{"baseline":"#95a5a6","exp":"#e74c3c","scaled_exp":"#e67e22","additive":"#f39c12","softplus":"#2ecc71","tanh":"#3498db","sigmoid":"#9b59b6","dot_product":"#1abc9c"}'
      DEFAULT_TIME="8:00:00"
      ;;
    dimensionality_increase/vector_size)
      TARGET_DIR='dimensionality_increase/vector_size'
      TRAIN_CMD_TEMPLATE='python dimensionality_increase.py --data_path "__DATA_PATH__" --output_dir results'
      RESULTS_PATTERNS='dimensionality_increase/vector_size/results/dimensionality_analysis_*.pkl'
      FIGURES_DIR='figures/dimensionality_vector'
      DEFAULT_LABELS_JSON='{"baseline":"Dim=1 (Baseline)","dim_2":"Dim=2","dim_5":"Dim=5","dim_6":"Dim=6","dim_8":"Dim=8","dim_10":"Dim=10","dim_12":"Dim=12","dim_15":"Dim=15","dim_20":"Dim=20","dim_50":"Dim=50"}'
      DEFAULT_COLORS_JSON='{"baseline":"#95a5a6","dim_2":"#824e05","dim_5":"#e74c3c","dim_6":"#e67e22","dim_8":"#f39c12","dim_10":"#f1c40f","dim_12":"#a2f10f","dim_15":"#2ecc71","dim_20":"#1d8d4b","dim_50":"#3498db"}'
      DEFAULT_TIME="8:00:00"
      ;;
    *)
      return 1
      ;;
  esac

  return 0
}

submit_baseline() {
  local baseline_train_dir="$(dirname "$SCRIPT_DIR")"
  local baseline_results_dir="$baseline_train_dir/results/baseline"
  mkdir -p "$baseline_results_dir"

  local walltime="6:00:00"
  if [[ -n "$TIME_OVERRIDE" ]]; then
    walltime="$TIME_OVERRIDE"
  fi

  local safe_baseline="baseline"
  local job_file="$JOB_DIR/${safe_baseline}_$(date +%Y%m%d_%H%M%S).sbatch"
  local job_name="baseline"

  cat > "$job_file" <<EOF
#!/usr/bin/env bash
#SBATCH --gres=gpu:$GPU
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$walltime
#SBATCH --job-name=$job_name
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

cd "$baseline_train_dir"
echo "[$(date)] Training baseline model"
python src/train.py --model baseline
echo "[$(date)] Baseline training completed"
EOF

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] Generated baseline job: $job_file"
    echo "[DRY-RUN] Would submit: sbatch $job_file"
    echo "[DRY-RUN] Baseline results will be saved to: $baseline_results_dir/"
    return 0
  fi

  local sbatch_output
  sbatch_output="$(sbatch "$job_file")"
  echo "$sbatch_output"
  echo "Submitted baseline training with script: $job_file"
  echo "Results will be saved to: $baseline_results_dir/"
}

submit_target() {
  local target="$1"

  if ! resolve_target "$target"; then
    echo "Unsupported target: $target" >&2
    return 1
  fi

  local target_dir_abs="$SCRIPT_DIR/$TARGET_DIR"
  if [[ ! -d "$target_dir_abs" ]]; then
    echo "Target directory does not exist: $target_dir_abs" >&2
    return 1
  fi

  local wandb_flag=""
  if [[ "$NO_WANDB" == "1" ]]; then
    wandb_flag="--no_wandb"
  fi

  local labels_json="$DEFAULT_LABELS_JSON"
  local colors_json="$DEFAULT_COLORS_JSON"
  if [[ -n "$LABELS_JSON_OVERRIDE" ]]; then
    labels_json="$LABELS_JSON_OVERRIDE"
  fi
  if [[ -n "$COLORS_JSON_OVERRIDE" ]]; then
    colors_json="$COLORS_JSON_OVERRIDE"
  fi

  local baseline_results_abs=""
  if [[ -n "$BASELINE_RESULTS" ]]; then
    if [[ "$BASELINE_RESULTS" = /* ]]; then
      baseline_results_abs="$BASELINE_RESULTS"
    elif [[ -e "$BASELINE_RESULTS" ]]; then
      baseline_results_abs="$PWD/$BASELINE_RESULTS"
    else
      baseline_results_abs="$SCRIPT_DIR/$BASELINE_RESULTS"
    fi
  fi

  local train_cmd="$TRAIN_CMD_TEMPLATE"
  local optional_data_arg=""
  if [[ "$DATA_PATH_SET" == "1" ]]; then
    optional_data_arg="$OPTIONAL_DATA_ARG_TEMPLATE"
  fi
  train_cmd="${train_cmd//__OPTIONAL_DATA_ARG__/$optional_data_arg}"
  train_cmd="${train_cmd//__DATA_PATH__/$DATA_PATH}"
  train_cmd="${train_cmd//__NO_WANDB__/$wandb_flag}"

  local walltime="$DEFAULT_TIME"
  if [[ -n "$TIME_OVERRIDE" ]]; then
    walltime="$TIME_OVERRIDE"
  fi

  local safe_target
  safe_target="${target//\//_}"

  local job_file="$JOB_DIR/${safe_target}_$(date +%Y%m%d_%H%M%S).sbatch"
  local job_name="${safe_target}"

  cat > "$job_file" <<EOF
#!/usr/bin/env bash
#SBATCH --gres=gpu:$GPU
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$walltime
#SBATCH --job-name=$job_name
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

cd "$target_dir_abs"
echo "[$(date)] Target: $target"
echo "[$(date)] Data path: $DATA_PATH"
echo "[$(date)] Train command: $train_cmd"
$train_cmd

cd "$SCRIPT_DIR"
RESULT_PATTERNS='$RESULTS_PATTERNS'
INCLUDE_BASELINE='$INCLUDE_BASELINE'
BASELINE_RESULTS='$baseline_results_abs'
BASELINE_KEY='$BASELINE_KEY'
LABELS_JSON='$labels_json'
COLORS_JSON='$colors_json'

IFS=';' read -r -a result_patterns <<< "\$RESULT_PATTERNS"
RESULT_FILES=()
shopt -s nullglob
for pattern in "\${result_patterns[@]}"; do
  for file in "$SCRIPT_DIR"/\$pattern; do
    RESULT_FILES+=("\$file")
  done
done
shopt -u nullglob

if [[ \${#RESULT_FILES[@]} -eq 0 ]]; then
  echo "[$(date)] No result files matched patterns: \$RESULT_PATTERNS" >&2
  exit 1
fi

if [[ "\$INCLUDE_BASELINE" == "1" && -n "\$BASELINE_RESULTS" ]]; then
  if [[ -f "\$BASELINE_RESULTS" ]]; then
    BASELINE_NORMALIZED="$JOB_DIR/${safe_target}_baseline_normalized.pkl"
    python - "\$BASELINE_RESULTS" "\$BASELINE_NORMALIZED" "\$BASELINE_KEY" <<'PY'
import pickle
import sys

baseline_path, output_path, key = sys.argv[1], sys.argv[2], sys.argv[3]
with open(baseline_path, "rb") as f:
    payload = pickle.load(f)

if isinstance(payload, dict) and "predictions" in payload and "targets" in payload:
    normalized = {key: payload}
else:
    normalized = payload

with open(output_path, "wb") as f:
    pickle.dump(normalized, f)
PY
    RESULT_FILES=("\$BASELINE_NORMALIZED" "\${RESULT_FILES[@]}")
  else
    echo "[$(date)] Baseline file not found: \$BASELINE_RESULTS (continuing without baseline)"
  fi
fi

echo "[$(date)] Visualization inputs: \${RESULT_FILES[*]}"
viz_cmd=(python visualize_results.py --results_paths "\${RESULT_FILES[@]}" --output_dir "$FIGURES_DIR")
if [[ -n "\$LABELS_JSON" ]]; then
  viz_cmd+=(--labels "\$LABELS_JSON")
fi
if [[ -n "\$COLORS_JSON" ]]; then
  viz_cmd+=(--colors "\$COLORS_JSON")
fi
"\${viz_cmd[@]}"
echo "[$(date)] Completed target: $target"
EOF

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] Generated: $job_file"
    echo "[DRY-RUN] Would submit: sbatch $job_file"
    return 0
  fi

  local sbatch_output
  sbatch_output="$(sbatch "$job_file")"
  echo "$sbatch_output"
  echo "Submitted target '$target' with script: $job_file"
}

# Execute baseline training if requested
if [[ "$BASELINE_TRAIN" == "1" ]]; then
  submit_baseline
fi

# Execute target submissions
if [[ ${#TARGETS[@]} -gt 0 ]]; then
  for target in "${TARGETS[@]}"; do
    submit_target "$target"
  done
fi
