#!/usr/bin/env bash
set -euo pipefail

# SAFE ma-QAOA family sweep: per instance, stage the problem JSON, run the SAFE
# pipeline (LWPP pre-train -> exact fine-tune + exact-only baseline), then
# parameter distillation. One worker per instance id, resume-safe.
#
# Usage:
#   FAMILY=sk|grid2d|maxcut [MW=4] bash sweeps/run_family_sweep.sh
#
# Environment knobs (defaults are the paper settings):
#   MW=4                      LWPP truncation weight w_max
#   QUBITS_LIST="12 16 20"    sk/maxcut sizes     GRIDS_LIST="3x4 4x4 4x5" for grid2d
#   LAYERS_LIST="2 3 4"       circuit depths p    INSTANCE_IDS="0 1 2 3 4"
#   EXACT_DEVICE=auto         auto: <=12 qubits -> lightning.qubit, otherwise lightning.amdgpu
#   LWPP_DTYPE=float16        CONDA_ENV=SAFE      FORCE_RERUN=0
#   DISTILL_THRESHOLDS=0.01,0.3
#
# Training seed: grid2d passes the case seed (120490+instance) to the trainer;
# sk/maxcut use the trainer default (0). The case seed always drives problem generation.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FAMILY="${FAMILY:?set FAMILY=sk|grid2d|maxcut}"
MW="${MW:-4}"
CONDA_ENV="${CONDA_ENV:-SAFE}"

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda not found in PATH; source <miniconda>/etc/profile.d/conda.sh first" >&2
  exit 1
}

PY_TRAIN="safe_maqaoa/run_ma_qaoa_lwpp_exact_compare.py"
PY_MAKE="safe_maqaoa/make_problem.py"
PY_DISTILL="safe_maqaoa/run_distill_finetune.py"

QUBITS_LIST=(${QUBITS_LIST:-12 16 20})
GRIDS_LIST=(${GRIDS_LIST:-3x4 4x4 4x5})
LAYERS_LIST=(${LAYERS_LIST:-2 3 4})
INSTANCE_IDS=(${INSTANCE_IDS:-0 1 2 3 4})

LWPP_STEPS="${LWPP_STEPS:-500}"
EXACT_STEPS="${EXACT_STEPS:-100}"
EXACT_ONLY_STEPS="${EXACT_ONLY_STEPS:-100}"
LWPP_DTYPE="${LWPP_DTYPE:-float16}"
LWPP_PRESET="${LWPP_PRESET:-gpu}"
QAOA_RELAX_STEPS="${QAOA_RELAX_STEPS:-200}"
SNAPSHOT_EVERY_EXACT="${SNAPSHOT_EVERY_EXACT:-10}"
EXACT_DEVICE="${EXACT_DEVICE:-auto}"
DISTILL_THRESHOLDS="${DISTILL_THRESHOLDS:-0.01,0.3}"
PLOT_REFRESH_EVERY="${PLOT_REFRESH_EVERY:-50}"
FORCE_RERUN="${FORCE_RERUN:-0}"

SWEEP_TAG="mw${MW}/${FAMILY}"
BASE_OUT_ROOT="results/${SWEEP_TAG}"
PROBLEM_ROOT="data/problems/${FAMILY}"
LOG_ROOT="${LOG_ROOT:-results/logs/${SWEEP_TAG}}"
mkdir -p "$BASE_OUT_ROOT" "$PROBLEM_ROOT" "$LOG_ROOT"

stage_problem() {
  local case_tag="$1" problem_json="$2" n_qubits="$3" seed="$4" rows="${5:-}" cols="${6:-}"
  if [[ -f "$problem_json" ]]; then
    return 0
  fi
  case "$FAMILY" in
    sk)
      conda run -n "$CONDA_ENV" python "$PY_MAKE" \
        --problem-type sk --n-qubits "$n_qubits" --seed "$seed" --preset paper \
        --output "$problem_json"
      ;;
    maxcut)
      conda run -n "$CONDA_ENV" python "$PY_MAKE" \
        --problem-type maxcut --n-qubits "$n_qubits" --seed "$seed" --preset paper \
        --output "$problem_json"
      ;;
    grid2d)
      conda run -n "$CONDA_ENV" python "$PY_MAKE" \
        --problem-type grid2d --rows "$rows" --cols "$cols" --seed "$seed" --preset paper \
        --output "$problem_json"
      ;;
  esac
}

resolve_exact_device() {
  local n_qubits="$1"
  if [[ "$EXACT_DEVICE" != "auto" ]]; then
    echo "$EXACT_DEVICE"
  elif (( n_qubits <= 12 )); then
    echo "lightning.qubit"
  else
    echo "lightning.amdgpu"
  fi
}

run_one_case() {
  local case_tag="$1" problem_json="$2" p_layers="$3" log_file="$4" seed="$5" n_qubits="$6"
  local run_dir="${BASE_OUT_ROOT}/${case_tag}"
  local exact_device
  exact_device="$(resolve_exact_device "$n_qubits")"

  local train_seed_args=()
  if [[ "$FAMILY" == "grid2d" ]]; then
    train_seed_args=(--seed "$seed")
  fi

  {
    echo "[$(date +'%F %T')] START ${case_tag} (mw=${MW}, exact_device=${exact_device})"

    if [[ -f "${run_dir}/results.json" && "$FORCE_RERUN" != "1" ]]; then
      echo "[$(date +'%F %T')] SKIP TRAIN (exists): ${run_dir}/results.json"
    else
      conda run -n "$CONDA_ENV" python "$PY_TRAIN" \
        --resume-run-dir "$run_dir" \
        --problem-json "$problem_json" \
        --p-layers "$p_layers" \
        --theta-dtype float64 \
        --lwpp-dtype "$LWPP_DTYPE" \
        --exact-device "$exact_device" \
        "${train_seed_args[@]}" \
        --init-strategy baseline11 \
        --n-random-inits 5 \
        --const-values 0.01,0.05,0.1,0.2,0.4 \
        --n-qaoa-relax-inits 1 \
        --qaoa-relax-steps "$QAOA_RELAX_STEPS" \
        --tqa-dt-range 0.5 0.5 \
        --lwpp-max-weight "$MW" \
        --lwpp-steps "$LWPP_STEPS" \
        --exact-steps "$EXACT_STEPS" \
        --exact-only-steps "$EXACT_ONLY_STEPS" \
        --snapshot-every-exact "$SNAPSHOT_EVERY_EXACT" \
        --lwpp-preset "$LWPP_PRESET" \
        --live-plot \
        --plot-refresh-every "$PLOT_REFRESH_EVERY" \
        --no-plot-window \
        --lwpp-exact-eval-every "${LWPP_EXACT_EVAL_EVERY:-20}" \
        --stream-save-every "${STREAM_SAVE_EVERY:-10}" \
        --exact-chunk-size "${EXACT_CHUNK_SIZE:-1}" || return 1
    fi

    local distill_force=()
    if [[ "$FORCE_RERUN" == "1" ]]; then
      distill_force=(--force)
    fi
    conda run -n "$CONDA_ENV" python "$PY_DISTILL" \
      --run-dir "$run_dir" \
      --thresholds "$DISTILL_THRESHOLDS" \
      --steps "$EXACT_STEPS" \
      --lr 0.02 \
      --exact-device "$exact_device" \
      "${distill_force[@]}" || return 1

    echo "[$(date +'%F %T')] DONE  ${case_tag}"
  } >>"$log_file" 2>&1
  return 0
}

run_worker() {
  local worker_id="$1"
  local worker_log="${LOG_ROOT}/worker_${worker_id}.log"
  local failed=0
  {
    echo "[$(date +'%F %T')] Worker ${worker_id} launched (family=${FAMILY} mw=${MW})"
    if [[ "$FAMILY" == "grid2d" ]]; then
      local seed=$((120490 + worker_id))
      for g in "${GRIDS_LIST[@]}"; do
        local rows="${g%x*}" cols="${g#*x}"
        for p in "${LAYERS_LIST[@]}"; do
          local case_tag="grid${g}_p${p}_inst${worker_id}_seed${seed}"
          local problem_json="${PROBLEM_ROOT}/${case_tag}.json"
          stage_problem "$case_tag" "$problem_json" "$((rows * cols))" "$seed" "$rows" "$cols" || { echo "[FAIL] stage ${case_tag}"; failed=1; continue; }
          run_one_case "$case_tag" "$problem_json" "$p" "${LOG_ROOT}/${case_tag}.log" "$seed" "$((rows * cols))" || { echo "[FAIL] ${case_tag} (see ${LOG_ROOT}/${case_tag}.log)"; failed=1; }
        done
      done
    else
      for n in "${QUBITS_LIST[@]}"; do
        for p in "${LAYERS_LIST[@]}"; do
          local seed=$((700000 + n * 1000 + p * 100 + worker_id))
          local case_tag="${FAMILY}_q${n}_p${p}_inst${worker_id}_seed${seed}"
          local problem_json="${PROBLEM_ROOT}/${case_tag}.json"
          stage_problem "$case_tag" "$problem_json" "$n" "$seed" || { echo "[FAIL] stage ${case_tag}"; failed=1; continue; }
          run_one_case "$case_tag" "$problem_json" "$p" "${LOG_ROOT}/${case_tag}.log" "$seed" "$n" || { echo "[FAIL] ${case_tag} (see ${LOG_ROOT}/${case_tag}.log)"; failed=1; }
        done
      done
    fi
    echo "[$(date +'%F %T')] Worker ${worker_id} finished (failed=${failed})"
  } >>"$worker_log" 2>&1
  return "$failed"
}

echo "Launching ${#INSTANCE_IDS[@]} parallel workers (family=${FAMILY}, mw=${MW})..."
pids=()
for wid in "${INSTANCE_IDS[@]}"; do
  run_worker "$wid" &
  pids+=("$!")
  echo "  worker ${wid} pid=$!"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
echo "Logs:    ${LOG_ROOT}"
echo "Outputs: ${BASE_OUT_ROOT}"
if [[ "$status" != "0" ]]; then
  echo "Some cases failed; grep '\[FAIL\]' ${LOG_ROOT}/worker_*.log" >&2
  exit 1
fi
echo "All workers finished."
