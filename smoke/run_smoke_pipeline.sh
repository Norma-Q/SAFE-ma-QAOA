#!/usr/bin/env bash
set -euo pipefail

# Reduced end-to-end run of the SAFE pipeline on SK q12 p2 inst0:
# CPU checks -> LWPP pre-train -> exact fine-tune + exact-only -> distillation
# -> aggregation -> summary tables. Writes to results_smoke/ (never results/).
#
#   bash smoke/run_smoke_pipeline.sh                                        # GPU
#   EXACT_DEVICE=lightning.qubit LWPP_PRESET=cpu bash smoke/run_smoke_pipeline.sh   # CPU-only

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-SAFE}"
EXACT_DEVICE="${EXACT_DEVICE:-lightning.amdgpu}"
LWPP_PRESET="${LWPP_PRESET:-gpu}"
LWPP_DTYPE="${LWPP_DTYPE:-float16}"

OUT_ROOT="results_smoke/mw4/sk"
CASE="sk_q12_p2_inst0_seed712200"
RUN_DIR="${OUT_ROOT}/${CASE}"
rm -rf "$RUN_DIR"
mkdir -p "$OUT_ROOT"

echo "=== [smoke 1/6] CPU checks: problem regeneration + engine vs PennyLane ==="
conda run -n "$CONDA_ENV" python smoke/verify_problem_regeneration.py
conda run -n "$CONDA_ENV" python smoke/verify_engine_vs_pennylane.py

echo "=== [smoke 2/6] SAFE pipeline (reduced steps) ==="
conda run -n "$CONDA_ENV" python safe_maqaoa/run_ma_qaoa_lwpp_exact_compare.py \
  --resume-run-dir "$RUN_DIR" \
  --problem-json "data/problems/sk/${CASE}.json" \
  --p-layers 2 \
  --theta-dtype float64 \
  --lwpp-dtype "$LWPP_DTYPE" \
  --exact-device "$EXACT_DEVICE" \
  --init-strategy baseline11 \
  --n-random-inits 5 \
  --const-values 0.01,0.05,0.1,0.2,0.4 \
  --n-qaoa-relax-inits 1 \
  --qaoa-relax-steps 20 \
  --tqa-dt-range 0.5 0.5 \
  --lwpp-max-weight 4 \
  --lwpp-steps 30 \
  --exact-steps 10 \
  --exact-only-steps 10 \
  --snapshot-every-exact 5 \
  --lwpp-preset "$LWPP_PRESET" \
  --live-plot \
  --plot-refresh-every 10 \
  --no-plot-window \
  --stream-save-every 5

echo "=== [smoke 3/6] distillation fine-tune (thr 0.01, 0.3) ==="
conda run -n "$CONDA_ENV" python safe_maqaoa/run_distill_finetune.py \
  --run-dir "$RUN_DIR" \
  --thresholds 0.01,0.3 \
  --steps 10 \
  --lr 0.02 \
  --exact-device "$EXACT_DEVICE"

echo "=== [smoke 4/6] aggregation ==="
conda run -n "$CONDA_ENV" python safe_maqaoa/aggregate.py \
  --results-root results_smoke \
  --out results_smoke/runs_long.csv \
  --mw 4

echo "=== [smoke 5/6] computation-cost aggregation (terms / VRAM / time) ==="
conda run -n "$CONDA_ENV" python safe_maqaoa/aggregate_cost.py \
  --results-root results_smoke \
  --out results_smoke/cost_metrics.csv

echo "=== [smoke 6/6] summary tables ==="
conda run -n "$CONDA_ENV" python figures/make_summary_tables.py \
  --long-csv results_smoke/runs_long.csv \
  --out-dir results_smoke/figout

echo
echo "smoke artifacts:"
ls -1 "$RUN_DIR" | sed 's/^/  /'
echo "smoke PASS"
