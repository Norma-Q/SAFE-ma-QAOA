#!/usr/bin/env bash
set -euo pipefail

# Full SAFE ma-QAOA reproduction sweep: w_max = 4 then w_max = 3, each over all
# three families, q in {12,16,20} (grids 3x4/4x4/4x5), p in {2,3,4}, 5 instances.
# Each family sweep is resume-safe, so this script can be interrupted and relaunched.
# Environment knobs are forwarded to sweeps/run_family_sweep.sh.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

for mw in 4 3; do
  echo "=== w_max=${mw}, full grid ==="
  for fam in sk grid2d maxcut; do
    echo "--- family=${fam} mw=${mw} ---"
    FAMILY="$fam" MW="$mw" bash sweeps/run_family_sweep.sh
  done
done

echo "All paper sweeps finished."
