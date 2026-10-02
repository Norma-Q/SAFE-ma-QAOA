"""Benchmark PennyLane exact devices for the ma-QAOA fine-tuning workload.

Times one batched energy+gradient evaluation (the inner loop of exact
fine-tuning: the batch of initializations, default 11) on each candidate device, per
(q, p) setting, and prints a per-setting and overall recommendation.

  python smoke/benchmark_exact_device.py
  python smoke/benchmark_exact_device.py --devices lightning.qubit
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from safe_maqaoa.problem_factory import load_problem
from safe_maqaoa.run_ma_qaoa_lwpp_exact_compare import (
    _build_exact_qnode,
    _build_ma_qaoa_circuit,
    _exact_energy_and_grad_batch,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--devices", type=str, default="lightning.amdgpu,lightning.qubit")
    ap.add_argument("--qubits", type=str, default="12,16,20")
    ap.add_argument("--layers", type=str, default="2,4")
    ap.add_argument("--batch", type=int, default=11)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=str, default="results_smoke/exact_device_benchmark.csv")
    args = ap.parse_args()

    devices = [d.strip() for d in str(args.devices).split(",") if d.strip()]
    qubits = [int(x) for x in str(args.qubits).split(",") if x.strip()]
    layers = [int(x) for x in str(args.layers).split(",") if x.strip()]

    rows = []
    totals = {d: 0.0 for d in devices}
    for q in qubits:
        for p in layers:
            seed = 700000 + q * 1000 + p * 100 + 0
            problem_path = _REPO_ROOT / "data" / "problems" / "sk" / f"sk_q{q}_p{p}_inst0_seed{seed}.json"
            problem = load_problem(problem_path)
            circuit, n_params = _build_ma_qaoa_circuit(problem, p_layers=p)
            rng = np.random.default_rng(0)
            theta_np = rng.uniform(-0.5, 0.5, size=(int(args.batch), n_params))

            for dev_name in devices:
                try:
                    qnode, dev, diff, lib = _build_exact_qnode(circuit, problem, dev_name)
                except Exception as exc:
                    print(f"q={q} p={p} {dev_name}: UNAVAILABLE ({exc})")
                    rows.append({"q": q, "p": p, "device": dev_name, "diff": "-", "mean_s": None})
                    totals[dev_name] = float("inf")
                    continue
                qnode_device = "cuda" if "gpu" in dev.lower() else "cpu"
                thetas = torch.as_tensor(theta_np, dtype=torch.float64, device=qnode_device)

                _exact_energy_and_grad_batch(qnode, lib, thetas, dev)  # warmup
                times = []
                for _ in range(int(args.reps)):
                    t0 = time.perf_counter()
                    _exact_energy_and_grad_batch(qnode, lib, thetas, dev)
                    times.append(time.perf_counter() - t0)
                mean_s = float(np.mean(times))
                totals[dev_name] += mean_s
                rows.append({"q": q, "p": p, "device": dev_name, "diff": diff, "mean_s": mean_s})
                print(f"q={q} p={p} {dev:18s} diff={diff:8s} mean={mean_s:8.3f}s (reps={args.reps})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["q", "p", "device", "diff", "mean_s"])
        w.writeheader()
        w.writerows(rows)

    finite = {d: t for d, t in totals.items() if np.isfinite(t)}
    if finite:
        best = min(finite, key=lambda d: finite[d])
        print("\nsummed batched-step time per device:")
        for d, t in sorted(finite.items(), key=lambda kv: kv[1]):
            print(f"  {d:18s} {t:8.3f}s")
        print(f"recommendation: EXACT_DEVICE={best}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
