"""The padopauli surrogate agrees with an exact reference.

Builds a small (6-qubit, p=2) SK ma-QAOA circuit, compiles it with padopauli at
full operator weight (no LWPP truncation) in float64 on the CPU, and compares
expectation values for random parameter batches against PennyLane's
default.qubit state-vector reference. Agreement below 1e-7 validates the
surrogate against the PennyLane state-vector reference.

Also checks that the truncated (w_max = 4) surrogate stays finite and close to
exact at small angles, where truncation effects are mild.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from padopauli import compile_program
from safe_maqaoa.problem_factory import build_sk_problem
from safe_maqaoa.run_ma_qaoa_lwpp_exact_compare import (
    _build_ising_observable,
    _build_ma_qaoa_circuit,
)


def main() -> int:
    torch.manual_seed(0)
    problem = build_sk_problem(n_qubits=6, seed=706200)
    obs = _build_ising_observable(problem)
    circuit, n_params = _build_ma_qaoa_circuit(problem, p_layers=2)

    rng = np.random.default_rng(0)
    thetas = torch.as_tensor(rng.uniform(-0.5, 0.5, size=(4, n_params)), dtype=torch.float64)

    # max_weight above n_qubits: no truncation
    program = compile_program(
        circuit=circuit,
        observables=[obs],
        preset="cpu",
        preset_overrides={"dtype": "float64", "max_weight": 1_000_000_000},
    )
    ours = np.array([float(program.expval(t, obs_index=0)) for t in thetas])
    ref = np.array([float(program.expvals_reference(t)[0]) for t in thetas])
    err_full = float(np.max(np.abs(ours - ref)))
    ok_full = err_full < 1e-7
    print(f"[{'OK' if ok_full else 'FAIL'}] full-weight vs PennyLane: max |diff| = {err_full:.3e}")

    # Autograd path used by LWPP pre-training.
    t = thetas.clone().requires_grad_(True)
    e = program.expval(t, obs_index=0)
    e.sum().backward()
    grad_finite = bool(torch.isfinite(t.grad).all())
    print(f"[{'OK' if grad_finite else 'FAIL'}] autograd gradient finite: {grad_finite}")

    # LWPP truncation (w_max = 4): surrogate, not exact; check it is finite and
    # tracks the exact value at small angles.
    program_lwpp = compile_program(
        circuit=circuit,
        observables=[obs],
        preset="cpu",
        preset_overrides={"dtype": "float64", "max_weight": 4},
    )
    small = 0.05 * thetas
    ours_lwpp = np.array([float(program_lwpp.expval(t, obs_index=0)) for t in small])
    ref_small = np.array([float(program.expvals_reference(t)[0]) for t in small])
    err_lwpp = float(np.max(np.abs(ours_lwpp - ref_small)))
    ok_lwpp = np.isfinite(ours_lwpp).all() and err_lwpp < 0.05
    print(f"[{'OK' if ok_lwpp else 'FAIL'}] w_max=4 surrogate at small angles: max |diff| = {err_lwpp:.3e}")

    ok = ok_full and grad_finite and ok_lwpp
    print("engine check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
