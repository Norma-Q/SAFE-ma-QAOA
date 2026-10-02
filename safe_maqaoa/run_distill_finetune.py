"""Parameter-distillation exact fine-tuning for SAFE ma-QAOA (paper Sec. III-C.2/3).

Given a completed run directory produced by ``run_ma_qaoa_lwpp_exact_compare.py``,
this script takes the LWPP-final parameters of every initialization, removes the
angles whose magnitude is at or below a distillation threshold, and fine-tunes
the remaining active parameters for a fixed number of steps on the exact energy
objective (PennyLane state-vector device) with Adam.

Distilled rotation gates are not built: for each initialization the
ma-QAOA circuit is reconstructed from only the active rotations (parameter
indices remapped to 0..n_active-1), and a fresh exact qnode is compiled for
that pruned circuit. Because active sets differ per initialization, each
initialization is fine-tuned sequentially, on its own pruned circuit.

Outputs (inside the run directory), per threshold tag T (e.g. 0.3 -> "0p3"):
  - distill_thr{T}_trace.csv            step, exact_energy_mean, exact_energy_min
                                        (mean/min across initializations)
  - distill_thr{T}_trace_per_seed.csv   stage, step, init_index, init_label, exact_energy
  - distill_thr{T}_init_energy.csv      per-init energy at the pruned starting point,
                                        active/pruned parameter counts
  - distill_thr{T}_result.json          config, prune summary, final thetas/energies
                                        (thetas reported in full width, pruned slots = 0)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from padopauli import CliffordGate, PauliRotation
from safe_maqaoa.problem_factory import IsingProblem, load_problem
from safe_maqaoa.run_ma_qaoa_lwpp_exact_compare import (
    _append_csv_row,
    _build_exact_qnode,
    _build_ma_qaoa_circuit,
    _cuda_peak_bytes,
    _cuda_reset_peak,
    _exact_energy_and_grad_batch,
    _exact_energy_batch,
    _now_tag,
)


def _threshold_tag(threshold: float) -> str:
    return f"{float(threshold):g}".replace(".", "p").replace("-", "m")


def _load_lwpp_final_thetas(run_dir: Path) -> Tuple[np.ndarray, List[str], str]:
    """LWPP-final thetas per init, with provenance. Tries, in order:

    1. lwpp_final_thetas.json      (written by the runner at the last LWPP step)
    2. results.json                (lwpp_initialized_exact_finetuning.pre_snapshots[-1])
    3. theta_checkpoints.pt        (theta_lwpp_end)
    """
    p1 = run_dir / "lwpp_final_thetas.json"
    if p1.exists():
        raw = json.loads(p1.read_text(encoding="utf-8"))
        thetas = np.asarray(raw["thetas"], dtype=np.float64)
        labels = [str(x) for x in raw.get("init_labels", [])]
        return thetas, labels, "lwpp_final_thetas.json"

    p2 = run_dir / "results.json"
    if p2.exists():
        raw = json.loads(p2.read_text(encoding="utf-8"))
        snaps = raw.get("lwpp_initialized_exact_finetuning", {}).get("pre_snapshots", [])
        labels = [str(x) for x in raw.get("config", {}).get("init", {}).get("labels", [])]
        if snaps:
            thetas = np.asarray(snaps[-1]["thetas"], dtype=np.float64)
            return thetas, labels, "results.json:pre_snapshots[-1]"

    p3 = run_dir / "theta_checkpoints.pt"
    if p3.exists():
        ckpt = torch.load(p3, map_location="cpu")
        if "theta_lwpp_end" in ckpt:
            thetas = np.asarray(torch.as_tensor(ckpt["theta_lwpp_end"]).numpy(), dtype=np.float64)
            labels = [str(x) for x in ckpt.get("theta_init_labels", [])]
            return thetas, labels, "theta_checkpoints.pt:theta_lwpp_end"

    raise FileNotFoundError(
        f"No LWPP-final thetas found in {run_dir} "
        "(expected lwpp_final_thetas.json, results.json pre_snapshots, or theta_checkpoints.pt)"
    )


def _prune_circuit(circuit: Sequence[Any], active_indices: Sequence[int]) -> List[Any]:
    """Rebuild the circuit with only the active rotation gates.

    Clifford gates are kept as-is. A PauliRotation is kept only when its
    param_idx is active, and its param_idx is remapped to the gate's rank in
    `active_indices` (0..n_active-1). Distilled rotations are simply not built.
    """
    remap = {int(old): new for new, old in enumerate(int(i) for i in active_indices)}
    pruned: List[Any] = []
    for gate in circuit:
        if isinstance(gate, PauliRotation):
            new_idx = remap.get(int(gate.param_idx))
            if new_idx is None:
                continue
            pruned.append(PauliRotation(str(gate.pauli), list(gate.qubits), param_idx=int(new_idx)))
        elif isinstance(gate, CliffordGate):
            pruned.append(gate)
        else:
            raise TypeError(f"Unsupported gate class in circuit: {type(gate).__name__}")
    return pruned


def _finetune_one_init(
    *,
    theta_active0: np.ndarray,
    pruned_circuit: Sequence[Any],
    problem: IsingProblem,
    exact_device: str,
    steps: int,
    lr: float,
) -> Tuple[np.ndarray, List[float], float, float, str, str]:
    """Fine-tune one initialization on its pruned circuit.

    Returns (theta_active_final, per-step pre-update energies, start energy,
    post-loop final energy, device, diff).
    """
    qnode, exact_dev, exact_diff, exact_lib = _build_exact_qnode(
        pruned_circuit, problem, str(exact_device).strip()
    )
    qnode_device = "cuda" if "gpu" in str(exact_dev).lower() else "cpu"

    n_active = int(theta_active0.shape[0])
    if n_active == 0:
        # Every rotation was distilled away: the circuit is the bare H layer and
        # nothing can train. Record the constant energy for all steps.
        with torch.no_grad():
            e0_t = _exact_energy_batch(
                qnode, exact_lib, torch.zeros((1, 0), dtype=torch.float64, device=qnode_device), exact_dev
            )
        e0 = float(e0_t.reshape(-1)[0].cpu().item())
        return theta_active0.copy(), [e0] * int(steps), e0, e0, str(exact_dev), str(exact_diff)

    thetas = torch.nn.Parameter(
        torch.as_tensor(theta_active0.reshape(1, n_active), dtype=torch.float64, device=qnode_device)
    )
    optimizer = torch.optim.Adam([thetas], lr=float(lr))

    with torch.no_grad():
        e0_t = _exact_energy_batch(qnode, exact_lib, thetas.detach(), exact_dev)
    e0 = float(e0_t.reshape(-1)[0].cpu().item())

    # Row k records the pre-update energy E(theta_k).
    energies: List[float] = []
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        exact_e, exact_g = _exact_energy_and_grad_batch(qnode, exact_lib, thetas, exact_dev)
        thetas.grad = exact_g
        optimizer.step()
        energies.append(float(exact_e.reshape(-1)[0].detach().cpu().item()))

    with torch.no_grad():
        e_final_t = _exact_energy_batch(qnode, exact_lib, thetas.detach(), exact_dev)
    e_final = float(e_final_t.reshape(-1)[0].cpu().item())

    return (
        thetas.detach().cpu().numpy().reshape(-1).astype(np.float64),
        energies,
        e0,
        e_final,
        str(exact_dev),
        str(exact_diff),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Distill (remove near-zero angles; pruned gates are not built) and "
            "exact-fine-tune every initialization of a completed run."
        )
    )
    p.add_argument("--run-dir", type=str, required=True, help="Completed run directory of run_ma_qaoa_lwpp_exact_compare.py")
    p.add_argument("--thresholds", type=str, default="0.01,0.3", help="Comma-separated distillation thresholds.")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--exact-device", type=str, default="", help="Optional PennyLane device override.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--force", action="store_true", help="Redo thresholds whose result JSON already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    problem: IsingProblem = load_problem(run_dir / "problem_copy.json")
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    p_layers = int(run_meta["p_layers"])

    thetas_np, init_labels, theta_source = _load_lwpp_final_thetas(run_dir)
    circuit, n_params = _build_ma_qaoa_circuit(problem, p_layers=p_layers)
    if int(thetas_np.shape[1]) != int(n_params):
        raise ValueError(
            f"theta width {int(thetas_np.shape[1])} does not match circuit n_params {int(n_params)} "
            f"(p_layers={p_layers})"
        )
    n_inits = int(thetas_np.shape[0])
    if not init_labels:
        init_labels = [f"init_{i + 1}" for i in range(n_inits)]

    print(f"[distill] run={run_dir.name} inits={n_inits} n_params={n_params} theta_source={theta_source}")

    thresholds = [float(x) for x in str(args.thresholds).split(",") if x.strip() != ""]

    distill_cost_records: List[Dict[str, Any]] = []
    for thr in thresholds:
        tag = _threshold_tag(thr)
        result_path = run_dir / f"distill_thr{tag}_result.json"
        if result_path.exists() and not bool(args.force):
            print(f"[distill] thr={thr:g} already done, skip ({result_path.name})")
            continue

        trace_csv = run_dir / f"distill_thr{tag}_trace.csv"
        trace_seed_csv = run_dir / f"distill_thr{tag}_trace_per_seed.csv"
        init_energy_csv = run_dir / f"distill_thr{tag}_init_energy.csv"
        for f in (trace_csv, trace_seed_csv, init_energy_csv):
            if f.exists():
                f.unlink()

        # Distillation: drop angles with |theta| <= threshold (paper Sec. III-C.2).
        active_sets: List[np.ndarray] = [
            np.flatnonzero(np.abs(thetas_np[i]) > float(thr)) for i in range(n_inits)
        ]
        n_active = [int(a.shape[0]) for a in active_sets]
        print(
            f"[distill] thr={thr:g} active(mean)={float(np.mean(n_active)):.1f}/{n_params} "
            f"pruned_frac(mean)={1.0 - float(np.mean(n_active)) / float(n_params):.3f}"
        )

        theta_final_full = np.zeros_like(thetas_np)
        energy_series: List[List[float]] = []
        e_start: List[float] = []
        e_final: List[float] = []
        exact_dev_used = ""
        exact_diff_used = ""

        _cuda_reset_peak()
        _t_distill0 = time.perf_counter()
        for i in range(n_inits):
            label = init_labels[i] if i < len(init_labels) else f"init_{i + 1}"
            active_idx = active_sets[i]
            if int(active_idx.shape[0]) == 0:
                print(
                    f"[distill][warn] thr={thr:g} init {i} ({label}): all parameters pruned; "
                    "circuit reduces to the bare H layer"
                )

            pruned_circuit = _prune_circuit(circuit, active_idx.tolist())
            theta_active_final, energies, e0, e_end, dev_used, diff_used = _finetune_one_init(
                theta_active0=thetas_np[i, active_idx],
                pruned_circuit=pruned_circuit,
                problem=problem,
                exact_device=str(args.exact_device),
                steps=int(args.steps),
                lr=float(args.lr),
            )
            exact_dev_used, exact_diff_used = dev_used, diff_used
            theta_final_full[i, active_idx] = theta_active_final
            energy_series.append(energies)
            e_start.append(float(e0))
            e_final.append(float(e_end))

            _append_csv_row(
                init_energy_csv,
                {
                    "init_index": int(i),
                    "init_label": str(label),
                    "exact_energy_at_pruned_init": float(e0),
                    "n_params": int(n_params),
                    "n_active": int(active_idx.shape[0]),
                    "n_pruned": int(n_params) - int(active_idx.shape[0]),
                },
                fieldnames=[
                    "init_index",
                    "init_label",
                    "exact_energy_at_pruned_init",
                    "n_params",
                    "n_active",
                    "n_pruned",
                ],
            )
            for step, e in enumerate(energies):
                _append_csv_row(
                    trace_seed_csv,
                    {
                        "stage": "distill_finetune",
                        "step": int(step),
                        "init_index": int(i),
                        "init_label": str(label),
                        "exact_energy": float(e),
                    },
                    fieldnames=["stage", "step", "init_index", "init_label", "exact_energy"],
                )
            print(
                f"[distill] thr={thr:g} init {i:02d} ({label}) "
                f"n_active={int(active_idx.shape[0]):4d} start={e0:.6f} final={e_final[-1]:.6f}"
            )

        distill_wall_s = float(time.perf_counter() - _t_distill0)
        distill_peak_vram_bytes = _cuda_peak_bytes()

        # Mean/min across initializations per step (same layout as the runner traces).
        series = np.asarray(energy_series, dtype=np.float64)  # (n_inits, steps)
        for step in range(int(args.steps)):
            _append_csv_row(
                trace_csv,
                {
                    "step": int(step),
                    "exact_energy_mean": float(np.mean(series[:, step])),
                    "exact_energy_min": float(np.min(series[:, step])),
                },
                fieldnames=["step", "exact_energy_mean", "exact_energy_min"],
            )

        active_mask = np.zeros((n_inits, int(n_params)), dtype=np.int8)
        for i, idx in enumerate(active_sets):
            active_mask[i, idx] = 1

        result = {
            "schema_version": "distill_finetune_v1",
            "created_at": _now_tag(),
            "run_dir": str(run_dir),
            "threshold": float(thr),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "exact_device": str(exact_dev_used),
            "exact_diff_method": str(exact_diff_used),
            "theta_source": str(theta_source),
            "p_layers": int(p_layers),
            "n_params": int(n_params),
            "init_labels": list(init_labels),
            "n_active_per_init": [int(x) for x in n_active],
            "active_indices_per_init": [a.tolist() for a in active_sets],
            "energy_at_pruned_init": [float(x) for x in e_start],
            "final_exact_energy": [float(x) for x in e_final],
            "final_thetas": theta_final_full.tolist(),
            "active_mask": active_mask.tolist(),
            "cost_metrics": {
                "wall_s": distill_wall_s,
                "peak_vram_bytes": distill_peak_vram_bytes,
                "n_inits": int(n_inits),
                "steps": int(args.steps),
                "n_active_mean": float(np.mean(n_active)) if n_active else 0.0,
                "exact_device": str(exact_dev_used),
                "vram_note": (
                    "torch.cuda peak (torch allocator). PennyLane lightning "
                    "statevector uses a separate allocator not captured here."
                ),
            },
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        distill_cost_records.append(
            {
                "threshold": float(thr),
                "wall_s": distill_wall_s,
                "peak_vram_bytes": distill_peak_vram_bytes,
                "n_active_mean": float(np.mean(n_active)) if n_active else 0.0,
            }
        )
        print(f"[distill] thr={thr:g} done -> {result_path.name}")

    if distill_cost_records:
        (run_dir / "cost_metrics_distill.json").write_text(
            json.dumps(
                {
                    "schema": "safe_maqaoa.cost_metrics_distill/v1",
                    "created_at": _now_tag(),
                    "run_dir": str(run_dir),
                    "exact_device": str(args.exact_device),
                    "per_threshold": distill_cost_records,
                    "total_wall_s": float(sum(r["wall_s"] for r in distill_cost_records)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
