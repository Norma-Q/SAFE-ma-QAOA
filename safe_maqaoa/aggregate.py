"""Aggregation of SAFE ma-QAOA sweep results into paper metrics.

Walks ``results/mw{MW}/{family}/{case}/`` run directories and produces one long
table with a row per (run, method, init): approximation-ratio trajectory
metrics, first-hit step, active parameter count, and the fine-tuning cost
ballpark estimate. Figs. 4-6 and the summary tables read this table.

Metric definitions (paper Sec. III-D):
  alpha(theta)   = (E_max - E(theta)) / (E_max - E_min), with E_min/E_max from
                   exhaustive enumeration of the 2^n spin configurations.
  tau_0.99       = min { t : alpha(t) >= 0.99 * alpha_best } over the method's
                   own exact fine-tuning trajectory. Exact-stage traces record
                   the pre-update energy, so step 0 is the starting point for
                   every method.
  C_ballpark     = N_active * tau_0.99 per (run, init); aggregated as the mean
                   over instances and initializations.

Methods:
  exact_only            direct exact optimization of all parameters
  safe_nodistill        LWPP pre-training -> exact fine-tuning (threshold 0)
  safe_distill_thr{T}   LWPP pre-training -> distillation at threshold T
                        -> exact fine-tuning of active parameters
  lwpp_only             exact energy at the LWPP-final parameters (no exact
                        fine-tuning); reference lines in Fig. 4
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from safe_maqaoa.problem_factory import IsingProblem, load_problem


# --------------------------------------------------------------------------
# Exact spectral bounds via exhaustive enumeration
# --------------------------------------------------------------------------

def energy_bounds(problem: IsingProblem) -> Tuple[float, float]:
    """(E_min, E_max) over all 2^n spin configurations."""
    n = int(problem.n_qubits)
    n_states = 1 << n
    state_ids = np.arange(n_states, dtype=np.uint32)
    energies = np.zeros((n_states,), dtype=np.float64)
    for u, v, j in problem.edges:
        su = 1.0 - 2.0 * ((state_ids >> np.uint32(u)) & 1).astype(np.float64)
        sv = 1.0 - 2.0 * ((state_ids >> np.uint32(v)) & 1).astype(np.float64)
        energies += float(j) * su * sv
    for q, h in problem.fields:
        sq = 1.0 - 2.0 * ((state_ids >> np.uint32(q)) & 1).astype(np.float64)
        energies += float(h) * sq
    return float(energies.min()), float(energies.max())


def alpha_from_energy(e: np.ndarray | float, e_min: float, e_max: float) -> np.ndarray | float:
    denom = float(e_max) - float(e_min)
    if denom <= 0.0:
        raise ValueError("degenerate spectrum: E_max <= E_min")
    return (float(e_max) - np.asarray(e, dtype=np.float64)) / denom


# --------------------------------------------------------------------------
# Run directory parsing
# --------------------------------------------------------------------------

_CASE_RE = re.compile(
    r"^(?P<family>sk|maxcut)_q(?P<q>\d+)_p(?P<p>\d+)_inst(?P<inst>\d+)_seed(?P<seed>\d+)$"
)
_GRID_RE = re.compile(
    r"^grid(?P<rows>\d)x(?P<cols>\d)_p(?P<p>\d+)_inst(?P<inst>\d+)_seed(?P<seed>\d+)$"
)


@dataclass(frozen=True)
class CaseKey:
    family: str      # sk | grid2d | maxcut
    size: str        # "12" / "16" / "20" for sk & maxcut, "3x4" ... for grid2d
    n_qubits: int
    p: int
    inst: int
    seed: int
    mw: int


def parse_case_dir(run_dir: Path, mw: int) -> Optional[CaseKey]:
    name = run_dir.name
    m = _CASE_RE.match(name)
    if m:
        q = int(m.group("q"))
        return CaseKey(
            family=str(m.group("family")),
            size=str(q),
            n_qubits=q,
            p=int(m.group("p")),
            inst=int(m.group("inst")),
            seed=int(m.group("seed")),
            mw=int(mw),
        )
    g = _GRID_RE.match(name)
    if g:
        rows, cols = int(g.group("rows")), int(g.group("cols"))
        return CaseKey(
            family="grid2d",
            size=f"{rows}x{cols}",
            n_qubits=rows * cols,
            p=int(g.group("p")),
            inst=int(g.group("inst")),
            seed=int(g.group("seed")),
            mw=int(mw),
        )
    return None


def _read_per_seed_energies(path: Path, energy_col: str = "exact_energy") -> Dict[int, List[Tuple[int, float]]]:
    """{init_index: [(step, energy), ...]} sorted by step."""
    out: Dict[int, List[Tuple[int, float]]] = {}
    if not path.exists():
        return out
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val = row.get(energy_col)
            if val in (None, "", "None"):
                continue
            idx = int(row["init_index"])
            out.setdefault(idx, []).append((int(row["step"]), float(val)))
    for idx in out:
        out[idx].sort(key=lambda t: t[0])
    return out


def _labels_from_per_seed(path: Path) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    if not path.exists():
        return labels
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.setdefault(int(row["init_index"]), str(row.get("init_label", "")))
    return labels


@dataclass
class MethodTrajectory:
    method: str
    init_index: int
    init_label: str
    alphas: np.ndarray          # trajectory alpha(t), t = t0 .. t0+len-1
    t0: int                     # 0 when the pre-update start is included, else 1
    n_active: int
    n_params: int


def _tau_first_hit(alphas: np.ndarray, t0: int, level: float = 0.99) -> Tuple[float, float, float]:
    """(tau, alpha_best, alpha_final) for one trajectory."""
    if alphas.size == 0:
        raise ValueError("empty trajectory")
    a_best = float(np.max(alphas))
    hit = np.flatnonzero(alphas >= float(level) * a_best)
    tau = float(int(hit[0]) + int(t0))
    return tau, a_best, float(alphas[-1])


def iter_method_trajectories(
    run_dir: Path,
    e_min: float,
    e_max: float,
    *,
    n_params_fallback: Optional[int] = None,
) -> Iterable[MethodTrajectory]:
    """Yield per-init trajectories for every method present in a run directory.

    Tolerates a corrupt/truncated results.json by falling back to run_meta.json
    for the init labels and to `n_params_fallback` for the parameter count.
    """
    n_params: Optional[int] = None
    labels_cfg: Dict[int, str] = {}
    try:
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        n_params = int(results["config"]["n_params"])
        labels_cfg = {i: str(l) for i, l in enumerate(results["config"]["init"].get("labels", []))}
    except (json.JSONDecodeError, KeyError, OSError):
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                labels_cfg = {i: str(l) for i, l in enumerate(meta.get("init", {}).get("labels", []))}
            except (json.JSONDecodeError, OSError):
                pass
    if n_params is None:
        if n_params_fallback is None:
            raise ValueError(f"cannot determine n_params for {run_dir} (corrupt results.json)")
        n_params = int(n_params_fallback)

    lwpp_seed = _read_per_seed_energies(run_dir / "lwpp_pre_trace_per_seed.csv")
    fine_seed = _read_per_seed_energies(run_dir / "exact_finetune_trace_per_seed.csv")
    direct_seed = _read_per_seed_energies(run_dir / "exact_only_trace_per_seed.csv")

    def _alpha(vals: Sequence[float]) -> np.ndarray:
        return np.asarray(alpha_from_energy(np.asarray(vals, dtype=np.float64), e_min, e_max))

    for idx, steps in sorted(direct_seed.items()):
        vals = [e for (_s, e) in steps]
        yield MethodTrajectory(
            method="exact_only",
            init_index=idx,
            init_label=labels_cfg.get(idx, f"init_{idx + 1}"),
            alphas=_alpha(vals),
            t0=0,
            n_active=n_params,
            n_params=n_params,
        )

    for idx, steps in sorted(fine_seed.items()):
        vals = [e for (_s, e) in steps]
        yield MethodTrajectory(
            method="safe_nodistill",
            init_index=idx,
            init_label=labels_cfg.get(idx, f"init_{idx + 1}"),
            alphas=_alpha(vals),
            t0=0,
            n_active=n_params,
            n_params=n_params,
        )

    # LWPP-only reference: single point at the LWPP-final parameters.
    for idx, steps in sorted(lwpp_seed.items()):
        if not steps:
            continue
        yield MethodTrajectory(
            method="lwpp_only",
            init_index=idx,
            init_label=labels_cfg.get(idx, f"init_{idx + 1}"),
            alphas=_alpha([steps[-1][1]]),
            t0=0,
            n_active=n_params,
            n_params=n_params,
        )

    # SAFE with distillation, one method per threshold found in the run dir.
    for init_csv in sorted(run_dir.glob("distill_thr*_init_energy.csv")):
        tag = init_csv.name[len("distill_thr"):-len("_init_energy.csv")]
        seed_csv = run_dir / f"distill_thr{tag}_trace_per_seed.csv"
        traj = _read_per_seed_energies(seed_csv)
        n_active: Dict[int, int] = {}
        with init_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                n_active[int(row["init_index"])] = int(row["n_active"])
        threshold = float(tag.replace("p", ".").replace("m", "-"))
        for idx, steps in sorted(traj.items()):
            vals = [e for (_s, e) in steps]
            yield MethodTrajectory(
                method=f"safe_distill_thr{threshold:g}",
                init_index=idx,
                init_label=labels_cfg.get(idx, f"init_{idx + 1}"),
                alphas=_alpha(vals),
                t0=0,
                n_active=n_active.get(idx, n_params),
                n_params=n_params,
            )


# --------------------------------------------------------------------------
# Long-table construction
# --------------------------------------------------------------------------

LONG_FIELDS = [
    "family", "size", "n_qubits", "p", "mw", "inst", "seed", "method",
    "init_index", "init_label", "alpha_start", "alpha_final", "alpha_best",
    "tau099", "n_active", "n_params", "c_ballpark", "e_min", "e_max",
]


def collect_long_rows(results_root: Path, mw_values: Sequence[int] = (3, 4)) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    bounds_cache: Dict[str, Tuple[float, float]] = {}

    for mw in mw_values:
        mw_root = results_root / f"mw{int(mw)}"
        if not mw_root.is_dir():
            continue
        for family_dir in sorted(p for p in mw_root.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
                key = parse_case_dir(run_dir, mw=int(mw))
                if key is None or not (run_dir / "results.json").exists():
                    continue

                problem_path = run_dir / "problem_copy.json"
                problem = load_problem(problem_path)
                cache_key = f"{key.family}/{run_dir.name}"
                if cache_key not in bounds_cache:
                    bounds_cache[cache_key] = energy_bounds(problem)
                e_min, e_max = bounds_cache[cache_key]
                n_params_fallback = int(key.p) * (
                    len(problem.edges) + len(problem.fields) + int(problem.n_qubits)
                )

                for traj in iter_method_trajectories(
                    run_dir, e_min, e_max, n_params_fallback=n_params_fallback
                ):
                    tau, a_best, a_final = _tau_first_hit(traj.alphas, traj.t0)
                    a_start = float(traj.alphas[0]) if traj.t0 == 0 else None
                    rows.append(
                        {
                            "family": key.family,
                            "size": key.size,
                            "n_qubits": key.n_qubits,
                            "p": key.p,
                            "mw": key.mw,
                            "inst": key.inst,
                            "seed": key.seed,
                            "method": traj.method,
                            "init_index": traj.init_index,
                            "init_label": traj.init_label,
                            "alpha_start": a_start,
                            "alpha_final": a_final,
                            "alpha_best": a_best,
                            "tau099": tau,
                            "n_active": traj.n_active,
                            "n_params": traj.n_params,
                            "c_ballpark": float(traj.n_active) * float(tau),
                            "e_min": e_min,
                            "e_max": e_max,
                        }
                    )
    return rows


def write_long_csv(rows: List[Dict[str, Any]], out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LONG_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in LONG_FIELDS})
    return out_csv


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate sweep results into the long metrics table.")
    ap.add_argument("--results-root", type=str, default="results")
    ap.add_argument("--out", type=str, default="figures/output/runs_long.csv")
    ap.add_argument("--mw", type=str, default="3,4", help="Comma-separated w_max values to scan.")
    args = ap.parse_args()

    mw_values = [int(x) for x in str(args.mw).split(",") if x.strip()]
    rows = collect_long_rows(Path(args.results_root), mw_values=mw_values)
    out = write_long_csv(rows, Path(args.out))
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
