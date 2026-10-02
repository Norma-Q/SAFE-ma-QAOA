"""Fig. 7: cost-angle structure in the largest exact-energy-gap cases.

For each family, selects the (run, init) with the largest absolute exact-energy
change during exact fine-tuning (|E(LWPP-final) - E(after fine-tuning)|), then

  left panel : that gap vs. the circular cosine similarity between the
               LWPP-final and fine-tuned cost-angle vectors,
               sim = mean_i cos(theta_i - phi_i)  over all cost angles
               (edge + field rotations, all layers)
  right panels: the selected cost layer's edge angles drawn on the problem
                graph, top = LWPP-final, bottom = after exact fine-tuning.

Layouts are deterministic: circular for SK/Max-Cut, lattice for the 2D grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from safe_maqaoa.problem_factory import load_problem


def _per_seed_last(path: Path) -> Dict[int, float]:
    out: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("exact_energy")
            if v in (None, "", "None"):
                continue
            out[int(row["init_index"])].append((int(row["step"]), float(v)))
    return {k: sorted(v)[-1][1] for k, v in out.items() if v}


def _load_thetas(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """(theta_lwpp_final, theta_fine_final), each (batch, n_params)."""
    lw = run_dir / "lwpp_final_thetas.json"
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if lw.exists():
        theta0 = np.asarray(json.loads(lw.read_text(encoding="utf-8"))["thetas"], dtype=np.float64)
    else:
        snaps = results["lwpp_initialized_exact_finetuning"]["pre_snapshots"]
        theta0 = np.asarray(snaps[-1]["thetas"], dtype=np.float64)
    theta1 = np.asarray(results["lwpp_initialized_exact_finetuning"]["final_thetas"], dtype=np.float64)
    return theta0, theta1


def _cost_angle_slices(problem, p_layers: int) -> List[slice]:
    """Slice of the edge-angle block for each layer (theta layout: [edges|fields|mixers] per layer)."""
    n_e, n_f, n_q = len(problem.edges), len(problem.fields), int(problem.n_qubits)
    block = n_e + n_f + n_q
    return [slice(l * block, l * block + n_e) for l in range(int(p_layers))]


def _all_cost_indices(problem, p_layers: int) -> np.ndarray:
    n_e, n_f, n_q = len(problem.edges), len(problem.fields), int(problem.n_qubits)
    block = n_e + n_f + n_q
    idx: List[int] = []
    for l in range(int(p_layers)):
        idx.extend(range(l * block, l * block + n_e + n_f))
    return np.asarray(idx, dtype=np.int64)


def circular_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.cos(np.asarray(a) - np.asarray(b))))


def _node_positions(problem) -> Dict[int, Tuple[float, float]]:
    meta = dict(problem.meta)
    if "rows" in meta and "cols" in meta:
        r, c = int(meta["rows"]), int(meta["cols"])
        return {i * c + j: (float(j), float(r - 1 - i)) for i in range(r) for j in range(c)}
    n = int(problem.n_qubits)
    return {
        q: (math.cos(2 * math.pi * q / n + math.pi / 2), math.sin(2 * math.pi * q / n + math.pi / 2))
        for q in range(n)
    }


def _draw_edge_angles(ax, problem, angles: Sequence[float], vmax: float) -> None:
    pos = _node_positions(problem)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    order = np.argsort(np.abs(np.asarray(angles)))
    for k in order:
        u, v, _w = problem.edges[int(k)]
        a = float(angles[int(k)])
        (x0, y0), (x1, y1) = pos[int(u)], pos[int(v)]
        ax.plot([x0, x1], [y0, y1], color=cmap(norm(a)), lw=0.6 + 3.2 * abs(a) / max(vmax, 1e-9), alpha=0.9)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=42, facecolor="white", edgecolor="dimgray", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")


def _select_case(mw_root: Path, family: str) -> Optional[Tuple[Path, int, float]]:
    """(run_dir, init_index, gap) with the largest fine-tuning energy gap."""
    best: Optional[Tuple[Path, int, float]] = None
    fam_dir = mw_root / family
    if not fam_dir.is_dir():
        return None
    for run_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
        if not (run_dir / "results.json").exists():
            continue
        lwpp_last = _per_seed_last(run_dir / "lwpp_pre_trace_per_seed.csv")
        fine_last = _per_seed_last(run_dir / "exact_finetune_trace_per_seed.csv")
        for idx in sorted(set(lwpp_last) & set(fine_last)):
            gap = abs(lwpp_last[idx] - fine_last[idx])
            if best is None or gap > best[2]:
                best = (run_dir, idx, gap)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=str, default="results")
    ap.add_argument("--mw", type=int, default=4)
    ap.add_argument(
        "--case",
        action="append",
        default=[],
        help="family:run_name:init_index:layer (repeatable). Defaults to the largest-gap case per family.",
    )
    ap.add_argument("--out", type=str, default="figures/output/fig7_cost_angle_structure.pdf")
    args = ap.parse_args()

    mw_root = Path(args.results_root) / f"mw{int(args.mw)}"
    families = ["maxcut", "grid2d", "sk"]
    fam_label = {"sk": "SK", "grid2d": "2D grid", "maxcut": "MaxCut"}

    cases: Dict[str, Tuple[Path, int, Optional[int]]] = {}
    for spec in args.case:
        fam, run_name, idx, layer = spec.split(":")
        cases[fam] = (mw_root / fam / run_name, int(idx), int(layer))
    for fam in families:
        if fam not in cases:
            sel = _select_case(mw_root, fam)
            if sel is not None:
                cases[fam] = (sel[0], sel[1], None)

    fig = plt.figure(figsize=(15, 6))
    ax_sc = fig.add_axes([0.05, 0.12, 0.24, 0.76])
    points = []
    point_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []

    for col, fam in enumerate(families):
        if fam not in cases:
            continue
        run_dir, idx, layer_opt = cases[fam]
        try:
            problem = load_problem(run_dir / "problem_copy.json")
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            p_layers = int(meta["p_layers"])
            theta0, theta1 = _load_thetas(run_dir)
        except (json.JSONDecodeError, KeyError, FileNotFoundError, OSError) as exc:
            print(f"[fig7] skip {fam} ({run_dir.name}): unreadable case ({type(exc).__name__})")
            continue

        cost_idx = _all_cost_indices(problem, p_layers)
        sim = circular_cosine_similarity(theta0[idx, cost_idx], theta1[idx, cost_idx])
        lwpp_last = _per_seed_last(run_dir / "lwpp_pre_trace_per_seed.csv")
        fine_last = _per_seed_last(run_dir / "exact_finetune_trace_per_seed.csv")
        gap = abs(lwpp_last.get(idx, np.nan) - fine_last.get(idx, np.nan))
        points.append((fam, gap, sim))
        point_rows.append(
            {"family": fam, "run": run_dir.name, "init_index": idx, "exact_energy_gap": gap, "circular_cosine_similarity": sim}
        )

        layer = int(layer_opt) if layer_opt is not None else p_layers - 1
        sl = _cost_angle_slices(problem, p_layers)[layer]
        ang0 = theta0[idx, sl]
        ang1 = theta1[idx, sl]
        vmax = float(max(np.abs(ang0).max(), np.abs(ang1).max(), 1e-9))
        for k, (u, v, _w) in enumerate(problem.edges):
            edge_rows.append(
                {
                    "family": fam,
                    "run": run_dir.name,
                    "init_index": idx,
                    "cost_layer": layer,
                    "edge_index": k,
                    "u": int(u),
                    "v": int(v),
                    "angle_lwpp_final": float(ang0[k]),
                    "angle_fine_final": float(ang1[k]),
                }
            )

        for rowi, angles in enumerate((ang0, ang1)):
            ax = fig.add_axes([0.34 + 0.22 * col, 0.52 - 0.44 * rowi, 0.20, 0.40])
            _draw_edge_angles(ax, problem, angles, vmax)
            if rowi == 0:
                ax.set_title(
                    f"{fam_label[fam]}  {run_dir.name}\ninit {idx}, cost layer {layer}", fontsize=9
                )
            side = "LWPP final" if rowi == 0 else "LWPP + Exact final"
            ax.text(1.02, 0.5, side, transform=ax.transAxes, rotation=90, va="center", fontsize=8)

    colors = {"sk": "teal", "grid2d": "tab:orange", "maxcut": "tab:blue"}
    for fam, gap, sim in points:
        ax_sc.scatter([gap], [sim], s=90, color=colors[fam], zorder=3)
        ax_sc.annotate(fam_label[fam], xy=(gap, sim), xytext=(5, 6), textcoords="offset points", fontsize=9)
    ax_sc.set_xlabel("Absolute exact-energy gap")
    ax_sc.set_ylabel("Circular cosine similarity")
    ax_sc.set_title("Selected large-gap cases", fontsize=10)
    ax_sc.grid(True, alpha=0.25)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")

    def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    data_out = out.with_name(out.stem + "_data.csv")
    edges_out = out.with_name(out.stem + "_edge_angles.csv")
    _write_rows(data_out, point_rows)
    _write_rows(edges_out, edge_rows)
    print(f"wrote {out}, {data_out}, {edges_out}")
    for fam, gap, sim in points:
        print(f"  {fam}: gap={gap:.3f} sim={sim:.4f}")


if __name__ == "__main__":
    main()
