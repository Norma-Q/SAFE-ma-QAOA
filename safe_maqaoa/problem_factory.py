from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

WeightedEdge = Tuple[int, int, float]
Field = Tuple[int, float]


@dataclass(frozen=True)
class IsingProblem:
    name: str
    n_qubits: int
    edges: List[WeightedEdge]
    fields: List[Field]
    meta: Dict[str, Any]


def _canonical_edge(u: int, v: int) -> Tuple[int, int]:
    uu = int(u)
    vv = int(v)
    if uu == vv:
        raise ValueError("Self-loop is not allowed.")
    return (uu, vv) if uu < vv else (vv, uu)


def build_sk_problem(
    *,
    n_qubits: int,
    seed: int,
    mean: float = 0.0,
    std: float | None = None,
    field_mean: float = 0.0,
    field_std: float = 0.0,
) -> IsingProblem:
    if int(n_qubits) < 2:
        raise ValueError("n_qubits must be >= 2")
    effective_std = float(std) if std is not None else (1.0 / math.sqrt(float(n_qubits)))
    if float(effective_std) <= 0.0:
        raise ValueError("std must be > 0")

    rng = np.random.default_rng(int(seed))
    edges: List[WeightedEdge] = []
    for u in range(int(n_qubits)):
        for v in range(u + 1, int(n_qubits)):
            j = float(rng.normal(loc=float(mean), scale=float(effective_std)))
            edges.append((int(u), int(v), j))

    fields: List[Field] = []
    if float(field_std) > 0.0:
        for q in range(int(n_qubits)):
            h = float(rng.normal(loc=float(field_mean), scale=float(field_std)))
            fields.append((int(q), h))

    return IsingProblem(
        name="sk",
        n_qubits=int(n_qubits),
        edges=edges,
        fields=fields,
        meta={
            "seed": int(seed),
            "distribution": "gaussian",
            "mean": float(mean),
            "std": float(effective_std),
            "field_mean": float(field_mean),
            "field_std": float(field_std),
        },
    )


def build_grid2d_problem(
    *,
    rows: int,
    cols: int,
    seed: int,
    nn_mean: float = 0.0,
    nn_std: float = 1.0,
    nnn_mean: float = 0.0,
    nnn_std: float = 0.3,
    include_nnn: bool = True,
    periodic: bool = False,
    field_mean: float = 0.0,
    field_std: float = 0.1,
) -> IsingProblem:
    r = int(rows)
    c = int(cols)
    if r < 2 or c < 2:
        raise ValueError("rows and cols must be >= 2")
    if float(nn_std) <= 0.0:
        raise ValueError("nn_std must be > 0")
    if bool(include_nnn) and float(nnn_std) <= 0.0:
        raise ValueError("nnn_std must be > 0 when include_nnn is enabled")

    rng = np.random.default_rng(int(seed))

    def qid(i: int, j: int) -> int:
        return int(i) * c + int(j)

    nn_pairs = set()
    nnn_pairs = set()
    for i in range(r):
        for j in range(c):
            u = qid(i, j)
            right_j = (j + 1) % c
            down_i = (i + 1) % r
            if periodic or j + 1 < c:
                nn_pairs.add(_canonical_edge(u, qid(i, right_j)))
            if periodic or i + 1 < r:
                nn_pairs.add(_canonical_edge(u, qid(down_i, j)))

            if bool(include_nnn):
                down_right_j = (j + 1) % c
                down_left_j = (j - 1) % c
                if periodic or (i + 1 < r and j + 1 < c):
                    nnn_pairs.add(_canonical_edge(u, qid(down_i, down_right_j)))
                if periodic or (i + 1 < r and j - 1 >= 0):
                    nnn_pairs.add(_canonical_edge(u, qid(down_i, down_left_j)))

    edges: List[WeightedEdge] = []
    for (u, v) in sorted(nn_pairs):
        j = float(rng.normal(loc=float(nn_mean), scale=float(nn_std)))
        edges.append((int(u), int(v), j))
    for (u, v) in sorted(nnn_pairs):
        j = float(rng.normal(loc=float(nnn_mean), scale=float(nnn_std)))
        edges.append((int(u), int(v), j))

    n_qubits = r * c
    fields: List[Field] = []
    if float(field_std) > 0.0:
        for q in range(int(n_qubits)):
            h = float(rng.normal(loc=float(field_mean), scale=float(field_std)))
            fields.append((int(q), h))

    return IsingProblem(
        name="grid2d",
        n_qubits=int(n_qubits),
        edges=edges,
        fields=fields,
        meta={
            "seed": int(seed),
            "rows": int(r),
            "cols": int(c),
            "periodic": bool(periodic),
            "distribution": "gaussian",
            "nn_mean": float(nn_mean),
            "nn_std": float(nn_std),
            "nnn_mean": float(nnn_mean),
            "nnn_std": float(nnn_std),
            "include_nnn": bool(include_nnn),
            "field_mean": float(field_mean),
            "field_std": float(field_std),
        },
    )


def build_maxcut_problem(
    *,
    n_qubits: int,
    seed: int,
    edge_prob: float = 0.3,
    weight: float = 1.0,
) -> IsingProblem:
    """Max-Cut on an Erdos-Renyi G(n, edge_prob) graph as an Ising problem.

    Each pair (u, v) with u < v is included independently with probability
    `edge_prob`, with uniform coupling J_uv = `weight` and no local fields,
    so minimizing the Ising energy maximizes the cut.
    """
    if int(n_qubits) < 2:
        raise ValueError("n_qubits must be >= 2")
    if not (0.0 < float(edge_prob) <= 1.0):
        raise ValueError("edge_prob must be in (0, 1]")

    rng = np.random.default_rng(int(seed))
    edges: List[WeightedEdge] = []
    for u in range(int(n_qubits)):
        for v in range(u + 1, int(n_qubits)):
            if float(rng.random()) < float(edge_prob):
                edges.append((int(u), int(v), float(weight)))

    if len(edges) == 0:
        raise ValueError(
            f"Erdos-Renyi draw produced an empty graph (n={int(n_qubits)}, p={float(edge_prob)}, seed={int(seed)})"
        )

    return IsingProblem(
        name="maxcut",
        n_qubits=int(n_qubits),
        edges=edges,
        fields=[],
        meta={
            "seed": int(seed),
            "graph": "erdos_renyi",
            "edge_prob": float(edge_prob),
            "weight": float(weight),
            "n_edges": int(len(edges)),
        },
    )


def save_problem(problem: IsingProblem, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": problem.name,
        "n_qubits": int(problem.n_qubits),
        "edges": [[int(u), int(v), float(w)] for (u, v, w) in problem.edges],
        "fields": [[int(q), float(h)] for (q, h) in problem.fields],
        "meta": dict(problem.meta),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_problem(path: str | Path) -> IsingProblem:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    return IsingProblem(
        name=str(raw["name"]),
        n_qubits=int(raw["n_qubits"]),
        edges=[(int(u), int(v), float(w)) for (u, v, w) in raw["edges"]],
        fields=[(int(q), float(h)) for (q, h) in raw.get("fields", [])],
        meta=dict(raw.get("meta", {})),
    )
