"""Shipped problem instances are exactly regenerable.

Regenerates one instance per family from its recorded seed and compares it
field-by-field against the JSON shipped in data/problems/. Checks one shipped
instance per family.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from safe_maqaoa.problem_factory import (
    build_grid2d_problem,
    build_maxcut_problem,
    build_sk_problem,
)


def _as_payload(problem) -> dict:
    return {
        "name": problem.name,
        "n_qubits": int(problem.n_qubits),
        "edges": [[int(u), int(v), float(w)] for (u, v, w) in problem.edges],
        "fields": [[int(q), float(h)] for (q, h) in problem.fields],
        "meta": dict(problem.meta),
    }


def main() -> int:
    checks = [
        (
            "sk/sk_q12_p2_inst0_seed712200.json",
            lambda: build_sk_problem(n_qubits=12, seed=712200),
        ),
        (
            "grid2d/grid3x4_p2_inst0_seed120490.json",
            lambda: build_grid2d_problem(rows=3, cols=4, seed=120490),
        ),
        (
            "maxcut/maxcut_q20_p4_inst4_seed720404.json",
            lambda: build_maxcut_problem(n_qubits=20, seed=720404, edge_prob=0.3),
        ),
    ]

    n_fail = 0
    for rel, builder in checks:
        shipped = json.loads((_REPO_ROOT / "data" / "problems" / rel).read_text(encoding="utf-8"))
        regen = _as_payload(builder())
        ok = shipped == regen
        print(f"[{'OK' if ok else 'FAIL'}] {rel}")
        if not ok:
            n_fail += 1
            for key in ("name", "n_qubits", "edges", "fields", "meta"):
                if shipped.get(key) != regen.get(key):
                    print(f"       mismatch in '{key}'")

    print("problem regeneration:", "PASS" if n_fail == 0 else f"FAIL ({n_fail})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
