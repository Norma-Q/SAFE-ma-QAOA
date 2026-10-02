from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from safe_maqaoa.problem_factory import (
    build_grid2d_problem,
    build_maxcut_problem,
    build_sk_problem,
    save_problem,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Ising problem instances for MA-QAOA experiments.")
    p.add_argument("--problem-type", type=str, required=True, choices=["sk", "grid2d", "maxcut"])
    p.add_argument(
        "--preset",
        type=str,
        default="paper",
        choices=["paper", "custom"],
        help="paper: SK std=1/sqrt(N); grid NNN/field std=0.3/0.1 with NNN enabled. Ignored for maxcut.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mean", type=float, default=0.0)
    p.add_argument("--std", type=float, default=None)
    p.add_argument("--nnn-mean", type=float, default=0.0)
    p.add_argument("--nnn-std", type=float, default=None)
    p.add_argument("--field-mean", type=float, default=0.0)
    p.add_argument("--field-std", type=float, default=None)

    p.add_argument("--n-qubits", type=int, default=12, help="Used for sk and maxcut")
    p.add_argument("--edge-prob", type=float, default=0.3, help="Used for maxcut (Erdos-Renyi edge probability)")
    p.add_argument("--edge-weight", type=float, default=1.0, help="Used for maxcut (uniform coupling)")
    p.add_argument("--rows", type=int, default=4, help="Used for grid2d")
    p.add_argument("--cols", type=int, default=4, help="Used for grid2d")
    p.add_argument("--periodic", action="store_true", help="Used for grid2d")
    p.add_argument(
        "--include-nnn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Used for grid2d; defaults to true for paper preset and false for custom.",
    )

    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_paper = str(args.preset) == "paper"

    if str(args.problem_type) == "sk":
        n_qubits = int(args.n_qubits)
        std = float(args.std) if args.std is not None else (1.0 / math.sqrt(float(n_qubits)) if use_paper else 1.0)
        field_std = float(args.field_std) if args.field_std is not None else 0.0
        problem = build_sk_problem(
            n_qubits=n_qubits,
            seed=int(args.seed),
            mean=float(args.mean),
            std=std,
            field_mean=float(args.field_mean),
            field_std=field_std,
        )
    elif str(args.problem_type) == "maxcut":
        problem = build_maxcut_problem(
            n_qubits=int(args.n_qubits),
            seed=int(args.seed),
            edge_prob=float(args.edge_prob),
            weight=float(args.edge_weight),
        )
    else:
        nn_std = float(args.std) if args.std is not None else 1.0
        nnn_std = float(args.nnn_std) if args.nnn_std is not None else (0.3 if use_paper else 0.0)
        field_std = float(args.field_std) if args.field_std is not None else (0.1 if use_paper else 0.0)
        include_nnn = bool(args.include_nnn) if args.include_nnn is not None else bool(use_paper)
        problem = build_grid2d_problem(
            rows=int(args.rows),
            cols=int(args.cols),
            seed=int(args.seed),
            nn_mean=float(args.mean),
            nn_std=nn_std,
            nnn_mean=float(args.nnn_mean),
            nnn_std=nnn_std,
            include_nnn=include_nnn,
            periodic=bool(args.periodic),
            field_mean=float(args.field_mean),
            field_std=field_std,
        )

    out = save_problem(problem, Path(args.output))
    print(f"saved problem: {out}")


if __name__ == "__main__":
    main()
