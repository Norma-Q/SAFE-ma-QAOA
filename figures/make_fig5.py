"""Fig. 5: average first-hit step tau_0.99 (w_max = 4, distillation threshold 0.3).

Rows: p = 2 and p = 4. Columns: q = 12, 16, 20 (2D grid: 3x4, 4x4, 4x5).
Each panel: grouped bars (SK, 2D grid, MaxCut) x (exact-only, SAFE without
distillation, SAFE with distillation), averaged over instances and inits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILIES = [("sk", "SK"), ("grid2d", "2D grid"), ("maxcut", "MaxCut")]
METHODS = [
    ("exact_only", "Exact only", "#4C72B0"),
    ("safe_nodistill", "No distillation", "#8C613C"),
    ("safe_distill_thr0.3", "Distillation", "#55A868"),
]
QS = [12, 16, 20]
PS = [2, 4]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long-csv", type=str, default="figures/output/runs_long.csv")
    ap.add_argument("--mw", type=int, default=4)
    ap.add_argument("--out", type=str, default="figures/output/fig5_first_hit.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.long_csv)
    df = df[df["mw"] == int(args.mw)]

    fig, axes = plt.subplots(len(PS), len(QS), figsize=(13, 6.5), sharey=True)
    data_rows = []

    for i, p in enumerate(PS):
        for j, q in enumerate(QS):
            ax = axes[i, j]
            x = np.arange(len(FAMILIES), dtype=float)
            w = 0.26
            for k, (method, mlabel, color) in enumerate(METHODS):
                vals = []
                for fam, _flabel in FAMILIES:
                    sub = df[
                        (df["family"] == fam)
                        & (df["n_qubits"] == q)
                        & (df["p"] == p)
                        & (df["method"] == method)
                    ]
                    vals.append(float(sub["tau099"].mean()) if len(sub) else np.nan)
                    data_rows.append(
                        {"mw": int(args.mw), "p": p, "n_qubits": q, "family": fam, "method": method, "tau099_mean": vals[-1]}
                    )
                ax.bar(x + (k - 1) * w, vals, width=w, color=color, label=mlabel if (i, j) == (0, 0) else None)

            ax.set_xticks(x)
            ax.set_xticklabels([fl for _f, fl in FAMILIES])
            ax.grid(True, axis="y", alpha=0.25)
            if j == 0:
                ax.set_ylabel(f"layer={p}\nAverage first-hit step")
            if i == len(PS) - 1:
                ax.set_xlabel(f"q={q}")
            else:
                ax.set_title(f"q={q}", fontsize=11)

    fig.legend(loc="upper center", ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    data_out = out.with_name(out.stem + "_data.csv")
    pd.DataFrame(data_rows).to_csv(data_out, index=False)
    print(f"wrote {out} and {data_out}")


if __name__ == "__main__":
    main()
