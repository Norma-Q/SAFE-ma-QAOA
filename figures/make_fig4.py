"""Fig. 4: final approximation ratio vs distillation threshold (20-qubit tier).

Rows: SK (q20), 2D grid (4x5), Max-Cut (q20).
Columns: (p=2, mw=3), (p=4, mw=3), (p=2, mw=4), (p=4, mw=4).
Per panel: mean/best final-ratio bars at thresholds {0, 0.01, 0.3}, horizontal
reference lines for exact-only (mean/best) and LWPP-only (mean/best), and the
average parameter-reduction fraction printed above each threshold column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILY_ROWS = [("sk", "20", "SK"), ("grid2d", "4x5", "2D grid"), ("maxcut", "20", "MaxCut")]
PANELS = [(2, 3), (4, 3), (2, 4), (4, 4)]  # (p, mw)
THRESHOLDS = [0.0, 0.01, 0.3]


def _method_for_thr(thr: float) -> str:
    return "safe_nodistill" if thr == 0.0 else f"safe_distill_thr{thr:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long-csv", type=str, default="figures/output/runs_long.csv")
    ap.add_argument("--out", type=str, default="figures/output/fig4_threshold_overview.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.long_csv)
    fig, axes = plt.subplots(len(FAMILY_ROWS), len(PANELS), figsize=(16, 8.5), sharey=True)
    data_rows = []

    for i, (fam, size, fam_label) in enumerate(FAMILY_ROWS):
        for j, (p, mw) in enumerate(PANELS):
            ax = axes[i, j]
            base = df[(df["family"] == fam) & (df["size"].astype(str) == size) & (df["p"] == p) & (df["mw"] == mw)]

            means, bests, reductions = [], [], []
            for thr in THRESHOLDS:
                sub = base[base["method"] == _method_for_thr(thr)]
                if len(sub) == 0:
                    means.append(np.nan)
                    bests.append(np.nan)
                    reductions.append(np.nan)
                    continue
                means.append(float(sub["alpha_final"].mean()))
                bests.append(float(sub["alpha_final"].max()))
                reductions.append(float((1.0 - sub["n_active"] / sub["n_params"]).mean()))

            x = np.arange(len(THRESHOLDS), dtype=float)
            w = 0.38
            ax.bar(x - w / 2, means, width=w, color="#4C72B0", label="Mean")
            ax.bar(x + w / 2, bests, width=w, color="#DD8452", label="Best")

            exact = base[base["method"] == "exact_only"]
            if len(exact) > 0:
                ax.axhline(float(exact["alpha_final"].mean()), color="gray", ls="--", lw=1.2, label="Exact-only mean")
                ax.axhline(float(exact["alpha_final"].max()), color="gray", ls=":", lw=1.2, label="Exact-only best")
            lwpp = base[base["method"] == "lwpp_only"]
            if len(lwpp) > 0:
                ax.axhline(float(lwpp["alpha_final"].mean()), color="black", ls="-", lw=1.1, label="LWPP-only exact mean")
                ax.axhline(float(lwpp["alpha_final"].max()), color="crimson", ls="-", lw=1.1, label="LWPP-only exact best")

            for thr, mean_v, best_v, red_v in zip(THRESHOLDS, means, bests, reductions):
                data_rows.append(
                    {
                        "family": fam,
                        "size": size,
                        "p": p,
                        "mw": mw,
                        "threshold": thr,
                        "alpha_final_mean": mean_v,
                        "alpha_final_best": best_v,
                        "param_reduction_mean": red_v,
                        "exact_only_mean": float(exact["alpha_final"].mean()) if len(exact) else np.nan,
                        "exact_only_best": float(exact["alpha_final"].max()) if len(exact) else np.nan,
                        "lwpp_only_mean": float(lwpp["alpha_final"].mean()) if len(lwpp) else np.nan,
                        "lwpp_only_best": float(lwpp["alpha_final"].max()) if len(lwpp) else np.nan,
                    }
                )

            for xx, red in zip(x, reductions):
                if np.isfinite(red):
                    ax.annotate(
                        f"{100.0 * red:.1f}%",
                        xy=(xx, 1.005),
                        xycoords=("data", "axes fraction"),
                        ha="center",
                        fontsize=8,
                        color="seagreen",
                        bbox=dict(boxstyle="round,pad=0.15", fc="honeydew", ec="seagreen", lw=0.6),
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([f"{t:g}" for t in THRESHOLDS])
            ax.set_ylim(0.6, 1.02)
            ax.grid(True, axis="y", alpha=0.25)
            if i == 0:
                ax.set_title(f"p={p}, mw={mw}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{fam_label}\nfinal approximation ratio")
            if i == len(FAMILY_ROWS) - 1:
                ax.set_xlabel("distillation threshold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    data_out = out.with_name(out.stem + "_data.csv")
    pd.DataFrame(data_rows).to_csv(data_out, index=False)
    print(f"wrote {out} and {data_out}")


if __name__ == "__main__":
    main()
