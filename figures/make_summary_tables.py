"""Tables I & II and the per-mw best-case summary CSVs from runs_long.csv.

Outputs (under figures/output/):
  MW{mw}_bestcase_threshold_steps_summary.csv   best-case progression per setting
  table1.csv / table1.tex                       Table I  (alpha step-0 -> final, +gain)
  table2.csv / table2.tex                       Table II (C_ballpark = N_active * tau_0.99)

Conventions:
  - Table I uses w_max = 4 and distillation threshold 0.3 (paper Sec. IV).
  - "Best case" selects, per (family, size, p, method), the (instance, init) row
    with the maximum final approximation ratio.
  - Table II cells are means of per-(instance, init) products N_active * tau_0.99;
    parenthesized factors are exact-only cell / SAFE cell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FAMILY_ORDER = ["sk", "grid2d", "maxcut"]
FAMILY_LABEL = {"sk": "SK", "grid2d": "2D grid", "maxcut": "Max-Cut"}
METHOD_LABEL = {
    "exact_only": "Exact only",
    "safe_nodistill": "SAFE without distillation",
    "safe_distill_thr0.3": "SAFE with distillation",
}
SIZES = [12, 16, 20]
PS = [2, 4]


def _bestcase(df: pd.DataFrame) -> pd.DataFrame:
    """Per (family, n_qubits, p, mw, method): the row with max alpha_final."""
    idx = df.groupby(["family", "n_qubits", "p", "mw", "method"])["alpha_final"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def make_bestcase_csv(df: pd.DataFrame, out_dir: Path, mw: int) -> Path:
    sub = df[df["mw"] == mw]
    best = _bestcase(sub)
    cols = [
        "family", "size", "n_qubits", "p", "mw", "method", "inst", "init_label",
        "alpha_start", "alpha_final", "alpha_best", "tau099", "n_active", "n_params",
    ]
    out = out_dir / f"MW{mw}_bestcase_threshold_steps_summary.csv"
    best[cols].sort_values(["family", "n_qubits", "p", "method"]).to_csv(out, index=False)
    return out


def make_table1(df: pd.DataFrame, out_dir: Path, mw: int = 4, thr_method: str = "safe_distill_thr0.3") -> None:
    sub = df[(df["mw"] == mw) & (df["method"].isin(["safe_nodistill", thr_method]))]
    best = _bestcase(sub)

    rows = []
    for fam in FAMILY_ORDER:
        for method in ["safe_nodistill", thr_method]:
            row = {"Family": FAMILY_LABEL[fam], "Method": METHOD_LABEL.get(method, method)}
            for p in PS:
                for q in SIZES:
                    hit = best[
                        (best["family"] == fam)
                        & (best["n_qubits"] == q)
                        & (best["p"] == p)
                        & (best["method"] == method)
                    ]
                    if len(hit) == 1 and pd.notna(hit.iloc[0]["alpha_start"]):
                        a0 = float(hit.iloc[0]["alpha_start"])
                        af = float(hit.iloc[0]["alpha_final"])
                        row[f"p{p}_q{q}"] = f"{a0:.3f}->{af:.3f} (+{af - a0:.3f})"
                    else:
                        row[f"p{p}_q{q}"] = "-"
            rows.append(row)
    t1 = pd.DataFrame(rows)
    t1.to_csv(out_dir / "table1.csv", index=False)

    with (out_dir / "table1.tex").open("w", encoding="utf-8") as f:
        f.write("% Table I: best-case approximation-ratio progression (step 0 -> final)\n")
        f.write("\\begin{tabular}{ll" + "c" * (len(PS) * len(SIZES)) + "}\n\\hline\n")
        header = ["Family", "Method"] + [f"$p={p}$, $q={q}$" for p in PS for q in SIZES]
        f.write(" & ".join(header) + " \\\\\n\\hline\n")
        for _, r in t1.iterrows():
            cells = [str(r["Family"]), str(r["Method"])] + [
                str(r[f"p{p}_q{q}"]).replace("->", "$\\rightarrow$") for p in PS for q in SIZES
            ]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")


def make_table2(df: pd.DataFrame, out_dir: Path, mw: int = 4, thr_method: str = "safe_distill_thr0.3") -> None:
    methods = ["exact_only", "safe_nodistill", thr_method]
    sub = df[(df["mw"] == mw) & (df["method"].isin(methods))]
    mean_c = (
        sub.groupby(["family", "n_qubits", "p", "method"])["c_ballpark"].mean().reset_index()
    )

    def _cell(fam: str, q: int, p: int, method: str) -> float | None:
        hit = mean_c[
            (mean_c["family"] == fam)
            & (mean_c["n_qubits"] == q)
            & (mean_c["p"] == p)
            & (mean_c["method"] == method)
        ]
        return float(hit.iloc[0]["c_ballpark"]) if len(hit) == 1 else None

    rows = []
    for fam in FAMILY_ORDER:
        for method in methods:
            row = {"Family": FAMILY_LABEL[fam], "Method": METHOD_LABEL.get(method, method)}
            for p in PS:
                for q in SIZES:
                    c = _cell(fam, q, p, method)
                    if c is None:
                        row[f"p{p}_q{q}"] = "-"
                    elif method == "exact_only":
                        row[f"p{p}_q{q}"] = f"{c:.1f}"
                    else:
                        c_exact = _cell(fam, q, p, "exact_only")
                        if c and c > 0 and c_exact:
                            row[f"p{p}_q{q}"] = f"{c:.1f} ({c_exact / c:.1f}x)"
                        else:
                            row[f"p{p}_q{q}"] = f"{c:.1f} (-)"
            rows.append(row)
    t2 = pd.DataFrame(rows)
    t2.to_csv(out_dir / "table2.csv", index=False)

    with (out_dir / "table2.tex").open("w", encoding="utf-8") as f:
        f.write("% Table II: fine-tuning cost ballpark estimate C = N_active * tau_0.99\n")
        f.write("\\begin{tabular}{ll" + "c" * (len(PS) * len(SIZES)) + "}\n\\hline\n")
        header = ["Family", "Method"] + [f"$p={p}$, $q={q}$" for p in PS for q in SIZES]
        f.write(" & ".join(header) + " \\\\\n\\hline\n")
        for _, r in t2.iterrows():
            cells = [str(r["Family"]), str(r["Method"])] + [
                str(r[f"p{p}_q{q}"]).replace("x)", "$\\times$)") for p in PS for q in SIZES
            ]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long-csv", type=str, default="figures/output/runs_long.csv")
    ap.add_argument("--out-dir", type=str, default="figures/output")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.long_csv)

    for mw in sorted(df["mw"].unique()):
        p = make_bestcase_csv(df, out_dir, int(mw))
        print(f"wrote {p}")
    make_table1(df, out_dir)
    make_table2(df, out_dir)
    print(f"wrote {out_dir / 'table1.csv'} / .tex and {out_dir / 'table2.csv'} / .tex")


if __name__ == "__main__":
    main()
