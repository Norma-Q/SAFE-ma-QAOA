"""Fig. 6: representative exact fine-tuning trajectories for one instance.

Black: exact-only from the raw initialization. Colored: exact fine-tuning
initialized with the LWPP pre-trained parameters, for the two initializations
with the lowest final exact energy (override with --inits).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _per_seed(path: Path) -> Dict[int, List[Tuple[int, float]]]:
    out: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("exact_energy")
            if v in (None, "", "None"):
                continue
            out[int(row["init_index"])].append((int(row["step"]), float(v)))
    for k in out:
        out[k].sort(key=lambda t: t[0])
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--inits", type=str, default="", help="Comma-separated init indices for the LWPP-initialized curves.")
    ap.add_argument("--exact-init", type=int, default=-1, help="Init index for the exact-only curve (default: first LWPP curve's init).")
    ap.add_argument("--out", type=str, default="figures/output/fig6_trajectories.pdf")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fine = _per_seed(run_dir / "exact_finetune_trace_per_seed.csv")
    direct = _per_seed(run_dir / "exact_only_trace_per_seed.csv")

    if str(args.inits).strip():
        chosen = [int(x) for x in str(args.inits).split(",") if x.strip()]
    else:
        finals = {idx: steps[-1][1] for idx, steps in fine.items() if steps}
        chosen = [idx for idx, _v in sorted(finals.items(), key=lambda kv: kv[1])[:2]]
    exact_idx = int(args.exact_init) if int(args.exact_init) >= 0 else chosen[0]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ds = direct.get(exact_idx, [])
    ax.plot([s for s, _ in ds], [e for _, e in ds], color="black", lw=1.8, label=f"exact-only (init {exact_idx})")
    for color, idx in zip(("tab:blue", "tab:orange"), chosen):
        fs = fine.get(idx, [])
        ax.plot(
            [s for s, _ in fs],
            [e for _, e in fs],
            color=color,
            lw=1.8,
            label=f"LWPP pre-trained, exact fine-tuning (init {idx})",
        )
        if fs:
            ax.plot([fs[-1][0]], [fs[-1][1]], marker="o", color=color, ms=5)

    ax.set_xlabel("Exact finetuning step")
    ax.set_ylabel("Exact energy")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")

    data_out = out.with_name(out.stem + "_data.csv")
    with data_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "curve", "init_index", "step", "exact_energy"])
        for s, e in ds:
            writer.writerow([run_dir.name, "exact_only", exact_idx, s, e])
        for idx in chosen:
            for s, e in fine.get(idx, []):
                writer.writerow([run_dir.name, "lwpp_finetune", idx, s, e])
    print(f"wrote {out} and {data_out} (run={run_dir.name}, lwpp inits={chosen}, exact init={exact_idx})")


if __name__ == "__main__":
    main()
