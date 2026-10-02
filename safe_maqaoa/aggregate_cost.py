"""Aggregate per-run cost_metrics.json (+ cost_metrics_distill.json) into one CSV.

Walks a results root for cost_metrics.json files written by
run_ma_qaoa_lwpp_exact_compare.py and emits a flat table for the paper's
computation-cost discussion: LWPP term counts before/after zero-filtering,
per-phase wall time, and peak VRAM.

Usage:
    python safe_maqaoa/aggregate_cost.py --results-root results \
        --out figures/output/cost_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _mb(x: Optional[int]) -> Optional[float]:
    return None if x is None else round(float(x) / (1024.0 * 1024.0), 2)


def _family_and_mw(path: Path) -> Dict[str, str]:
    """Infer family / mw from a .../mw{K}/{family}/{run}/cost_metrics.json path."""
    fam, mw = "", ""
    for part in path.parts:
        if part.startswith("mw") and part[2:].isdigit():
            mw = part[2:]
        if part in ("sk", "maxcut", "grid2d"):
            fam = part
    return {"family": fam, "mw": mw}


def _row(cm_path: Path) -> Dict[str, Any]:
    cm = json.loads(cm_path.read_text(encoding="utf-8"))
    sysd = cm.get("system", {})
    lwpp = cm.get("lwpp", {})
    fine = cm.get("exact_finetune", {})
    only = cm.get("exact_only", {})
    totals = cm.get("totals", {})
    meta = _family_and_mw(cm_path)

    distill_wall = None
    distill_path = cm_path.parent / "cost_metrics_distill.json"
    if distill_path.exists():
        try:
            distill_wall = float(json.loads(distill_path.read_text(encoding="utf-8")).get("total_wall_s"))
        except Exception:
            distill_wall = None

    return {
        "run": cm.get("run", cm_path.parent.name),
        "family": meta["family"],
        "mw": meta["mw"],
        "n_qubits": sysd.get("n_qubits"),
        "n_edges": sysd.get("n_edges"),
        "p_layers": sysd.get("p_layers"),
        "n_params": sysd.get("n_params"),
        "lwpp_max_weight": sysd.get("lwpp_max_weight"),
        "exact_device": sysd.get("exact_device"),
        "exact_chunk_size": sysd.get("exact_chunk_size"),
        "resumed_run": cm.get("resumed_run"),
        "terms_before_zero_filtering": lwpp.get("terms_before_zero_filtering"),
        "terms_after_zero_filtering": lwpp.get("terms_after_zero_filtering"),
        "keep_fraction": lwpp.get("keep_fraction"),
        "lwpp_compile_wall_s": lwpp.get("compile_wall_s"),
        "lwpp_train_wall_s": lwpp.get("train_wall_s"),
        "lwpp_compile_peak_vram_mb": _mb(lwpp.get("compile_peak_vram_bytes")),
        "lwpp_train_peak_vram_mb": _mb(lwpp.get("train_peak_vram_bytes")),
        "exact_finetune_wall_s": fine.get("wall_s"),
        "exact_finetune_peak_vram_mb": _mb(fine.get("peak_vram_bytes")),
        "exact_only_wall_s": only.get("wall_s"),
        "exact_only_peak_vram_mb": _mb(only.get("peak_vram_bytes")),
        "distill_wall_s": distill_wall,
        "total_train_wall_s": totals.get("train_wall_s"),
        "wall_including_compile_s": totals.get("wall_including_compile_s"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default="figures/output/cost_metrics.csv")
    args = ap.parse_args()

    root = Path(args.results_root).resolve()
    paths = sorted(root.rglob("cost_metrics.json"))
    if not paths:
        print(f"no cost_metrics.json under {root}")
        return

    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            rows.append(_row(p))
        except Exception as exc:  # keep going; report the bad file
            print(f"[warn] skipping {p}: {exc}")

    fields = list(rows[0].keys())
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
