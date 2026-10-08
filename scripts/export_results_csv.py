#!/usr/bin/env python3
"""
Export experiment result data to tidy CSVs for @writer's TikZ/PGF figures.

This emits DATA ONLY — no plotting. Figures (TikZ/PGF) are part of the writing
process and live in the paper repo; this script just turns the gitignored
`results/*/summary.json` into small, committed, render-ready tables under
`figure_data/` that `\\addplot table{...}` / `\\pgfplotstableread` can consume.

  python scripts/export_results_csv.py --kind rq2  --in results/rq2_multi/summary.json --out figure_data/rq2_multi.csv
  python scripts/export_results_csv.py --kind rq1  --in results/rq1_box2el/summary.json --out figure_data/rq1_box2el.csv
  python scripts/export_results_csv.py --kind slice --in results/slice/summary.json --out figure_data/slice.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def export_rq2(summary: dict, out: Path) -> None:
    # long format: one row per (rho, model)
    cols = ["rho", "model", "recovery", "recovery_L1", "recovery_L2", "recovery_L3",
            "out_loss", "total_loss", "param_count", "train_time_s"]
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in summary["rows"]:
            for m in ("A", "B"):
                d = r[m]
                w.writerow([r["rho"], m,
                            f"{d['recovery']:.4f}", f"{d['recovery_L1']:.4f}",
                            f"{d['recovery_L2']:.4f}", f"{d['recovery_L3']:.4f}",
                            f"{d['loss_out']:.3e}", f"{d['loss_total']:.3e}",
                            int(d.get('param_count', 0)), f"{d.get('train_time_s', 0):.2f}"])


def export_rq1(summary: dict, out: Path) -> None:
    cols = ["data_seed", "entailed", "mean_loss", "mean_completeness",
            "mean_spurious_rate", "intersection_completeness", "intersection_spurious_rate"]
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in summary["runs"]:
            w.writerow([r["data_seed"], r["entailed"], f"{r['mean_loss']:.3e}",
                        f"{r['mean_completeness']:.4f}", f"{r['mean_spurious_rate']:.4f}",
                        f"{r['intersection_completeness']:.4f}",
                        f"{r['intersection_spurious_rate']:.4f}"])
        # also an aggregate row
        w.writerow(["MEAN", "",
                    f"{summary['mean_final_loss']:.3e}",
                    f"{summary['mean_completeness']:.4f}",
                    f"{summary['mean_spurious_rate']:.4f}",
                    f"{summary['mean_intersection_completeness']:.4f}",
                    f"{summary['mean_intersection_spurious_rate']:.4f}"])


def export_slice(summary: dict, out: Path) -> None:
    r = summary["result"]
    # Report the dataset-side concept-name count (positions + P279 ancestors),
    # not result["concepts"], which is +1 larger because the model pads a
    # synthetic top (⊤) concept. The paper reports the dataset count.
    concepts = summary.get("dataset", {}).get("concepts", r["concepts"])
    cols = ["concepts", "ticks", "individuals", "completeness_sampled",
            "spurious_rate_sampled", "loss_subsumption", "loss_membership"]
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow([concepts, r["ticks"], r["individuals"],
                    f"{r['completeness_sampled']:.4f}", f"{r['spurious_rate_sampled']:.4f}",
                    f"{r['loss_subsumption']:.3e}", f"{r['loss_membership']:.3e}"])


def export_rq1bd(summary: dict, out: Path) -> None:
    """RQ1 breakdowns -> two CSVs (by operator, by rigid fraction) next to `out`."""
    metrics = ["loss", "completeness", "spurious_rate", "mrr", "hits1", "hits3", "hits10"]
    op_path = out.with_name(out.stem + "_by_operator.csv")
    with op_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["operator"] + metrics)
        for op, d in summary["by_operator"].items():
            w.writerow([op] + [f"{d[m]:.4f}" for m in metrics])
    rho_path = out.with_name(out.stem + "_by_rho.csv")
    with rho_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho"] + metrics)
        for rho, d in summary["by_rigid_fraction"].items():
            w.writerow([rho] + [f"{d[m]:.4f}" for m in metrics])
    print(f"wrote {op_path}")
    print(f"wrote {rho_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["rq2", "rq1", "slice", "rq1bd"], required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary = json.loads(Path(args.inp).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fn = {"rq2": export_rq2, "rq1": export_rq1, "slice": export_slice, "rq1bd": export_rq1bd}[args.kind]
    fn(summary, out)
    if args.kind != "rq1bd":
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
