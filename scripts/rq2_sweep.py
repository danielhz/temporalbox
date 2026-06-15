#!/usr/bin/env python3
"""
RQ2 — controlled support-reconstruction sweep: does faithfulness pay off?

A (per-time-point boxes) vs B (time-augmented, one convex interval), swept over
the rigid fraction ρ. Controlled = pure ABox (independent concepts, no TBox
packing confound), so A reconstructs every support exactly and B's failure is
purely its convex-time limitation. A and B see IDENTICAL data per (ρ, seed);
only model_cls differs (guardrail 4). Recovery is reported against achieved
loss (guardrail 1) and broken down by support level (guardrail 3); ρ=1 must
tie (guardrail 2).

Run twice for the mechanism panel:
  --support-max-intervals 1   (non-rigid = single interval, level 2: B ties A)
  --support-max-intervals 3   (non-rigid = multi-interval, level 3: B falls)

  python scripts/rq2_sweep.py --rhos 0 0.25 0.5 0.75 1 --seeds 0 1 2 \
      --support-max-intervals 3 --device cuda --out results/rq2_multi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ltel_box.model import TemporalBoxModel, TimeAugmentedBoxModel    # noqa: E402
from ltel_box.rq2 import (                                            # noqa: E402
    RQ2Config, RQ2TrainConfig, generate_rq2, reconstruction_metrics, train_rq2,
)

MODELS = {"A": TemporalBoxModel, "B": TimeAugmentedBoxModel}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--support-max-intervals", type=int, default=3)
    ap.add_argument("--n-concepts", type=int, default=16)
    ap.add_argument("--n-individuals", type=int, default=40)
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/rq2")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rho in args.rhos:
        agg = {m: {"recovery": [], "recovery_L1": [], "recovery_L2": [],
                   "recovery_L3": [], "loss_out": [], "loss_total": []} for m in MODELS}
        for seed in args.seeds:
            # controlled = pure ABox: no static/temporal TBox axioms
            ds = generate_rq2(RQ2Config(
                n_concepts=args.n_concepts, n_roles=1, rigid_fraction=rho, k=args.k,
                n_nf1=0, n_nf2=0, n_nf3=0, n_nf4=0, n_temporal=0,
                n_individuals=args.n_individuals, memberships_per_individual=4,
                support_max_intervals=args.support_max_intervals, seed=seed,
            ))
            tc = RQ2TrainConfig(dim=args.dim, epochs=args.epochs, device=args.device, seed=seed)
            for name, cls in MODELS.items():
                model, comps = train_rq2(ds, tc, model_cls=cls)
                rec = reconstruction_metrics(model, ds, eps=args.eps)
                for key in ("recovery", "recovery_L1", "recovery_L2", "recovery_L3"):
                    v = rec[key]
                    if v == v:  # skip NaN (level absent)
                        agg[name][key].append(v)
                agg[name]["loss_out"].append(comps["loss_membership_out"])
                agg[name]["loss_total"].append(comps["loss_total"])

        def mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        row = {"rho": rho}
        for name in MODELS:
            row[name] = {k: mean(v) for k, v in agg[name].items()}
        rows.append(row)
        a, b = row["A"], row["B"]
        print(f"ρ={rho:<4} | A rec={a['recovery']:.3f} (out-loss {a['loss_out']:.1e}) "
              f"| B rec={b['recovery']:.3f} (out-loss {b['loss_out']:.1e})  "
              f"L3: A={a['recovery_L3']:.3f} B={b['recovery_L3']:.3f}")

    summary = {"config": vars(args), "rows": rows}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {outdir}/summary.json")


if __name__ == "__main__":
    main()
