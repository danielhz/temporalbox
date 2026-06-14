#!/usr/bin/env python3
"""
Offices feasibility exhibit (§5): train Option A on a Wikidata office slice and
report subsumption preservation at scale.

  python scripts/slice_offices.py \
      --memberships OFFICES/memberships.tsv --p279 OFFICES/p279.tsv \
      --root Q48352 --granularity year --min-holders 0 \
      --epochs 3000 --dim 32 --device cuda --out results/slice

Preliminary feasibility only — not a test of the temporal operators.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ltel_box.logging_utils import ExperimentLogger                     # noqa: E402
from ltel_box.slice_trainer import (                                    # noqa: E402
    SliceTrainConfig, evaluate_slice, train_slice,
)
from ltel_box.wikidata import WikidataConfig, build_dataset, load_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Either build from the source TSVs, or load a pre-built dataset dir
    # (save_dataset output — lets the small dataset travel to the GPU host).
    ap.add_argument("--dataset-dir", default=None, help="load a pre-built dataset (skips TSVs)")
    ap.add_argument("--memberships", default=None)
    ap.add_argument("--p279", default=None)
    ap.add_argument("--root", default="Q48352")
    ap.add_argument("--granularity", choices=["year", "event"], default="event")
    ap.add_argument("--min-holders", type=int, default=0)
    ap.add_argument("--min-year", type=int, default=None)
    ap.add_argument("--max-year", type=int, default=None)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--offset-reg", type=float, default=1e-3)
    ap.add_argument("--membership-cap", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/slice")
    args = ap.parse_args()

    if args.dataset_dir:
        ds = load_dataset(args.dataset_dir)
    elif args.memberships and args.p279:
        ds = build_dataset(args.memberships, args.p279, WikidataConfig(
            granularity=args.granularity, root=args.root, min_holders=args.min_holders,
            min_year=args.min_year, max_year=args.max_year,
        ))
    else:
        ap.error("provide --dataset-dir or both --memberships and --p279")
    print("dataset:", json.dumps(ds.stats))

    cfg = SliceTrainConfig(
        dim=args.dim, epochs=args.epochs, lr=args.lr, offset_reg=args.offset_reg,
        membership_cap=args.membership_cap, device=args.device, seed=args.seed,
    )
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    run_cfg = {"rq": "slice", **vars(args), **ds.stats}
    run_cfg.pop("out", None)
    logger = ExperimentLogger.from_config(run_cfg, results_dir=outdir)

    model, st, final = train_slice(ds, cfg, logger)
    metrics = evaluate_slice(model, st, ds, cfg, eps=args.eps, seed=args.seed)
    record = {**final, **metrics}
    logger.log_eval("slice", record)
    logger.close()

    summary = {"config": vars(args), "dataset": ds.stats, "result": record}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsubsumption loss {final['loss_subsumption']:.2e}  "
          f"membership loss {final['loss_membership']:.2e}")
    print(f"completeness (sampled) {metrics['completeness_sampled']:.3f}  "
          f"spurious (sampled) {metrics['spurious_rate_sampled']:.4f}  "
          f"@ {metrics['concepts']}c × {metrics['ticks']}t, "
          f"{metrics['individuals']} individuals")
    print(f"summary -> {outdir}/summary.json")


if __name__ == "__main__":
    main()
