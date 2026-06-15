#!/usr/bin/env python3
"""
RQ1 breakdowns (§5: "completeness and spurious rate ... broken down by operator
and by rigid/non-rigid fraction"), plus the secondary MRR/Hits@k.

Two sweeps, both on the faithful Box²EL base:
  * by operator   — TBoxes isolating one temporal operator (○rhs/□rhs/○lhs/◇lhs/U)
                    over a static NF1–NF4 backbone; completeness/spurious/ranking.
  * by rigid frac — generate_rq2 TBoxes (confound rule: temporal axioms over
                    non-rigid names only) swept over ρ; RQ1 recovery vs ρ.

  python scripts/rq1_breakdowns.py --seeds 0 1 2 --out results/rq1_breakdowns
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ltel_box.induce import held_subsumptions, ranking_metrics, rq1_metrics  # noqa: E402
from ltel_box.model import Box2ELModel                                       # noqa: E402
from ltel_box.ontology import Vocabulary                                     # noqa: E402
from ltel_box.oracle import temporal_closure                                 # noqa: E402
from ltel_box.rq2 import RQ2Config, generate_rq2                             # noqa: E402
from ltel_box.synthetic import SyntheticConfig, generate_tbox               # noqa: E402
from ltel_box.trainer import TrainConfig, train                             # noqa: E402

OPERATORS = {  # label -> the single temporal-NF count to enable
    "next_rhs": "n_nf6", "globally_rhs": "n_nf7",
    "next_lhs": "n_nf10", "eventually_lhs": "n_nf12", "until_lhs": "n_nf13",
}


def _eval(tbox, k, loop, dim, epochs, seed, eps):
    vocab = Vocabulary.from_tbox(tbox)
    entailed = temporal_closure(tbox, k, loop)
    cfg = TrainConfig(dim=dim, k=k, loop=loop, epochs=epochs, seed=seed)
    model, final = train(tbox, vocab, cfg, model_cls=Box2ELModel)
    held = held_subsumptions(model, vocab, eps=eps)
    m = rq1_metrics(held, entailed, vocab, n_positions=k + 1)
    r = ranking_metrics(model, vocab, entailed)
    return {"loss": final["loss_total"], "entailed": m["entailed"],
            "completeness": m["completeness"], "spurious_rate": m["spurious_rate"], **r}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--n-concepts", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--loop", type=int, default=1)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--out", default="results/rq1_breakdowns")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    def mean(rows, key):
        xs = [r[key] for r in rows if r[key] == r[key]]
        return sum(xs) / len(xs) if xs else float("nan")

    # 1. by operator
    by_operator = {}
    for label, knob in OPERATORS.items():
        rows = []
        for s in args.seeds:
            kw = dict(n_nf1=10, n_nf2=4, n_nf3=4, n_nf4=4, n_nf5=0,
                      n_nf6=0, n_nf7=0, n_nf10=0, n_nf12=0, n_nf13=0)
            kw[knob] = 6
            tbox = generate_tbox(SyntheticConfig(n_concepts=args.n_concepts, n_roles=2,
                                                 seed=s, **kw)).tbox
            rows.append(_eval(tbox, args.k, args.loop, args.dim, args.epochs, s, args.eps))
        by_operator[label] = {k: mean(rows, k) for k in
                              ("loss", "completeness", "spurious_rate", "mrr", "hits1", "hits3", "hits10")}
        d = by_operator[label]
        print(f"[op {label:14s}] loss={d['loss']:.1e} compl={d['completeness']:.3f} "
              f"spur={d['spurious_rate']:.3f} MRR={d['mrr']:.3f} H@1={d['hits1']:.3f}")

    # 2. by rigid fraction (generate_rq2 TBoxes; confound rule built in)
    by_rho = {}
    for rho in args.rhos:
        rows = []
        for s in args.seeds:
            ds = generate_rq2(RQ2Config(n_concepts=args.n_concepts, n_roles=2, k=args.k,
                                        loop=args.loop, rigid_fraction=rho,
                                        n_individuals=1, seed=s))
            rows.append(_eval(ds.tbox, args.k, args.loop, args.dim, args.epochs, s, args.eps))
        by_rho[rho] = {k: mean(rows, k) for k in
                       ("loss", "completeness", "spurious_rate", "mrr", "hits1", "hits3", "hits10")}
        d = by_rho[rho]
        print(f"[rho {rho:<4}] loss={d['loss']:.1e} compl={d['completeness']:.3f} "
              f"spur={d['spurious_rate']:.3f} MRR={d['mrr']:.3f}")

    summary = {"config": vars(args), "by_operator": by_operator, "by_rigid_fraction": by_rho}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {outdir}/summary.json")


if __name__ == "__main__":
    main()
