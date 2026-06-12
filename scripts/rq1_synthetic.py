#!/usr/bin/env python3
"""
RQ1 — empirical validity of the faithful Option A construction.

For each seed: generate a synthetic LTEL TBox, compute the oracle entailments
(ground-then-classify, independent code path), train Option A to (near-)zero
loss, read the induced interpretation off the boxes, and report completeness
and spurious rate against the achieved loss (total and per NF type).

  python scripts/rq1_synthetic.py --seeds 0 1 2 --k 3 --loop 1 \
      --n-concepts 20 --epochs 3000 --out results/rq1

Writes one JSONL per run (epoch losses + final eval) plus a summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ltel_box.induce import held_subsumptions, rq1_metrics          # noqa: E402
from ltel_box.logging_utils import ExperimentLogger                 # noqa: E402
from ltel_box.oracle import temporal_closure                        # noqa: E402
from ltel_box.ontology import Vocabulary                            # noqa: E402
from ltel_box.synthetic import SyntheticConfig, generate_tbox       # noqa: E402
from ltel_box.trainer import TrainConfig, train                     # noqa: E402


def run_one(seed: int, args: argparse.Namespace, outdir: Path) -> dict:
    def scaled(x: int) -> int:
        return max(0, round(x * args.gci_scale))

    # Defaults scaled down from SyntheticConfig: dense random TBoxes over few
    # concepts saturate the closure (everything ⊑ everything), which is both
    # unrealistic and degenerate for the spurious-rate denominator.
    syn = SyntheticConfig(
        n_concepts=args.n_concepts,
        n_roles=args.n_roles,
        n_nf1=scaled(15), n_nf2=scaled(8), n_nf3=scaled(8), n_nf4=scaled(8),
        n_nf5=scaled(2), n_nf6=scaled(5), n_nf7=scaled(4),
        n_nf10=scaled(3), n_nf12=scaled(2), n_nf13=scaled(2),
        seed=seed,
    )
    ds = generate_tbox(syn)
    entailed = temporal_closure(ds.tbox, args.k, args.loop)
    vocab = Vocabulary.from_tbox(ds.tbox)

    cfg = TrainConfig(
        dim=args.dim, k=args.k, loop=args.loop,
        lr=args.lr, epochs=args.epochs, seed=seed, device=args.device,
    )
    run_cfg = {"seed": seed, "rq": "rq1", **vars(args)}
    run_cfg.pop("out", None)
    logger = ExperimentLogger.from_config(run_cfg, results_dir=outdir)
    model, final_losses = train(ds.tbox, vocab, cfg, logger)

    held = held_subsumptions(model, vocab, eps=args.eps)
    metrics = rq1_metrics(held, entailed, vocab, n_positions=args.k + 1)
    # tolerance sweep: how much recovery sits just outside the primary eps
    sweep = {}
    for e in (1e-3, 1e-2):
        h = held_subsumptions(model, vocab, eps=e)
        m = rq1_metrics(h, entailed, vocab, n_positions=args.k + 1)
        sweep[f"completeness@{e:g}"] = m["completeness"]
        sweep[f"spurious_rate@{e:g}"] = m["spurious_rate"]
    record = {**final_losses, **metrics, **sweep, "eps": args.eps, "seed": seed}
    logger.log_eval("rq1", record)
    logger.close()
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-concepts", type=int, default=30)
    ap.add_argument("--n-roles", type=int, default=3)
    ap.add_argument("--gci-scale", type=float, default=0.6,
                    help="multiplier on the per-NF axiom counts (density)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--loop", type=int, default=1)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/rq1")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        r = run_one(seed, args, outdir)
        rows.append(r)
        print(
            f"seed={seed}  loss={r['loss_total']:.2e}  "
            f"entailed={r['entailed']}  completeness={r['completeness']:.3f}  "
            f"spurious_rate={r['spurious_rate']:.4f}"
        )

    n = len(rows)
    summary = {
        "config": vars(args),
        "runs": rows,
        "mean_completeness": sum(r["completeness"] for r in rows) / n,
        "mean_spurious_rate": sum(r["spurious_rate"] for r in rows) / n,
        "mean_final_loss": sum(r["loss_total"] for r in rows) / n,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmean completeness {summary['mean_completeness']:.3f}, "
          f"mean spurious rate {summary['mean_spurious_rate']:.4f}, "
          f"mean final loss {summary['mean_final_loss']:.2e}")
    print(f"summary -> {outdir}/summary.json")


if __name__ == "__main__":
    main()
