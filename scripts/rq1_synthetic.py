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

from ltel_box.induce import across_seed_metrics, held_subsumptions, rq1_metrics  # noqa: E402
from ltel_box.logging_utils import ExperimentLogger                 # noqa: E402
from ltel_box.oracle import temporal_closure                        # noqa: E402
from ltel_box.ontology import Vocabulary                            # noqa: E402
from ltel_box.model import Box2ELModel, BoxELModel, TemporalBoxModel, TransBoxModel  # noqa: E402
from ltel_box.synthetic import SyntheticConfig, generate_tbox       # noqa: E402
from ltel_box.trainer import TrainConfig, train                     # noqa: E402

_MODELS = {"box2el": Box2ELModel, "boxel": BoxELModel, "transbox": TransBoxModel,
           "translational": TemporalBoxModel, "transe": TemporalBoxModel}


def run_tbox(data_seed: int, args: argparse.Namespace, outdir: Path) -> dict:
    """One TBox (data_seed), trained from `args.init_seeds` random inits.

    The across-init intersection of held subsumptions is the empirical proxy
    for "satisfied in every zero-loss model" (the universal quantifier in the
    entailment-phrased faithfulness theorem).
    """
    def scaled(x: int) -> int:
        return max(0, round(x * args.gci_scale))

    # Dense random TBoxes over few concepts saturate the closure; scale down.
    syn = SyntheticConfig(
        n_concepts=args.n_concepts,
        n_roles=args.n_roles,
        n_nf1=scaled(15), n_nf2=scaled(8), n_nf3=scaled(8), n_nf4=scaled(8),
        n_nf5=args.n_nf5, n_nf6=scaled(5), n_nf7=scaled(4),
        n_nf10=scaled(3), n_nf12=scaled(2), n_nf13=scaled(2),
        seed=data_seed,
    )
    ds = generate_tbox(syn)
    entailed = temporal_closure(ds.tbox, args.k, args.loop)
    vocab = Vocabulary.from_tbox(ds.tbox)
    npos = args.k + 1

    held_per_init: list[set] = []
    per_init: list[dict] = []
    for init in range(args.init_seeds):
        cfg = TrainConfig(
            dim=args.dim, k=args.k, loop=args.loop,
            lr=args.lr, epochs=args.epochs, seed=init, device=args.device,
        )
        run_cfg = {"data_seed": data_seed, "init_seed": init, "rq": "rq1", **vars(args)}
        run_cfg.pop("out", None)
        logger = ExperimentLogger.from_config(run_cfg, results_dir=outdir)
        model, final_losses = train(ds.tbox, vocab, cfg, logger, model_cls=_MODELS[args.model])
        held = held_subsumptions(model, vocab, eps=args.eps)
        m = rq1_metrics(held, entailed, vocab, n_positions=npos)
        held_per_init.append(held)
        per_init.append({**final_losses, **m, "init": init})
        logger.log_eval("rq1", {**final_losses, **m, "data_seed": data_seed, "init": init})
        logger.close()

    inter = across_seed_metrics(held_per_init, entailed, vocab, n_positions=npos)
    n = len(per_init)
    return {
        "data_seed": data_seed,
        "entailed": len(entailed),
        "mean_loss": sum(r["loss_total"] for r in per_init) / n,
        "mean_completeness": sum(r["completeness"] for r in per_init) / n,
        "mean_spurious_rate": sum(r["spurious_rate"] for r in per_init) / n,
        "max_spurious_rate": max(r["spurious_rate"] for r in per_init),
        "intersection_completeness": inter["intersection_completeness"],
        "intersection_spurious_rate": inter["intersection_spurious_rate"],
        "per_init": per_init,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="data seeds (each generates one TBox)")
    ap.add_argument("--init-seeds", type=int, default=3,
                    help="model initializations per TBox (intersected for the universal-quantifier proxy)")
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
    ap.add_argument("--model", choices=["box2el", "boxel", "transbox", "translational", "transe"], default="box2el",
                    help="base method: box2el (faithful) or translational (Phase-1)")
    ap.add_argument("--n-nf5", type=int, default=2,
                    help="role-composition (RIA) axioms per TBox. Box²EL supports "
                         "composition via NF7 (Head(r)⊆Head(t) ∧ Tail(s)⊆Tail(t), "
                         "proved sound in the Lean development); set 0 to scope RQ1 to NF1–NF4+temporal")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/rq1")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        r = run_tbox(seed, args, outdir)
        rows.append(r)
        print(
            f"tbox={seed}  loss={r['mean_loss']:.2e}  entailed={r['entailed']}  "
            f"compl(mean)={r['mean_completeness']:.3f}  "
            f"spurious(mean)={r['mean_spurious_rate']:.4f} "
            f"(max {r['max_spurious_rate']:.4f})  "
            f"∩-spurious={r['intersection_spurious_rate']:.4f}  "
            f"∩-compl={r['intersection_completeness']:.3f}"
        )

    n = len(rows)
    summary = {
        "config": vars(args),
        "runs": rows,
        "mean_completeness": sum(r["mean_completeness"] for r in rows) / n,
        "mean_spurious_rate": sum(r["mean_spurious_rate"] for r in rows) / n,
        "mean_intersection_spurious_rate": sum(r["intersection_spurious_rate"] for r in rows) / n,
        "mean_intersection_completeness": sum(r["intersection_completeness"] for r in rows) / n,
        "mean_final_loss": sum(r["mean_loss"] for r in rows) / n,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmean completeness {summary['mean_completeness']:.3f}  |  "
          f"per-seed-model spurious {summary['mean_spurious_rate']:.4f}  ->  "
          f"across-init intersection spurious {summary['mean_intersection_spurious_rate']:.4f}")
    print(f"summary -> {outdir}/summary.json")


if __name__ == "__main__":
    main()
