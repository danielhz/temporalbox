#!/usr/bin/env python3
"""
Materialize the synthetic datasets of the reported experiments as files
(for the DaRUS deposit).

The experiments regenerate their TBoxes from (config, seed) at run time, so
the datasets exist nowhere as files. This script reads the archived run
records (summary.json + per-run JSONL meta lines), rebuilds every distinct
dataset through the exact generator code paths the drivers use
(ltel_box.synthetic.generate_tbox / ltel_box.rq2.generate_rq2), recomputes the
oracle ground-truth closure, and writes one directory per dataset (see
ltel_box.dataset_io for the format) plus a manifest.json mapping run-id /
sweep cell -> dataset directory.

Where a run record carries the oracle's entailment count, the materialized
closure is cross-checked against it.

  python scripts/materialize_datasets.py \
      --results-root ../darus/staging/runs/kg1-ltel-validate/results \
      --out ../darus/staging/datasets/synthetic
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ltel_box.dataset_io import save_dataset_dir                    # noqa: E402
from ltel_box.oracle import temporal_closure                        # noqa: E402
from ltel_box.rq2 import RQ2Config, generate_rq2                    # noqa: E402
from ltel_box.synthetic import SyntheticConfig, generate_tbox       # noqa: E402

# RQ1 experiment dirs sharing the generate_tbox dataset family (one dataset
# per data_seed; the base model does not enter the dataset).
RQ1_DIRS = ("rq1_box2el", "rq1_boxel", "rq1_transbox")

# Mirror of scripts/rq1_breakdowns.py OPERATORS + its by-operator constants.
BREAKDOWN_OPERATORS = {
    "next_rhs": "n_nf6", "globally_rhs": "n_nf7",
    "next_lhs": "n_nf10", "eventually_lhs": "n_nf12", "until_lhs": "n_nf13",
}


def scaled(x: int, gci_scale: float) -> int:
    """Mirror of scripts/rq1_synthetic.py::run_tbox.scaled."""
    return max(0, round(x * gci_scale))


def rq1_synthetic_config(data_seed: int, cfg: dict) -> SyntheticConfig:
    """Mirror of scripts/rq1_synthetic.py::run_tbox's SyntheticConfig.

    Metas written before the --n-nf5 flag existed carry no "n_nf5" key; those
    runs used the then-hardcoded scaled(2).
    """
    gs = cfg["gci_scale"]
    n_nf5 = cfg.get("n_nf5", scaled(2, gs))
    return SyntheticConfig(
        n_concepts=cfg["n_concepts"], n_roles=cfg["n_roles"],
        n_nf1=scaled(15, gs), n_nf2=scaled(8, gs), n_nf3=scaled(8, gs),
        n_nf4=scaled(8, gs), n_nf5=n_nf5, n_nf6=scaled(5, gs),
        n_nf7=scaled(4, gs), n_nf10=scaled(3, gs), n_nf12=scaled(2, gs),
        n_nf13=scaled(2, gs), seed=data_seed,
    )


def materialize_tbox_dataset(out: Path, rel: str, syn: SyntheticConfig,
                             k: int, loop: int, family: str,
                             manifest: dict) -> int:
    """generate_tbox family: write tbox + temporal closure; return closure size."""
    if rel in manifest["datasets"]:
        return manifest["datasets"][rel]["closure_pairs"]
    ds = generate_tbox(syn)
    closure = temporal_closure(ds.tbox, k, loop)
    config = {
        "family": family,
        "generator": "ltel_box.synthetic.generate_tbox",
        "synthetic_config": vars(syn),
        "oracle": {"function": "ltel_box.oracle.temporal_closure", "k": k, "loop": loop},
    }
    save_dataset_dir(out / rel, tbox=ds.tbox, closure=closure, config=config)
    manifest["datasets"][rel] = {
        "family": family, "config": config,
        "n_gcis": len(ds.tbox.nf_gcis), "closure_pairs": len(closure),
    }
    return len(closure)


def materialize_rq2_dataset(out: Path, rel: str, cfg: RQ2Config, family: str,
                            manifest: dict) -> None:
    """generate_rq2 family: write tbox + temporal closure + ABox."""
    if rel in manifest["datasets"]:
        return
    ds = generate_rq2(cfg)
    config = {
        "family": family,
        "generator": "ltel_box.rq2.generate_rq2",
        "rq2_config": vars(cfg),
        "oracle": {"function": "ltel_box.oracle.temporal_closure",
                   "k": cfg.k, "loop": cfg.loop},
    }
    abox = {
        "concept_id_note": "concept ids follow Vocabulary.from_tbox order: "
                           "id 0 = TOP, then concept names sorted",
        "individuals": ds.individuals,
        "n_ticks": ds.n_ticks, "k": ds.k, "loop": ds.loop,
        "memberships": [[i, c, ticks] for (i, c, ticks) in ds.memberships],
        "pos_triples": [list(t) for t in ds.pos_triples],
        "neg_triples": [list(t) for t in ds.neg_triples],
        "rigid_concepts": sorted(ds.rigid_concepts),
        "support_level": ds.support_level,
        "stats": ds.stats,
    }
    save_dataset_dir(out / rel, tbox=ds.tbox, closure=ds.closure,
                     config=config, abox=abox)
    manifest["datasets"][rel] = {
        "family": family, "config": config,
        "n_gcis": len(ds.tbox.nf_gcis), "closure_pairs": len(ds.closure),
        "memberships": len(ds.memberships),
    }


def jsonl_meta_and_entailed(path: Path) -> tuple[dict, int | None]:
    """First (meta) line's config + the eval line's 'entailed' count if present."""
    meta, entailed = {}, None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "meta":
                meta = rec.get("config", {})
            elif rec.get("type") == "eval" and "entailed" in rec:
                entailed = rec["entailed"]
    return meta, entailed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True,
                    help="archived results root (contains rq1_box2el/, rq2_multi/, ...)")
    ap.add_argument("--out", required=True,
                    help="output root, e.g. .../datasets/synthetic")
    args = ap.parse_args()
    root = Path(args.results_root)
    out = Path(args.out)

    manifest: dict = {
        "generated_by": "scripts/materialize_datasets.py",
        "source_results_root": str(root),
        "python": f"{platform.python_version()} ({platform.python_implementation()})",
        "determinism": (
            "Datasets are regenerated from (config, seed) via Python's "
            "random.Random (MT19937) only — no numpy/torch RNG — so they are "
            "bit-identical across platforms for any CPython 3.x."
        ),
        "datasets": {},   # dataset dir -> description
        "runs": {},       # "<experiment>/<run_id>" -> {dataset, reported}
        "sweeps": {},     # summary-only experiments: sweep cell -> dataset dir
    }
    checked = mismatches = 0

    # ── RQ1 main (three bases over the same 5 TBoxes) ────────────────────────
    summary_cfgs = {}
    for d in RQ1_DIRS:
        s = json.loads((root / d / "summary.json").read_text())
        summary_cfgs[d] = s["config"]
        # cross-check: summary rows carry the oracle 'entailed' count
        for row in s["runs"]:
            syn = rq1_synthetic_config(row["data_seed"], s["config"])
            rel = f"rq1/seed{row['data_seed']}_nf5-{syn.n_nf5}"
            n = materialize_tbox_dataset(out, rel, syn, s["config"]["k"],
                                         s["config"]["loop"], "rq1", manifest)
            checked += 1
            if n != row["entailed"]:
                mismatches += 1
                print(f"MISMATCH {d} data_seed={row['data_seed']}: "
                      f"materialized {n} entailed, summary says {row['entailed']}")

    # map every archived run-id to its dataset (incl. interim configs)
    for d in RQ1_DIRS:
        reported_cfg = summary_cfgs[d]
        reported_nf5 = reported_cfg.get("n_nf5", scaled(2, reported_cfg["gci_scale"]))
        for jl in sorted((root / d).glob("*.jsonl")):
            meta, entailed = jsonl_meta_and_entailed(jl)
            if "data_seed" not in meta:   # oldest metas: data seed under "seed"
                meta = {**meta, "data_seed": meta.get("seed")}
            syn = rq1_synthetic_config(meta["data_seed"], meta)
            rel = f"rq1/seed{meta['data_seed']}_nf5-{syn.n_nf5}"
            n = materialize_tbox_dataset(out, rel, syn, meta["k"], meta["loop"],
                                         "rq1", manifest)
            run_id = jl.stem
            manifest["runs"][f"{d}/{run_id}"] = {
                "dataset": rel,
                "reported": syn.n_nf5 == reported_nf5,
            }
            if entailed is not None:
                checked += 1
                if n != entailed:
                    mismatches += 1
                    print(f"MISMATCH {d}/{run_id}: materialized {n} entailed, "
                          f"run record says {entailed}")

    # ── RQ1 breakdowns (summary-only) ────────────────────────────────────────
    s = json.loads((root / "rq1_breakdowns" / "summary.json").read_text())
    cfg = s["config"]
    sweep: dict[str, str] = {}
    for label, knob in BREAKDOWN_OPERATORS.items():
        for seed in cfg["seeds"]:
            # mirror of scripts/rq1_breakdowns.py (by-operator constants)
            kw = dict(n_nf1=10, n_nf2=4, n_nf3=4, n_nf4=4, n_nf5=0,
                      n_nf6=0, n_nf7=0, n_nf10=0, n_nf12=0, n_nf13=0)
            kw[knob] = 6
            syn = SyntheticConfig(n_concepts=cfg["n_concepts"], n_roles=2,
                                  seed=seed, **kw)
            rel = f"rq1_breakdowns/op-{label}_seed{seed}"
            materialize_tbox_dataset(out, rel, syn, cfg["k"], cfg["loop"],
                                     "rq1_breakdowns/by_operator", manifest)
            sweep[f"by_operator/{label}/seed{seed}"] = rel
    for rho in cfg["rhos"]:
        for seed in cfg["seeds"]:
            rq2cfg = RQ2Config(n_concepts=cfg["n_concepts"], n_roles=2,
                               k=cfg["k"], loop=cfg["loop"],
                               rigid_fraction=rho, n_individuals=1, seed=seed)
            rel = f"rq1_breakdowns/rho{rho}_seed{seed}"
            materialize_rq2_dataset(out, rel, rq2cfg,
                                    "rq1_breakdowns/by_rigid_fraction", manifest)
            sweep[f"by_rigid_fraction/rho{rho}/seed{seed}"] = rel
    manifest["sweeps"]["rq1_breakdowns"] = sweep

    # ── RQ2 single / multi (summary-only) ────────────────────────────────────
    for d in ("rq2_single", "rq2_multi"):
        s = json.loads((root / d / "summary.json").read_text())
        cfg = s["config"]
        sweep = {}
        for rho in cfg["rhos"]:
            for seed in cfg["seeds"]:
                # mirror of scripts/rq2_sweep.py (controlled = pure ABox)
                rq2cfg = RQ2Config(
                    n_concepts=cfg["n_concepts"], n_roles=1,
                    rigid_fraction=rho, k=cfg["k"],
                    n_nf1=0, n_nf2=0, n_nf3=0, n_nf4=0, n_temporal=0,
                    n_individuals=cfg["n_individuals"],
                    memberships_per_individual=4,
                    support_max_intervals=cfg["support_max_intervals"],
                    seed=seed,
                )
                rel = f"{d}/rho{rho}_seed{seed}"
                materialize_rq2_dataset(out, rel, rq2cfg, d, manifest)
                sweep[f"rho{rho}/seed{seed}"] = rel
        manifest["sweeps"][d] = sweep

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\n{len(manifest['datasets'])} datasets -> {out}")
    print(f"{len(manifest['runs'])} run-ids mapped; oracle cross-checks: "
          f"{checked} checked, {mismatches} mismatches")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
