#!/usr/bin/env python3
"""
Step 4 — build the LTEL dataset from the exported TSVs and materialise it.

  python 04_build.py --memberships OFFICES/memberships.tsv \
                     --p279 OFFICES/p279.tsv --out DATASET_DIR \
                     [--granularity year|event] [--min-holders N] \
                     [--root QID] [--min-year Y] [--max-year Y]

Pure Python (no torch); runs anywhere. See src/ltel_box/wikidata.py.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# allow running from the repo without install
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ltel_box.wikidata import WikidataConfig, build_dataset, save_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memberships", required=True)
    ap.add_argument("--p279", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--granularity", choices=["year", "event"], default="year")
    ap.add_argument("--min-holders", type=int, default=0)
    ap.add_argument("--root", default=None, help="restrict to P279* descendants of this QID")
    ap.add_argument("--min-year", type=int, default=None)
    ap.add_argument("--max-year", type=int, default=None)
    args = ap.parse_args()

    cfg = WikidataConfig(
        granularity=args.granularity,
        min_holders=args.min_holders,
        root=args.root,
        min_year=args.min_year,
        max_year=args.max_year,
    )
    ds = build_dataset(args.memberships, args.p279, cfg)
    save_dataset(ds, args.out)
    print(f"wrote dataset to {args.out}")
    for k, v in ds.stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
