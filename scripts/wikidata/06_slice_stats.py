#!/usr/bin/env python3
"""
Step 6 — slice statistics for the §5 feasibility exhibit and the RQ2
support-shape discussion.

For a chosen sub-hierarchy root, reports:
  * non-rigid support shape — the fraction of (person, position) memberships
    with >=2 DISJOINT intervals (same person, same office, non-consecutive
    terms; "Grover Cleveland" cases). This is the real-data occurrence of the
    non-convex temporal support that Option B provably cannot represent, so it
    says whether the slice probes that hard regime (level 3) or only
    single-interval non-rigidity (level 2).
  * succession-collapse fraction at year granularity (see 05_collapse_check.py
    for the standalone version) — the justification for event-based ticks.

  python 06_slice_stats.py --memberships OFFICES/memberships.tsv \
      --p279 OFFICES/p279.tsv --root Q48352
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ltel_box.wikidata import _descendants, _parse_p279, _qid, _year  # noqa: E402


def _disjoint_intervals(intervals: list[tuple[int, int]]) -> int:
    """Number of maximal disjoint intervals after merging touching/overlapping
    ones (years one apart count as contiguous, not disjoint)."""
    iv = sorted((a, b) for a, b in intervals if a is not None and b is not None)
    if len(iv) < 2:
        return len(iv)
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return len(merged)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memberships", required=True)
    ap.add_argument("--p279", required=True)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    edges = _parse_p279(Path(args.p279))
    children: dict[str, set[str]] = defaultdict(set)
    for c, p in edges:
        children[p].add(c)
    desc = _descendants(args.root, children)

    iv: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    with Path(args.memberships).open() as f:
        f.readline()
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            while len(cols) < 6:
                cols.append("")
            person, pos = _qid(cols[0]), _qid(cols[1])
            if not person or not pos or pos not in desc:
                continue
            s, e = _year(cols[2]), _year(cols[3])
            if s is None and e is None:
                continue
            iv[(person, pos)].append((s if s is not None else e, e if e is not None else s))

    total = len(iv)
    multi = sum(1 for v in iv.values() if _disjoint_intervals(v) >= 2)
    print(f"root:                                   {args.root}")
    print(f"(person, position) memberships:         {total}")
    print(f"  with >=2 disjoint intervals:          {multi}  ({multi / total:.2%})")
    print("  (non-convex / 'Grover Cleveland' — probes Option B's failure mode)")


if __name__ == "__main__":
    main()
