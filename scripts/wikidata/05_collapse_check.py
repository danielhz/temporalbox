#!/usr/bin/env python3
"""
Step 5 — succession collapse check for the calendar-year discretisation.

For candidate sub-hierarchy roots, counts the explicit succession pairs
(P1366 replaced-by) whose transition collapses onto the same yearly tick,
i.e. year(end of predecessor's term) == year(start of successor's term on the
same position). A high collapse fraction means year-rounding smears the
succession chains (the most LTEL-flavoured ground truth) and argues for the
event-based discretisation instead.

  python 05_collapse_check.py --memberships OFFICES/memberships.tsv \
      --p279 OFFICES/p279.tsv --roots Q48352 Q2285706 Q486839 ...

Output: one row per root with sub-hierarchy size, membership/succession
counts, and the collapse fraction.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ltel_box.wikidata import _parse_p279, _descendants, _qid, _year  # noqa: E402


def parse_memberships_full(path: Path):
    """Rows (person, position, start_year, end_year, replaced_by_person)."""
    rows = []
    with path.open() as f:
        f.readline()
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            while len(cols) < 6:
                cols.append("")
            person, position = _qid(cols[0]), _qid(cols[1])
            if person is None or position is None:
                continue
            rows.append((person, position, _year(cols[2]), _year(cols[3]), _qid(cols[5])))
    return rows


def collapse_stats(rows, positions: set[str]):
    """Succession-pair collapse statistics within a position set."""
    in_dom = [r for r in rows if r[1] in positions]
    # successor lookup: (person, position) -> sorted start years
    starts: dict[tuple[str, str], list[int]] = defaultdict(list)
    for person, pos, s, _e, _rb in in_dom:
        if s is not None:
            starts[(person, pos)].append(s)
    for v in starts.values():
        v.sort()

    succ_links = sum(1 for r in in_dom if r[4] is not None)
    resolvable = collapsed = 0
    for _person, pos, _s, e, rb in in_dom:
        if rb is None or e is None:
            continue
        cand = [y for y in starts.get((rb, pos), []) if y >= e - 1]
        if not cand:
            continue
        resolvable += 1
        if cand[0] == e:
            collapsed += 1
    return {
        "memberships": len(in_dom),
        "positions_used": len({r[1] for r in in_dom}),
        "succession_links": succ_links,
        "resolvable_pairs": resolvable,
        "collapsed_pairs": collapsed,
        "collapse_fraction": (collapsed / resolvable) if resolvable else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memberships", required=True)
    ap.add_argument("--p279", required=True)
    ap.add_argument("--roots", nargs="+", required=True, help="candidate root QIDs")
    args = ap.parse_args()

    rows = parse_memberships_full(Path(args.memberships))
    edges = _parse_p279(Path(args.p279))
    children: dict[str, set[str]] = defaultdict(set)
    for c, p in edges:
        children[p].add(c)

    header = ("root", "descendants", "positions_used", "memberships",
              "succession_links", "resolvable_pairs", "collapsed_pairs", "collapse_fraction")
    print("\t".join(header))
    for root in args.roots:
        desc = _descendants(root, children)
        st = collapse_stats(rows, desc)
        print("\t".join(str(x) for x in (
            root, len(desc), st["positions_used"], st["memberships"],
            st["succession_links"], st["resolvable_pairs"], st["collapsed_pairs"],
            f"{st['collapse_fraction']:.3f}",
        )))


if __name__ == "__main__":
    main()
