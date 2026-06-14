#!/usr/bin/env python3
"""
Cross-check the in-house EL+ closure against ELK on grounded LTEL ontologies.

For each synthetic seed: generate a TBox, ground it to static EL+ over the
lasso (oracle.ground_tbox), then compare two independent classifications of
the SAME grounded ontology —

  * the in-house forward-chaining closure (ltel_box.closure), and
  * ELK 0.6.0 (reasoner of record), via the OWL functional export.

ELK emits the transitively-reduced hierarchy, so its edges are transitively
closed before comparison. Disagreement on any seed = a bug in the grounding's
EL+ handling or in the closure; the RQ1 ground truth depends on these
agreeing, so this is a hard gate.

  ELK_JAR=.../elk.jar python scripts/oracle/crosscheck_elk.py --seeds 0..9
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ltel_box.closure import compute_el_closure                      # noqa: E402
from ltel_box.oracle import export_functional_owl, ground_tbox       # noqa: E402
from ltel_box.ontology import TOP_NAME                               # noqa: E402
from ltel_box.synthetic import SyntheticConfig, generate_tbox        # noqa: E402

_PREFIX = "http://ltel.example.org/"
_SUB_RE = re.compile(rf"SubClassOf\(<{re.escape(_PREFIX)}([^>]+)>\s+<{re.escape(_PREFIX)}([^>]+)>\)")
_EQ_RE = re.compile(rf"EquivalentClasses\(((?:<{re.escape(_PREFIX)}[^>]+>\s*)+)\)")
_CLS_RE = re.compile(rf"<{re.escape(_PREFIX)}([^>]+)>")


def parse_elk(path: Path) -> set[tuple[str, str]]:
    """Direct (sub, sup) edges from ELK output: SubClassOf + expanded
    EquivalentClasses; owl:* and reflexive pairs dropped."""
    text = path.read_text()
    edges: set[tuple[str, str]] = set()
    for m in _SUB_RE.finditer(text):
        edges.add((urllib.parse.unquote(m.group(1)), urllib.parse.unquote(m.group(2))))
    for m in _EQ_RE.finditer(text):
        members = [urllib.parse.unquote(x) for x in _CLS_RE.findall(m.group(1))]
        for a in members:
            for b in members:
                if a != b:
                    edges.add((a, b))
    return {(a, b) for a, b in edges if a != b}


def transitive_closure(edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    succ: dict[str, set[str]] = {}
    for a, b in edges:
        succ.setdefault(a, set()).add(b)
    out: set[tuple[str, str]] = set()
    for start in succ:
        seen, stack = set(), list(succ[start])
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            out.add((start, n))
            stack.extend(succ.get(n, ()))
    return {(a, b) for a, b in out if a != b}


def named_pairs(pairs) -> set[tuple[str, str]]:
    """Restrict to genuine named-class pairs (drop ⊤ and reflexive)."""
    return {(a, b) for (a, b) in pairs if a != b and a != TOP_NAME and b != TOP_NAME}


def run_seed(seed: int, k: int, loop: int, elk_jar: str, workdir: Path) -> dict:
    ds = generate_tbox(SyntheticConfig(n_concepts=20, n_roles=3, seed=seed))
    grounded = ground_tbox(ds.tbox, k, loop)

    mine = named_pairs(compute_el_closure(grounded))

    ofn = workdir / f"seed{seed}.ofn"
    tax = workdir / f"seed{seed}_tax.ofn"
    export_functional_owl(grounded, ofn)
    subprocess.run(["java", "-jar", elk_jar, "-i", str(ofn), "-c", "-o", str(tax), "-q"],
                   check=True, capture_output=True)
    elk = named_pairs(transitive_closure(parse_elk(tax)))

    only_mine = mine - elk
    only_elk = elk - mine
    return {
        "seed": seed, "grounded_concepts": len(grounded.concept_names),
        "grounded_gcis": len(grounded.nf_gcis), "mine": len(mine), "elk": len(elk),
        "only_mine": sorted(only_mine)[:5], "only_elk": sorted(only_elk)[:5],
        "agree": not only_mine and not only_elk,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--loop", type=int, default=1)
    ap.add_argument("--elk-jar", default=os.environ.get("ELK_JAR", ""))
    args = ap.parse_args()
    if not args.elk_jar or not Path(args.elk_jar).is_file():
        sys.exit("set --elk-jar or ELK_JAR to the unpacked elk.jar")

    all_agree = True
    with tempfile.TemporaryDirectory() as d:
        for seed in args.seeds:
            r = run_seed(seed, args.k, args.loop, args.elk_jar, Path(d))
            status = "OK " if r["agree"] else "MISMATCH"
            print(f"[{status}] seed={seed:2d}  grounded={r['grounded_concepts']}c/"
                  f"{r['grounded_gcis']}gci  mine={r['mine']}  elk={r['elk']}")
            if not r["agree"]:
                all_agree = False
                print(f"         only-mine(≤5): {r['only_mine']}")
                print(f"         only-elk (≤5): {r['only_elk']}")
    print("\nALL SEEDS AGREE" if all_agree else "\nDISAGREEMENT FOUND")
    sys.exit(0 if all_agree else 1)


if __name__ == "__main__":
    main()
