"""
Ground-then-classify oracle for LTEL entailment (RQ1 ground truth).

Grounds a normal-form LTEL TBox to a *static* EL+ TBox over time-indexed
concept names `A@p` on a lasso (positions 0..k, loop index ℓ, successor
σ(t) = t+1 if t < k else ℓ), following the §3.3 compilation:

  static NF1–NF5    copied per position (role inclusions once — roles rigid)
  A ⊑ ○B            A@t ⊑ B@σ(t)
  ○A ⊑ B            A@σ(t) ⊑ B@t
  A ⊑ □B            A@t ⊑ B@s            for every s ∈ future(t)
  ◇A ⊑ B            A@s ⊑ B@t            for every s ∈ future(t)
  (A U B) ⊑ C       B@s ⊓ ⨅_{p∈V(t,s)} A@p ⊑ C@t   for every s ∈ future(t)

with  future(t) = {t..k} ∪ {ℓ..k}  (the positions visited at or after t) and
the between-set  V(t,s) = {t..s−1} when s ≥ t (no wrap), and
{t..k} ∪ {ℓ..s−1} when s < t (after the wrap; only arises for s ∈ {ℓ..t−1}).
Multi-wrap variants are intentionally NOT emitted: a larger left-hand meet is
implied by the smaller one.

The n-ary U-meets are chained into binary NF2 GCIs with fresh names (a
conservative extension), so the grounded TBox is plain EL+ and can be fed both
to the in-house closure (`compute_el_closure`) and, via the OWL export, to ELK
— the two must agree (cross-check).

INDEPENDENCE NOTE: this module is the oracle code path. It must not import
from, or share lasso/loss machinery with, the embedding construction (loss.py
or the trainer) — a shared bug would make the RQ1 validity check vacuous. The
lasso arithmetic below is deliberately (re-)implemented here from the spec.
"""

from __future__ import annotations
import urllib.parse
from pathlib import Path

from .closure import compute_el_closure
from .ontology import (
    TOP_NAME,
    NF1, NF2, NF3, NF4, NF5, NF6, NF7, NF10, NF12, NF13,
    NormalFormGCI, NormalFormTBox,
)

# ── lasso arithmetic (from the spec; private to the oracle) ──────────────────

def successor(t: int, k: int, loop: int) -> int:
    """σ(t) = t+1 if t < k, else ℓ."""
    return t + 1 if t < k else loop


def future(t: int, k: int, loop: int) -> list[int]:
    """Positions the run visits at or after t: {t..k} ∪ {ℓ..k}, sorted."""
    return sorted(set(range(t, k + 1)) | set(range(loop, k + 1)))


def between(t: int, s: int, k: int, loop: int) -> list[int]:
    """V(t,s): positions visited strictly before reaching witness s from t."""
    if s >= t:
        return list(range(t, s))
    # s < t: reached after the wrap (s ∈ {ℓ..t−1})
    return sorted(set(range(t, k + 1)) | set(range(loop, s)))


# ── grounding ────────────────────────────────────────────────────────────────

def _at(name: str, t: int) -> str:
    """Time-indexed concept name; ⊤ is rigid and stays ⊤."""
    return TOP_NAME if name == TOP_NAME else f"{name}@{t}"


def ground_tbox(tbox: NormalFormTBox, k: int, loop: int) -> NormalFormTBox:
    """Ground a normal-form LTEL TBox to static EL+ over the (k, ℓ)-lasso."""
    if not (0 <= loop <= k):
        raise ValueError(f"need 0 <= loop <= k, got k={k}, loop={loop}")

    gcis: list[NormalFormGCI] = []
    fresh_count = 0

    def fresh() -> str:
        nonlocal fresh_count
        name = f"_g{fresh_count}"
        fresh_count += 1
        return name

    def emit_chain_meet(conjuncts: list[str], rhs: str) -> None:
        """Emit `⨅ conjuncts ⊑ rhs` as binary NF2 chains (fresh names)."""
        if len(conjuncts) == 1:
            gcis.append(NF1(conjuncts[0], rhs))
            return
        cur = conjuncts[0]
        for c in conjuncts[1:-1]:
            g = fresh()
            gcis.append(NF2(cur, c, g))
            cur = g
        gcis.append(NF2(cur, conjuncts[-1], rhs))

    positions = range(k + 1)
    for g in tbox.nf_gcis:
        if isinstance(g, NF1):
            for t in positions:
                gcis.append(NF1(_at(g.lhs, t), _at(g.rhs, t)))
        elif isinstance(g, NF2):
            for t in positions:
                gcis.append(NF2(_at(g.lhs1, t), _at(g.lhs2, t), _at(g.rhs, t)))
        elif isinstance(g, NF3):
            for t in positions:
                gcis.append(NF3(_at(g.lhs, t), g.role, _at(g.rhs, t)))
        elif isinstance(g, NF4):
            for t in positions:
                gcis.append(NF4(g.role, _at(g.lhs, t), _at(g.rhs, t)))
        elif isinstance(g, NF5):
            gcis.append(g)  # roles are rigid: once, un-indexed
        elif isinstance(g, NF6):    # A ⊑ ○B
            for t in positions:
                gcis.append(NF1(_at(g.lhs, t), _at(g.rhs, successor(t, k, loop))))
        elif isinstance(g, NF10):   # ○A ⊑ B
            for t in positions:
                gcis.append(NF1(_at(g.lhs, successor(t, k, loop)), _at(g.rhs, t)))
        elif isinstance(g, NF7):    # A ⊑ □B
            for t in positions:
                for s in future(t, k, loop):
                    gcis.append(NF1(_at(g.lhs, t), _at(g.rhs, s)))
        elif isinstance(g, NF12):   # ◇A ⊑ B
            for t in positions:
                for s in future(t, k, loop):
                    gcis.append(NF1(_at(g.lhs, s), _at(g.rhs, t)))
        elif isinstance(g, NF13):   # (A U B) ⊑ C
            for t in positions:
                for s in future(t, k, loop):
                    conjuncts = [_at(g.until_right, s)] + [
                        _at(g.until_left, p) for p in between(t, s, k, loop)
                    ]
                    emit_chain_meet(conjuncts, _at(g.rhs, t))
        else:
            raise TypeError(f"Unhandled normal form in grounding: {type(g)}")

    concepts = {
        _at(c, t) for c in tbox.concept_names for t in positions
    } | {f"_g{i}" for i in range(fresh_count)}
    return NormalFormTBox(
        concept_names=frozenset(concepts),
        role_names=tbox.role_names,
        nf_gcis=gcis,
    )


# ── oracle entailments ───────────────────────────────────────────────────────

def temporal_closure(
    tbox: NormalFormTBox, k: int, loop: int
) -> frozenset[tuple[str, str, int]]:
    """All entailed temporal subsumptions (A, B, t): A ⊑ B holds at position t.

    Computed with the in-house EL+ closure on the grounded TBox; only pairs of
    *original* concept names at equal positions are reported (fresh grounding
    names and cross-position pairs are internal).
    """
    grounded = ground_tbox(tbox, k, loop)
    pairs = compute_el_closure(grounded)
    out: set[tuple[str, str, int]] = set()
    for sub, sup in pairs:
        if "@" not in sub or "@" not in sup:
            continue
        a, ta = sub.rsplit("@", 1)
        b, tb = sup.rsplit("@", 1)
        if ta != tb or a.startswith("_g") or b.startswith("_g"):
            continue
        if a in tbox.concept_names and b in tbox.concept_names and a != b:
            out.add((a, b, int(ta)))
    return frozenset(out)


# ── OWL functional-syntax export (for the ELK cross-check) ──────────────────

_PREFIX = "http://ltel.example.org/"


def _iri(name: str) -> str:
    return f"<{_PREFIX}{urllib.parse.quote(name, safe='')}>"


def _cls(name: str) -> str:
    return "owl:Thing" if name == TOP_NAME else _iri(name)


def export_functional_owl(tbox: NormalFormTBox, path: str | Path) -> None:
    """Write a (static EL+) NormalFormTBox as OWL 2 functional syntax for ELK.

    Only NF1–NF5 are supported — ground the TBox first.
    """
    lines = [
        "Prefix(owl:=<http://www.w3.org/2002/07/owl#>)",
        f"Ontology(<{_PREFIX}ontology>",
    ]
    for c in sorted(tbox.concept_names):
        if c != TOP_NAME:
            lines.append(f"Declaration(Class({_iri(c)}))")
    for r in sorted(tbox.role_names):
        lines.append(f"Declaration(ObjectProperty({_iri(r)}))")
    for g in tbox.nf_gcis:
        if isinstance(g, NF1):
            lines.append(f"SubClassOf({_cls(g.lhs)} {_cls(g.rhs)})")
        elif isinstance(g, NF2):
            lines.append(
                f"SubClassOf(ObjectIntersectionOf({_cls(g.lhs1)} {_cls(g.lhs2)}) {_cls(g.rhs)})"
            )
        elif isinstance(g, NF3):
            lines.append(
                f"SubClassOf({_cls(g.lhs)} ObjectSomeValuesFrom({_iri(g.role)} {_cls(g.rhs)}))"
            )
        elif isinstance(g, NF4):
            lines.append(
                f"SubClassOf(ObjectSomeValuesFrom({_iri(g.role)} {_cls(g.lhs)}) {_cls(g.rhs)})"
            )
        elif isinstance(g, NF5):
            lines.append(
                f"SubObjectPropertyOf(ObjectPropertyChain({_iri(g.role1)} {_iri(g.role2)}) {_iri(g.rhs)})"
            )
        else:
            raise TypeError(
                f"export_functional_owl expects a grounded (static) TBox; got {type(g)}"
            )
    lines.append(")")
    Path(path).write_text("\n".join(lines) + "\n")


def parse_elk_taxonomy(path: str | Path) -> frozenset[tuple[str, str]]:
    """Read SubClassOf pairs back from an ELK taxonomy output (functional syntax).

    Returns (sub, sup) pairs over the original grounded names; owl:Thing /
    owl:Nothing bounds are dropped.
    """
    import re

    sub_re = re.compile(
        rf"SubClassOf\(<{re.escape(_PREFIX)}([^>]+)>\s+<{re.escape(_PREFIX)}([^>]+)>\)"
    )
    pairs: set[tuple[str, str]] = set()
    text = Path(path).read_text()
    for m in sub_re.finditer(text):
        sub = urllib.parse.unquote(m.group(1))
        sup = urllib.parse.unquote(m.group(2))
        pairs.add((sub, sup))
    return frozenset(pairs)
