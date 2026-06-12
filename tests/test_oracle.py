"""Ground-then-classify oracle: lasso arithmetic, wrap cases, sanity checks.

The two hand-check cases prescribed in the §3.3 spec note are covered:
(1) t = k: `A ⊑ ○B` must ground to `A@k ⊑ B@ℓ`;
(2) t > ℓ with witness s < t: the U-expansion must include the wrap set.
Plus the sanity property: `◇B ⊑ C` and `(⊤ U B) ⊑ C` induce identical
entailments.
"""

from ltel_box.ontology import TOP_NAME, NF1, NF2, NF6, NF7, NF13, NF12, NormalFormTBox
from ltel_box.oracle import (
    successor, future, between, ground_tbox, temporal_closure,
    export_functional_owl, parse_elk_taxonomy,
)


def _tbox(concepts, gcis, roles=()):
    return NormalFormTBox(
        concept_names=frozenset(concepts),
        role_names=frozenset(roles),
        nf_gcis=list(gcis),
    )


# ── lasso arithmetic ─────────────────────────────────────────────────────────

def test_successor_wraps_to_loop():
    assert successor(0, k=2, loop=1) == 1
    assert successor(1, k=2, loop=1) == 2
    assert successor(2, k=2, loop=1) == 1   # wrap


def test_future_sets():
    assert future(0, k=2, loop=1) == [0, 1, 2]
    assert future(1, k=2, loop=1) == [1, 2]
    assert future(2, k=2, loop=1) == [1, 2]   # t ≥ ℓ: {ℓ..k}


def test_between_direct_and_wrap():
    assert between(0, 2, k=2, loop=1) == [0, 1]      # direct
    assert between(1, 1, k=2, loop=1) == []          # s = t
    assert between(2, 1, k=2, loop=1) == [2]         # wrap: {2..2} ∪ {1..0}
    assert between(3, 2, k=4, loop=1) == [1, 3, 4]   # wrap: {3..4} ∪ {1..1}


# ── prescribed wrap checks ───────────────────────────────────────────────────

def test_next_grounds_through_wrap():
    # (1) t = k: A ⊑ ○B must yield A@k ⊑ B@ℓ.
    tb = _tbox({"A", "B"}, [NF6("A", "B")])
    grounded = ground_tbox(tb, k=2, loop=1)
    assert NF1("A@2", "B@1") in grounded.nf_gcis


def test_until_includes_wrap_witness():
    # (2) t = 2 > ℓ = 1, witness s = 1 < t: V(2,1) = {2}, so the grounded
    # GCI for this witness is B@1 ⊓ A@2 ⊑ C@2.
    tb = _tbox({"A", "B", "C"}, [NF13("A", "B", "C")])
    grounded = ground_tbox(tb, k=2, loop=1)
    assert NF2("B@1", "A@2", "C@2") in grounded.nf_gcis


def test_eventually_equals_top_until():
    # Sanity: ◇B ⊑ C natively vs (⊤ U B) ⊑ C — identical entailments.
    base = [NF1("D", "B")]  # something to entail: D ⊑ B everywhere
    tb_native = _tbox({"B", "C", "D"}, base + [NF12("B", "C")])
    tb_until = _tbox({"B", "C", "D"}, base + [NF13(TOP_NAME, "B", "C")])
    for k, loop in [(2, 0), (2, 1), (3, 2)]:
        assert temporal_closure(tb_native, k, loop) == temporal_closure(tb_until, k, loop)


# ── semantic checks via the closure ──────────────────────────────────────────

def test_next_composes_through_wrap():
    # A ⊑ ○B, B ⊑ ○C on k=1, ℓ=0: A@0→B@1→C@0, so (A, C, 0) and (A, C, 1);
    # cross-position pairs like A@0 ⊑ B@1 must NOT surface as (A, B, t).
    tb = _tbox({"A", "B", "C"}, [NF6("A", "B"), NF6("B", "C")])
    tc = temporal_closure(tb, k=1, loop=0)
    assert ("A", "C", 0) in tc and ("A", "C", 1) in tc
    assert not any(a == "A" and b == "B" for (a, b, _) in tc)


def test_globally_holds_at_all_future_positions():
    # A ⊑ □B: t ∈ future(t) always, so (A, B, t) for every t.
    tb = _tbox({"A", "B"}, [NF7("A", "B")])
    tc = temporal_closure(tb, k=2, loop=1)
    assert {("A", "B", 0), ("A", "B", 1), ("A", "B", 2)} <= tc


# ── OWL export round-trip ────────────────────────────────────────────────────

def test_owl_export_and_parse_roundtrip(tmp_path):
    tb = _tbox({"A", "B"}, [NF6("A", "B")])
    grounded = ground_tbox(tb, k=1, loop=0)
    path = tmp_path / "g.ofn"
    export_functional_owl(grounded, path)
    text = path.read_text()
    assert "Ontology(" in text and "SubClassOf(" in text
    # our own parser reads back exactly the NF1 pairs we exported
    pairs = parse_elk_taxonomy(path)
    assert ("A@0", "B@1") in pairs and ("A@1", "B@0") in pairs


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
