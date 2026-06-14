"""EL+ closure completeness — regression guard for the rules, esp. CR5
(role composition), whose absence the ELK cross-check caught."""

from ltel_box.closure import compute_el_closure
from ltel_box.ontology import NF1, NF2, NF3, NF4, NF5, NormalFormTBox


def _tbox(concepts, gcis, roles):
    return NormalFormTBox(frozenset(concepts), frozenset(roles), list(gcis))


def _closure(concepts, gcis, roles=()):
    return {(a, b) for (a, b) in compute_el_closure(_tbox(concepts, gcis, roles))
            if a != b and b != "⊤"}


def test_transitivity():
    c = _closure({"A", "B", "C"}, [NF1("A", "B"), NF1("B", "C")])
    assert ("A", "C") in c


def test_conjunction_rule():
    c = _closure({"A", "B1", "B2", "D"},
                 [NF1("A", "B1"), NF1("A", "B2"), NF2("B1", "B2", "D")])
    assert ("A", "D") in c


def test_existential_cr4():
    # A ⊑ ∃r.B, B ⊑ C, ∃r.C ⊑ D  ⇒  A ⊑ D
    c = _closure({"A", "B", "C", "D"},
                 [NF3("A", "r", "B"), NF1("B", "C"), NF4("r", "C", "D")],
                 roles={"r"})
    assert ("A", "D") in c


def test_role_composition_cr5():
    # A ⊑ ∃r.B, B ⊑ ∃s.E, r∘s ⊑ t, ∃t.E ⊑ G  ⇒  A ⊑ G
    # (the case the ELK cross-check flagged as missing before CR5 was added)
    c = _closure(
        {"A", "B", "E", "G"},
        [NF3("A", "r", "B"), NF3("B", "s", "E"), NF5("r", "s", "t"), NF4("t", "E", "G")],
        roles={"r", "s", "t"},
    )
    assert ("A", "G") in c


def test_no_spurious_without_composition():
    # Same as above but WITHOUT the r∘s⊑t axiom: A ⊑ G must NOT be derived.
    c = _closure(
        {"A", "B", "E", "G"},
        [NF3("A", "r", "B"), NF3("B", "s", "E"), NF4("t", "E", "G")],
        roles={"r", "s", "t"},
    )
    assert ("A", "G") not in c


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
