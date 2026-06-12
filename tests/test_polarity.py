"""Polarity restriction: right of ⊑ allows ○/□ only; left allows ○/U/◇ only."""

import pytest

from ltel_box.ontology import (
    ConceptName, Conjunction, Eventually, Globally, NextC, Until,
    GCI, LTELTBox, NF7, NF12,
)
from ltel_box.normalize import normalize, PolarityError

A, B, C = ConceptName("A"), ConceptName("B"), ConceptName("C")


def _tbox(*gcis: GCI) -> LTELTBox:
    return LTELTBox(
        concept_names=frozenset({"A", "B", "C"}),
        role_names=frozenset(),
        gcis=list(gcis),
        role_inclusions=[],
    )


def test_right_eventually_rejected():
    with pytest.raises(PolarityError):
        normalize(_tbox(GCI(A, Eventually(B))))


def test_right_until_rejected():
    with pytest.raises(PolarityError):
        normalize(_tbox(GCI(A, Until(B, C))))


def test_left_globally_rejected():
    with pytest.raises(PolarityError):
        normalize(_tbox(GCI(Globally(A), B)))


def test_allowed_forms_pass():
    nf = normalize(_tbox(
        GCI(A, NextC(B)),          # right ○
        GCI(A, Globally(B)),       # right □
        GCI(NextC(A), B),          # left ○
        GCI(Eventually(A), B),     # left ◇
        GCI(Until(A, B), C),       # left U
    ))
    assert len(nf.nf_gcis) == 5


def test_nested_left_eventually_allowed():
    # ◇A inside a left-side conjunction: aux must be defined as ◇A ⊑ aux
    # (left polarity), yielding an NF12 — not a forbidden right-◇.
    nf = normalize(_tbox(GCI(Conjunction(Eventually(A), B), C)))
    assert any(isinstance(g, NF12) for g in nf.nf_gcis)


def test_nested_right_eventually_rejected():
    # ◇C inside a right-side conjunction is a right-◇: no normal form.
    with pytest.raises(PolarityError):
        normalize(_tbox(GCI(A, Conjunction(B, Eventually(C)))))


def test_nested_right_globally_allowed():
    nf = normalize(_tbox(GCI(A, Conjunction(B, Globally(C)))))
    assert any(isinstance(g, NF7) for g in nf.nf_gcis)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
