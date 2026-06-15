"""Trainer + induced-interpretation harness (requires torch)."""

import pytest

torch = pytest.importorskip("torch")

from ltel_box.induce import held_subsumptions, rq1_metrics            # noqa: E402
from ltel_box.model import TemporalBoxModel, between, future, successor  # noqa: E402
from ltel_box.ontology import (                                       # noqa: E402
    NF1, NF6, NF7, NF13, NormalFormTBox, Vocabulary,
)
from ltel_box.oracle import temporal_closure                          # noqa: E402
from ltel_box.trainer import TrainConfig, compile_tbox, loss_components, train  # noqa: E402


def _tbox(concepts, gcis, roles=()):
    return NormalFormTBox(
        concept_names=frozenset(concepts),
        role_names=frozenset(roles),
        nf_gcis=list(gcis),
    )


def test_construction_lasso_matches_spec():
    # Construction-side lasso arithmetic (model.py) must agree with the spec
    # examples — independently of oracle.py, which has its own copy.
    assert successor(2, k=2, loop=1) == 1
    assert future(2, k=2, loop=1) == [1, 2]
    assert between(2, 1, k=2, loop=1) == [2]


def test_compile_counts():
    tb = _tbox({"A", "B", "C"}, [NF1("A", "B"), NF6("A", "B"), NF7("A", "B"),
                                 NF13("A", "B", "C")])
    vocab = Vocabulary.from_tbox(tb)
    comp = compile_tbox(tb, vocab, k=2, loop=1)
    assert comp.containment["nf1"].shape[0] == 3          # one per position
    assert comp.containment["nf6"].shape[0] == 3
    # NF7: sum over |future(t)| = 3 + 2 + 2
    assert comp.containment["nf7"].shape[0] == 7
    # NF13: one row per (t, s ∈ future(t)) = 7, grouped by |V|
    assert sum(v.shape[0] for v in comp.nf13.values()) == 7


def test_nf6_compile_wraps():
    tb = _tbox({"A", "B"}, [NF6("A", "B")])
    vocab = Vocabulary.from_tbox(tb)
    comp = compile_tbox(tb, vocab, k=2, loop=1)
    rows = comp.containment["nf6"].tolist()
    a, b = vocab.concept_to_id["A"], vocab.concept_to_id["B"]
    assert [a, b, 2, 1] in rows                            # σ(k)=ℓ wrap


def test_loss_decreases():
    tb = _tbox({"A", "B", "C"}, [NF1("A", "B"), NF6("B", "C"), NF7("A", "C")])
    vocab = Vocabulary.from_tbox(tb)
    cfg = TrainConfig(dim=8, k=2, loop=1, epochs=300, lr=5e-2, seed=0)
    comp = compile_tbox(tb, vocab, cfg.k, cfg.loop)
    model0 = TemporalBoxModel(vocab.n_concepts, 1, cfg.k, cfg.loop, cfg.dim,
                              top_id=vocab.top_id, seed=0)
    with torch.no_grad():
        before = float(sum(loss_components(model0, comp).values()))
    _, final = train(tb, vocab, cfg)
    assert final["loss_total"] < before
    assert final["loss_total"] < 1e-3


def test_end_to_end_recovery_tiny():
    # A ⊑ B, B ⊑ ○C on a (k=1, ℓ=0) lasso: oracle entails (A,B,t) and the
    # ○-composition; near-zero loss must recover all entailed pairs.
    tb = _tbox({"A", "B", "C"}, [NF1("A", "B"), NF6("B", "C")])
    vocab = Vocabulary.from_tbox(tb)
    cfg = TrainConfig(dim=8, k=1, loop=0, epochs=1500, lr=5e-2, seed=0)
    model, final = train(tb, vocab, cfg)
    assert final["loss_total"] < 1e-5

    entailed = temporal_closure(tb, cfg.k, cfg.loop)
    held = held_subsumptions(model, vocab, eps=1e-4)
    m = rq1_metrics(held, entailed, vocab, n_positions=cfg.k + 1)
    assert m["completeness"] == pytest.approx(1.0)
    assert m["entailed"] >= 2          # at least the NF1 pairs at both positions


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))


def test_box2el_base_trains_with_existentials():
    # The faithful Box²EL base must drive a TBox with existentials + role
    # composition to low loss (the Phase-2 base, distinct from translational).
    from ltel_box.model import Box2ELModel
    from ltel_box.ontology import NF1, NF3, NF4, NF5, NormalFormTBox, Vocabulary

    tb = NormalFormTBox(
        frozenset({"A", "B", "C", "D"}), frozenset({"r", "s", "t"}),
        [NF1("A", "B"), NF3("A", "r", "C"), NF4("r", "C", "D"), NF5("r", "s", "t")],
    )
    vocab = Vocabulary.from_tbox(tb)
    cfg = TrainConfig(dim=16, k=2, loop=1, epochs=600, lr=5e-2, seed=0)
    _, final = train(tb, vocab, cfg, model_cls=Box2ELModel)
    assert final["loss_total"] < 5e-2     # Box²EL path runs and optimizes
