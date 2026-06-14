"""Slice trainer on a tiny synthetic office dataset (requires torch)."""

import pytest

torch = pytest.importorskip("torch")

from ltel_box.loss import point_in_box_loss                          # noqa: E402
from ltel_box.ontology import NF1, NormalFormTBox, Vocabulary        # noqa: E402
from ltel_box.slice_trainer import (                                 # noqa: E402
    SliceTrainConfig, build_slice_tensors, evaluate_slice, train_slice,
)
from ltel_box.wikidata import MembershipInterval, WikidataConfig, WikidataOfficeDataset  # noqa: E402


def _toy_dataset() -> WikidataOfficeDataset:
    # Hierarchy: Q100 ⊑ Q300 ⊑ Q900 ; two people holding Q100 in disjoint terms.
    tbox = NormalFormTBox(
        concept_names=frozenset({"Q100", "Q300", "Q900"}),
        role_names=frozenset(),
        nf_gcis=[NF1("Q100", "Q300"), NF1("Q300", "Q900")],
    )
    vocab = Vocabulary.from_tbox(tbox)
    closure = frozenset({("Q100", "Q300"), ("Q300", "Q900"), ("Q100", "Q900")})
    return WikidataOfficeDataset(
        tbox=tbox, vocab=vocab,
        individuals=["P1", "P2"],
        time_points=[2000, 2001, 2002, 2003],
        memberships=[
            MembershipInterval("P1", "Q100", 0, 1),
            MembershipInterval("P2", "Q100", 2, 3),
        ],
        subsumption_closure=closure,
        config=WikidataConfig(),
        stats={},
    )


def test_point_in_box_loss():
    c = torch.zeros(1, 2)
    o = torch.ones(1, 2)
    assert point_in_box_loss(torch.zeros(1, 2), c, o).item() == pytest.approx(0.0)
    assert point_in_box_loss(torch.tensor([[2.0, 0.0]]), c, o).item() == pytest.approx(1.0)


def test_build_tensors_expands_intervals():
    st = build_slice_tensors(_toy_dataset())
    # P1: ticks 0,1 ; P2: ticks 2,3  -> 4 membership triples
    assert st.membership.shape == (4, 3)
    assert st.n_ticks == 4
    assert st.nf1.shape == (2, 2)


def test_train_and_preserve_subsumptions():
    ds = _toy_dataset()
    cfg = SliceTrainConfig(dim=8, epochs=800, lr=5e-2, eval_pairs=3,
                           eval_ticks_per_pair=4, eval_neg_pairs=10)
    model, st, final = train_slice(ds, cfg)
    assert final["loss_subsumption"] < 1e-2
    assert final["loss_membership"] < 1e-2
    m = evaluate_slice(model, st, ds, cfg, eps=1e-2)
    # the rigid hierarchy must be preserved at the sampled ticks
    assert m["completeness_sampled"] == pytest.approx(1.0)
    # individuals are in their office box: P1's point in Q100 at tick 0
    pt = model.individual_point(torch.tensor([0]))
    c, o = model.concept_box(torch.tensor([st.vocab.concept_to_id["Q100"]]), torch.tensor([0]))
    assert point_in_box_loss(pt, c, o).item() < 1e-2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
