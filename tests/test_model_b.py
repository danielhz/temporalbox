"""Option B (TimeAugmentedBoxModel) — protocol conformance + the convex-support
limitation that RQ2 measures (requires torch)."""

import pytest

torch = pytest.importorskip("torch")

from ltel_box.model import BoxScorer, TemporalBoxModel, TimeAugmentedBoxModel  # noqa: E402


def test_b_satisfies_scorer_protocol():
    m = TimeAugmentedBoxModel(n_concepts=5, n_roles=1, k=3, loop=3, dim=8)
    assert isinstance(m, BoxScorer)
    ids = torch.tensor([1, 2])
    pos = torch.tensor([0, 1])
    c, o = m.concept_box(ids, pos)
    assert c.shape == (2, 8) and o.shape == (2, 8)


def test_b_time_gating_is_convex():
    # A concept's box is "active" (large offset) only on a contiguous tick
    # window around its time-center; offsets decay away from it. So the set of
    # active ticks is an interval — B cannot represent non-contiguous support.
    torch.manual_seed(0)
    m = TimeAugmentedBoxModel(n_concepts=1, n_roles=1, k=10, loop=10, dim=4, gate_beta=8.0)
    with torch.no_grad():
        m.time_center[0] = 5.0
        m.time_offset_raw[0] = torch.tensor(1.0)  # softplus(1)≈1.31 half-width
    cid = torch.zeros(11, dtype=torch.long)
    ticks = torch.arange(11)
    _, o = m.concept_box(cid, ticks)
    vol = o.sum(-1)  # per-tick box size
    active = (vol > 0.5 * vol.max()).int().tolist()
    # the active region is a single contiguous block (convex support)
    runs = sum(1 for i in range(1, len(active)) if active[i] and not active[i - 1])
    runs += active[0]
    assert runs == 1


def test_b_shares_box_across_ticks_spatially():
    # The spatial center is shared across ticks (one box augmented by time);
    # only the time gate varies. Centers at different ticks coincide.
    m = TimeAugmentedBoxModel(n_concepts=3, n_roles=1, k=4, loop=4, dim=4)
    c0, _ = m.concept_box(torch.tensor([1]), torch.tensor([0]))
    c1, _ = m.concept_box(torch.tensor([1]), torch.tensor([3]))
    assert torch.allclose(c0, c1)


def test_a_and_b_both_train_via_same_harness():
    # Smoke: both models expose concept_box and train under the shared loss.
    from ltel_box.ontology import NF1, NormalFormTBox, Vocabulary
    from ltel_box.trainer import compile_tbox, loss_components

    tb = NormalFormTBox(frozenset({"A", "B"}), frozenset(), [NF1("A", "B")])
    vocab = Vocabulary.from_tbox(tb)
    comp = compile_tbox(tb, vocab, k=2, loop=1)
    for cls in (TemporalBoxModel, TimeAugmentedBoxModel):
        m = cls(vocab.n_concepts, 1, 2, 1, 8, top_id=vocab.top_id, seed=0)
        loss = sum(loss_components(m, comp).values())
        assert loss.item() >= 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
