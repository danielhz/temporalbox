"""Numeric checks for the Option A temporal losses (requires torch; the
suite skips cleanly on CPU-only hosts without it — run on the GPU cluster)."""

import pytest

torch = pytest.importorskip("torch")

from ltel_box.loss import (  # noqa: E402
    box_meet_many, nf6_loss, nf7_loss, nf10_loss, nf12_loss, nf13_witness_loss,
)


def box(lo, hi):
    """1-D box [lo, hi] as (center, offset) tensors of shape (1, 1)."""
    c = torch.tensor([[(lo + hi) / 2]])
    o = torch.tensor([[(hi - lo) / 2]])
    return c, o


def stack(boxes):
    cs = torch.stack([c for c, _ in boxes], dim=-2)
    os_ = torch.stack([o for _, o in boxes], dim=-2)
    return cs, os_


def test_box_meet_many_nonempty_and_empty():
    cs, os_ = stack([box(0, 4), box(2, 6), box(3, 5)])
    cm, om, ne = box_meet_many(cs, os_)
    assert ne.all()
    assert torch.allclose(cm, torch.tensor([[3.5]]))   # meet = [3, 4]
    assert torch.allclose(om, torch.tensor([[0.5]]))
    _, _, ne2 = box_meet_many(*stack([box(0, 1), box(2, 3)]))
    assert not ne2.any()


def test_nf6_zero_iff_contained():
    a, b_good, b_bad = box(1, 2), box(0, 3), box(0, 1.5)
    assert nf6_loss(*a, *b_good).item() == pytest.approx(0.0)
    assert nf6_loss(*a, *b_bad).item() > 0


def test_nf7_sums_over_future_positions():
    a = box(1, 2)
    futures_ok = stack([box(0, 3), box(0.5, 2.5)])
    futures_one_bad = stack([box(0, 3), box(1.5, 3)])
    assert nf7_loss(*a, *futures_ok).item() == pytest.approx(0.0)
    assert nf7_loss(*a, *futures_one_bad).item() > 0


def test_nf10_is_containment_of_shifted_a():
    a_next, b = box(1, 2), box(0, 3)
    assert nf10_loss(*a_next, *b).item() == pytest.approx(0.0)


def test_nf12_every_future_a_inside_b_now():
    futures = stack([box(1, 2), box(2, 3)])
    b_good, b_bad = box(0, 4), box(0, 2.5)
    assert nf12_loss(*futures, *b_good).item() == pytest.approx(0.0)
    assert nf12_loss(*futures, *b_bad).item() > 0


def test_nf13_witness_empty_between_set():
    b_s, c_t = box(1, 2), box(0, 3)
    assert nf13_witness_loss(*b_s, None, None, *c_t).item() == pytest.approx(0.0)


def test_nf13_witness_meet_containment_and_vacuity():
    b_s = box(0, 4)
    a_between = stack([box(2, 6)])           # meet with B = [2, 4]
    c_contains = box(1, 5)
    c_misses = box(0, 3)
    assert nf13_witness_loss(*b_s, *a_between, *c_contains).item() == pytest.approx(0.0)
    assert nf13_witness_loss(*b_s, *a_between, *c_misses).item() > 0
    # empty meet => vacuously satisfied regardless of C
    a_disjoint = stack([box(10, 11)])
    assert nf13_witness_loss(*b_s, *a_disjoint, *c_misses).item() == pytest.approx(0.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
