"""RQ2 generator + A-vs-B core property (requires torch)."""

import pytest

torch = pytest.importorskip("torch")

from ltel_box.model import TemporalBoxModel, TimeAugmentedBoxModel       # noqa: E402
from ltel_box.ontology import NF6, NF7, NF10, NF12, NF13                 # noqa: E402
from ltel_box.rq2 import RQ2Config, generate_rq2  # noqa: E402

_TEMPORAL = (NF6, NF7, NF10, NF12, NF13)


def test_confound_temporal_count_invariant_across_rho():
    # Non-degenerate temporal-GCI count must be constant across ρ<1 (the
    # confound rule), and zero at ρ=1 (no non-rigid names left).
    counts = {}
    for rho in (0.0, 0.25, 0.5, 0.75):
        ds = generate_rq2(RQ2Config(rigid_fraction=rho, n_temporal=12, seed=1))
        counts[rho] = sum(1 for g in ds.tbox.nf_gcis if isinstance(g, _TEMPORAL))
    assert len(set(counts.values())) == 1 and next(iter(counts.values())) == 12
    ds1 = generate_rq2(RQ2Config(rigid_fraction=1.0, n_temporal=12, seed=1))
    assert sum(1 for g in ds1.tbox.nf_gcis if isinstance(g, _TEMPORAL)) == 0


def test_negatives_are_hierarchy_consistent():
    # No negative (i, c, t) may coincide with a positive membership (i, c, t):
    # a genuine non-member is never simultaneously an (expanded) member.
    ds = generate_rq2(RQ2Config(rigid_fraction=0.0, support_max_intervals=3, seed=2))
    pos = set(ds.pos_triples)
    assert pos.isdisjoint(set(ds.neg_triples))


def test_support_levels_assigned():
    ds_multi = generate_rq2(RQ2Config(rigid_fraction=0.0, support_max_intervals=3, seed=0))
    assert set(ds_multi.support_level.values()) <= {1, 2, 3}
    assert any(v == 3 for v in ds_multi.support_level.values())  # multi-interval present


def test_isolated_nonconvex_support_A_succeeds_B_fails():
    # The load-bearing claim, isolated: on a non-convex temporal support, the
    # per-time-point model (A) reaches zero exclusion loss while the
    # time-augmented model (B) — one convex interval — plateaus above zero.
    from ltel_box.loss import point_in_box_loss, point_outside_box_loss

    k, n_ind, active = 7, 12, [1, 4, 7]
    inactive = [t for t in range(8) if t not in active]
    pos = torch.tensor([(i, 1, t) for i in range(n_ind) for t in active])
    neg = torch.tensor([(i, 1, t) for i in range(n_ind) for t in inactive])

    def fit(cls):
        torch.manual_seed(0)
        m = cls(n_concepts=2, n_roles=1, k=k, loop=k, dim=8, top_id=0, n_individuals=n_ind, seed=0)
        opt = torch.optim.Adam(m.parameters(), lr=5e-2)
        for _ in range(1500):
            opt.zero_grad()
            ci, oi = m.concept_box(pos[:, 1], pos[:, 2])
            cn, on = m.concept_box(neg[:, 1], neg[:, 2])
            loss = (point_in_box_loss(m.individual_point(pos[:, 0]), ci, oi).mean()
                    + point_outside_box_loss(m.individual_point(neg[:, 0]), cn, on, 0.1).mean())
            loss.backward()
            opt.step()
        with torch.no_grad():
            cn, on = m.concept_box(neg[:, 1], neg[:, 2])
            return float(point_outside_box_loss(m.individual_point(neg[:, 0]), cn, on, 0.1).mean())

    a_out = fit(TemporalBoxModel)
    b_out = fit(TimeAugmentedBoxModel)
    assert a_out < 1e-3            # A represents non-convex support exactly
    assert b_out > 10 * a_out + 1e-2  # B's exclusion loss plateaus well above A


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
