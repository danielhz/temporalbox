"""
Option A model: one box per concept per lasso position, rigid role translations.

The trainer and the induced-interpretation checker depend only on the
`BoxScorer` protocol, so any box-producing scorer (e.g. a time-augmented
Option B model for the RQ2 comparison) can be plugged in without touching the
harness.

Role semantics (translational, TransE-style): each role r has one vector δ_r
shared across all positions (roles are rigid); r is interpreted as
{(x, x + δ_r)}.  Hence `A ⊑ ∃r.B` requires Box(A) + δ_r ⊆ Box(B), and
`∃r.A ⊑ B` requires Box(A) − δ_r ⊆ Box(B) (the SAME δ_r, negated), which is
what makes role-mediated subsumption chains compose geometrically.

⊤ is a fixed, non-trainable box large enough to contain every trainable box,
so GCIs involving ⊤ need no special-casing in the compiler.

INDEPENDENCE NOTE: this is the construction-side code path. The lasso
arithmetic below is deliberately separate from oracle.py (the ground-truth
path re-implements the same spec); do not unify the two.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

TOP_EXTENT = 1e3  # half-width of the fixed ⊤ box; trainable boxes stay far inside


# ── construction-side lasso arithmetic ───────────────────────────────────────

def successor(t: int, k: int, loop: int) -> int:
    """σ(t) = t+1 if t < k, else ℓ."""
    return t + 1 if t < k else loop


def future(t: int, k: int, loop: int) -> list[int]:
    """Positions visited at or after t: {t..k} ∪ {ℓ..k}, sorted."""
    return sorted(set(range(t, k + 1)) | set(range(loop, k + 1)))


def between(t: int, s: int, k: int, loop: int) -> list[int]:
    """V(t,s): positions visited strictly before reaching witness s from t."""
    if s >= t:
        return list(range(t, s))
    return sorted(set(range(t, k + 1)) | set(range(loop, s)))


# ── scorer protocol (method-agnostic harness boundary) ───────────────────────

@runtime_checkable
class BoxScorer(Protocol):
    """Anything that yields a box per (concept, lasso position) and a vector
    per role. Implemented by TemporalBoxModel (Option A); an Option B scorer
    implements the same surface for RQ2."""

    n_concepts: int
    n_positions: int

    def concept_box(self, concept_ids: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """(centers, offsets), each of shape concept_ids.shape + (d,)."""
        ...

    def role_delta(self, role_ids: Tensor) -> Tensor:
        """Translation vectors, shape role_ids.shape + (d,)."""
        ...


# ── Option A model ───────────────────────────────────────────────────────────

class TemporalBoxModel(nn.Module):
    """Per-time-point boxes: centers/offsets of shape [n_concepts, k+1, d]."""

    def __init__(
        self,
        n_concepts: int,
        n_roles: int,
        k: int,
        loop: int,
        dim: int,
        top_id: int = 0,
        n_individuals: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.n_concepts = n_concepts
        self.n_positions = k + 1
        self.k = k
        self.loop = loop
        self.dim = dim
        self.top_id = top_id
        self.n_individuals = n_individuals

        self.centers = nn.Parameter(torch.rand(n_concepts, k + 1, dim))
        # softplus(-1.5) ≈ 0.20: boxes start as modest cubes
        self.offsets_raw = nn.Parameter(torch.full((n_concepts, k + 1, dim), -1.5))
        self.role_emb = nn.Parameter(torch.randn(max(n_roles, 1), dim) * 0.1)
        # Constant domain: one rigid point per individual, shared across ticks.
        # Non-rigid membership comes from the per-tick boxes moving, not the
        # points — this is the coupling that makes the per-time-point boxes do
        # real work on the slice.
        if n_individuals > 0:
            self.points = nn.Parameter(torch.rand(n_individuals, dim))

    def concept_box(self, concept_ids: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        c = self.centers[concept_ids, positions]
        o = F.softplus(self.offsets_raw[concept_ids, positions])
        is_top = (concept_ids == self.top_id).unsqueeze(-1)
        c = torch.where(is_top, torch.zeros_like(c), c)
        o = torch.where(is_top, torch.full_like(o, TOP_EXTENT), o)
        return c, o

    def role_delta(self, role_ids: Tensor) -> Tensor:
        return self.role_emb[role_ids]

    def individual_point(self, individual_ids: Tensor) -> Tensor:
        return self.points[individual_ids]


class TimeAugmentedBoxModel(nn.Module):
    """Option B comparator (RQ2): one box per concept augmented with a single
    learned time interval [τ−ω, τ+ω] (the "extra time axis").

    Same BoxScorer surface as Option A, so it drops into the harness. The box
    "at position t" is the spatial box whose offset is smoothly gated to ~0 as
    t leaves the concept's time interval:

        offset(t) = spatial_offset · σ(β·(ω − |t − τ|))

    This is the load-bearing limitation B has and A does not: a concept's
    temporal support is forced CONVEX (one interval), so a concept that is
    active at non-contiguous ticks (multi-interval / Grover-Cleveland support)
    cannot be represented — which is what makes recovery decline as the
    non-rigid, multi-interval fraction grows. Parameter count is independent of
    the number of ticks (one box + (τ, ω) per concept), unlike A.
    """

    def __init__(
        self,
        n_concepts: int,
        n_roles: int,
        k: int,
        loop: int,
        dim: int,
        top_id: int = 0,
        n_individuals: int = 0,
        gate_beta: float = 4.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.n_concepts = n_concepts
        self.n_positions = k + 1
        self.k = k
        self.loop = loop
        self.dim = dim
        self.top_id = top_id
        self.n_individuals = n_individuals
        self.gate_beta = gate_beta

        self.centers = nn.Parameter(torch.rand(n_concepts, dim))
        self.offsets_raw = nn.Parameter(torch.full((n_concepts, dim), -1.5))
        # time interval per concept: center τ ≈ k/2, half-width ω initialised
        # WIDE (covers the whole lasso) so B starts as static BoxEL — gates open.
        # It then only needs to NARROW for non-rigid support; convex support
        # (rigid / single-interval) stays cheap, which is what makes B tie A at
        # ρ=1 (guardrail) while still failing to carve non-convex support.
        self.time_center = nn.Parameter(torch.full((n_concepts,), k / 2.0))
        self.time_offset_raw = nn.Parameter(torch.full((n_concepts,), float(k)))
        self.role_emb = nn.Parameter(torch.randn(max(n_roles, 1), dim) * 0.1)
        if n_individuals > 0:
            self.points = nn.Parameter(torch.rand(n_individuals, dim))

    def concept_box(self, concept_ids: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        c = self.centers[concept_ids]
        o = F.softplus(self.offsets_raw[concept_ids])
        tau = self.time_center[concept_ids]
        omega = F.softplus(self.time_offset_raw[concept_ids])
        gate = torch.sigmoid(self.gate_beta * (omega - (positions.float() - tau).abs()))
        o = o * gate.unsqueeze(-1)
        is_top = (concept_ids == self.top_id).unsqueeze(-1)
        c = torch.where(is_top, torch.zeros_like(c), c)
        o = torch.where(is_top, torch.full_like(o, TOP_EXTENT), o)
        return c, o

    def role_delta(self, role_ids: Tensor) -> Tensor:
        return self.role_emb[role_ids]

    def individual_point(self, individual_ids: Tensor) -> Tensor:
        return self.points[individual_ids]
