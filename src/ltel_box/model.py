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
    """Per-time-point boxes: centers/offsets of shape [n_concepts, k+1, d].

    Rigid-box collapse (optional): concepts in `rigid_ids` share ONE box across
    all ticks (their box is read at position 0 regardless of the requested
    tick), realising the time-invariance of rigid concepts. Their per-tick
    positions 1..k become tied (no free parameters), so `free_param_count()`
    drops toward static-BoxEL cost as the rigid fraction → 1 — the empirical
    echo of the tractability theorem. With `rigid_ids=None` (default) behaviour
    is unchanged (all concepts per-time)."""

    def __init__(
        self,
        n_concepts: int,
        n_roles: int,
        k: int,
        loop: int,
        dim: int,
        top_id: int = 0,
        n_individuals: int = 0,
        rigid_ids: set[int] | None = None,
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

        rigid = torch.zeros(n_concepts, dtype=torch.bool)
        if rigid_ids:
            rigid[list(rigid_ids)] = True
        self.register_buffer("rigid_mask", rigid)
        self.n_rigid = int(rigid.sum())

    def _eff_pos(self, concept_ids: Tensor, positions: Tensor) -> Tensor:
        # rigid concepts always read their position-0 box (time-invariant)
        return torch.where(self.rigid_mask[concept_ids], torch.zeros_like(positions), positions)

    def concept_box(self, concept_ids: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        pos = self._eff_pos(concept_ids, positions)
        c = self.centers[concept_ids, pos]
        o = F.softplus(self.offsets_raw[concept_ids, pos])
        is_top = (concept_ids == self.top_id).unsqueeze(-1)
        c = torch.where(is_top, torch.zeros_like(c), c)
        o = torch.where(is_top, torch.full_like(o, TOP_EXTENT), o)
        return c, o

    def free_param_count(self) -> int:
        """Free parameters: rigid concepts contribute 1 box (not k+1). Tied
        rigid positions and dead params are excluded; this is the model-size
        measure the tractability claim is about (a ragged store realises it
        physically)."""
        n_nonrigid = self.n_concepts - self.n_rigid
        box_params = (n_nonrigid * self.n_positions + self.n_rigid) * self.dim * 2
        roles = self.role_emb.numel()
        pts = self.points.numel() if hasattr(self, "points") else 0
        return int(box_params + roles + pts)

    def role_delta(self, role_ids: Tensor) -> Tensor:
        return self.role_emb[role_ids]

    def individual_point(self, individual_ids: Tensor) -> Tensor:
        return self.points[individual_ids]


class Box2ELModel(nn.Module):
    """Option A on the Box²EL base (Jackermeier et al., WWW 2024) — the faithful
    static method the paper's theorem and @prover's Lean use.

    Per-time-point concept boxes (non-rigid), plus a RIGID relational layer
    (Q5 requirement): each concept has a rigid bump vector, each role a rigid
    Head and Tail box. Existentials use the dual-box + bump scoring; the concept
    meet/containment lattice (NF1/NF2) is the shared axisBoxAlgebra and uses no
    bumps. See docs/base-method.md.
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
        self.offsets_raw = nn.Parameter(torch.full((n_concepts, k + 1, dim), -1.5))
        # rigid relational layer (shared across time)
        self.bumps = nn.Parameter(torch.randn(n_concepts, dim) * 0.1)
        self.head_center = nn.Parameter(torch.rand(max(n_roles, 1), dim))
        self.head_offset_raw = nn.Parameter(torch.full((max(n_roles, 1), dim), -1.0))
        self.tail_center = nn.Parameter(torch.rand(max(n_roles, 1), dim))
        self.tail_offset_raw = nn.Parameter(torch.full((max(n_roles, 1), dim), -1.0))
        if n_individuals > 0:
            self.points = nn.Parameter(torch.rand(n_individuals, dim))

    def concept_box(self, concept_ids: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        c = self.centers[concept_ids, positions]
        o = F.softplus(self.offsets_raw[concept_ids, positions])
        is_top = (concept_ids == self.top_id).unsqueeze(-1)
        c = torch.where(is_top, torch.zeros_like(c), c)
        o = torch.where(is_top, torch.full_like(o, TOP_EXTENT), o)
        return c, o

    def bump(self, concept_ids: Tensor) -> Tensor:
        return self.bumps[concept_ids]

    def head_box(self, role_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.head_center[role_ids], F.softplus(self.head_offset_raw[role_ids])

    def tail_box(self, role_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.tail_center[role_ids], F.softplus(self.tail_offset_raw[role_ids])

    # kept for harness compatibility; Box²EL existentials use the dual boxes,
    # so role_delta is unused by the Box²EL loss path.
    def role_delta(self, role_ids: Tensor) -> Tensor:
        return torch.zeros(role_ids.shape + (self.dim,), device=self.centers.device)

    def individual_point(self, individual_ids: Tensor) -> Tensor:
        return self.points[individual_ids]


class BoxELModel(nn.Module):
    """Option A on the BoxEL base (Xiong et al., ISWC 2022) — the static method
    the paper's faithfulness theorem cites and §5 instantiates ("BoxEL
    throughout").

    Per-time-point concept boxes (non-rigid), plus a RIGID relational layer:
    each role r is an affine transformation (per-dimension scaling s_r > 0 and
    translation t_r) — BoxEL's role model, as opposed to Box²EL's dual
    head/tail boxes + bumps. Existentials transform boxes:
        NF3  C⊑∃r.D :  f_r(Box(C)) ⊆ Box(D),         f_r(x)=s_r⊙x+t_r
        NF4  ∃r.C⊑D :  f_r⁻¹(Box(C)) ⊆ Box(D)
    The concept meet/containment lattice (NF1/NF2) is the shared axisBoxAlgebra.

    NOTE: the exact BoxEL NF3/NF4 affine forms are being confirmed with @prover
    against Xiong et al. 2022 (same process that corrected the Box²EL NF4
    Tail→Head). See docs/base-method.md.
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
        self.offsets_raw = nn.Parameter(torch.full((n_concepts, k + 1, dim), -1.5))
        # rigid affine role transforms: scaling (via softplus, >0) + translation
        self.role_scale_raw = nn.Parameter(torch.zeros(max(n_roles, 1), dim))  # softplus(0)≈0.69 → ~1 after +offset
        self.role_trans = nn.Parameter(torch.zeros(max(n_roles, 1), dim))
        if n_individuals > 0:
            self.points = nn.Parameter(torch.rand(n_individuals, dim))

    def concept_box(self, concept_ids: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        c = self.centers[concept_ids, positions]
        o = F.softplus(self.offsets_raw[concept_ids, positions])
        is_top = (concept_ids == self.top_id).unsqueeze(-1)
        c = torch.where(is_top, torch.zeros_like(c), c)
        o = torch.where(is_top, torch.full_like(o, TOP_EXTENT), o)
        return c, o

    def role_scale(self, role_ids: Tensor) -> Tensor:
        # keep scaling strictly positive and ~1 at init
        return F.softplus(self.role_scale_raw[role_ids]) + 0.5

    def role_translation(self, role_ids: Tensor) -> Tensor:
        return self.role_trans[role_ids]

    # harness compatibility (unused by the BoxEL affine loss path)
    def role_delta(self, role_ids: Tensor) -> Tensor:
        return torch.zeros(role_ids.shape + (self.dim,), device=self.centers.device)

    def individual_point(self, individual_ids: Tensor) -> Tensor:
        return self.points[individual_ids]


class TransBoxModel(nn.Module):
    """Option A on the TransBox base (Yang, Chen, Sattler, WWW 2025) — the third
    faithful base on @prover's list, alongside BoxEL and Box²EL.

    Roles are **translation boxes**, not single vectors: r(a,b) iff
    x_a − x_b ∈ Box(r) = (c_r, o_r). The existential is a Minkowski sum of boxes
    (∃r.D = Box(D) ⊕ Box(r) = box(c_D+c_r, o_D+o_r)), and composition is closed
    (Box(r∘s) = box(c_r+c_s, o_r+o_s)) — its many-to-many capability is exactly
    what a single TransE vector lacks. Per-time-point concept boxes; rigid role
    boxes. NF1/NF2 = the shared axisBoxAlgebra.

    The role losses are derived from TransBox's relation semantics; exact match
    to Yang 2025 is being confirmed with @prover.
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
        rigid_ids: set[int] | None = None,
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
        self.offsets_raw = nn.Parameter(torch.full((n_concepts, k + 1, dim), -1.5))
        # rigid translation box per role
        self.role_center = nn.Parameter(torch.randn(max(n_roles, 1), dim) * 0.1)
        self.role_offset_raw = nn.Parameter(torch.full((max(n_roles, 1), dim), -1.5))
        if n_individuals > 0:
            self.points = nn.Parameter(torch.rand(n_individuals, dim))
        rigid = torch.zeros(n_concepts, dtype=torch.bool)
        if rigid_ids:
            rigid[list(rigid_ids)] = True
        self.register_buffer("rigid_mask", rigid)
        self.n_rigid = int(rigid.sum())

    def concept_box(self, concept_ids: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        pos = torch.where(self.rigid_mask[concept_ids], torch.zeros_like(positions), positions)
        c = self.centers[concept_ids, pos]
        o = F.softplus(self.offsets_raw[concept_ids, pos])
        is_top = (concept_ids == self.top_id).unsqueeze(-1)
        c = torch.where(is_top, torch.zeros_like(c), c)
        o = torch.where(is_top, torch.full_like(o, TOP_EXTENT), o)
        return c, o

    def role_box(self, role_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.role_center[role_ids], F.softplus(self.role_offset_raw[role_ids])

    def role_delta(self, role_ids: Tensor) -> Tensor:  # harness compat (unused)
        return torch.zeros(role_ids.shape + (self.dim,), device=self.centers.device)

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
