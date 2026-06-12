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

        self.centers = nn.Parameter(torch.rand(n_concepts, k + 1, dim))
        # softplus(-1.5) ≈ 0.20: boxes start as modest cubes
        self.offsets_raw = nn.Parameter(torch.full((n_concepts, k + 1, dim), -1.5))
        self.role_emb = nn.Parameter(torch.randn(max(n_roles, 1), dim) * 0.1)

    def concept_box(self, concept_ids: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        c = self.centers[concept_ids, positions]
        o = F.softplus(self.offsets_raw[concept_ids, positions])
        is_top = (concept_ids == self.top_id).unsqueeze(-1)
        c = torch.where(is_top, torch.zeros_like(c), c)
        o = torch.where(is_top, torch.full_like(o, TOP_EXTENT), o)
        return c, o

    def role_delta(self, role_ids: Tensor) -> Tensor:
        return self.role_emb[role_ids]
