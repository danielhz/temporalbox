"""
BoxEL-style loss terms for LTEL box embeddings.

Static loss terms (NF1–NF5) are complete and match the BoxEL construction
(Xiong et al., ISWC 2022).  Temporal loss terms (NF6–NF13) have the correct
signatures but raise NotImplementedError until the architecture decisions
(single-space-per-time-point vs. time-augmented box) are finalized.

All functions operate on raw tensors rather than a model object so they can
be composed freely during architecture exploration.

Box representation:
    center: (..., d)   real-valued
    offset: (..., d)   strictly positive (caller ensures this, e.g. via softplus)

    lower bound = center - offset
    upper bound = center + offset
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


# ── Geometry helpers ─────────────────────────────────────────────────────────

def box_contains(
    center_inner: Tensor,
    offset_inner: Tensor,
    center_outer: Tensor,
    offset_outer: Tensor,
) -> Tensor:
    """
    Differentiable penalty for inner ⊄ outer (should be 0 when inner ⊆ outer).

    Returns shape (...,) — one scalar loss per batch element.
    """
    lb_inner = center_inner - offset_inner
    ub_inner = center_inner + offset_inner
    lb_outer = center_outer - offset_outer
    ub_outer = center_outer + offset_outer
    # Violation: inner's lower bound below outer's, or inner's upper bound above outer's.
    loss = F.relu(lb_outer - lb_inner) + F.relu(ub_inner - ub_outer)
    return loss.sum(dim=-1)


def box_meet(
    center_a: Tensor,
    offset_a: Tensor,
    center_b: Tensor,
    offset_b: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Coordinate-wise intersection of two boxes.

    Returns (center_meet, offset_meet, nonempty) where nonempty is a boolean
    tensor of shape (...,) indicating whether each meet is non-empty.
    """
    lb_a = center_a - offset_a
    ub_a = center_a + offset_a
    lb_b = center_b - offset_b
    ub_b = center_b + offset_b
    lb_meet = torch.maximum(lb_a, lb_b)
    ub_meet = torch.minimum(ub_a, ub_b)
    center_meet = (lb_meet + ub_meet) / 2
    offset_meet = (ub_meet - lb_meet) / 2
    nonempty = (offset_meet >= 0).all(dim=-1)
    return center_meet, offset_meet, nonempty


# ── Static loss terms (NF1–NF5) ──────────────────────────────────────────────

def nf1_loss(
    center_A: Tensor, offset_A: Tensor,
    center_B: Tensor, offset_B: Tensor,
) -> Tensor:
    """A ⊑ B  — box_A must be contained in box_B."""
    return box_contains(center_A, offset_A, center_B, offset_B)


def nf2_loss(
    center_A1: Tensor, offset_A1: Tensor,
    center_A2: Tensor, offset_A2: Tensor,
    center_B: Tensor, offset_B: Tensor,
) -> Tensor:
    """A1 ⊓ A2 ⊑ B — meet(box_A1, box_A2) must be contained in box_B.

    Loss is 0 for a batch element when the meet is empty (vacuously satisfied).
    """
    cm, om, nonempty = box_meet(center_A1, offset_A1, center_A2, offset_A2)
    containment = box_contains(cm, om, center_B, offset_B)
    return containment * nonempty.float()


def nf3_loss(
    center_A: Tensor, offset_A: Tensor,
    delta_r_head: Tensor,
    center_B: Tensor, offset_B: Tensor,
) -> Tensor:
    """A ⊑ ∃r.B — translation of box_A by role r's head vector must lie in box_B.

    The existential witness is box_A shifted by delta_r_head.  This is the
    simplest faithful encoding; more expressive role representations (e.g. full
    affine maps) can replace delta_r_head later without changing this signature.
    """
    return box_contains(center_A + delta_r_head, offset_A, center_B, offset_B)


def nf4_loss(
    delta_r_tail: Tensor,
    center_A: Tensor, offset_A: Tensor,
    center_B: Tensor, offset_B: Tensor,
) -> Tensor:
    """∃r.A ⊑ B — translation of box_A by role r's tail vector must lie in box_B."""
    return box_contains(center_A + delta_r_tail, offset_A, center_B, offset_B)


def nf5_loss(
    delta_r1: Tensor,
    delta_r2: Tensor,
    delta_t: Tensor,
) -> Tensor:
    """r ∘ s ⊑ t — the composed role displacement should equal t's displacement."""
    return (delta_r1 + delta_r2 - delta_t).pow(2).sum(dim=-1)


# ── Temporal loss terms (NF6–NF13) — stubs ───────────────────────────────────
# Signatures are fixed; implementations will be added once the architecture
# (single-space-per-time-point vs. time-augmented box) is decided.

def nf6_loss(*args, **kwargs) -> Tensor:
    """A ⊑ ○B — Next operator on rhs.  NOT YET IMPLEMENTED."""
    raise NotImplementedError("nf6_loss: architecture not yet decided")


def nf7_loss(*args, **kwargs) -> Tensor:
    """A ⊑ □B — Globally on rhs.  NOT YET IMPLEMENTED."""
    raise NotImplementedError("nf7_loss: architecture not yet decided")


def nf8_loss(*args, **kwargs) -> Tensor:
    """A ⊑ ◇B — Eventually on rhs.  NOT YET IMPLEMENTED."""
    raise NotImplementedError("nf8_loss: architecture not yet decided")


def nf9_loss(*args, **kwargs) -> Tensor:
    """A ⊑ B U C — Until on rhs.  NOT YET IMPLEMENTED."""
    raise NotImplementedError("nf9_loss: architecture not yet decided")


def nf10_loss(*args, **kwargs) -> Tensor:
    """○A ⊑ B — Next on lhs.  NOT YET IMPLEMENTED."""
    raise NotImplementedError("nf10_loss: architecture not yet decided")


def nf11_loss(*args, **kwargs) -> Tensor:
    """□A ⊑ B — Globally on lhs.  NOT YET IMPLEMENTED."""
    raise NotImplementedError("nf11_loss: architecture not yet decided")


def nf12_loss(*args, **kwargs) -> Tensor:
    """◇A ⊑ B — Eventually on lhs.  NOT YET IMPLEMENTED."""
    raise NotImplementedError("nf12_loss: architecture not yet decided")


def nf13_loss(*args, **kwargs) -> Tensor:
    """(A U C) ⊑ B — Until on lhs.  NOT YET IMPLEMENTED."""
    raise NotImplementedError("nf13_loss: architecture not yet decided")
