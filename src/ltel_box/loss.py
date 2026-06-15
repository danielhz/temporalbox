"""
BoxEL-style loss terms for LTEL box embeddings (Option A).

Static loss terms (NF1–NF5) match the BoxEL construction (Xiong et al.,
ISWC 2022).  Temporal loss terms implement Option A — one box per concept per
lasso position — for the polarity-restricted normal forms only (right of ⊑:
○, □; left of ⊑: ○, U, ◇).  The trainer enumerates lasso positions/witnesses
and passes already-indexed tensors.

All functions operate on raw tensors rather than a model object, so any
box-producing scorer can be plugged in (method-agnostic, for the RQ2
comparison).

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


def point_in_box_loss(
    point: Tensor,
    center: Tensor,
    offset: Tensor,
) -> Tensor:
    """Penalty for `point` lying outside the box (0 when inside).

    Used for ABox membership: an individual's point must lie in the box of
    each concept it belongs to at the relevant time point. Shape (...,).
    """
    return F.relu((point - center).abs() - offset).sum(dim=-1)


def point_outside_box_loss(
    point: Tensor,
    center: Tensor,
    offset: Tensor,
    margin: float = 0.0,
) -> Tensor:
    """Penalty for `point` lying inside the box (0 when outside by `margin`).

    Closed-world ABox term: at a time point where an individual is NOT a member
    of a concept, its point must lie OUTSIDE that concept's box. A point is
    outside iff some coordinate exceeds the box; we penalise the least-violated
    coordinate. This is what forces a concept's temporal support to be
    faithful — and what Option B's single convex time interval cannot honour
    when the true support is non-convex (it must keep the box open across the
    gap, trapping the points it should exclude). Shape (...,).
    """
    return F.relu((offset - (point - center).abs()).min(dim=-1).values + margin)


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


# ── Temporal loss terms (Option A: per-time-point boxes over a lasso) ────────
#
# Each concept A has one box Box(A)^(t) per lasso position t ∈ {0..k}. The
# *trainer* enumerates positions and witnesses (σ(t), future(t), V(t,s)) and
# calls these functions with the already-indexed tensors, so the losses stay
# method-agnostic (any box-producing scorer can be plugged in for RQ2).
#
# Only the polarity-restricted forms exist (right: ○, □; left: ○, U, ◇);
# right-◇/right-U/left-□ have no normal form and hence no loss.

def box_meet_many(centers: Tensor, offsets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Iterated coordinate-wise intersection of n boxes.

    centers/offsets: (..., n, d) — n boxes per batch element.
    Returns (center_meet, offset_meet, nonempty) with shapes
    (..., d), (..., d), (...,).  offset_meet may have negative entries when
    the meet is empty; nonempty reports emptiness, and callers must mask with
    it (as in nf2_loss).
    """
    lb = (centers - offsets).max(dim=-2).values
    ub = (centers + offsets).min(dim=-2).values
    center_meet = (lb + ub) / 2
    offset_meet = (ub - lb) / 2
    nonempty = (offset_meet >= 0).all(dim=-1)
    return center_meet, offset_meet, nonempty


# ── BoxEL role/existential losses (affine-transform base) ───────────────────
# Each role r is an affine map f_r(x) = scale_r ⊙ x + trans_r. A box transforms
# as center -> scale⊙center + trans, offset -> scale⊙offset.

def boxel_nf3_loss(
    cC: Tensor, oC: Tensor, scale: Tensor, trans: Tensor, cD: Tensor, oD: Tensor,
) -> Tensor:
    """C ⊑ ∃r.D — T_r(Box(C)) ⊆ Box(D), T_r(x)=scale⊙x+trans (diagonal affine).

    Exact Xiong-2022 BoxEL form, @prover-confirmed (geometry is the BoxEL paper's,
    not machine-checked)."""
    return box_contains(scale * cC + trans, scale * oC, cD, oD)


def boxel_nf4_loss(
    cC: Tensor, oC: Tensor, scale: Tensor, trans: Tensor, cD: Tensor, oD: Tensor,
) -> Tensor:
    """∃r.C ⊑ D — T_r⁻¹(Box(C)) ⊆ Box(D): the r-predecessors of C are
    T_r⁻¹(Box(C)) (center=(cC−trans)/scale, offset=oC/scale), forced into Box(D).

    Exact Xiong-2022 BoxEL form, @prover-confirmed. Needs scale>0 for the inverse."""
    return box_contains((cC - trans) / scale, oC / scale, cD, oD)


# ── Box²EL role/existential losses (faithful base) ──────────────────────────
# Dual-box + bump construction (Jackermeier et al., WWW 2024); the concept
# boxes are the per-time boxes, the bumps and Head/Tail boxes are rigid.

def box2el_nf3_loss(
    cC: Tensor, oC: Tensor, cD: Tensor, oD: Tensor,
    bump_C: Tensor, bump_D: Tensor,
    head_c: Tensor, head_o: Tensor, tail_c: Tensor, tail_o: Tensor,
) -> Tensor:
    """C ⊑ ∃r.D — Box(C)⊕Bump(D) ⊆ Head(r)  and  Box(D)⊕Bump(C) ⊆ Tail(r)."""
    return (box_contains(cC + bump_D, oC, head_c, head_o)
            + box_contains(cD + bump_C, oD, tail_c, tail_o))


def box2el_nf4_loss(
    head_c: Tensor, head_o: Tensor,
    bump_X: Tensor,
    cY: Tensor, oY: Tensor,
) -> Tensor:
    """∃r.X ⊑ Y — Head(r) ⊖ Bump(X) ⊆ Box(Y).

    Sound by the Box²EL role semantics: NF3 fills Head(r) with source boxes
    bumped by the target's bump (`Box(·)⊕Bump(X)⊆Head(r)`), so any a ∈ ∃r.X
    satisfies a⊕Bump(X) ∈ Head(r), i.e. a ∈ Head(r)⊖Bump(X). Enforcing
    Head(r)⊖Bump(X) ⊆ Box(Y) therefore forces ∃r.X ⊆ Y. This is a *containment*
    constraint (unlike ELBE's unsound NF4 overlap)."""
    return box_contains(head_c - bump_X, head_o, cY, oY)


def box2el_nf5_loss(
    head_r_c: Tensor, head_r_o: Tensor,
    tail_s_c: Tensor, tail_s_o: Tensor,
    head_t_c: Tensor, head_t_o: Tensor,
    tail_t_c: Tensor, tail_t_o: Tensor,
) -> Tensor:
    """r ∘ s ⊑ t — Head(r) ⊆ Head(t) and Tail(s) ⊆ Tail(t).

    Exact Box²EL form (their NF7 loss), confirmed sound by @prover
    (2026-06-15, after correcting an earlier mistaken "drop it" note): the
    composed chain inherits r's source side (Head) and s's target side (Tail),
    so `Head(r)⊆Head(t) ∧ Tail(s)⊆Tail(t)` realises `r∘s⊑t` (Box²EL Thm 2).
    Box embeddings DO support role composition. (Role *hierarchy* r⊑s is the
    related NF6: `Head(r)⊆Head(s) ∧ Tail(r)⊆Tail(s)`.) Geometry is the Box²EL
    paper's, not machine-checked in the Lean. See docs/base-method.md."""
    return (box_contains(head_r_c, head_r_o, head_t_c, head_t_o)
            + box_contains(tail_s_c, tail_s_o, tail_t_c, tail_t_o))


def nf6_loss(
    center_A_t: Tensor, offset_A_t: Tensor,
    center_B_next: Tensor, offset_B_next: Tensor,
) -> Tensor:
    """A ⊑ ○B at position t — Box(A)^(t) ⊆ Box(B)^(σ(t)).

    The caller passes B's box at the *successor* position σ(t) (which wraps
    to the loop index ℓ at t = k).
    """
    return box_contains(center_A_t, offset_A_t, center_B_next, offset_B_next)


def nf7_loss(
    center_A_t: Tensor, offset_A_t: Tensor,
    centers_B_future: Tensor, offsets_B_future: Tensor,
) -> Tensor:
    """A ⊑ □B at position t — Box(A)^(t) ⊆ Box(B)^(s) for every s ∈ future(t).

    centers_B_future/offsets_B_future: (..., m, d), B's boxes stacked over the
    m = |future(t)| positions.  Sum of containment violations over s.
    """
    c_A = center_A_t.unsqueeze(-2)
    o_A = offset_A_t.unsqueeze(-2)
    return box_contains(c_A, o_A, centers_B_future, offsets_B_future).sum(dim=-1)


def nf10_loss(
    center_A_next: Tensor, offset_A_next: Tensor,
    center_B_t: Tensor, offset_B_t: Tensor,
) -> Tensor:
    """○A ⊑ B at position t — Box(A)^(σ(t)) ⊆ Box(B)^(t)."""
    return box_contains(center_A_next, offset_A_next, center_B_t, offset_B_t)


def nf12_loss(
    centers_A_future: Tensor, offsets_A_future: Tensor,
    center_B_t: Tensor, offset_B_t: Tensor,
) -> Tensor:
    """◇A ⊑ B at position t — Box(A)^(s) ⊆ Box(B)^(t) for every s ∈ future(t).

    centers_A_future/offsets_A_future: (..., m, d).  Sum over s: an element
    eventually in A (at any visited s) must be in B now.
    """
    c_B = center_B_t.unsqueeze(-2)
    o_B = offset_B_t.unsqueeze(-2)
    return box_contains(centers_A_future, offsets_A_future, c_B, o_B).sum(dim=-1)


def nf13_witness_loss(
    center_B_s: Tensor, offset_B_s: Tensor,
    centers_A_between: Tensor | None, offsets_A_between: Tensor | None,
    center_C_t: Tensor, offset_C_t: Tensor,
) -> Tensor:
    """(A U B) ⊑ C at position t, for ONE witness position s ∈ future(t):

        Box(B)^(s) ⊓ ⨅_{p ∈ V(t,s)} Box(A)^(p)  ⊆  Box(C)^(t)

    centers_A_between/offsets_A_between: (..., |V|, d) — A's boxes over the
    between-set V(t,s); pass None when V is empty (s = t), reducing the
    constraint to Box(B)^(t) ⊆ Box(C)^(t).  Vacuously satisfied (zero loss)
    when the meet is empty.  The trainer sums this over all witnesses s.
    """
    if centers_A_between is None:
        return box_contains(center_B_s, offset_B_s, center_C_t, offset_C_t)
    all_c = torch.cat([center_B_s.unsqueeze(-2), centers_A_between], dim=-2)
    all_o = torch.cat([offset_B_s.unsqueeze(-2), offsets_A_between], dim=-2)
    cm, om, nonempty = box_meet_many(all_c, all_o)
    # Clamp the (possibly negative) empty-meet offsets so box_contains gets a
    # well-formed box; the nonempty mask zeroes those terms anyway.
    om = om.clamp_min(0.0)
    return box_contains(cm, om, center_C_t, offset_C_t) * nonempty.float()
