"""
TBox → tensor compilation and the Option A training loop.

`compile_tbox` enumerates every lasso position (and, for □/◇/U, every
witness) once, producing flat index tensors per normal form. The loss is then
a fixed differentiable expression over the scorer's boxes — full batch, no
sampling — and reports a per-NF-type breakdown every epoch (required by the
RQ1 protocol: recovery numbers are only interpretable against the achieved
loss, total and per type).

Temporal constraints (Option A, per-time-point boxes; see model.py for the
role semantics):

  NF1  A@t ⊑ B@t                             containment, every t
  NF2  A1@t ⊓ A2@t ⊑ B@t                     meet containment, every t
  NF3  Box(A)^t + δ_r ⊆ Box(B)^t             every t
  NF4  Box(A)^t − δ_r ⊆ Box(B)^t             every t
  NF5  δ_r + δ_s = δ_t                       once (roles rigid)
  NF6  Box(A)^t ⊆ Box(B)^σ(t)                every t
  NF7  Box(A)^t ⊆ Box(B)^s                   every t, s ∈ future(t)
  NF10 Box(A)^σ(t) ⊆ Box(B)^t                every t
  NF12 Box(A)^s ⊆ Box(B)^t                   every t, s ∈ future(t)
  NF13 Box(B)^s ⊓ ⨅_{p∈V(t,s)} Box(A)^p ⊆ Box(C)^t   every t, s ∈ future(t)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .logging_utils import ExperimentLogger
from .loss import (
    box2el_nf3_loss, box2el_nf4_loss, box2el_nf5_loss,
    box_contains, box_meet, box_meet_many, nf5_loss,
)
from .model import BoxScorer, TemporalBoxModel, between, future, successor
from .ontology import (
    NF1, NF2, NF3, NF4, NF5, NF6, NF7, NF10, NF12, NF13,
    NormalFormTBox, Vocabulary,
)


@dataclass
class TrainConfig:
    dim: int = 16
    k: int = 3
    loop: int = 1
    lr: float = 2e-2
    lr_min: float = 1e-4          # cosine-annealed floor
    epochs: int = 8000
    seed: int = 0
    device: str = "cpu"
    log_every: int = 200


@dataclass
class CompiledTBox:
    """Flat index tensors per normal form (see module docstring)."""

    # (a, b, t_a, t_b) generic cross-position containment pairs for
    # NF1/NF6/NF7/NF10/NF12 — kept separate per NF for the loss breakdown.
    containment: dict[str, Tensor] = field(default_factory=dict)
    nf2: Tensor | None = None            # (a1, a2, b, t)
    nf3: Tensor | None = None            # (a, r, b, t)
    nf4: Tensor | None = None            # (a, r, b, t)  — δ negated in the loss
    nf5: Tensor | None = None            # (r1, r2, rt)
    # NF13 witnesses grouped by |V|: v_len -> (b, c, t, s, v_1..v_len) per row;
    # the until-left concept a is column-broadcast over V positions.
    nf13: dict[int, Tensor] = field(default_factory=dict)
    nf13_a: dict[int, Tensor] = field(default_factory=dict)   # v_len -> (a,) ids


def compile_tbox(
    tbox: NormalFormTBox, vocab: Vocabulary, k: int, loop: int, device: str = "cpu"
) -> CompiledTBox:
    cid = vocab.concept_to_id
    rid = vocab.role_to_id
    positions = range(k + 1)

    pairs: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    nf2_rows: list[tuple[int, int, int, int]] = []
    nf3_rows: list[tuple[int, int, int, int]] = []
    nf4_rows: list[tuple[int, int, int, int]] = []
    nf5_rows: list[tuple[int, int, int]] = []
    nf13_rows: dict[int, list[list[int]]] = defaultdict(list)
    nf13_a_rows: dict[int, list[int]] = defaultdict(list)

    for g in tbox.nf_gcis:
        if isinstance(g, NF1):
            for t in positions:
                pairs["nf1"].append((cid[g.lhs], cid[g.rhs], t, t))
        elif isinstance(g, NF2):
            for t in positions:
                nf2_rows.append((cid[g.lhs1], cid[g.lhs2], cid[g.rhs], t))
        elif isinstance(g, NF3):
            for t in positions:
                nf3_rows.append((cid[g.lhs], rid[g.role], cid[g.rhs], t))
        elif isinstance(g, NF4):
            for t in positions:
                nf4_rows.append((cid[g.lhs], rid[g.role], cid[g.rhs], t))
        elif isinstance(g, NF5):
            nf5_rows.append((rid[g.role1], rid[g.role2], rid[g.rhs]))
        elif isinstance(g, NF6):
            for t in positions:
                pairs["nf6"].append((cid[g.lhs], cid[g.rhs], t, successor(t, k, loop)))
        elif isinstance(g, NF7):
            for t in positions:
                for s in future(t, k, loop):
                    pairs["nf7"].append((cid[g.lhs], cid[g.rhs], t, s))
        elif isinstance(g, NF10):
            for t in positions:
                pairs["nf10"].append((cid[g.lhs], cid[g.rhs], successor(t, k, loop), t))
        elif isinstance(g, NF12):
            for t in positions:
                for s in future(t, k, loop):
                    pairs["nf12"].append((cid[g.lhs], cid[g.rhs], s, t))
        elif isinstance(g, NF13):
            for t in positions:
                for s in future(t, k, loop):
                    v = between(t, s, k, loop)
                    nf13_rows[len(v)].append(
                        [cid[g.until_right], cid[g.rhs], t, s, *v]
                    )
                    nf13_a_rows[len(v)].append(cid[g.until_left])
        else:
            raise TypeError(f"Unhandled normal form in compile: {type(g)}")

    def lt(rows):  # LongTensor on device
        return torch.tensor(rows, dtype=torch.long, device=device)

    comp = CompiledTBox()
    for key, rows in pairs.items():
        comp.containment[key] = lt(rows)
    if nf2_rows:
        comp.nf2 = lt(nf2_rows)
    if nf3_rows:
        comp.nf3 = lt(nf3_rows)
    if nf4_rows:
        comp.nf4 = lt(nf4_rows)
    if nf5_rows:
        comp.nf5 = lt(nf5_rows)
    for v_len, rows in nf13_rows.items():
        comp.nf13[v_len] = lt(rows)
        comp.nf13_a[v_len] = lt(nf13_a_rows[v_len])
    return comp


def loss_components(scorer: BoxScorer, comp: CompiledTBox) -> dict[str, Tensor]:
    """Per-NF mean losses (zero tensors for absent NF types)."""
    zero = torch.zeros((), device=next(iter(comp.containment.values())).device
                       if comp.containment else None)
    out: dict[str, Tensor] = {}

    for key, idx in comp.containment.items():
        ca, oa = scorer.concept_box(idx[:, 0], idx[:, 2])
        cb, ob = scorer.concept_box(idx[:, 1], idx[:, 3])
        out[key] = box_contains(ca, oa, cb, ob).mean()

    if comp.nf2 is not None:
        idx = comp.nf2
        c1, o1 = scorer.concept_box(idx[:, 0], idx[:, 3])
        c2, o2 = scorer.concept_box(idx[:, 1], idx[:, 3])
        cb, ob = scorer.concept_box(idx[:, 2], idx[:, 3])
        cm, om, nonempty = box_meet(c1, o1, c2, o2)
        om = om.clamp_min(0.0)
        out["nf2"] = (box_contains(cm, om, cb, ob) * nonempty.float()).mean()

    # Existentials: Box²EL dual-box + bump path when the model exposes bumps
    # (the faithful base); otherwise the translational path. Role-free
    # experiments (slice, RQ2) never reach here.
    box2el = hasattr(scorer, "bump") and hasattr(scorer, "head_box")
    if comp.nf3 is not None:
        idx = comp.nf3
        cC, oC = scorer.concept_box(idx[:, 0], idx[:, 3])
        cD, oD = scorer.concept_box(idx[:, 2], idx[:, 3])
        if box2el:
            hc, ho = scorer.head_box(idx[:, 1])
            tc, to = scorer.tail_box(idx[:, 1])
            out["nf3"] = box2el_nf3_loss(cC, oC, cD, oD, scorer.bump(idx[:, 0]),
                                         scorer.bump(idx[:, 2]), hc, ho, tc, to).mean()
        else:
            out["nf3"] = box_contains(cC + scorer.role_delta(idx[:, 1]), oC, cD, oD).mean()
    if comp.nf4 is not None:
        idx = comp.nf4   # columns: (filler X, role r, rhs Y, tick)
        if box2el:
            hc, ho = scorer.head_box(idx[:, 1])
            cY, oY = scorer.concept_box(idx[:, 2], idx[:, 3])
            out["nf4"] = box2el_nf4_loss(hc, ho, scorer.bump(idx[:, 0]), cY, oY).mean()
        else:
            cC, oC = scorer.concept_box(idx[:, 0], idx[:, 3])
            cD, oD = scorer.concept_box(idx[:, 2], idx[:, 3])
            out["nf4"] = box_contains(cC - scorer.role_delta(idx[:, 1]), oC, cD, oD).mean()

    if comp.nf5 is not None:
        idx = comp.nf5   # columns: (role r, role s, role t)
        if box2el:
            hr, hro = scorer.head_box(idx[:, 0])
            ts, tso = scorer.tail_box(idx[:, 1])
            ht, hto = scorer.head_box(idx[:, 2])
            tt, tto = scorer.tail_box(idx[:, 2])
            out["nf5"] = box2el_nf5_loss(hr, hro, ts, tso, ht, hto, tt, tto).mean()
        else:
            out["nf5"] = nf5_loss(
                scorer.role_delta(idx[:, 0]),
                scorer.role_delta(idx[:, 1]),
                scorer.role_delta(idx[:, 2]),
            ).mean()

    nf13_terms = []
    for v_len, idx in comp.nf13.items():
        a_ids = comp.nf13_a[v_len]
        cb, ob = scorer.concept_box(idx[:, 0], idx[:, 3])      # B at witness s
        cc, oc = scorer.concept_box(idx[:, 1], idx[:, 2])      # C at t
        if v_len == 0:
            nf13_terms.append(box_contains(cb, ob, cc, oc))
            continue
        v_pos = idx[:, 4:]                                     # (m, v_len)
        a_rep = a_ids.unsqueeze(1).expand(-1, v_len)
        cav, oav = scorer.concept_box(a_rep, v_pos)            # (m, v_len, d)
        all_c = torch.cat([cb.unsqueeze(1), cav], dim=1)
        all_o = torch.cat([ob.unsqueeze(1), oav], dim=1)
        cm, om, nonempty = box_meet_many(all_c, all_o)
        om = om.clamp_min(0.0)
        nf13_terms.append(box_contains(cm, om, cc, oc) * nonempty.float())
    if nf13_terms:
        out["nf13"] = torch.cat(nf13_terms).mean()

    if not out:
        out["nf1"] = zero
    return out


def train(
    tbox: NormalFormTBox,
    vocab: Vocabulary,
    cfg: TrainConfig,
    logger: ExperimentLogger | None = None,
    model_cls=TemporalBoxModel,
) -> tuple[torch.nn.Module, dict[str, float]]:
    """Train an Option A model to satisfy the TBox; returns (model, final losses).
    model_cls selects the base method (TemporalBoxModel = translational roles;
    Box2ELModel = the faithful Box²EL base)."""
    torch.manual_seed(cfg.seed)
    comp = compile_tbox(tbox, vocab, cfg.k, cfg.loop, cfg.device)
    model = model_cls(
        n_concepts=vocab.n_concepts,
        n_roles=max(vocab.n_roles, 1),
        k=cfg.k,
        loop=cfg.loop,
        dim=cfg.dim,
        top_id=vocab.top_id,
        seed=cfg.seed,
    ).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr_min
    )

    final: dict[str, float] = {}
    for epoch in range(cfg.epochs):
        opt.zero_grad()
        comps = loss_components(model, comp)
        total = sum(comps.values())
        total.backward()
        opt.step()
        sched.step()
        if logger is not None and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            record = {f"loss_{k}": float(v.detach()) for k, v in comps.items()}
            record["loss_total"] = float(total.detach())
            logger.log(epoch, record)
    with torch.no_grad():
        comps = loss_components(model, comp)
        final = {f"loss_{k}": float(v) for k, v in comps.items()}
        final["loss_total"] = float(sum(comps.values()))
    return model, final
