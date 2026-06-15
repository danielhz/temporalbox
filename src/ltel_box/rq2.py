"""
RQ2 — does faithfulness pay off? A (per-time-point boxes) vs B (time-augmented,
single convex time interval), swept over the rigid fraction ρ.

Synthetic temporal interpretations (TBox + ABox), trained slice-style:

  * TBox: static backbone (NF1–NF4) over all concepts; temporal axioms
    (NF6/7/10/12/13) over NON-RIGID concepts only, with the non-degenerate
    temporal-GCI count held FIXED across ρ (the confound rule — otherwise
    raising ρ silently deletes temporal constraints and B improves for the
    wrong reason).
  * ABox: individuals with memberships whose temporal support encodes the
    three levels — rigid concept → full-time (convex) support (level 1);
    non-rigid single-interval → convex (level 2); non-rigid multi-interval →
    NON-convex (level 3), the case B provably cannot represent.

ρ = fraction of concepts that are rigid. Rigidity is realised in the DATA
(support shape), so A and B share architecture and the comparison is clean:
B should TIE A at ρ=1 (all convex) and FAIL as ρ→0 with multi-interval support.

Both models train through the identical pipeline; only model_cls differs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from .closure import compute_el_closure
from .loss import box_contains, point_in_box_loss, point_outside_box_loss
from .model import TemporalBoxModel
from .ontology import (
    TOP_NAME, NF1, NF2, NF3, NF4, NF6, NF7, NF10, NF12, NF13,
    NormalFormTBox, Vocabulary,
)
from .oracle import temporal_closure
from .trainer import compile_tbox, loss_components


@dataclass
class RQ2Config:
    n_concepts: int = 24
    n_roles: int = 2
    rigid_fraction: float = 0.0
    k: int = 7                       # lasso length; n_ticks = k+1
    loop: int = 3
    # static backbone (over ALL concepts)
    n_nf1: int = 16
    n_nf2: int = 4
    n_nf3: int = 4
    n_nf4: int = 4
    # temporal axioms (over NON-RIGID concepts only) — fixed count across ρ
    n_temporal: int = 12
    # ABox
    n_individuals: int = 60
    memberships_per_individual: int = 4
    support_max_intervals: int = 1   # 1 = convex (level 2); >=2 = multi-interval (level 3)
    neg_ratio: int = 2               # closed-world negatives per positive membership triple
    seed: int = 0


@dataclass
class RQ2Dataset:
    tbox: NormalFormTBox
    vocab: Vocabulary
    individuals: list[str]
    n_ticks: int
    k: int
    loop: int
    memberships: list[tuple[int, int, list[int]]]  # (individual_idx, concept_id, active_ticks) — direct
    pos_triples: list[tuple[int, int, int]]        # (ind, concept, tick) member — hierarchy-expanded
    neg_triples: list[tuple[int, int, int]]        # (ind, concept, tick) genuinely NOT a member
    rigid_concepts: frozenset[str]
    support_level: dict[str, int]                  # concept -> 1/2/3
    closure: frozenset[tuple[str, str, int]]
    stats: dict = field(default_factory=dict)


def _support_ticks(rng: random.Random, n_ticks: int, n_intervals: int) -> list[int]:
    """Active ticks forming `n_intervals` DISJOINT intervals (gap >= 1 between).

    n_intervals=1 → a single contiguous interval (convex, level 2).
    n_intervals>=2 → that many separated intervals (non-convex, level 3): the
    case Option B's single time interval cannot represent.
    """
    if n_intervals <= 1:
        length = rng.randint(1, max(1, n_ticks // 2))
        start = rng.randint(0, n_ticks - length)
        return list(range(start, start + length))
    # pick `n_intervals` start ticks pairwise >= 2 apart, each a singleton
    # (singletons separated by gaps are unambiguously non-convex)
    starts: list[int] = []
    for pos in rng.sample(range(n_ticks), n_ticks):
        if all(abs(pos - s) >= 2 for s in starts):
            starts.append(pos)
            if len(starts) >= n_intervals:
                break
    return sorted(starts)


def generate_rq2(cfg: RQ2Config | None = None) -> RQ2Dataset:
    cfg = cfg or RQ2Config()
    rng = random.Random(cfg.seed)
    concepts = [f"C{i}" for i in range(cfg.n_concepts)]
    roles = [f"r{i}" for i in range(cfg.n_roles)]

    n_rigid = round(cfg.rigid_fraction * cfg.n_concepts)
    rigid = set(concepts[:n_rigid])
    nonrigid = [c for c in concepts if c not in rigid]

    def rc():
        return rng.choice(concepts)

    def rnr():
        return rng.choice(nonrigid) if nonrigid else rng.choice(concepts)

    gcis: list = []
    for _ in range(cfg.n_nf1):
        gcis.append(NF1(rc(), rc()))
    for _ in range(cfg.n_nf2):
        gcis.append(NF2(rc(), rc(), rc()))
    for _ in range(cfg.n_nf3):
        gcis.append(NF3(rc(), rng.choice(roles), rc()))
    for _ in range(cfg.n_nf4):
        gcis.append(NF4(rng.choice(roles), rc(), rc()))
    # temporal axioms over NON-RIGID concepts only (confound rule). With no
    # non-rigid concepts (ρ=1) there are none — degenerate, and B==A there.
    temporal_ctors = [
        lambda: NF6(rnr(), rnr()), lambda: NF7(rnr(), rnr()),
        lambda: NF10(rnr(), rnr()), lambda: NF12(rnr(), rnr()),
        lambda: NF13(rnr(), rnr(), rnr()),
    ]
    n_temporal = cfg.n_temporal if nonrigid else 0
    for i in range(n_temporal):
        gcis.append(temporal_ctors[i % len(temporal_ctors)]())

    tbox = NormalFormTBox(frozenset(concepts), frozenset(roles), gcis)
    vocab = Vocabulary.from_tbox(tbox)
    n_ticks = cfg.k + 1

    # support level per concept
    support_level = {}
    for c in concepts:
        if c in rigid:
            support_level[c] = 1
        elif cfg.support_max_intervals >= 2:
            support_level[c] = 3
        else:
            support_level[c] = 2

    # ABox memberships
    cid = vocab.concept_to_id
    individuals = [f"i{j}" for j in range(cfg.n_individuals)]
    memberships: list[tuple[int, int, list[int]]] = []
    for j in range(cfg.n_individuals):
        chosen = rng.sample(concepts, min(cfg.memberships_per_individual, len(concepts)))
        for c in chosen:
            if c in rigid:
                active = list(range(n_ticks))                     # full-time convex
            elif cfg.support_max_intervals >= 2:
                active = _support_ticks(rng, n_ticks, rng.randint(2, cfg.support_max_intervals))
            else:
                active = _support_ticks(rng, n_ticks, 1)
            if active:
                memberships.append((j, cid[c], active))

    closure = temporal_closure(tbox, cfg.k, cfg.loop)

    # Hierarchy-consistent ABox: expand each individual's membership at each
    # tick by static superclasses (member of c ⇒ member of every c ⊒ c), then
    # derive genuine non-membership for the closed-world (out-of-box) negatives.
    static_sup: dict[str, set[str]] = {c: {c} for c in concepts}
    for sub, sup in compute_el_closure(tbox):
        if sub in static_sup and sup != TOP_NAME:
            static_sup[sub].add(sup)
    # member_at[(ind, tick)] = set of concept-ids the individual is in (expanded)
    member_at: dict[tuple[int, int], set[int]] = {}
    for (i, c, ticks) in memberships:
        cname = vocab.id_to_concept[c]
        for t in ticks:
            s = member_at.setdefault((i, t), set())
            for sup in static_sup[cname]:
                s.add(cid[sup])
    pos_triples = sorted({(i, c, t) for (i, t), cs in member_at.items() for c in cs})

    # negatives: (ind, concept, tick) where the concept has memberships somewhere
    # (a "relevant" concept) but the individual is genuinely not in it at t.
    membership_concepts = sorted({c for (_i, c, _ts) in memberships})
    neg_pool = []
    for i in range(cfg.n_individuals):
        for t in range(n_ticks):
            here = member_at.get((i, t), set())
            for c in membership_concepts:
                if c not in here:
                    neg_pool.append((i, c, t))
    rng.shuffle(neg_pool)
    neg_triples = neg_pool[: max(len(pos_triples), 1) * cfg.neg_ratio]

    nondegen_temporal = sum(
        1 for g in gcis
        if isinstance(g, (NF6, NF7, NF10, NF12, NF13))
    )
    stats = {
        "rigid_fraction": cfg.rigid_fraction, "n_rigid": n_rigid,
        "concepts": cfg.n_concepts, "temporal_gcis": nondegen_temporal,
        "individuals": cfg.n_individuals, "memberships": len(memberships),
        "support_max_intervals": cfg.support_max_intervals,
        "closure_pairs": len(closure), "n_ticks": n_ticks,
        "pos_triples": len(pos_triples), "neg_triples": len(neg_triples),
    }
    return RQ2Dataset(
        tbox=tbox, vocab=vocab, individuals=individuals, n_ticks=n_ticks,
        k=cfg.k, loop=cfg.loop, memberships=memberships,
        pos_triples=pos_triples, neg_triples=neg_triples,
        rigid_concepts=frozenset(rigid), support_level=support_level,
        closure=closure, stats=stats,
    )


@dataclass
class RQ2TrainConfig:
    dim: int = 16
    lr: float = 2e-2
    lr_min: float = 1e-4
    epochs: int = 4000
    offset_reg: float = 1e-3
    device: str = "cpu"
    seed: int = 0


def train_rq2(ds: RQ2Dataset, cfg: RQ2TrainConfig, model_cls=TemporalBoxModel):
    """Train A or B on the same dataset: full TBox loss + faithful ABox fit
    (members in-box, genuine non-members out-of-box; hierarchy-consistent)."""
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    comp = compile_tbox(ds.tbox, ds.vocab, ds.k, ds.loop, dev)
    model = model_cls(
        n_concepts=ds.vocab.n_concepts, n_roles=max(ds.vocab.n_roles, 1),
        k=ds.k, loop=ds.loop, dim=cfg.dim, top_id=ds.vocab.top_id,
        n_individuals=len(ds.individuals), seed=cfg.seed,
    ).to(dev)

    pos = torch.tensor(ds.pos_triples, dtype=torch.long, device=dev)
    neg = torch.tensor(ds.neg_triples, dtype=torch.long, device=dev) if ds.neg_triples else None

    def mem_losses():
        cm, om = model.concept_box(pos[:, 1], pos[:, 2])
        l_in = point_in_box_loss(model.individual_point(pos[:, 0]), cm, om).mean()
        if neg is None:
            return l_in, torch.zeros((), device=dev)
        cn, on = model.concept_box(neg[:, 1], neg[:, 2])
        l_out = point_outside_box_loss(model.individual_point(neg[:, 0]), cn, on, margin=0.1).mean()
        return l_in, l_out

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)
    for _ in range(cfg.epochs):
        opt.zero_grad()
        l_tbox = sum(loss_components(model, comp).values())
        l_in, l_out = mem_losses()
        reg = cfg.offset_reg * torch.nn.functional.softplus(model.offsets_raw).mean()
        total = l_tbox + l_in + l_out + reg
        total.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        comps = {k: float(v) for k, v in loss_components(model, comp).items()}
        l_in, l_out = mem_losses()
        comps["loss_membership_in"] = float(l_in)
        comps["loss_membership_out"] = float(l_out)
        comps["loss_total"] = sum(comps.values())
    return model, comps


@torch.no_grad()
def evaluate_rq2(model, ds: RQ2Dataset, eps: float = 1e-3) -> dict:
    """Subsumption recovery (completeness) overall and by support level of the
    sub-concept. Enumerates the temporal closure (small in synthetic RQ2)."""
    dev = next(model.parameters()).device
    cid = ds.vocab.concept_to_id

    by_level = {1: [0, 0], 2: [0, 0], 3: [0, 0]}  # level -> [held, total]
    held_all = total_all = 0
    # batch the closure triples
    triples = list(ds.closure)
    if not triples:
        return {"completeness": float("nan"), "by_level": {}}
    a = torch.tensor([cid[x[0]] for x in triples], device=dev)
    b = torch.tensor([cid[x[1]] for x in triples], device=dev)
    t = torch.tensor([x[2] for x in triples], device=dev)
    ca, oa = model.concept_box(a, t)
    cb, ob = model.concept_box(b, t)
    held = (box_contains(ca, oa, cb, ob) <= eps).tolist()
    for (sub, _sup, _t), h in zip(triples, held):
        lvl = ds.support_level[sub]
        by_level[lvl][1] += 1
        by_level[lvl][0] += int(h)
        total_all += 1
        held_all += int(h)
    return {
        "completeness": held_all / total_all if total_all else float("nan"),
        "by_level": {lvl: (hc / tc if tc else float("nan"))
                     for lvl, (hc, tc) in by_level.items()},
        "level_counts": {lvl: tc for lvl, (_h, tc) in by_level.items()},
    }
