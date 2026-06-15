"""
Train Option A on a Wikidata office slice (the §5 feasibility exhibit).

This is experiment (A) from the slice-design decision: fit per-time-point
boxes to the non-rigid membership intervals while enforcing the (rigid) P279
subclass hierarchy, then check the ontology's subsumptions are preserved at
every time point. It is explicitly NOT a test of the temporal operators (the
office TBox has none); it exercises the per-time-point/lasso machinery and
constant-domain non-rigid memberships at real scale (~1.7k concepts).

Model coupling: individuals are rigid points shared across ticks (constant
domain); a person's point must lie in its office-concept's box at each tick of
its term; the hierarchy forces sub-boxes inside super-boxes at every tick.
A light offset (volume) regularizer curbs the trivial "all boxes huge" optimum
(there is no ⊥ to oppose overcommitment — see the RQ1 spurious discussion).

Scale: concepts × ticks boxes and closure × ticks entailments are far too
large to enumerate, so evaluation is SAMPLING-based (sampled subsumption×tick
checks for completeness, sampled non-entailed pairs for the spurious proxy).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .induce import held_subsumptions  # noqa: F401  (re-exported convenience)
from .logging_utils import ExperimentLogger
from .loss import box_contains, point_in_box_loss
from .model import TemporalBoxModel
from .ontology import Vocabulary
from .wikidata import WikidataOfficeDataset

# default model factory; RQ2 passes TimeAugmentedBoxModel (Option B) instead.
DEFAULT_MODEL = TemporalBoxModel


@dataclass
class SliceTrainConfig:
    dim: int = 32
    lr: float = 2e-2
    lr_min: float = 1e-4
    epochs: int = 3000
    seed: int = 0
    device: str = "cpu"
    offset_reg: float = 1e-3        # volume penalty (curbs overcommitment)
    membership_cap: int = 0         # 0 = expand all; >0 = random subsample per build
    eval_pairs: int = 5000          # sampled closure pairs for completeness
    eval_ticks_per_pair: int = 16   # sampled ticks per pair
    eval_neg_pairs: int = 5000      # sampled non-entailed pairs for spurious
    log_every: int = 200


@dataclass
class SliceTensors:
    vocab: Vocabulary
    individuals: list[str]
    nf1: torch.Tensor          # (n, 2) concept-id pairs (child, parent)
    membership: torch.Tensor   # (m, 3) (individual_idx, concept_id, tick)
    n_ticks: int


def build_slice_tensors(
    ds: WikidataOfficeDataset, device: str = "cpu", cap: int = 0, seed: int = 0
) -> SliceTensors:
    vocab = Vocabulary.from_tbox(ds.tbox)
    ind_id = {q: i for i, q in enumerate(ds.individuals)}
    cid = vocab.concept_to_id

    nf1 = [(cid[g.lhs], cid[g.rhs]) for g in ds.tbox.nf_gcis if hasattr(g, "lhs")]

    triples: list[tuple[int, int, int]] = []
    for m in ds.memberships:
        c = cid.get(m.position)
        if c is None:
            continue
        i = ind_id[m.person]
        for t in range(m.start_tick, m.end_tick + 1):
            triples.append((i, c, t))
    if cap and len(triples) > cap:
        g = torch.Generator().manual_seed(seed)
        keep = torch.randperm(len(triples), generator=g)[:cap].tolist()
        triples = [triples[j] for j in keep]

    return SliceTensors(
        vocab=vocab,
        individuals=list(ds.individuals),
        nf1=torch.tensor(nf1, dtype=torch.long, device=device),
        membership=torch.tensor(triples, dtype=torch.long, device=device),
        n_ticks=len(ds.time_points),
    )


def train_slice(
    ds: WikidataOfficeDataset,
    cfg: SliceTrainConfig,
    logger: ExperimentLogger | None = None,
    model_cls=DEFAULT_MODEL,
) -> tuple[torch.nn.Module, SliceTensors, dict[str, float]]:
    torch.manual_seed(cfg.seed)
    st = build_slice_tensors(ds, cfg.device, cfg.membership_cap, cfg.seed)
    k = st.n_ticks - 1
    model = model_cls(
        n_concepts=st.vocab.n_concepts, n_roles=1, k=k, loop=k, dim=cfg.dim,
        top_id=st.vocab.top_id, n_individuals=len(st.individuals), seed=cfg.seed,
    ).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)

    nf1, mem = st.nf1, st.membership
    # NF1 is rigid: enforce containment at every tick. Built once as a flat
    # (pair, tick) index; large but vectorized.
    ticks = torch.arange(st.n_ticks, device=cfg.device)
    nf1_a = nf1[:, 0].repeat_interleave(st.n_ticks)
    nf1_b = nf1[:, 1].repeat_interleave(st.n_ticks)
    nf1_t = ticks.repeat(nf1.shape[0])

    final: dict[str, float] = {}
    for epoch in range(cfg.epochs):
        opt.zero_grad()
        ca, oa = model.concept_box(nf1_a, nf1_t)
        cb, ob = model.concept_box(nf1_b, nf1_t)
        l_sub = box_contains(ca, oa, cb, ob).mean()

        pts = model.individual_point(mem[:, 0])
        cm, om = model.concept_box(mem[:, 1], mem[:, 2])
        l_mem = point_in_box_loss(pts, cm, om).mean()

        l_reg = cfg.offset_reg * torch.nn.functional.softplus(model.offsets_raw).mean()
        total = l_sub + l_mem + l_reg
        total.backward()
        opt.step()
        sched.step()
        if logger is not None and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            logger.log(epoch, {
                "loss_total": float(total.detach()),
                "loss_subsumption": float(l_sub.detach()),
                "loss_membership": float(l_mem.detach()),
                "loss_offset_reg": float(l_reg.detach()),
            })
    with torch.no_grad():
        ca, oa = model.concept_box(nf1_a, nf1_t)
        cb, ob = model.concept_box(nf1_b, nf1_t)
        l_sub = float(box_contains(ca, oa, cb, ob).mean())
        pts = model.individual_point(mem[:, 0])
        cm, om = model.concept_box(mem[:, 1], mem[:, 2])
        l_mem = float(point_in_box_loss(pts, cm, om).mean())
    final = {"loss_subsumption": l_sub, "loss_membership": l_mem,
             "memberships_expanded": int(mem.shape[0])}
    return model, st, final


@torch.no_grad()
def evaluate_slice(
    model: TemporalBoxModel, st: SliceTensors, ds: WikidataOfficeDataset,
    cfg: SliceTrainConfig, eps: float = 1e-3, seed: int = 0,
) -> dict[str, float]:
    """Sampling-based subsumption preservation: of sampled (A⊑B, tick) from the
    closure, what fraction hold? Spurious proxy: of sampled non-entailed pairs
    (at a random tick), what fraction the model asserts."""
    g = torch.Generator().manual_seed(seed)
    dev = next(model.parameters()).device
    cid = st.vocab.concept_to_id
    closure = [(cid[a], cid[b]) for a, b in ds.subsumption_closure
               if a in cid and b in cid]
    n_concepts, T = st.vocab.n_concepts, st.n_ticks

    def held(a_ids, b_ids, t_ids) -> torch.Tensor:
        ca, oa = model.concept_box(a_ids.to(dev), t_ids.to(dev))
        cb, ob = model.concept_box(b_ids.to(dev), t_ids.to(dev))
        return box_contains(ca, oa, cb, ob) <= eps

    # completeness over sampled closure pairs × sampled ticks
    npairs = min(cfg.eval_pairs, len(closure))
    idx = torch.randperm(len(closure), generator=g)[:npairs].tolist()
    a = torch.tensor([closure[j][0] for j in idx for _ in range(cfg.eval_ticks_per_pair)])
    b = torch.tensor([closure[j][1] for j in idx for _ in range(cfg.eval_ticks_per_pair)])
    t = torch.randint(0, T, (a.shape[0],), generator=g)
    completeness = float(held(a, b, t).float().mean()) if a.numel() else float("nan")

    # spurious proxy over sampled non-entailed ordered pairs at a random tick
    closure_set = set(closure)
    negs = []
    while len(negs) < cfg.eval_neg_pairs:
        x = int(torch.randint(0, n_concepts, (1,), generator=g))
        y = int(torch.randint(0, n_concepts, (1,), generator=g))
        if x != y and x != st.vocab.top_id and y != st.vocab.top_id and (x, y) not in closure_set:
            negs.append((x, y))
    na = torch.tensor([p[0] for p in negs])
    nb = torch.tensor([p[1] for p in negs])
    nt = torch.randint(0, T, (len(negs),), generator=g)
    spurious = float(held(na, nb, nt).float().mean())

    return {
        "concepts": n_concepts, "ticks": T, "individuals": len(st.individuals),
        "closure_pairs": len(closure), "sampled_pairs": npairs,
        "completeness_sampled": completeness, "spurious_rate_sampled": spurious, "eps": eps,
    }
