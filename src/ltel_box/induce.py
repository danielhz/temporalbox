"""
Read the induced interpretation off a trained box scorer and score it
against the oracle (RQ1 protocol).

A subsumption A ⊑ B is *held* at position t iff Box(A)^(t) ⊆ Box(B)^(t) up to
a tolerance eps on the summed coordinate violations (the same violation
measure the containment loss optimizes, so "held at eps" is exactly "this
pair's loss term would be ≤ eps").

Metrics (always interpret against the achieved training loss):
  completeness  |entailed ∧ held| / |entailed|      — recovery of the oracle
  spurious_rate |held ∧ ¬entailed| / |¬entailed|    — wrongly asserted pairs

The universe is all ordered pairs of distinct non-⊤ concept names at every
lasso position. Works for any BoxScorer (method-agnostic).
"""

from __future__ import annotations

import torch

from .model import BoxScorer
from .ontology import Vocabulary


@torch.no_grad()
def held_subsumptions(
    scorer: BoxScorer,
    vocab: Vocabulary,
    eps: float = 1e-4,
    chunk: int = 512,
) -> set[tuple[str, str, int]]:
    """All (A, B, t) with Box(A)^(t) ⊆ Box(B)^(t) up to eps; ⊤ excluded."""
    n, T = scorer.n_concepts, scorer.n_positions
    names = vocab.id_to_concept
    top = vocab.top_id
    held: set[tuple[str, str, int]] = set()

    ids = torch.arange(n)
    for t in range(T):
        ts = torch.full((n,), t, dtype=torch.long)
        c, o = scorer.concept_box(ids, ts)
        lb, ub = c - o, c + o
        for a0 in range(0, n, chunk):
            a1 = min(a0 + chunk, n)
            # violation of A ⊆ B for the chunk: (a, b) -> sum_d relu terms
            viol = (
                torch.relu(lb[None, :, :] - lb[a0:a1, None, :])
                + torch.relu(ub[a0:a1, None, :] - ub[None, :, :])
            ).sum(-1)
            contained = viol <= eps
            for ai, bi in contained.nonzero(as_tuple=False).tolist():
                a = a0 + ai
                if a == bi or a == top or bi == top:
                    continue
                held.add((names[a], names[bi], t))
    return held


def rq1_metrics(
    held: set[tuple[str, str, int]],
    entailed: set[tuple[str, str, int]],
    vocab: Vocabulary,
    n_positions: int,
) -> dict[str, float]:
    """Completeness and spurious rate over the full (A, B, t) universe."""
    names = [c for i, c in enumerate(vocab.id_to_concept) if i != vocab.top_id]
    universe = len(names) * (len(names) - 1) * n_positions
    n_entailed = len(entailed)
    n_negative = universe - n_entailed

    recovered = len(held & entailed)
    spurious = len(held - entailed)
    return {
        "universe": universe,
        "entailed": n_entailed,
        "held": len(held),
        "recovered": recovered,
        "spurious": spurious,
        "completeness": recovered / n_entailed if n_entailed else float("nan"),
        "spurious_rate": spurious / n_negative if n_negative else float("nan"),
    }


@torch.no_grad()
def ranking_metrics(
    scorer: BoxScorer,
    vocab: Vocabulary,
    entailed: set[tuple[str, str, int]],
    chunk: int = 256,
) -> dict[str, float]:
    """Secondary KG-style ranking (§5): for each entailed (A,B,t), rank the
    true superconcept B among all candidate concepts by the A⊑·@t containment
    violation (lower = better); report MRR and Hits@{1,3,10}.

    Filtered ranking: other *entailed* superconcepts of A@t are removed from the
    candidate list before ranking B (standard filtered protocol)."""
    n, T = scorer.n_concepts, scorer.n_positions
    top = vocab.top_id
    # group entailed superconcepts by (A, t) for filtering
    sup_by: dict[tuple[str, int], set[str]] = {}
    for a, b, t in entailed:
        sup_by.setdefault((a, t), set()).add(b)

    ranks: list[int] = []
    dev = next(scorer.parameters()).device if hasattr(scorer, "parameters") else None
    ids = torch.arange(n, device=dev)
    # precompute per-tick boxes
    for t in range(T):
        ts = torch.full((n,), t, dtype=torch.long, device=dev)
        c, o = scorer.concept_box(ids, ts)
        lb, ub = c - o, c + o
        queries = [(a, b) for (a, b, tt) in entailed if tt == t]
        for a, b in queries:
            ai = vocab.concept_to_id[a]
            bi = vocab.concept_to_id[b]
            # violation of A ⊆ candidate for every candidate, at tick t
            viol = (torch.relu(lb - lb[ai]) + torch.relu(ub[ai] - ub)).sum(-1)
            v_true = viol[bi]
            # filtered: ignore ⊤, A itself, and other entailed supers of (a,t)
            ignore = {top, ai} | {vocab.concept_to_id[s] for s in sup_by.get((a, t), ()) if s != b}
            better = 0
            for j in range(n):
                if j in ignore or j == bi:
                    continue
                if viol[j] < v_true or (viol[j] == v_true and j < bi):
                    better += 1
            ranks.append(better + 1)

    if not ranks:
        return {"mrr": float("nan"), "hits1": float("nan"),
                "hits3": float("nan"), "hits10": float("nan")}
    rt = torch.tensor(ranks, dtype=torch.float)
    return {
        "mrr": float((1.0 / rt).mean()),
        "hits1": float((rt <= 1).float().mean()),
        "hits3": float((rt <= 3).float().mean()),
        "hits10": float((rt <= 10).float().mean()),
    }


def across_seed_metrics(
    held_per_seed: list[set[tuple[str, str, int]]],
    entailed: set[tuple[str, str, int]],
    vocab: Vocabulary,
    n_positions: int,
) -> dict[str, float]:
    """Across-seed aggregates for the faithfulness theorem's universal quantifier.

    `T ⊨ C⊑D` iff *every* zero-loss model satisfies it; the empirical proxy is
    the intersection of held pairs over all trained seeds. The intersection's
    spurious rate should be far below any single seed's (different models
    overcommit on different pairs), approximating the entailment closure.
    Also reports the union (any seed) as the optimistic bound.
    """
    if not held_per_seed:
        return {}
    names = [c for i, c in enumerate(vocab.id_to_concept) if i != vocab.top_id]
    universe = len(names) * (len(names) - 1) * n_positions
    n_negative = universe - len(entailed)

    inter = set.intersection(*held_per_seed)
    union = set.union(*held_per_seed)
    return {
        "seeds": len(held_per_seed),
        "entailed": len(entailed),
        "intersection_held": len(inter),
        "intersection_recovered": len(inter & entailed),
        "intersection_spurious": len(inter - entailed),
        "intersection_completeness": len(inter & entailed) / len(entailed) if entailed else float("nan"),
        "intersection_spurious_rate": len(inter - entailed) / n_negative if n_negative else float("nan"),
        "union_spurious_rate": len(union - entailed) / n_negative if n_negative else float("nan"),
    }
