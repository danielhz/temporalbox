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
