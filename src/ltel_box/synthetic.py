"""
Synthetic LTEL TBox generator.

Generates random TBoxes directly in normal form, avoiding the need to
normalize.  The temporal GCI mix is controlled by SyntheticConfig.

Ground-truth subsumption pairs are computed by the EL+ closure (static
part only).  Temporal entailment is left for a future module once the
embedding design is finalized.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field

from .ontology import (
    TOP_NAME,
    NF1, NF2, NF3, NF4, NF5, NF6, NF7, NF8, NF9, NF10, NF11, NF12, NF13,
    NormalFormGCI, NormalFormTBox,
)
from .closure import compute_el_closure


@dataclass
class SyntheticConfig:
    n_concepts: int = 20
    n_roles: int = 3
    # Approximate GCI counts per type.  Set to 0 to skip a type.
    n_nf1: int = 15   # A ⊑ B
    n_nf2: int = 8    # A1 ⊓ A2 ⊑ B
    n_nf3: int = 8    # A ⊑ ∃r.B
    n_nf4: int = 8    # ∃r.A ⊑ B
    n_nf5: int = 2    # r ∘ s ⊑ t
    n_nf6: int = 5    # A ⊑ ○B
    n_nf7: int = 4    # A ⊑ □B
    n_nf8: int = 4    # A ⊑ ◇B
    n_nf9: int = 3    # A ⊑ B U C
    n_nf10: int = 3   # ○A ⊑ B
    n_nf11: int = 2   # □A ⊑ B
    n_nf12: int = 2   # ◇A ⊑ B
    n_nf13: int = 2   # (A U C) ⊑ B
    seed: int = 0


@dataclass
class SyntheticDataset:
    tbox: NormalFormTBox
    # (sub, sup) concept-name pairs entailed by the static part of the TBox.
    positive_pairs: frozenset[tuple[str, str]]
    # Random unentailed pairs used as negatives (excluding self-subsumptions).
    negative_pairs: frozenset[tuple[str, str]]
    concept_names: list[str]
    role_names: list[str]


def generate_tbox(cfg: SyntheticConfig | None = None) -> SyntheticDataset:
    if cfg is None:
        cfg = SyntheticConfig()
    rng = random.Random(cfg.seed)

    concepts = [f"C{i}" for i in range(cfg.n_concepts)]
    roles = [f"r{i}" for i in range(cfg.n_roles)]

    def rc() -> str:  # random concept name
        return rng.choice(concepts)

    def rc2() -> tuple[str, str]:  # two distinct concept names
        a = rng.choice(concepts)
        b = rng.choice(concepts)
        return a, b

    def rr() -> str:  # random role name
        return rng.choice(roles)

    gcis: list[NormalFormGCI] = []

    # Static GCIs
    for _ in range(cfg.n_nf1):
        a, b = rc2()
        gcis.append(NF1(a, b))

    for _ in range(cfg.n_nf2):
        a1, a2 = rc2()
        b = rc()
        gcis.append(NF2(a1, a2, b))

    for _ in range(cfg.n_nf3):
        a, b = rc2()
        gcis.append(NF3(a, rr(), b))

    for _ in range(cfg.n_nf4):
        a, b = rc2()
        gcis.append(NF4(rr(), a, b))

    for _ in range(cfg.n_nf5):
        r1, r2 = rng.choice(roles), rng.choice(roles)
        t = rng.choice(roles)
        gcis.append(NF5(r1, r2, t))

    # Temporal GCIs
    for _ in range(cfg.n_nf6):
        a, b = rc2()
        gcis.append(NF6(a, b))

    for _ in range(cfg.n_nf7):
        a, b = rc2()
        gcis.append(NF7(a, b))

    for _ in range(cfg.n_nf8):
        a, b = rc2()
        gcis.append(NF8(a, b))

    for _ in range(cfg.n_nf9):
        a, bl, br = rc(), rc(), rc()
        gcis.append(NF9(a, bl, br))

    for _ in range(cfg.n_nf10):
        a, b = rc2()
        gcis.append(NF10(a, b))

    for _ in range(cfg.n_nf11):
        a, b = rc2()
        gcis.append(NF11(a, b))

    for _ in range(cfg.n_nf12):
        a, b = rc2()
        gcis.append(NF12(a, b))

    for _ in range(cfg.n_nf13):
        al, ar, b = rc(), rc(), rc()
        gcis.append(NF13(al, ar, b))

    tbox = NormalFormTBox(
        concept_names=frozenset(concepts),
        role_names=frozenset(roles),
        nf_gcis=gcis,
    )

    positives = compute_el_closure(tbox)
    # Only keep pairs that are non-trivial (not X ⊑ X or X ⊑ ⊤)
    positives = frozenset(
        (s, p) for (s, p) in positives
        if s != p and p != TOP_NAME and s != TOP_NAME
    )

    # Negative pairs: random pairs not in positives and not X ⊑ X
    all_pairs = {(a, b) for a in concepts for b in concepts if a != b}
    neg_pool = list(all_pairs - set(positives))
    rng.shuffle(neg_pool)
    n_neg = min(len(neg_pool), max(len(positives) * 5, 50))
    negatives = frozenset(neg_pool[:n_neg])

    return SyntheticDataset(
        tbox=tbox,
        positive_pairs=positives,
        negative_pairs=negatives,
        concept_names=concepts,
        role_names=roles,
    )
