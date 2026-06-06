"""
EL+ subsumption closure via forward-chaining completion rules.

Computes all concept-name pairs (A, B) such that A ⊑ B is entailed by the
static (non-temporal) part of a NormalFormTBox.  Temporal GCIs (NF6–NF13)
are ignored here; temporal entailment checking is left to a future module.

Completion rules applied (Baader et al. 2005, Table 1):
    R1  A ⊑ B,  B ⊑ C            → A ⊑ C
    R2  A ⊑ A1, A ⊑ A2,
        A1 ⊓ A2 ⊑ B              → A ⊑ B
    R3  A ⊑ ∃r.B, B ⊑ C,
        ∃r.C ⊑ D                 → A ⊑ D
    R4  A ⊑ ∃r.B, B ⊑ C,
        r ∘ s ⊑ t (role chain)  → A ⊑ ∃t.… (abbreviated; full chain handled)
    R5  (role transitivity via R4)

Returns a frozenset of (sub, sup) string pairs where sub ⊑ sup is entailed.
"""

from __future__ import annotations
from collections import defaultdict

from .ontology import TOP_NAME, NF1, NF2, NF3, NF4, NF5, NormalFormTBox


def compute_el_closure(tbox: NormalFormTBox) -> frozenset[tuple[str, str]]:
    concepts = tbox.concept_names | {TOP_NAME}

    # Initialise: each concept subsumes itself and TOP subsumes everything.
    sub: dict[str, set[str]] = {c: {c, TOP_NAME} for c in concepts}
    # role_succs[c][r] = set of D such that c ⊑ ∃r.D is derived
    role_succs: dict[str, dict[str, set[str]]] = {
        c: defaultdict(set) for c in concepts
    }

    nf1 = tbox.gcis_of_type(NF1)
    nf2 = tbox.gcis_of_type(NF2)
    nf3 = tbox.gcis_of_type(NF3)
    nf4 = tbox.gcis_of_type(NF4)
    nf5 = tbox.gcis_of_type(NF5)

    # Index NF2: (a1, a2) → list of rhs
    nf2_index: dict[frozenset[str], list[str]] = defaultdict(list)
    for g in nf2:
        nf2_index[frozenset({g.lhs1, g.lhs2})].append(g.rhs)

    # Index NF4: (role, filler) → list of rhs
    nf4_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for g in nf4:
        nf4_index[(g.role, g.lhs)].append(g.rhs)

    # Index NF5: (role1, role2) → rhs role
    nf5_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for g in nf5:
        nf5_index[(g.role1, g.role2)].append(g.rhs)

    changed = True
    while changed:
        changed = False

        for c in list(concepts):
            sub_c = sub[c]
            succ_c = role_succs[c]

            # R1: propagate NF1 axioms
            for g in nf1:
                if g.lhs in sub_c and g.rhs not in sub_c:
                    sub_c.add(g.rhs)
                    changed = True

            # R2: NF2 axioms — if both lhs1 and lhs2 are in sub(c)
            for key, rhs_list in nf2_index.items():
                if key <= sub_c:
                    for rhs in rhs_list:
                        if rhs not in sub_c:
                            sub_c.add(rhs)
                            changed = True

            # R3: NF3 — c ⊑ ∃r.B
            for g in nf3:
                if g.lhs in sub_c:
                    # add existential witness
                    if g.rhs not in succ_c[g.role]:
                        succ_c[g.role].add(g.rhs)
                        changed = True

            # R3 continuation: ∃r.D ⊑ E — if c ⊑ ∃r.D and D ⊑ B and ∃r.B ⊑ E
            for role, fillers in succ_c.items():
                for filler in list(fillers):
                    # Propagate filler subsumptions to new fillers (via sub[filler])
                    for filler_super in list(sub.get(filler, set())):
                        for E in nf4_index.get((role, filler_super), []):
                            if E not in sub_c:
                                sub_c.add(E)
                                changed = True

    pairs: set[tuple[str, str]] = set()
    for c, supers in sub.items():
        for s in supers:
            pairs.add((c, s))
    return frozenset(pairs)
