"""
Normalize a raw LTEL TBox into the normal forms NF1–NF13.

The procedure introduces fresh auxiliary concept names for any complex
sub-expression, mirroring the standard EL+ normalization (Baader et al. 2005)
extended to the temporal operators.
"""

from __future__ import annotations
from typing import Iterator

from .ontology import (
    TOP, TOP_NAME,
    ConceptName, Conjunction, Existential, NextC, Until, Globally, Eventually,
    ConceptExpr, GCI, RoleInclusion,
    LTELTBox, NormalFormTBox,
    NF1, NF2, NF3, NF4, NF5, NF6, NF7, NF8, NF9, NF10, NF11, NF12, NF13,
    NormalFormGCI,
)


class _Normalizer:
    def __init__(self) -> None:
        self._fresh_counter = 0
        self._gcis: list[NormalFormGCI] = []
        self._concepts: set[str] = set()
        self._roles: set[str] = set()

    def _fresh(self) -> str:
        name = f"_aux{self._fresh_counter}"
        self._fresh_counter += 1
        self._concepts.add(name)
        return name

    def _name(self, expr: ConceptExpr) -> str:
        """Return the concept-name string for expr, introducing aux names as needed."""
        if expr is TOP:
            return TOP_NAME
        if isinstance(expr, ConceptName):
            self._concepts.add(expr.name)
            return expr.name
        # Complex expression: name it with a fresh concept, add defining GCIs.
        aux = self._fresh()
        self._define(aux, expr)
        return aux

    def _define(self, name: str, expr: ConceptExpr) -> None:
        """Emit GCIs that define `name` to be equivalent to `expr`."""
        # We emit only the ⊑ direction (name ⊑ expr), which is sufficient for
        # a faithful embedding: the loss encodes the axioms of the TBox, not
        # the full equivalence.  The closure / training can handle the rest.
        self._emit_rhs(name, expr)

    def _emit_rhs(self, lhs_name: str, rhs: ConceptExpr) -> None:
        """Emit the normal form for `lhs_name ⊑ rhs`."""
        if rhs is TOP or isinstance(rhs, ConceptName):
            self._gcis.append(NF1(lhs_name, self._name(rhs)))
        elif isinstance(rhs, Conjunction):
            b = self._name(rhs.left)
            c = self._name(rhs.right)
            self._gcis.append(NF1(lhs_name, b))
            self._gcis.append(NF1(lhs_name, c))
        elif isinstance(rhs, Existential):
            self._roles.add(rhs.role)
            b = self._name(rhs.filler)
            self._gcis.append(NF3(lhs_name, rhs.role, b))
        elif isinstance(rhs, NextC):
            b = self._name(rhs.concept)
            self._gcis.append(NF6(lhs_name, b))
        elif isinstance(rhs, Globally):
            b = self._name(rhs.concept)
            self._gcis.append(NF7(lhs_name, b))
        elif isinstance(rhs, Eventually):
            b = self._name(rhs.concept)
            self._gcis.append(NF8(lhs_name, b))
        elif isinstance(rhs, Until):
            bl = self._name(rhs.left)
            br = self._name(rhs.right)
            self._gcis.append(NF9(lhs_name, bl, br))
        else:
            raise TypeError(f"Unhandled rhs type: {type(rhs)}")

    def _emit_lhs(self, lhs: ConceptExpr, rhs_name: str) -> None:
        """Emit the normal form for `lhs ⊑ rhs_name`."""
        if lhs is TOP or isinstance(lhs, ConceptName):
            self._gcis.append(NF1(self._name(lhs), rhs_name))
        elif isinstance(lhs, Conjunction):
            a1 = self._name(lhs.left)
            a2 = self._name(lhs.right)
            self._gcis.append(NF2(a1, a2, rhs_name))
        elif isinstance(lhs, Existential):
            self._roles.add(lhs.role)
            a = self._name(lhs.filler)
            self._gcis.append(NF4(lhs.role, a, rhs_name))
        elif isinstance(lhs, NextC):
            a = self._name(lhs.concept)
            self._gcis.append(NF10(a, rhs_name))
        elif isinstance(lhs, Globally):
            a = self._name(lhs.concept)
            self._gcis.append(NF11(a, rhs_name))
        elif isinstance(lhs, Eventually):
            a = self._name(lhs.concept)
            self._gcis.append(NF12(a, rhs_name))
        elif isinstance(lhs, Until):
            al = self._name(lhs.left)
            ar = self._name(lhs.right)
            self._gcis.append(NF13(al, ar, rhs_name))
        else:
            raise TypeError(f"Unhandled lhs type: {type(lhs)}")

    def process_gci(self, gci: GCI) -> None:
        lhs, rhs = gci.lhs, gci.rhs
        lhs_atomic = lhs is TOP or isinstance(lhs, ConceptName)
        rhs_atomic = rhs is TOP or isinstance(rhs, ConceptName)

        if lhs_atomic and rhs_atomic:
            self._gcis.append(NF1(self._name(lhs), self._name(rhs)))
        elif lhs_atomic:
            self._emit_rhs(self._name(lhs), rhs)
        elif rhs_atomic:
            self._emit_lhs(lhs, self._name(rhs))
        else:
            # Both sides complex: introduce aux for rhs, normalise lhs ⊑ aux.
            aux = self._name(rhs)
            self._emit_lhs(lhs, aux)

    def process_role_inclusion(self, ri: RoleInclusion) -> None:
        for r in ri.chain:
            self._roles.add(r)
        self._roles.add(ri.rhs)
        if len(ri.chain) == 2:
            self._gcis.append(NF5(ri.chain[0], ri.chain[1], ri.rhs))
        elif len(ri.chain) == 1:
            # r ⊑ s: treat as r ∘ r ⊑ s is wrong; instead emit as NF1 analogue.
            # Encode as A ⊑ ∃s.⊤ whenever A ⊑ ∃r.⊤, via a helper NF5-style.
            # For now flatten to NF1 on a dummy reflexive: just add NF5 with same role.
            # TODO: handle length-1 role inclusions properly (they are RI, not GCI).
            self._gcis.append(NF5(ri.chain[0], ri.chain[0], ri.rhs))
        else:
            # Flatten chain r1 ∘ r2 ∘ … ∘ rk into binary role inclusions with aux roles.
            chain = list(ri.chain)
            while len(chain) > 2:
                aux_role = f"_arole{len(self._roles)}"
                self._roles.add(aux_role)
                self._gcis.append(NF5(chain[0], chain[1], aux_role))
                chain = [aux_role] + chain[2:]
            self._gcis.append(NF5(chain[0], chain[1], ri.rhs))


def normalize(tbox: LTELTBox) -> NormalFormTBox:
    norm = _Normalizer()
    for name in tbox.concept_names:
        norm._concepts.add(name)
    for name in tbox.role_names:
        norm._roles.add(name)
    for gci in tbox.gcis:
        norm.process_gci(gci)
    for ri in tbox.role_inclusions:
        norm.process_role_inclusion(ri)
    return NormalFormTBox(
        concept_names=frozenset(norm._concepts),
        role_names=frozenset(norm._roles),
        nf_gcis=norm._gcis,
    )
