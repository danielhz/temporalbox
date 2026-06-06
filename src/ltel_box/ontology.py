"""
LTEL concept expressions, GCIs, TBox, and vocabulary.

LTEL concept descriptions:
    C, D ::= ⊤ | A | C ⊓ D | ∃r.C | ○C | C U D | □C | ◇C

Normal-form GCI shapes (lhs/rhs are concept names or ⊤ unless stated):
    NF1   A ⊑ B
    NF2   A₁ ⊓ A₂ ⊑ B
    NF3   A ⊑ ∃r.B
    NF4   ∃r.A ⊑ B
    NF5   r ∘ s ⊑ t              (role chain inclusion)
    NF6   A ⊑ ○B                 (next on rhs)
    NF7   A ⊑ □B                 (globally on rhs)
    NF8   A ⊑ ◇B                 (eventually on rhs)
    NF9   A ⊑ B U C              (until on rhs)
    NF10  ○A ⊑ B                 (next on lhs)
    NF11  □A ⊑ B                 (globally on lhs)
    NF12  ◇A ⊑ B                 (eventually on lhs)
    NF13  (A U C) ⊑ B            (until on lhs)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Sequence

# ── Concept expressions ──────────────────────────────────────────────────────

class _Top:
    """Singleton ⊤."""
    _inst: _Top | None = None

    def __new__(cls) -> _Top:
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:
        return "⊤"

    def __hash__(self) -> int:
        return hash("⊤")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Top)


TOP: _Top = _Top()


@dataclass(frozen=True)
class ConceptName:
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Conjunction:
    left: ConceptExpr
    right: ConceptExpr

    def __repr__(self) -> str:
        return f"({self.left} ⊓ {self.right})"


@dataclass(frozen=True)
class Existential:
    role: str
    filler: ConceptExpr

    def __repr__(self) -> str:
        return f"∃{self.role}.{self.filler}"


@dataclass(frozen=True)
class NextC:
    concept: ConceptExpr

    def __repr__(self) -> str:
        return f"○{self.concept}"


@dataclass(frozen=True)
class Until:
    left: ConceptExpr
    right: ConceptExpr

    def __repr__(self) -> str:
        return f"({self.left} U {self.right})"


@dataclass(frozen=True)
class Globally:
    concept: ConceptExpr

    def __repr__(self) -> str:
        return f"□{self.concept}"


@dataclass(frozen=True)
class Eventually:
    concept: ConceptExpr

    def __repr__(self) -> str:
        return f"◇{self.concept}"


ConceptExpr = Union[_Top, ConceptName, Conjunction, Existential, NextC, Until, Globally, Eventually]


# ── Axioms ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GCI:
    """General concept inclusion C ⊑ D."""
    lhs: ConceptExpr
    rhs: ConceptExpr

    def __repr__(self) -> str:
        return f"{self.lhs} ⊑ {self.rhs}"


@dataclass(frozen=True)
class RoleInclusion:
    """Role chain inclusion r₁ ∘ … ∘ rₖ ⊑ s (k ≥ 1)."""
    chain: tuple[str, ...]
    rhs: str

    def __repr__(self) -> str:
        return f"{' ∘ '.join(self.chain)} ⊑ {self.rhs}"


# ── Normal-form GCIs ─────────────────────────────────────────────────────────
# All concept arguments are concept-name strings or the sentinel "⊤".

TOP_NAME = "⊤"


@dataclass(frozen=True)
class NF1:
    lhs: str
    rhs: str


@dataclass(frozen=True)
class NF2:
    lhs1: str
    lhs2: str
    rhs: str


@dataclass(frozen=True)
class NF3:
    lhs: str
    role: str
    rhs: str


@dataclass(frozen=True)
class NF4:
    role: str
    lhs: str
    rhs: str


@dataclass(frozen=True)
class NF5:
    role1: str
    role2: str
    rhs: str


@dataclass(frozen=True)
class NF6:
    lhs: str
    rhs: str


@dataclass(frozen=True)
class NF7:
    lhs: str
    rhs: str


@dataclass(frozen=True)
class NF8:
    lhs: str
    rhs: str


@dataclass(frozen=True)
class NF9:
    lhs: str
    until_left: str
    until_right: str


@dataclass(frozen=True)
class NF10:
    lhs: str
    rhs: str


@dataclass(frozen=True)
class NF11:
    lhs: str
    rhs: str


@dataclass(frozen=True)
class NF12:
    lhs: str
    rhs: str


@dataclass(frozen=True)
class NF13:
    until_left: str
    until_right: str
    rhs: str


NormalFormGCI = Union[NF1, NF2, NF3, NF4, NF5, NF6, NF7, NF8, NF9, NF10, NF11, NF12, NF13]


# ── TBox ─────────────────────────────────────────────────────────────────────

@dataclass
class LTELTBox:
    concept_names: frozenset[str]
    role_names: frozenset[str]
    gcis: list[GCI]
    role_inclusions: list[RoleInclusion]

    def concept_count(self) -> int:
        return len(self.concept_names)

    def role_count(self) -> int:
        return len(self.role_names)


@dataclass
class NormalFormTBox:
    concept_names: frozenset[str]  # excludes TOP_NAME
    role_names: frozenset[str]
    nf_gcis: list[NormalFormGCI]

    def concept_count(self) -> int:
        return len(self.concept_names)

    def role_count(self) -> int:
        return len(self.role_names)

    def gcis_of_type(self, nf_type: type) -> list[NormalFormGCI]:
        return [g for g in self.nf_gcis if isinstance(g, nf_type)]


# ── Vocabulary ───────────────────────────────────────────────────────────────

@dataclass
class Vocabulary:
    """Integer ID mappings for concepts (including ⊤) and roles."""

    concept_to_id: dict[str, int] = field(default_factory=dict)
    role_to_id: dict[str, int] = field(default_factory=dict)
    id_to_concept: list[str] = field(default_factory=list)
    id_to_role: list[str] = field(default_factory=list)

    @classmethod
    def from_tbox(cls, tbox: NormalFormTBox) -> Vocabulary:
        vocab = cls()
        vocab._add_concept(TOP_NAME)
        for c in sorted(tbox.concept_names):
            vocab._add_concept(c)
        for r in sorted(tbox.role_names):
            vocab._add_role(r)
        return vocab

    def _add_concept(self, name: str) -> int:
        if name not in self.concept_to_id:
            idx = len(self.id_to_concept)
            self.concept_to_id[name] = idx
            self.id_to_concept.append(name)
        return self.concept_to_id[name]

    def _add_role(self, name: str) -> int:
        if name not in self.role_to_id:
            idx = len(self.id_to_role)
            self.role_to_id[name] = idx
            self.id_to_role.append(name)
        return self.role_to_id[name]

    @property
    def n_concepts(self) -> int:
        return len(self.id_to_concept)

    @property
    def n_roles(self) -> int:
        return len(self.id_to_role)

    @property
    def top_id(self) -> int:
        return self.concept_to_id[TOP_NAME]
