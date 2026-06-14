"""
Build an LTEL dataset from a Wikidata political-office extract.

Input: two TSV files exported (by SPARQL, see scripts/wikidata/) from the
pinned Wikidata 2026-04 dump:

  memberships.tsv : ?person ?position ?start ?end ?replaces ?replacedBy
                    (one row per time-qualified `position held` (P39)
                    statement; ?start/?end are P580/P582 datetimes; the
                    succession columns are P1365/P1366 holder links)
  p279.tsv        : ?child ?parent   (all `subclass of` (P279) edges)

Modelling (point-based LTEL, per the project README and paper):

  * Concepts  = positions (the P39 values) and their P279 ancestors.
  * Individuals = office-holders (people).
  * TBox      = the upward P279 closure over the *used* positions, emitted as
                NF1 GCIs (position ⊑ superclass). Rigid (time-invariant).
  * Temporal ABox = a person is a member of a position-concept only during the
                [start, end] interval of the statement — the non-rigid
                membership LTEL needs.

Two discretisations of the time line (both committed; the paper defers which
to report):

  * "year"  — fixed calendar granularity; ticks are calendar years.
  * "event" — event-based; ticks are the sorted distinct change-points
              (all interval endpoints); ○ (Next) = next change-point.

The subclass closure is computed here in Python (deterministic BFS) rather
than with a SPARQL property path, so the build is fast and reproducible.
"""

from __future__ import annotations
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ontology import TOP_NAME, NF1, NormalFormTBox, Vocabulary
from .closure import compute_el_closure

_QID_RE = re.compile(r"Q\d+")
_YEAR_RE = re.compile(r'^"?(-?\d+)')


def _qid(term: str) -> str | None:
    """Extract a bare Q-id from an N-Triples/TSV IRI term, else None."""
    if not term:
        return None
    m = _QID_RE.search(term)
    return m.group(0) if m else None


def _year(term: str) -> int | None:
    """Extract the (possibly negative/BCE) year from an xsd:dateTime literal."""
    if not term:
        return None
    m = _YEAR_RE.match(term)
    return int(m.group(1)) if m else None


@dataclass
class MembershipInterval:
    """A person is a member of `position` for ticks [start_tick, end_tick]
    (inclusive indices into WikidataOfficeDataset.time_points)."""
    person: str
    position: str
    start_tick: int
    end_tick: int


@dataclass
class WikidataConfig:
    granularity: str = "year"          # "year" | "event"
    min_holders: int = 0               # keep positions with >= this many distinct holders
    root: str | None = None            # if set, keep only positions that are P279* descendants of this QID
    min_year: int | None = None        # clip the time line (inclusive)
    max_year: int | None = None
    open_interval_span: int = 1        # ticks to extend a statement with only one endpoint


@dataclass
class WikidataOfficeDataset:
    tbox: NormalFormTBox
    vocab: Vocabulary
    individuals: list[str]
    time_points: list[int]
    memberships: list[MembershipInterval]
    subsumption_closure: frozenset[tuple[str, str]]
    config: WikidataConfig
    stats: dict = field(default_factory=dict)

    def memberships_at(self, tick_idx: int) -> list[tuple[str, str]]:
        """(person, position) pairs holding at the given tick index."""
        return [
            (m.person, m.position)
            for m in self.memberships
            if m.start_tick <= tick_idx <= m.end_tick
        ]


# ── parsing ──────────────────────────────────────────────────────────────────

@dataclass
class _RawMembership:
    person: str
    position: str
    start_year: int | None
    end_year: int | None


def _parse_memberships(path: Path) -> list[_RawMembership]:
    rows: list[_RawMembership] = []
    with path.open() as f:
        f.readline()  # header: ?person ?position ?start ?end ?replaces ?replacedBy
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            # Unbound trailing optionals may be omitted; pad to the full width.
            while len(cols) < 6:
                cols.append("")
            person = _qid(cols[0])
            position = _qid(cols[1])
            if person is None or position is None:
                continue
            rows.append(_RawMembership(person, position, _year(cols[2]), _year(cols[3])))
    return rows


def _parse_p279(path: Path) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    with path.open() as f:
        f.readline()  # header
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            child, parent = _qid(cols[0]), _qid(cols[1])
            if child and parent and child != parent:
                edges.append((child, parent))
    return edges


# ── closure helpers ────────────────────────────────────────────────────────

def _upward_closure(seeds: set[str], parents: dict[str, set[str]]) -> set[str]:
    """All concepts reachable from `seeds` following P279 edges upward."""
    seen: set[str] = set()
    stack = list(seeds)
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        stack.extend(parents.get(c, ()))
    return seen


def _descendants(root: str, children: dict[str, set[str]]) -> set[str]:
    """All concepts reachable from `root` following P279 edges downward."""
    seen: set[str] = set()
    stack = [root]
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        stack.extend(children.get(c, ()))
    return seen


# ── build ────────────────────────────────────────────────────────────────────

def build_dataset(
    memberships_tsv: str | Path,
    p279_tsv: str | Path,
    config: WikidataConfig | None = None,
) -> WikidataOfficeDataset:
    cfg = config or WikidataConfig()
    if cfg.granularity not in ("year", "event"):
        raise ValueError(f"granularity must be 'year' or 'event', got {cfg.granularity!r}")

    raw = _parse_memberships(Path(memberships_tsv))
    edges = _parse_p279(Path(p279_tsv))

    parents: dict[str, set[str]] = {}
    children: dict[str, set[str]] = {}
    for c, p in edges:
        parents.setdefault(c, set()).add(p)
        children.setdefault(p, set()).add(c)

    # Optional restriction to a sub-hierarchy under `root`.
    allowed_positions: set[str] | None = None
    if cfg.root is not None:
        allowed_positions = _descendants(cfg.root, children)

    # Count holders per position; apply min_holders / root filters.
    holders: dict[str, set[str]] = {}
    for r in raw:
        if allowed_positions is not None and r.position not in allowed_positions:
            continue
        holders.setdefault(r.position, set()).add(r.person)
    used_positions = {
        pos for pos, hs in holders.items() if len(hs) >= cfg.min_holders
    }

    # Filtered raw memberships (position kept, and at least one usable year).
    kept = [
        r for r in raw
        if r.position in used_positions and (r.start_year is not None or r.end_year is not None)
    ]

    # Concept universe = used positions + their P279 ancestors; TBox = the
    # induced subclass edges among that universe.
    concepts = _upward_closure(used_positions, parents)
    nf_gcis = [NF1(c, p) for (c, p) in edges if c in concepts and p in concepts]
    tbox = NormalFormTBox(
        concept_names=frozenset(concepts),
        role_names=frozenset(),
        nf_gcis=nf_gcis,
    )
    vocab = Vocabulary.from_tbox(tbox)

    # Time line.
    years = [y for r in kept for y in (r.start_year, r.end_year) if y is not None]
    if cfg.min_year is not None:
        years = [y for y in years if y >= cfg.min_year]
    if cfg.max_year is not None:
        years = [y for y in years if y <= cfg.max_year]
    if not years:
        raise ValueError("no usable dated memberships after filtering")
    lo, hi = min(years), max(years)

    if cfg.granularity == "year":
        # Guard against garbage Wikidata dates (e.g. year -2,343,126) blowing
        # the calendar timeline into millions of empty ticks. Event granularity
        # is immune (it only uses occurring change-points); for year, clip the
        # range with min_year/max_year.
        span = hi - lo + 1
        if span > 10_000:
            raise ValueError(
                f"year granularity would create {span} ticks (year range [{lo}, {hi}]); "
                "likely outlier dates — set min_year/max_year to clip, or use "
                "granularity='event' (bounded by occurring change-points)"
            )
        time_points = list(range(lo, hi + 1))
    else:  # event-based: ticks are the sorted distinct endpoints
        time_points = sorted({y for y in years})
    tick_index = {t: i for i, t in enumerate(time_points)}

    def _nearest_tick(year: int, *, floor: bool) -> int:
        """Map a year to a tick index (event mode may not have an exact tick)."""
        if year in tick_index:
            return tick_index[year]
        # binary-search-free: clamp into range, then snap
        if year <= time_points[0]:
            return 0
        if year >= time_points[-1]:
            return len(time_points) - 1
        # linear scan is fine; time_points is modest
        for i in range(len(time_points) - 1):
            if time_points[i] <= year < time_points[i + 1]:
                return i if floor else i + 1
        return len(time_points) - 1

    individuals_set: set[str] = set()
    memberships: list[MembershipInterval] = []
    for r in kept:
        s, e = r.start_year, r.end_year
        if s is None:
            s = e - cfg.open_interval_span if e is not None else lo
        if e is None:
            e = s + cfg.open_interval_span
        s = max(lo, min(hi, s))
        e = max(lo, min(hi, e))
        if e < s:
            s, e = e, s
        if cfg.min_year is not None and e < cfg.min_year:
            continue
        if cfg.max_year is not None and s > cfg.max_year:
            continue
        st = _nearest_tick(s, floor=True)
        et = _nearest_tick(e, floor=False)
        if et < st:
            st, et = et, st
        memberships.append(MembershipInterval(r.person, r.position, st, et))
        individuals_set.add(r.person)

    closure = compute_el_closure(tbox)
    closure = frozenset(
        (a, b) for (a, b) in closure if a != b and a != TOP_NAME and b != TOP_NAME
    )

    stats = {
        "raw_memberships": len(raw),
        "kept_memberships": len(memberships),
        "used_positions": len(used_positions),
        "concepts": len(concepts),
        "tbox_gcis": len(nf_gcis),
        "individuals": len(individuals_set),
        "time_points": len(time_points),
        "year_range": [lo, hi],
        "closure_pairs": len(closure),
    }

    return WikidataOfficeDataset(
        tbox=tbox,
        vocab=vocab,
        individuals=sorted(individuals_set),
        time_points=time_points,
        memberships=memberships,
        subsumption_closure=closure,
        config=cfg,
        stats=stats,
    )


# ── serialisation ──────────────────────────────────────────────────────────

def save_dataset(ds: WikidataOfficeDataset, outdir: str | Path) -> None:
    """Materialise a built dataset as plain text files (FAIR-friendly)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "tbox.tsv").open("w") as f:
        f.write("child\tparent\n")
        for g in ds.tbox.nf_gcis:
            f.write(f"{g.lhs}\t{g.rhs}\n")
    with (out / "closure.tsv").open("w") as f:
        f.write("sub\tsup\n")
        for a, b in sorted(ds.subsumption_closure):
            f.write(f"{a}\t{b}\n")
    with (out / "memberships.tsv").open("w") as f:
        f.write("person\tposition\tstart_tick\tend_tick\n")
        for m in ds.memberships:
            f.write(f"{m.person}\t{m.position}\t{m.start_tick}\t{m.end_tick}\n")
    with (out / "time_points.tsv").open("w") as f:
        f.write("tick_index\tvalue\n")
        for i, t in enumerate(ds.time_points):
            f.write(f"{i}\t{t}\n")
    with (out / "meta.json").open("w") as f:
        json.dump({"config": asdict(ds.config), "stats": ds.stats}, f, indent=2)


def load_dataset(indir: str | Path) -> WikidataOfficeDataset:
    """Load a dataset materialised by save_dataset (lets the small built dataset
    travel to the GPU host instead of the multi-hundred-MB source TSVs)."""
    d = Path(indir)
    meta = json.loads((d / "meta.json").read_text())

    nf_gcis: list = []
    concepts: set[str] = set()
    with (d / "tbox.tsv").open() as f:
        f.readline()
        for line in f:
            child, parent = line.rstrip("\n").split("\t")
            nf_gcis.append(NF1(child, parent))
            concepts.update((child, parent))
    tbox = NormalFormTBox(frozenset(concepts), frozenset(), nf_gcis)
    vocab = Vocabulary.from_tbox(tbox)

    closure: set[tuple[str, str]] = set()
    with (d / "closure.tsv").open() as f:
        f.readline()
        for line in f:
            a, b = line.rstrip("\n").split("\t")
            closure.add((a, b))

    individuals: set[str] = set()
    memberships: list[MembershipInterval] = []
    with (d / "memberships.tsv").open() as f:
        f.readline()
        for line in f:
            person, position, s, e = line.rstrip("\n").split("\t")
            memberships.append(MembershipInterval(person, position, int(s), int(e)))
            individuals.add(person)

    time_points: list[int] = []
    with (d / "time_points.tsv").open() as f:
        f.readline()
        for line in f:
            time_points.append(int(line.rstrip("\n").split("\t")[1]))

    return WikidataOfficeDataset(
        tbox=tbox, vocab=vocab, individuals=sorted(individuals),
        time_points=time_points, memberships=memberships,
        subsumption_closure=frozenset(closure),
        config=WikidataConfig(**meta["config"]), stats=meta["stats"],
    )
