"""Tests for the Wikidata political-offices LTEL builder."""

from pathlib import Path

from ltel_box.wikidata import (
    WikidataConfig, build_dataset, save_dataset, load_dataset, _qid, _year,
)

FIX = Path(__file__).parent / "fixtures" / "wikidata"
MEM = FIX / "memberships.tsv"
P279 = FIX / "p279.tsv"


def test_qid_and_year_parsing():
    assert _qid("<http://www.wikidata.org/entity/Q42>") == "Q42"
    assert _qid("") is None
    assert _year('"2000-01-01T00:00:00Z"^^<...>') == 2000
    assert _year('"-0044-03-15T00:00:00Z"^^<...>') == -44
    assert _year("") is None


def test_concept_universe_excludes_unused_positions():
    ds = build_dataset(MEM, P279, WikidataConfig(granularity="year"))
    # Used positions Q100,Q200,Q300 + ancestors Q400,Q900; Q999 is unused.
    assert ds.tbox.concept_names == {"Q100", "Q200", "Q300", "Q400", "Q900"}
    assert "Q999" not in ds.tbox.concept_names


def test_tbox_edges_and_closure():
    ds = build_dataset(MEM, P279, WikidataConfig(granularity="year"))
    edges = {(g.lhs, g.rhs) for g in ds.tbox.nf_gcis}
    assert edges == {("Q100", "Q300"), ("Q300", "Q900"),
                     ("Q200", "Q400"), ("Q400", "Q900")}
    # transitive subsumptions must be in the closure
    assert ("Q100", "Q900") in ds.subsumption_closure
    assert ("Q200", "Q900") in ds.subsumption_closure


def test_individuals_and_timeline():
    ds = build_dataset(MEM, P279, WikidataConfig(granularity="year"))
    assert set(ds.individuals) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    # years span 1998..2012 at yearly granularity
    assert ds.time_points[0] == 1998 and ds.time_points[-1] == 2012
    # Q1 (Q100, 2000-2005) should be active in 2003 but not 1999
    def active(year):
        idx = ds.time_points.index(year)
        return {(p, c) for (p, c) in ds.memberships_at(idx)}
    assert ("Q1", "Q100") in active(2003)
    assert ("Q1", "Q100") not in active(1999)


def test_event_granularity_ticks_sorted_unique():
    ds = build_dataset(MEM, P279, WikidataConfig(granularity="event"))
    assert ds.time_points == sorted(set(ds.time_points))


def test_min_holders_filter():
    # Q100 has 3 holders, Q200/Q300 have 1 each.
    ds = build_dataset(MEM, P279, WikidataConfig(min_holders=2))
    assert "Q100" in ds.tbox.concept_names
    assert "Q200" not in ds.tbox.concept_names  # filtered out (1 holder)


def test_save_roundtrip(tmp_path):
    ds = build_dataset(MEM, P279, WikidataConfig(granularity="year"))
    save_dataset(ds, tmp_path)
    for fn in ("tbox.tsv", "closure.tsv", "memberships.tsv", "time_points.tsv", "meta.json"):
        assert (tmp_path / fn).exists()
    lines = (tmp_path / "tbox.tsv").read_text().splitlines()
    assert lines[0] == "child\tparent" and len(lines) == 1 + len(ds.tbox.nf_gcis)


def test_save_load_roundtrip(tmp_path):
    ds = build_dataset(MEM, P279, WikidataConfig(granularity="year"))
    save_dataset(ds, tmp_path)
    ds2 = load_dataset(tmp_path)
    assert ds2.tbox.concept_names == ds.tbox.concept_names
    assert {(g.lhs, g.rhs) for g in ds2.tbox.nf_gcis} == {(g.lhs, g.rhs) for g in ds.tbox.nf_gcis}
    assert ds2.subsumption_closure == ds.subsumption_closure
    assert set(ds2.individuals) == set(ds.individuals)
    assert ds2.time_points == ds.time_points
    assert len(ds2.memberships) == len(ds.memberships)


if __name__ == "__main__":  # allow running without pytest
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            import inspect
            if "tmp_path" in inspect.signature(fn).parameters:
                import tempfile
                fn(Path(tempfile.mkdtemp()))
            else:
                fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
