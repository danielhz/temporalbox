# Wikidata political-offices dataset — reproduction pipeline

End-to-end, reproducible regeneration of the LTEL political-offices dataset
(Phase 2) from the pinned Wikidata dump. Committed to satisfy FAIR: every step
needed to reproduce the dataset is here. The large intermediate files
(`subset.nt.gz`, the oxigraph `store/`, the `offices/*.tsv`) are **not**
committed — they are regenerable from these scripts and the pinned source, and
are gitignored.

## Pinned source

- **Dump:** Wikidata 2026-04 full RDF dump, `latest-all.nt.bz2` (N-Triples,
  `-all` export with the full statement/qualifier model).
- **Path (kg1):** `/data/ki_nfs_data0/ac/wikidata-dump-2026-04/latest-all.nt.bz2`
- **Size:** 191,612,079,272 bytes; file date 2026-04-10.

## Tooling (present on kg1)

`lbzip2`, `pigz`, `grep`; `oxigraph_server` 0.3.23 (`~/.cargo/bin`); Python ≥3.10.

## Steps

All of steps 1–3 run on **kg1** (that is where the dump and the disk space
are). Step 4 is pure Python and runs anywhere.

```bash
# 1. Extract the temporally-qualified predicate subset (one streaming pass; ~2h40m, NFS-bound)
bash 01_extract_subset.sh        # -> ~/wd-extract/subset.nt.gz  (51.5M triples, ~1.1 GB)

# 2. Load the subset into a local oxigraph store (~4 min)
bash 02_load_oxigraph.sh         # -> ~/wd-extract/store  (~14 GB)

# 3. Export the office-membership and subclass TSVs via SPARQL (~6 min)
bash 03_export.sh                # -> ~/wd-extract/offices/{memberships,p279}.tsv

# 4. Build the LTEL dataset (deterministic; subclass closure computed in Python)
python 04_build.py \
    --memberships ~/wd-extract/offices/memberships.tsv \
    --p279        ~/wd-extract/offices/p279.tsv \
    --out         ~/wd-extract/dataset-offices-1950 \
    --granularity year --min-holders 50 --min-year 1950 --max-year 2026
```

The build above is the reference configuration used in the paper
(dense positions, modern window). It yields ≈655k memberships over 1,961
positions, 2,786 concepts, 4,007 subclass GCIs, 343k individuals, 77 yearly
ticks, 104,884 entailed subsumptions. Drop `--min-holders`/`--min-year` for the
full slice; use `--granularity event` for the event-based discretisation;
`--root QID` restricts to one office sub-hierarchy.

## Modelling

- Concepts = positions (P39 values) + their P279 ancestors; TBox = the induced
  subclass edges as NF1 GCIs (rigid).
- Individuals = office-holders; a person is a member of a position-concept only
  during the statement's [P580, P582] interval (non-rigid membership).
- Two discretisations: `year` (calendar) and `event` (change-point ticks).

## Provenance of the predicate filter

`patterns.txt` lists the exact predicate IRIs kept in step 1 (P39/P54 office &
team statements, P580/P582/P585 time qualifiers, P155/P156/P1365/P1366
succession, P279 subclass). P54 (sports) is kept for the secondary domain; the
offices builder uses P39 + P279 only.

## Output dataset files (from step 4)

| File | Contents |
|---|---|
| `tbox.tsv` | `child <TAB> parent` — NF1 subclass GCIs |
| `closure.tsv` | `sub <TAB> sup` — entailed subsumptions (EL+ closure) |
| `memberships.tsv` | `person <TAB> position <TAB> start_tick <TAB> end_tick` |
| `time_points.tsv` | `tick_index <TAB> value` |
| `meta.json` | config + summary stats |
