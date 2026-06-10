# Wikidata temporal subset — extraction and domain profiling

Profiling of candidate domains for the LTEL real-data experiment (Phase 2).
This records what is on kg1, how the subset was extracted, and the density
profile that motivates the seed-domain choice. All numbers from the pinned
dump below; reproduce with the queries in `queries/`.

## Pinned data source

- **Dump:** Wikidata 2026-04 full RDF dump, `latest-all.nt.bz2`
  (N-Triples, the `-all` export with the full statement/qualifier model).
- **Path (kg1):** `/data/ki_nfs_data0/ac/wikidata-dump-2026-04/latest-all.nt.bz2`
- **Size:** 191,612,079,272 bytes compressed; file date 2026-04-10.
- **Disk:** the dump lives on the *remote* NFS mount
  (`ki-nfs.tik.uni-stuttgart.de:/ki_nfs_data0`), which is the slow/archival
  disk — hence we stream it exactly once. (A 2023-05-22 Jena TDB2 temporal
  index also exists on kg1's fast local disk but is owned by another user and
  not writable, so TDB2 cannot open it; the dump path is dependency-free and
  gives fresher 2026-04 data.)

## Why the RDF dump, not the JSON

The temporal qualifiers (`start time` P580 / `end time` P582 /
`point in time` P585) live only in the full statement model
(`p:` / `ps:` / `pq:`), which the `-all` N-Triples export carries and the
`-truthy` export (direct `wdt:` triples only) does not. N-Triples is
line-oriented, so a streaming `grep` filter works; the JSON dump would need a
streaming parser. Statement node IDs encode their entity QID
(`.../statement/Q31-…`), so the subject entity is recoverable without a join.

## Extraction pipeline (one streaming pass)

```
lbzip2 -dc -n 6 latest-all.nt.bz2 | grep -F -f patterns.txt | pigz > subset.nt.gz
```

- `lbzip2` — parallel bzip2 decompress (6 of 16 cores, polite on the shared box).
- `grep -F -f patterns.txt` — fixed-string filter on the **predicate slot**
  (patterns anchored with surrounding spaces). Predicates kept: `p:P39`,
  `ps:P39`, `p:P54`, `ps:P54`, `pq:P580`, `pq:P582`, `pq:P585`, `pq:P155`,
  `pq:P156`, `pq:P1365`, `pq:P1366`, `wdt:P279`.
- Result: **51,501,678 triples, 1.1 GB gz**. Streaming pass wall time ≈ 2h40m
  (NFS-read bound).

The subset is then loaded into a local **oxigraph** store (own store, on kg1
fast local disk — no permission issue) and profiled with SPARQL:

```
pigz -dc subset.nt.gz | oxigraph_server -l ./store load --format nt --lenient   # 51.5M triples in 256s
oxigraph_server -l ./store query --query-file queries/<q>.rq --results-format tsv
```

## Domain profile (2026-04 dump)

| Metric | Domain 1 — political office (P39) | Domain 2 — sports team (P54) |
|---|---:|---:|
| Statements | 1,844,856 | 1,758,817 |
| Distinct people | 956,929 | 646,463 |
| Distinct positions / teams | 144,196 | 50,364 |
| … with a P279 superclass | 124,328 (86%) | — |
| With start (P580) | 1,171,479 (63%) | 1,248,128 (71%) |
| With end (P582) | 871,639 (47%) | 1,112,014 (63%) |
| **With full interval (start+end)** | **860,022 (47%)** | **1,103,521 (63%)** |
| With point-in-time (P585) | 7,997 | 7,290 |
| **With succession (P1365/P1366)** | **325,920 (18%)** | **5 (~0%)** |

Backbone (whole subset): **5,171,744** `wdt:P279` edges over **4,421,392**
classes with a parent — a large, deep subclass hierarchy for the TBox.

## Reading of the profile

- Both domains are dense enough to train on (≈1.8M statements, hundreds of
  thousands of people each).
- **Domain 1 (political offices) is the recommended seed.** It is the only
  one with explicit **succession links** (325,920 statements carry
  P1365/P1366 replaces/replaced-by), which give ground-truth `○`/`U` chains
  rather than chains inferred from adjacent intervals; and 86% of positions
  sit in the P279 hierarchy, so temporal subsumptions like
  *President of France ⊑ head of state* are recoverable. Interval coverage 47%.
- **Domain 2 (sports) is the high-volume secondary.** Higher interval
  coverage (63%) but essentially no P1365/P1366 succession — "next club" would
  have to be inferred from consecutive memberships, not read off a link.

## Next step

Use the same oxigraph store as the working substrate for the Phase-2
extractor: query it to emit the `NormalFormTBox` (P279 → NF1) plus the
time-qualified membership facts, in both discretisations (calendar-bucketed
and event-based ticks), per the plan in the project `README.org`.
