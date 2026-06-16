# Reproducibility summary (for the appendix "Hyperparameters and Reproducibility")

`@coder`→`@writer` handoff: pinned versions, per-experiment hyperparameters,
seeds, and data provenance. All experiments are regenerable from the committed
scripts; see also `docs/base-method.md` (base methods) and
`scripts/wikidata/README.md` (data pipeline).

## Environments & pinned versions

| Component | Version | Where used |
|---|---|---|
| Python | 3.11.3 (GPU runs), 3.10.12 (CPU/data) | training / extraction |
| PyTorch | 2.5.1+cu121 (GPU), 2.12.0+cpu (CPU) | all training |
| oxigraph_server | 0.3.23 | Wikidata subset SPARQL store |
| ELK | 0.6.0 (`io.github.liveontologies:elk-distribution-cli`) | RQ1 oracle cross-check |
| Java | OpenJDK 17.0.19 | ELK |
| GPU | NVIDIA A40 48 GB (slice); A100 80 GB / A40 available via SLURM | slice training |

Determinism: every run fixes `torch.manual_seed`; data seeds and model-init
seeds are reported per experiment below. CPU sweeps pin `OMP/MKL_NUM_THREADS=1`
(reproducible + avoids the tiny-tensor thread-thrash).

## Faithfulness oracle (RQ1)

Ground-then-classify: TBox grounded to static EL+ over the lasso, classified by
(a) the in-house EL+ closure and (b) ELK 0.6.0, cross-checked for agreement —
**25/25 synthetic seeds agree** across lasso shapes (k=3 ℓ=1, k=4 ℓ=2). The
grounding is implemented from the §2.2 semantics, independently of the embedding
loss (no shared machinery).

## Per-experiment hyperparameters

**RQ1 (validity, `scripts/rq1_synthetic.py`)** — bases: Box²EL, BoxEL (reported),
TransBox (extra). dim 16, lasso k=3 ℓ=1, Adam lr 2e-2 → cosine 1e-4, 8000
epochs, eps 1e-4. TBoxes: 30 concepts, 3 roles, gci-scale 0.6, NF5 (RIAs)=2.
Data seeds 0–4; 3 model inits per TBox (across-init intersection = the
universal-quantifier proxy).

**RQ1 breakdowns (`scripts/rq1_breakdowns.py`)** — base Box²EL, dim 16, k=3 ℓ=1,
6000 epochs, 20 concepts, seeds 0–2. By operator: one temporal NF isolated over
a static NF1–NF4 backbone. By rigid fraction: `generate_rq2` TBoxes (confound
rule), ρ ∈ {0,.25,.5,.75,1}. Secondary MRR/Hits@{1,3,10} (filtered ranking).

**RQ2 (faithfulness pays off, `scripts/rq2_sweep.py`)** — A vs B, dim 32, lasso
k=7, Adam lr 2e-2 → cosine 1e-4, 3000 epochs, offset-reg 1e-3, closed-world
negatives ratio 2. 24 concepts, 60 individuals, 4 memberships each. ρ ∈
{0,.25,.5,.75,1}, seeds 0–2; support_max_intervals 3 (multi, headline) and 1
(single, companion). Cost panel reports free-parameter count (rigid collapse)
and training time.

**Slice (offices feasibility, `scripts/slice_offices.py`)** — base = per-time
boxes; event-tick discretisation; root Q48352 (head of state), years clipped
[-1000, 2026]; dim 64, 4000 epochs, lr 2e-2, offset-reg 1e-3; sampled evaluation
(5000 closure pairs × 16 ticks; 5000 negatives).

## Data provenance

- **Wikidata 2026-04 dump** `latest-all.nt.bz2` (191,612,079,272 bytes, dumped
  2026-04-08/10), from <https://dumps.wikimedia.org/wikidatawiki/entities/>.
  Pipeline: `scripts/wikidata/01–06`. Predicate subset, oxigraph store, SPARQL
  export, builder — all committed; large intermediates regenerable, not committed.
- Synthetic RQ1/RQ2 data is generated in-code from the listed seeds (no external
  data).

## Result tables

Committed render-ready in `figure_data/` (`rq1_box2el.csv`, `rq1_boxel.csv`,
`rq1_breakdowns_by_{operator,rho}.csv`, `rq2_{multi,single}.csv`, `slice.csv`);
regenerate via `scripts/export_results_csv.py`. Raw per-epoch logs and
`summary.json` are gitignored/regenerable.
