# figure_data/ — render-ready result tables for the paper figures

Curated CSVs of the experiment results, for `@writer` to consume directly in
TikZ/PGF (`\pgfplotstableread` / `\addplot table{...}`). **Data only** — the
figures themselves (TikZ/PGF) are part of the writing process and live in the
paper repo; this directory is the `@coder`→`@writer` data handover.

Regenerate any table with `scripts/export_results_csv.py` from the corresponding
`results/*/summary.json` (the summaries are gitignored/regenerable; these small
tables are committed as the deliverable).

| File | Experiment | Status |
|---|---|---|
| `slice.csv` | Offices feasibility slice (§5) | **final** — role-free, on the faithful core |
| `rq2_multi.csv` | RQ2 A-vs-B, non-rigid = multi-interval (headline) | **final** — A flat 1.0; B's L1 (rigid)=1.0 but L3 (multi-interval)≈0.85; B ties A at ρ=1 |
| `rq2_single.csv` | RQ2 A-vs-B, non-rigid = single-interval | **final** — note: per-membership single intervals still aggregate to non-convex *concept-level* support across individuals, so B also declines here (less steeply); the clean convex control is ρ=1 + the isolated {1,4,7} test |
| `rq1_box2el.csv` | RQ1 recovery/spurious, faithful Box²EL base | re-running with **NF4 = exact Box²EL** (@prover-confirmed, Head form) and **NF5/RIAs dropped** (box embeddings don't realise composition); supersedes the interim version |

Columns:
- `slice.csv`: concepts, ticks, individuals, completeness_sampled, spurious_rate_sampled, loss_subsumption, loss_membership
- `rq2_*.csv`: rho, model (A/B), recovery, recovery_L1/2/3, out_loss, total_loss
- `rq1_box2el.csv`: data_seed, entailed, mean_loss, mean_completeness, mean_spurious_rate, intersection_completeness, intersection_spurious_rate (+ a MEAN row)
