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
| `rq2_multi.csv` | RQ2 A-vs-B, non-rigid = multi-interval (the headline sweep) | added when the full sweep lands |
| `rq2_single.csv` | RQ2 A-vs-B, non-rigid = single-interval (level-2 control) | added when the full sweep lands |
| `rq1_box2el.csv` | RQ1 recovery/spurious, faithful Box²EL base | **PRELIMINARY** — uses interim NF4/role-composition (NF3 exact); final pending @prover's exact Box²EL conditions. See `docs/base-method.md`. |

Columns:
- `slice.csv`: concepts, ticks, individuals, completeness_sampled, spurious_rate_sampled, loss_subsumption, loss_membership
- `rq2_*.csv`: rho, model (A/B), recovery, recovery_L1/2/3, out_loss, total_loss
- `rq1_box2el.csv`: data_seed, entailed, mean_loss, mean_completeness, mean_spurious_rate, intersection_completeness, intersection_spurious_rate (+ a MEAN row)
