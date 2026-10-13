# ltel-box: faithful box embeddings for Linear Temporal EL+

Experiment code for the paper *Faithful Box Embeddings for Linear Temporal EL+* (NeSy 2026). The package lifts a faithful static box embedding method (BoxEL, Box²EL or TransBox) to the temporal description logic LTEL by compiling each temporal axiom to box containments over a lasso of time points. The paper calls this construction the generic temporal lifting; the method is named LassoBox.

The machine-checked proofs live in a separate Lean 4 repository. Both repositories are archived together in the DaRUS deposit [doi:10.18419/DARUS-6407](https://doi.org/10.18419/DARUS-6407), which also holds the materialized datasets and the per-run logs behind the paper's tables and figures. See `CITATION.cff` for how to cite.

## Layout

- `src/ltel_box/` — the package: LTEL syntax (`ontology.py`), normalization and polarity check (`normalize.py`), the grounding oracle (`oracle.py`, `closure.py`), the temporal box model on the three base methods (`model.py`, `loss.py`), training (`trainer.py`, `slice_trainer.py`), the synthetic TBox generator (`synthetic.py`) and the Wikidata offices slice (`wikidata.py`).
- `scripts/` — experiment drivers (`rq1_synthetic.py`, `rq2_sweep.py`, `rq1_breakdowns.py`), the Wikidata extraction pipeline (`scripts/wikidata/`, see its README), the ELK cross-check (`scripts/oracle/`), the dataset materializer and the CSV exporter.
- `tests/` — pytest suite; CPU only, no GPU or network required.
- `figure_data/` — curated result CSVs that the paper's figures read directly (see its README for the table-to-run mapping).
- `docs/` — `base-method.md` (which static base method each result rests on, and what is machine-checked) and `reproducibility.md` (pinned versions, hyperparameters, seeds, data provenance).
- `results/`, `logs/` — produced by runs; large outputs are git-ignored. The complete run records are in the deposit's `runs.tar.gz`.

## Installation

Python 3.10 or newer and PyTorch 2.1 or newer.

```
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
```

## Running

- `make test` runs the pytest suite on the CPU. Run it before committing.
- `make lint` runs `ruff check src/ tests/` (line length 100, target Python 3.10).
- `make datasets` materializes the synthetic TBoxes and oracle closures.
- `python scripts/rq1_synthetic.py --help` and `python scripts/rq2_sweep.py --help` list the experiment options. Every run is seeded; the mapping from paper tables and figures to result directories is in `figure_data/README.md`.
- The `sync-*` Makefile targets rsync code and results to the authors' compute hosts and are not needed to reproduce the experiments.

## Conventions

Type hints throughout, ruff-clean, deterministic seeds. Keep commits small and messaged by intent. Development tooling of individual contributors (editor or assistant configuration) is not tracked in this repository.

## License

MIT, see `LICENSE`.
