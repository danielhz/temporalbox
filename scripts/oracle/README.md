# Oracle: ground-then-classify + ELK cross-check

The RQ1 ground truth (which temporal subsumptions an LTEL TBox entails) comes
from `ltel_box.oracle`: ground the normal-form TBox to a static EL+ ontology
over the lasso (§3.3 compilation), then classify. Two independent classifiers
run on the *same* grounded ontology and must agree:

- **In-house** forward-chaining EL+ closure (`ltel_box.closure`) — used in the
  eval loop (fast, no JVM).
- **ELK 0.6.0** (reasoner of record; EL+ is its native fragment) via the OWL
  functional-syntax export.

## Reasoner of record (pinned)

- **ELK 0.6.0**, `io.github.liveontologies:elk-distribution-cli:0.6.0`
  (Apache-2.0). The runnable `elk.jar` lives inside the distribution **zip**
  (the bare `-cli.jar` on Maven is an aggregator pom). `run_elk.sh` fetches and
  unpacks it on first use; Java 11+ required (tested on 17).

## Cross-check

```bash
ELK_JAR=scripts/oracle/tools/elk-distribution-cli-0.6.0/elk.jar \
    python scripts/oracle/crosscheck_elk.py --seeds 0..24 --k 3 --loop 1
```

For each synthetic seed it grounds the TBox, classifies with both, transitively
closes ELK's (reduced) taxonomy, and asserts the named-class entailment sets
are identical. **Status: 25/25 seeds agree** across lasso shapes (k=3 ℓ=1 and
k=4 ℓ=2).

### Bug this caught

The first run disagreed on 2/10 seeds — ELK derived *more* pairs than the
in-house closure. The closure was missing the **CR5 role-composition rule**
(`A ⊑ ∃r.B, B ⊑ ∃s.E, r∘s⊑t ⇒ A ⊑ ∃t.E`); seeds with `r∘s⊑t` axioms
under-derived. Fixed in `closure.py` and guarded by
`tests/test_closure.py::test_role_composition_cr5`. The same closure generates
the Wikidata subsumption ground truth, so the fix matters beyond RQ1.

## Independence (no circularity)

`oracle.py` re-implements the lasso arithmetic (σ/future/V) separately from the
construction side (`model.py`/`trainer.py`): a shared bug there would make the
RQ1 validity check vacuous. The ELK cross-check additionally guards the closure
step against the in-house implementation.
