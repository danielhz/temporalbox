# Base method: which static box embedding the temporal construction sits on

The LTEL temporal embedding is a *lifting* of a static box method: per-time-point
concept boxes glued by the temporal-operator losses, over a lasso. The lifting
only transfers faithfulness if the **static base method** has zero-loss
soundness (every zero-loss embedding is a model). This note records which base
method the code uses, the history, and what each experiment rests on — so we do
not confuse the early translational results with the faithful-base ones.

## The faithful list (per @prover, machine-checked in `lean/LTEL/`)

- **Faithful:** BoxEL (Xiong et al., ISWC 2022), Box²EL (Jackermeier et al.,
  WWW 2024), TransBox.
- **NOT faithful:** ELBE — `@prover` gave a machine-checked counterexample
  (`LTEL/ELBE.lean`): its `∃R.C ⊑ D` loss is an *overlap* (non-disjointness)
  constraint, not containment, so zero loss need not give a model; and its
  `C ⊑ ⊥` loss collapses to a singleton box. (The ⊥ clause is moot for us —
  LTEL is ⊥-free.)
- The **concept lattice** shared by BoxEL/Box²EL/TransBox is `axisBoxAlgebra`:
  axis-parallel boxes in ℝᵈ with coordinate-wise intersection as exact meet.
  This is precisely the NF1 (containment) / NF2 (meet) core in `loss.py`.

## History of this repository's base method

**Phase 1 (initial scaffold, commits up to ~`f39b8dd`): translational roles.**
Roles were a single translation vector δ_r per role:
`NF3  Box(A)+δ_r ⊆ Box(B)`, `NF4  Box(A)−δ_r ⊆ Box(B)`, `NF5  δ_r+δ_s = δ_t`.
The NF1/NF2 concept core was already `axisBoxAlgebra` (the faithful lattice).
The role encoding is *translational* — closest to TransBox, and NOT BoxEL's
affine maps nor Box²EL's dual role-boxes + bumps. Note our NF4 used
**containment** (`Box(A)−δ ⊆ Box(B)`), unlike ELBE's overlap loss, so it is not
the ELBE non-example; but it was never reconciled against the paper's named
base method, and "the simplest faithful encoding" comment in early `loss.py`
overclaimed (translational-role faithfulness was not established here).

**Phase 2 (this change): Box²EL roles** — `model.Box2ELModel`. Per @prover's
Q5 result, the faithful base for the construction is Box²EL: each concept C has
a (per-time, possibly non-rigid) box `Box(C)` AND a **rigid** bump `Bump(C)`;
each role r has **rigid** Head(r)/Tail(r) boxes. Rigidity of bumps and role
boxes is required (Q5): non-rigid concept boxes, rigid relational layer. The
per-normal-form losses (`loss.box2el_*`):

- `NF3  C⊑∃r.D`: `Box(C)⊕Bump(D) ⊆ Head(r)` and `Box(D)⊕Bump(C) ⊆ Tail(r)`.
  Exact Box²EL form.
- `NF4  ∃r.X⊑Y`: `Head(r)⊖Bump(X) ⊆ Box(Y)` — **confirmed by @prover** as the
  exact Box²EL form (an earlier interim wrongly used Tail; Head is correct
  because NF4 constrains the r-*predecessors* of X-fillers, which live in
  `Head(r)⊖Bump(X)`). Sound *containment* (the clause where ELBE was unsound
  via *overlap*).
- `NF5  r∘s⊑t` (role composition / RIA): **DROPPED.** @prover confirmed box
  embeddings realise role *hierarchy*, not composition — head/tail-box inclusion
  does not imply edge chaining, so any box encoding of `r∘s⊑t` is unsound. RQ1
  therefore omits RIAs (`scripts/rq1_synthetic.py --n-nf5 0`, the default); the
  slice and RQ2 are role-free. `loss.box2el_nf5_loss` is retained, marked unsound
  for composition, and not exercised.

**BoxEL base** — `model.BoxELModel` (the method §5 cites, "BoxEL throughout").
Per-time concept boxes + a **rigid** affine role transform per role
(per-dimension scale `s_r>0` + translation `t_r`); `f_r(x)=s_r⊙x+t_r`:
- `NF3 C⊑∃r.D`: `f_r(Box(C)) ⊆ Box(D)`.
- `NF4 ∃r.X⊑Y`: `f_r⁻¹(Box(X)) ⊆ Box(Y)` (predecessors of X under r).
These are **interim** — the exact Xiong-2022 affine convention is being
confirmed with @prover (same process that corrected the Box²EL NF4). BoxEL and
Box²EL are both faithful bases; the lifting is base-agnostic, so RQ1 can report
either (or both). Slice + RQ2 are role-free → base-independent.

**Verification provenance (important for paper prose):** NF1/NF2 = `axisBoxAlgebra`
are **machine-checked** in @prover's Lean; the **NF3/NF4 geometry is the Box²EL
paper's (Thm 2), not machine-checked** — @prover's `t1_soundness` assumes the
base method's static ∃-soundness as a hypothesis. So RQ1 prose must say NF3/NF4
are *exact-from-Box²EL*, not "machine-checked."

## What each experiment rests on — and the faithfulness status of each

| Experiment | Normal forms exercised | Base-method dependence | Status |
|---|---|---|---|
| **Slice (offices, §5)** | NF1 + point-membership only (no roles) | the faithful `axisBoxAlgebra` core + BoxEL-style point individuals | **faithful, unaffected by the role change** |
| **RQ2 (A-vs-B sweep)** | NF1 + point-membership only (no roles) | same faithful core | **faithful, unaffected** |
| **RQ1 (synthetic recovery)** | NF1–NF5 incl. temporal | used the **translational** roles (Phase 1) | **superseded** — re-run on the Box²EL base (Phase 2); see below |

The headline empirical results — slice completeness 1.0, and the RQ2 A-vs-B
support-reconstruction sweep — are **role-free** and therefore already sit on
the faithful BoxEL/Box²EL concept lattice. The role-construction change does not
touch them. Only RQ1 (which uses existentials/role-composition) is re-run.

## Preserved Phase-1 (translational) results — DO NOT confuse with Phase 2

The Phase-1 RQ1 numbers (translational roles) are kept for the record under
`results/rq1/` (and in the agent-messages log). They are **not** the paper's
faithful-base numbers; the paper reports the Phase-2 (Box²EL) RQ1 re-run, stored
under `results/rq1_box2el/`. Any figure/table must cite the Phase-2 run. This
file is the index that disambiguates the two.
