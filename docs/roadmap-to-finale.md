# Roadmap to the grand finale — 6 December 2026

Written 2026-09-10. Nothing in the brief is being cut; this is the order that makes
"everything" reachable with one operator and one Claude account.

## The ordering rule

There is no parallelism here. The brief's dependency graph assumes five people; with one
account it is a queue, and the only question is sequence. One rule decides it:

> **Anything that changes labels or features goes early.
> Anything that consumes them goes late.**

Because:

- **B1 (IMD rainfall) changes the bust label itself.** Every number Workstream D produces
  is computed against those labels. B1 in November means every confidence interval,
  reliability diagram and economic-value curve is recomputed in December.
- **C4 (district descriptors) changes the feature set**, so it must land before the final
  retrain or the model is trained twice.
- **D5 (ladder rungs) feeds F2**; **D1–D4 feed F3 and F4.** Building those screens before
  D exists means building them twice.
- **F1 (boundary depiction) is research that could force map rework.** It costs nothing
  to start now and a great deal to discover in November.

## Sequence

### Now — this week
- **C5 retrain at district grain.** In flight. Until it works the product is either 36
  districts or saturated nonsense.
- **A4 branch protection on `main`.** Browser, 5 minutes. A2 is done.
- **A1** push the six local commits to `develop`, open the PR. Tag what production
  actually runs (`a24a663`), not `d0916bb` — that is 21 commits behind it.
- **A5** serving memory check. Set the budget from a fresh measurement; the 440 MB in the
  brief is below the 442 MB already recorded in `known-issues.md`, and the CI harness
  cannot resolve below ~±55 MB. Measure first, then pick the number.

### Weeks 1–3 — things that change the label
- **B1 IMD gauge rainfall.** The highest-value change in the project and the answer to the
  reviewer's first attack. Budget real time for the 0830 IST accumulation convention —
  that is exactly the class of problem that has already cost this project a day.
- **F1 boundary research.** No code. Runs alongside B1 because it is reading, not building.
- **B2 finish.** Largely done via CDS rather than Zarr; the native-grid limitation is
  already beaten. What remains is Z500 for the synoptic bust definition.

### Weeks 3–5 — things that change the features
- **C1 jumpiness**, **C2 time-lagged ensemble**, **C3 MJO/MISO**, **C4 district
  descriptors.** Score the ladder before and after *each* family so it is clear which
  earned its place.
- Then the retrain that the rest of the project quotes.

### Weeks 5–8 — verification, on labels and features that have stopped moving
- **D1** block-bootstrap CIs, resampled **by cycle**, never by row.
- **D2** binormal Z-AUC alongside trapezoidal.
- **D3** CORP reliability diagrams and the Brier decomposition.
- **D4** relative economic value and SEDI.
- **D5** logistic regression, EMOS, IDR, analog — scored through the existing gate.

### Weeks 8–10 — make it visible
- **F2** ladder to the first screen, with error bars from D1 and the CNN's refusal shown.
- **F4** CORP diagram, **F3** economic-value slider, **F5** disaster replay (Ockhi,
  Chennai 2015, Kerala 2018), **F6** attribution, **F7** IMD badge.

### Weeks 10–11 — the ambitious tail
- **D6 conformal prediction**, **B4 IPED**, **A3 workflow split.**
  These are last not because they are unimportant but because everything above is either
  a prerequisite or a larger score. If the schedule slips, it shows up here, visibly,
  with time left to react.

### Week 12 — freeze
No new code. Rehearse the demo, re-run the suite, re-read the limitations doc.

## Standing constraints

- **Workstream E is finished.** The CNN was trained on real data, scored on identical
  held-out rows and lost by roughly 4x the seed spread; the unmodified gate refused it.
  Reopening it costs time and changes no conclusion.
- **Batch the long jobs.** A 3-hour ingest ties up the one account. Start those when
  stepping away; use desk time for work that needs the back-and-forth.
- **Never spend the account on browser work.** Branch protection, dashboard settings and
  registrations are minutes by hand and expensive by agent.
