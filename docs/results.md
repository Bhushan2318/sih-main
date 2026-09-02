# Sanket — measured results

## Baselines

_Generated 2026-09-02 00:21 UTC · run `run_20260902T001857Z` · git `ea466f0`_

Held-out test split: **1,050 events** over **3 forecast cycles**, bust rate **0.430**. Baselines fitted on the 4,200 training events (12 cycles) and evaluated on rows they never saw. Brier skill score is against climatology — the training base rate — so 0.000 means no skill beyond knowing how often busts happen.

A bust is defined against the 90th percentile of each variable's own error distribution, not an absolute error, so the label does not simply grow with lead time: measured correlation between lead day and bust is **-0.075 on train** and **+0.070 on test**. That weak, sign-flipping relationship is why a lead-day-only model can score below chance here — and it is also direct evidence that the classifier's skill is not simply a rediscovery of lead time.

| model | Brier ↓ | BSS vs climatology ↑ | ROC-AUC ↑ | F1 ↑ |
|---|---|---|---|---|
| climatology | 0.2452 | 0.0000 | 0.5000 | 0.0000 |
| lead_day | 0.2492 | -0.0162 | 0.4592 | 0.0000 |
| spread | 0.2461 | -0.0035 | 0.5427 | 0.2682 |
| lead+spread+season | 0.2553 | -0.0411 | 0.5127 | 0.2157 |
| **Sanket bust classifier** | 0.1949 | 0.2052 | 0.7621 | 0.6628 |

**Brier skill score by lead day** (lead days with under 20 test events, or no bust, are omitted):

| model | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| climatology | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lead_day | -0.042 | -0.023 | 0.001 | -0.006 | -0.001 | 0.007 | -0.003 | -0.004 | -0.022 | -0.064 |
| spread | 0.012 | 0.008 | 0.025 | -0.016 | -0.042 | 0.026 | 0.018 | -0.039 | -0.016 | -0.008 |
| lead+spread+season | -0.012 | 0.013 | 0.001 | -0.023 | -0.028 | 0.013 | -0.026 | -0.076 | -0.093 | -0.168 |
| **Sanket bust classifier** | 0.228 | 0.391 | 0.398 | 0.189 | 0.037 | 0.114 | 0.175 | 0.144 | 0.156 | 0.216 |
