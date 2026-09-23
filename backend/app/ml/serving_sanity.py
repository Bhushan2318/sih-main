"""Does the candidate model still say different things about different districts?

The promotion gate compares held-out ROC-AUC. That catches a model that has got worse at
ranking, and it is the right first question. It cannot catch a model that ranks perfectly
well and serves one number, because ROC-AUC is invariant under any monotone transform of
the scores: crush every probability to within a hair of 1 and the ranking, and therefore
the AUC, does not move at all.

That is not hypothetical. run_20260910T064804Z held out at ROC-AUC 0.8411 and Brier
0.1656, with its held-out probabilities spread evenly over the five calibration bins
(11.3 / 23.2 / 16.5 / 20.9 / 28.1 per cent). It was promoted on those numbers, and it
served 642 of 666 districts in the bust band at a median probability of 0.977. Both
things were true simultaneously. The held-out rows were the 34 districts that had
observations at training time, scored against the observation set that existed then; the
store has since been replaced with 666-district ERA5-CDS. The model was never wrong about
its own test set - it was wrong about the world it was asked to serve.

So the check has to run the real serving path over the real store and look at what comes
out. Measure, do not estimate (CLAUDE.md rule 5), applied to the gate itself.

WHAT THIS DOES NOT MEASURE, and why
-----------------------------------
The obvious criteria both fail on real data, so neither is used here. Measured over the
2018-12-31 cycle, all ten lead days, 6,660 events:

                       median    IQR    >0.99   largest band
  run_20260910 broken   0.977  0.054    46.2%   high 93.8%
  run_20260918 broken   0.959  0.073    25.6%   high 87.0%
  run_20260922 good     0.187  0.119     0.0%   low  90.4%

"Concentrated in one band" is not the signal: the good model is 90.4% low, which is
tighter than one of the broken models. A quiet day is supposed to look like that.
Interquartile range is not the signal either - the good model's is 0.119, *narrower* than
you might expect and well inside the range a naive floor would refuse. Refusing on either
would have thrown away the run that actually fixed the problem.

What does separate them is which rail the mass is pinned against. A working model on a
quiet day piles up near zero and keeps a thin upper tail; a broken one piles up against
one, and the bust band swallows the country. So the checks below are asymmetric on
purpose, and the bust-band share is compared against what the band cuts are built to
produce: the cuts are the 50th and 80th percentiles of the validation predictions, so
roughly a fifth of events should land in the top band by construction. Three times that
is damage, not weather.

WHAT THIS CANNOT CATCH, and why that is not a hole to widen
-----------------------------------------------------------
Measured 2026-09-23 on run_20260922T100055Z, whose temperature regressor is broken: it
predicts absolute temperature errors from -2218.9 to +438.99 where the true range is 0 to
2.69, held-out r2 -682,626. Scored through this same path on a real cycle it PASSES, and
its served distribution is nearly identical to the good model's - median 0.279 against
0.287, IQR 0.401 against 0.407, bust band 13.1% against 13.4%.

It passes because the classifier was TRAINED on those broken values. Its split thresholds
sit wherever that distribution put them, and at serving time the same values land in the
same leaves. The model is internally consistent and externally nonsense, so the
distribution it produces is not degenerate and there is nothing here to see.

Note the mechanism precisely, because a plausible wrong version of it circulated first.
`conf` does saturate - `(1.0 - pred_err/p90).clip(0.0, 1.0)` is 0.0 for every pred_err at
or above p90, so 4.7 and 2218.9 are indistinguishable through that feature - but the
classifier reads `pred_err_*` directly as well as `conf_*`, so the magnitude does reach
it. Saturation absorbs part of it; having been trained on it absorbs the rest.

That draws the line this check actually sits on:
  - broken BEFORE training, so the model learned the broken values: internally consistent,
    invisible here. It belongs at the regressor stage, where a held-out r2 worse than
    predicting the mean should refuse the artifact outright - one comparison, no store, no
    scoring, and it would have caught this at training time.
  - broken AFTER training, because the store changed underneath it: inconsistent, and this
    is what catches it. run_20260910T064804Z was that case.

Widening this to cover the first would turn a distribution check into a second gate with a
second set of thresholds, policing a cause it cannot observe.

WHAT IT COSTS
-------------
Measured on the real store, 666 districts x 10 lead days, scoring one cycle end to end:
16.5 s wall, peak RSS 1,443 MB against a 154 MB baseline. That is a CI-only cost - this
runs inside `full_retrain`, after training has finished and freed its frame, on the 16 GB
runner. It must never be called on the serving box, which is killed at 512 MB.
"""

from __future__ import annotations

import numpy as np

# Share of events allowed in the top (bust) band before this is read as damage. The cuts
# put ~20% there by construction; the broken runs served 93.8% and 87.0%, the working one
# 1.8%. 0.60 sits in the gap with room on both sides, nearer the broken end because this
# refuses a model - it should fire on damage rather than on an unusual but working day.
MAX_BUST_BAND_SHARE = 0.60

# Share allowed to sit against either rail. Saturation shows here before it reaches the
# bands, and it catches a model whose cuts are themselves degenerate. Broken: 46.2% and
# 25.6% above 0.99. Working: 0.0% above, 0.0% below.
MAX_RAIL_SHARE = 0.10

# A model returning one constant. Deliberately far below the 0.119 the working run
# serves: this is here to catch a flat line, not to judge how spread out a real day is.
MIN_SERVED_IQR = 0.02

# Fewer scored districts than this and the distribution cannot show degeneracy either
# way, so there is nothing to pass. Refuse rather than wave it through: a scoring path
# that suddenly returns a handful of rows is itself the kind of failure this exists to
# notice - see the serving check that read a top-level `regions` key from a payload that
# nests its rows under `days[]`, and refused a model that was scoring fine.
MIN_SCORED_DISTRICTS = 50

BUST_BAND = "high"


def degeneracy_verdict(probabilities, bands=None) -> tuple[bool, str]:
    """Whether a served probability distribution is usable, and why.

    `bands` is the risk band per event, when the caller has them; without it the band
    check is skipped and only the rail and flatness checks run.

    Returns (ok, reason). The reason always carries the numbers it decided on, because a
    refusal that cannot be acted on only moves the confusion somewhere else.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    finite = np.isfinite(p)
    if bands is not None:
        bands = np.asarray(bands, dtype=object).ravel()[finite]
    p = p[finite]

    if p.size == 0:
        return False, ("no scored districts: the serving path returned nothing finite to "
                       "check. This is a scoring failure, not a quiet model.")

    if p.size < MIN_SCORED_DISTRICTS:
        return False, (f"too few scored districts: {p.size} of at least "
                       f"{MIN_SCORED_DISTRICTS} needed before a distribution means "
                       f"anything. Check what the serving path returned before reading "
                       f"this as a verdict on the model.")

    q1, med, q3 = (float(x) for x in np.percentile(p, [25, 50, 75]))
    iqr = q3 - q1
    high_rail = float(np.mean(p > 0.99))
    low_rail = float(np.mean(p < 0.01))
    shape = (f"median {med:.3f}, quartiles {q1:.3f}/{q3:.3f}, {100 * high_rail:.1f}% "
             f"above 0.99, {100 * low_rail:.1f}% below 0.01, over {p.size} scored events")

    why_rank = ("Held-out ROC-AUC is not evidence against this: ROC-AUC is rank-based and "
                "does not move when probabilities are crushed together. The usual cause is "
                "a model trained against a different observation set than the one now in "
                "the store.")

    if bands is not None and bands.size == p.size:
        share = float(np.mean(bands == BUST_BAND))
        if share > MAX_BUST_BAND_SHARE:
            return False, (
                f"NOT promoted: {100 * share:.1f}% of scored events land in the "
                f"'{BUST_BAND}' band, above the ceiling of {100 * MAX_BUST_BAND_SHARE:.0f}%. "
                f"The band cuts are the 50th and 80th percentiles of this model's own "
                f"validation predictions, so about 20% belongs there by construction. "
                f"{shape.capitalize()}. {why_rank}"
            )

    if high_rail > MAX_RAIL_SHARE or low_rail > MAX_RAIL_SHARE:
        rail = "1.0" if high_rail > MAX_RAIL_SHARE else "0.0"
        worst = max(high_rail, low_rail)
        return False, (
            f"NOT promoted: {100 * worst:.1f}% of served probabilities are pinned against "
            f"{rail}, above the ceiling of {100 * MAX_RAIL_SHARE:.0f}%. A model saturated "
            f"at a rail has stopped distinguishing between districts. {shape.capitalize()}. "
            f"{why_rank}"
        )

    if iqr <= MIN_SERVED_IQR:
        return False, (
            f"NOT promoted: the served probabilities are effectively one constant - "
            f"interquartile range {iqr:.4f}, at or below {MIN_SERVED_IQR}. {shape.capitalize()}. "
            f"{why_rank}"
        )

    return True, f"served distribution looks usable: {shape}, interquartile range {iqr:.3f}."


def served_probabilities(state, init_date=None):
    """Score a real cycle from the store with `state`; return (probabilities, bands).

    Imported lazily: this module is imported by the training pipeline, and
    `app.ml.inference` pulls in the serving stack, which training has no other reason to
    hold in memory.
    """
    from app.ml import inference

    scored = inference.score_cycle(state, init_date=init_date)
    if scored is None or scored.events.empty:
        return np.array([], dtype=float), np.array([], dtype=object)
    ev = scored.events
    bands = (ev["risk_band"].to_numpy(dtype=object) if "risk_band" in ev
             else np.array([], dtype=object))
    return ev["bust_probability"].to_numpy(dtype=float), bands


def check_served_model(state, init_date=None) -> tuple[bool, str]:
    """The whole check: score a real cycle, judge the distribution it produces."""
    p, bands = served_probabilities(state, init_date=init_date)
    return degeneracy_verdict(p, bands if bands.size else None)
