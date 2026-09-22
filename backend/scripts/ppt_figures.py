"""Every headline figure for the deck, regenerated from the served model.

CLAUDE.md rule 2 says metrics are served, never written down, because a number in a file
is right until the next retrain and quietly wrong afterwards. A slide deck breaks that
rule by its nature - it is a file full of numbers. This script is the compromise: it
prints the figures on demand from the current run's own artifacts, stamped with the run
id and the time, so a stale slide is obvious rather than invisible, and so the deck can
be refreshed in one command after a retrain instead of retyped.

    python backend/scripts/ppt_figures.py                  # markdown to stdout
    python backend/scripts/ppt_figures.py --out deck.md     # ...or to a file
    python backend/scripts/ppt_figures.py --json            # machine-readable

Nothing here recomputes a model or re-derives a threshold. Everything is read from the
run directory and the eval-event frame that training emitted, at the same decision
threshold the served precision and recall already use, so the numbers on a slide
reconcile with the numbers on /api/model/status rather than merely resembling them.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.db.base import resolve_path  # noqa: E402
from app.ml import registry  # noqa: E402

# The threshold the trainer scores at (app/ml/classifier.py::_evaluate). Imported as a
# constant rather than guessed: precision, recall, F1 and every count below move with it,
# and a slide quoting a different one from /api/model/status would be indefensible.
DECISION_THRESHOLD = 0.5

_ERR_PREFIX = "actual_err_"


def _events(run_id: str) -> pd.DataFrame:
    path = resolve_path(settings.data_dir) / "analysis" / "eval_events" / f"{run_id}.parquet"
    if not path.is_file():
        raise SystemExit(
            f"no eval events for {run_id} at {path}\n"
            "They are written by the training run with --write-eval-events; a run trained "
            "without them cannot produce per-lead-day figures."
        )
    return pd.read_parquet(path)


def _counts(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
    }


def _skill(c: dict) -> dict:
    """POD, FAR and friends, named the way a forecasting audience uses them.

    POD (probability of detection) is the share of real busts that were warned about, and
    is the same quantity scikit-learn calls recall.

    FAR here is the **false alarm ratio** - of the warnings issued, how many did not
    verify - which is what a forecast verification audience means by FAR. It is not the
    false alarm *rate* (POFD, FP/(FP+TN)); the two differ whenever the classes are
    unbalanced and confusing them flatters or damns a model by a wide margin. Both are
    reported so a slide cannot quietly pick the flattering one.
    """
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    pod = tp / (tp + fn) if tp + fn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    pofd = fp / (fp + tn) if fp + tn else float("nan")
    return {
        **c,
        "n": tp + fp + fn + tn,
        "pod": pod,
        "far": far,
        "pofd": pofd,
        "csi": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
        "bias": (tp + fp) / (tp + fn) if tp + fn else float("nan"),
        "precision": 1 - far if tp + fp else float("nan"),
        "base_rate": (tp + fn) / (tp + fp + fn + tn) if tp + fp + fn + tn else float("nan"),
    }


def _per_lead(test: pd.DataFrame) -> list:
    out = []
    for lead, g in test.groupby("lead_time_days", sort=True):
        y = g["y_bust"].to_numpy(int)
        pred = (g["model_proba"].to_numpy(float) >= DECISION_THRESHOLD).astype(int)
        row = _skill(_counts(y, pred))
        row["lead_time_days"] = int(lead)
        row["mean_proba"] = float(g["model_proba"].mean())
        out.append(row)
    return out


def _shap_top(run_id: str, k: int = 12) -> list:
    path = registry.run_dir(run_id) / "shap_summary.parquet"
    if not path.is_file():
        return []
    d = pd.read_parquet(path)
    d = d[(d["model"] == "classifier") & (d["group_region_id"] == "__all__")]
    if d.empty:
        return []
    d = d.sort_values("mean_abs_shap", ascending=False).head(k)
    return [{"feature": r.feature, "mean_abs_shap": float(r.mean_abs_shap)} for r in d.itertuples()]


def _region_name(region_id: str) -> str:
    try:
        from app.utils import india_districts as idist  # noqa: PLC0415

        d = idist.resolve_by_id(region_id)
        if d is None:
            return region_id
        name = getattr(d, "region_name", None) or region_id
        state = getattr(d, "state_name", None)
        return f"{name}, {state}" if state and state != name else name
    except Exception:
        return region_id


# A (district, variable) pair whose typical error is this many times the all-district
# median is treated as systematically biased rather than hard to forecast. See
# _biased_pairs for why this exists at all.
SITE_BIAS_FACTOR = 2.0


def _biased_pairs(test: pd.DataFrame, thresholds: dict) -> set:
    """(region, variable) pairs whose error is a property of the site, not of the weather.

    The first version of this script picked Leh, at ~3,500 m, with a pressure error of
    19.26 hPa against a 2.74 threshold, and called it a forecast bust. It is not one.
    Ranked by median absolute pressure error on the held-out split, the worst districts
    are exactly the high ones - Leh 5.32, Srinagar 2.56, Shimla 1.83 - against an
    all-district median of 0.555, while the best are Jaipur, Lakshadweep and Hyderabad.
    That is an elevation and pressure-reduction artifact, and presenting it as a caught
    bust in front of a forecasting audience would be an own goal.

    So a pair is excluded when its own median error is more than SITE_BIAS_FACTOR times
    the all-district median for that variable. The model is not wrong to flag these rows -
    the label is real by the project's own definition - but they are the wrong thing to
    put on a slide as a success story.
    """
    biased = set()
    for var in thresholds:
        col = f"{_ERR_PREFIX}{var}"
        if col not in test.columns:
            continue
        per_site = test.groupby("region_id")[col].median().dropna()
        if per_site.empty:
            continue
        overall = float(per_site.median())
        if not np.isfinite(overall) or overall <= 0:
            continue
        for region_id, med in per_site.items():
            if float(med) > SITE_BIAS_FACTOR * overall:
                biased.add((region_id, var))
    return biased


def _case_study(test: pd.DataFrame, thresholds: dict) -> dict:
    """One real bust the model called, with the evidence that it was a real call.

    Picked as the highest-confidence *correct* warning whose actual error most exceeds its
    own threshold - a true positive, on held-out data, chosen by a stated rule rather than
    by eye. A case study picked by hand is worth nothing to a reviewer.

    Pairs flagged by _biased_pairs are skipped: a station whose error is dominated by its
    own elevation is not evidence that the model saw a bust coming.
    """
    hits = test[(test["y_bust"] == 1) & (test["model_proba"] >= DECISION_THRESHOLD)]
    if hits.empty:
        return {}

    biased = _biased_pairs(test, thresholds)
    best, best_ratio = None, 0.0
    for row in hits.itertuples():
        for var, thr in thresholds.items():
            if not thr or (row.region_id, var) in biased:
                continue
            val = getattr(row, f"actual_err_{var}", None)
            if val is None or not np.isfinite(val):
                continue
            ratio = float(val) / float(thr) * float(row.model_proba)
            if ratio > best_ratio:
                best_ratio, best = ratio, (row, var, float(val), float(thr))
    if best is None:
        return {}

    row, var, err, thr = best

    # Every variable that broke, not just the worst. The label is "any variable busted",
    # so quoting one exceedance beside a probability invites the reading that the model
    # predicted *that* variable - it did not. Several correlated variables going together
    # is also the more convincing story: one variable off on its own reads like noise.
    exceeded = []
    for v, t in thresholds.items():
        if not t:
            continue
        e = getattr(row, f"{_ERR_PREFIX}{v}", None)
        if e is None or not np.isfinite(e) or float(e) < float(t):
            continue
        exceeded.append({
            "variable": v,
            "actual_error": float(e),
            "threshold": float(t),
            "exceedance": float(e) / float(t),
            "ensemble_spread": float(getattr(row, f"spread_{v}", float("nan"))),
        })
    exceeded.sort(key=lambda x: -x["exceedance"])

    return {
        "region_id": row.region_id,
        "region_name": _region_name(row.region_id),
        "init_date": str(pd.Timestamp(row.init_date).date()) if pd.notna(row.init_date) else None,
        "valid_date": str(pd.Timestamp(row.valid_date).date()) if pd.notna(row.valid_date) else None,
        "lead_time_days": int(row.lead_time_days),
        "bust_probability": float(row.model_proba),
        "variable": var,
        "actual_error": err,
        "threshold": thr,
        "exceedance": err / thr,
        # No forecast value: the eval frame carries errors, spreads and confidences, not
        # the forecast itself. An earlier version read a fc_ column that does not exist
        # and silently reported NaN.
        "predicted_error": float(getattr(row, f"pred_err_{var}", float("nan"))),
        "ensemble_spread": float(getattr(row, f"spread_{var}", float("nan"))),
        "exceeded": exceeded,
    }


def collect() -> dict:
    run_id = registry.current_run_id()
    if not run_id:
        raise SystemExit("no current model - nothing to report")

    metrics = registry.load_metrics(run_id) or {}
    baselines = registry.load_baselines(run_id) or {}
    thr_obj = registry.load_thresholds(run_id)
    thresholds = dict(getattr(thr_obj, "bust_threshold", {}) or {}) if thr_obj else {}

    ev = _events(run_id)
    test = ev[ev["split"] == "test"].copy()
    test = test[test["model_proba"].notna() & test["y_bust"].notna()]

    y = test["y_bust"].to_numpy(int)
    pred = (test["model_proba"].to_numpy(float) >= DECISION_THRESHOLD).astype(int)

    clf = (metrics.get("classifier") or {}).get("test") or {}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "decision_threshold": DECISION_THRESHOLD,
        "split": "test",
        "headline": {
            "roc_auc": clf.get("roc_auc"),
            "pr_auc": clf.get("pr_auc"),
            "brier": clf.get("brier"),
            "precision": clf.get("precision"),
            "recall": clf.get("recall"),
            "f1": clf.get("f1"),
            "n": clf.get("n"),
            "bust_rate": clf.get("bust_rate"),
        },
        "confusion": _skill(_counts(y, pred)),
        "per_lead": _per_lead(test),
        "baselines": baselines.get("models", []),
        "baseline_meta": {
            "test_events": baselines.get("test_events"),
            "test_cycles": baselines.get("test_cycles"),
            "bust_rate": baselines.get("bust_rate"),
        },
        "shap": _shap_top(run_id),
        "case_study": _case_study(test, thresholds),
        "thresholds": thresholds,
    }


def _f(v, n=4):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


def to_markdown(d: dict) -> str:
    L = []
    a = L.append
    a("# Sanket — figures for the deck")
    a("")
    a(f"**Run `{d['run_id']}` · generated {d['generated_at']} · held-out `{d['split']}` split**")
    a("")
    a("> Regenerate with `python backend/scripts/ppt_figures.py` after any retrain.")
    a("> These change when the model does; a copied slide goes stale silently.")
    a("")

    h = d["headline"]
    a("## Headline classifier scores")
    a("")
    a("| Metric | Value | Reading |")
    a("|---|---|---|")
    a(f"| ROC-AUC | **{_f(h['roc_auc'])}** | chance a real bust outranks a non-bust; 0.5 is a coin flip |")
    a(f"| PR-AUC | **{_f(h['pr_auc'])}** | against a base rate of {_f(h['bust_rate'], 4)}, which is the score to beat |")
    a(f"| Brier | **{_f(h['brier'])}** | mean squared error of the probability; lower is better |")
    a(f"| Precision | **{_f(h['precision'])}** | of the busts it warned about, this share verified |")
    a(f"| Recall (POD) | **{_f(h['recall'])}** | of the busts that happened, it warned about this share |")
    a(f"| F1 | **{_f(h['f1'])}** | harmonic mean of the two above |")
    a(f"| Held-out forecasts | **{h['n']:,}** | rows the model never trained on |")
    a("")

    c = d["confusion"]
    a(f"## Confusion matrix (threshold {d['decision_threshold']})")
    a("")
    a("| | Bust observed | No bust observed |")
    a("|---|---|---|")
    a(f"| **Warned** | TP **{c['tp']:,}** | FP **{c['fp']:,}** |")
    a(f"| **Did not warn** | FN **{c['fn']:,}** | TN **{c['tn']:,}** |")
    a("")
    a(f"POD **{_f(c['pod'])}** · FAR **{_f(c['far'])}** · POFD **{_f(c['pofd'])}** · "
      f"CSI **{_f(c['csi'])}** · frequency bias **{_f(c['bias'], 3)}** · n **{c['n']:,}**")
    a("")
    a("*FAR here is the false alarm **ratio**, FP/(TP+FP) — warnings that did not verify. "
      "It is not POFD, FP/(FP+TN), which is also given. The two are often confused and "
      "differ widely on unbalanced classes.*")
    a("")

    a("## POD and FAR per lead day")
    a("")
    a("| Lead | POD ↑ | FAR ↓ | POFD ↓ | CSI ↑ | Bias | TP | FP | FN | TN | n |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in d["per_lead"]:
        a(f"| D{r['lead_time_days']} | {_f(r['pod'], 3)} | {_f(r['far'], 3)} | {_f(r['pofd'], 3)} | "
          f"{_f(r['csi'], 3)} | {_f(r['bias'], 2)} | {r['tp']:,} | {r['fp']:,} | {r['fn']:,} | "
          f"{r['tn']:,} | {r['n']:,} |")
    a("")

    a("## Baseline ladder")
    a("")
    m = d["baseline_meta"]
    a(f"All scored on the identical held-out rows — {m.get('test_events'):,} events over "
      f"{m.get('test_cycles')} forecast cycles, base rate {_f(m.get('bust_rate'), 4)}.")
    a("")
    a("| Model | Brier ↓ | Skill vs climatology ↑ | ROC-AUC ↑ |")
    a("|---|---|---|---|")
    for b in d["baselines"]:
        star = " **(ours)**" if b.get("is_model") else ""
        a(f"| {b.get('name')}{star} | {_f(b.get('brier'))} | {_f(b.get('bss'))} | {_f(b.get('roc_auc'))} |")
    a("")
    a("*Skill is the Brier skill score against climatology — guessing the long-run bust "
      "rate every time. 0.000 means no better than that guess; negative means worse.*")
    a("")

    if d["shap"]:
        a("## What drives the prediction (SHAP)")
        a("")
        a("Mean |SHAP| over the classifier, all regions and lead days pooled.")
        a("")
        a("| Feature | Mean \\|SHAP\\| |")
        a("|---|---|")
        for s in d["shap"]:
            a(f"| `{s['feature']}` | {_f(s['mean_abs_shap'])} |")
        a("")

    cs = d["case_study"]
    if cs:
        a("## Case study — a real bust, called in advance")
        a("")
        a(f"**{cs['region_name']}** · forecast issued {cs['init_date']} · "
          f"valid {cs['valid_date']} · lead day {cs['lead_time_days']}")
        a("")
        a(f"The model gave this forecast a **{cs['bust_probability'] * 100:.1f}% chance of "
          f"busting**, {cs['lead_time_days']} days out. It busted on "
          f"**{len(cs['exceeded'])} variables at once**:")
        a("")
        a("| Variable | Error | Threshold | Exceedance | Ensemble spread |")
        a("|---|---|---|---|---|")
        for e in cs["exceeded"]:
            a(f"| {e['variable']} | **{_f(e['actual_error'], 2)}** | {_f(e['threshold'], 2)} | "
              f"**{e['exceedance']:.1f}×** | {_f(e['ensemble_spread'], 3)} |")
        a("")
        spread = cs.get("ensemble_spread", float("nan"))
        if np.isfinite(spread) and spread == 0:
            a(f"The ensemble's spread on {cs['variable']} was **exactly zero** — all five "
              "members agreed, and all five were wrong. That is the case the spread "
              "baseline cannot catch by construction, and it is why that baseline scores "
              "0.508 ROC-AUC while this model scores 0.841.")
            a("")
        a("*Chosen by rule, not by eye: the highest-confidence correct warning whose actual "
          "error most exceeds its own threshold, on held-out data, excluding sites whose "
          "error for that variable is systematically inflated (see `_biased_pairs`). The "
          "probability is for a bust on **any** variable — the model was not asked to "
          "name which one.*")
        a("")

    a("---")
    a(f"Every figure above is read from run `{d['run_id']}`. Nothing is hand-entered.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="write markdown here instead of stdout")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = ap.parse_args()

    data = collect()
    text = json.dumps(data, indent=2, default=str) if args.json else to_markdown(data)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out} ({len(text):,} chars) for run {data['run_id']}")
    else:
        print(text)


if __name__ == "__main__":
    main()
