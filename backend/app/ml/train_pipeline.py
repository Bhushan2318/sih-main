from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pathlib import Path

from app.config import settings
from app.db.base import resolve_path
from app.features import engineering as fe
from app.features import pivot as pv
from app.ml import classifier as clf_mod
from app.ml import explain as explain_mod
from app.ml import regressors as reg_mod
from app.ml import registry
from app.ml.thresholds import (
    Thresholds,
    compute_error_thresholds,
    compute_member_p90,
    compute_risk_bands,
)
from app.storage import parquet_store

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
_REG_CATEGORICAL = reg_mod.CATEGORICAL_FEATURES
_CLF_CATEGORICAL = clf_mod.CATEGORICAL


@dataclass
class TrainReport:
    run_id: str
    status: str
    made_current: bool = False
    data_rows: int = 0
    paired_rows: int = 0
    split_cycles: dict = field(default_factory=dict)
    modelled_variables: list = field(default_factory=list)
    skipped_variables: dict = field(default_factory=dict)
    regressor_metrics: dict = field(default_factory=dict)
    classifier_metrics: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0


_TRAINING_COLUMNS = [
    "region_id", "variable", "valid_date", "value", "value_type",
    "init_date", "lead_time_days", "ensemble_member_id", "verification_status",
]


_CHUNK_CYCLES = 12

_MAX_LEAD_DAYS = 10
_OBS_PAD_DAYS = 3


def _build_paired_in_chunks() -> "tuple[pd.DataFrame, int]":
    inits = parquet_store.read_dataset(
        value_types=["forecast"], columns=["init_date"], dedupe=False,
    )["init_date"].dropna().unique()
    if len(inits) == 0:
        return pd.DataFrame(), 0
    inits = sorted(pd.to_datetime(pd.Series(inits)).dt.normalize().unique())

    frames, rows_read = [], 0
    for i in range(0, len(inits), _CHUNK_CYCLES):
        chunk = inits[i : i + _CHUNK_CYCLES]
        fc = parquet_store.read_dataset(
            value_types=["forecast"], columns=_TRAINING_COLUMNS,
            init_dates=[pd.Timestamp(c).date() for c in chunk],
            exclude_provisional=True,
        )
        if fc.empty:
            continue
        ob = parquet_store.read_dataset(
            value_types=["observed"], columns=_TRAINING_COLUMNS,
            valid_date_min=(pd.Timestamp(chunk[0]) - pd.Timedelta(days=_OBS_PAD_DAYS)).date(),
            valid_date_max=(pd.Timestamp(chunk[-1])
                            + pd.Timedelta(days=_MAX_LEAD_DAYS + _OBS_PAD_DAYS)).date(),
            exclude_provisional=True,
        )
        rows_read += len(fc) + len(ob)
        part = fe.build_training_frame(
            pd.concat([fc, ob], ignore_index=True), historical_bust_freq=None)
        if not part.empty:
            frames.append(part)
        del fc, ob, part

    if not frames:
        return pd.DataFrame(), rows_read
    categorical = [c for c in frames[0].columns
                   if str(frames[0][c].dtype) == "category"]
    out = pd.concat(frames, ignore_index=True)
    for col in categorical:
        if str(out[col].dtype) != "category":
            out[col] = out[col].astype("category")
    return (out.sort_values(fe.MEMBER_KEYS[:-1] + ["variable", "ensemble_member_id",
                                                   "lead_time_days"])
               .reset_index(drop=True), rows_read)


def _split_by_cycle(paired: pd.DataFrame):
    cycles = sorted(paired["init_date"].dropna().unique())
    n = len(cycles)
    if n < 3:
        return set(cycles), set(), set()
    a = max(1, int(round(n * TRAIN_FRAC)))
    b = max(a + 1, int(round(n * (TRAIN_FRAC + VAL_FRAC))))
    return set(cycles[:a]), set(cycles[a:b]), set(cycles[b:])


def full_retrain(triggered_by_batch_id: str | None = None, make_current: bool = True,
                 *, emit_eval: bool = False) -> TrainReport:
    t0 = time.time()
    run_id = registry.new_run_id()
    report = TrainReport(run_id=run_id, status="failed")

    if not settings.allow_local_retrain:
        report.status = "refused"
        report.error = (
            "Training is disabled on this deployment. It peaks around 2.3 GB and this "
            "instance has 512 MB, so a retrain here would kill the process rather than "
            "slow it. Models are trained in CI and shipped as a release artifact. Set "
            "ALLOW_LOCAL_RETRAIN=true to train locally or in CI."
        )
        return report

    try:
        paired, report.data_rows = _build_paired_in_chunks()
        if paired.empty:
            report.status = "no_data"
            return report
        report.paired_rows = len(paired)
        if paired.empty:
            report.status = "no_data"
            return report

        train_c, val_c, test_c = _split_by_cycle(paired)
        report.split_cycles = {
            "train": len(train_c), "val": len(val_c), "test": len(test_c),
            "train_dates": [str(pd.Timestamp(c).date()) for c in sorted(train_c)],
        }
        tr = paired[paired["init_date"].isin(train_c)].copy()
        va = paired[paired["init_date"].isin(val_c)].copy()
        te = paired[paired["init_date"].isin(test_c)].copy()

        hbf = fe.compute_historical_bust_frequency(tr)
        for frame in (tr, va, te):
            if frame.empty:
                continue
            key = list(zip(frame["region_id"].astype(str), frame["season"].astype(str)))
            frame["historical_bust_frequency_region_season"] = [hbf.get(k, np.nan) for k in key]

        p90_error = compute_member_p90(tr[["variable", "abs_error"]])
        event_err_tr = _event_mean_error(tr)
        bust_threshold = compute_error_thresholds(event_err_tr, percentile=90.0)

        artifacts: dict = {}
        oof_tr = pd.Series(np.nan, index=tr.index, dtype=float)
        val_pred = pd.Series(np.nan, index=va.index, dtype=float)
        test_pred = pd.Series(np.nan, index=te.index, dtype=float)

        for var in sorted(tr["variable"].unique()):
            n_var = int((tr["variable"] == var).sum())
            if n_var < reg_mod.MIN_ROWS:
                report.skipped_variables[var] = f"only {n_var} paired train rows (<{reg_mod.MIN_ROWS})"
                continue
            art = reg_mod.train_variable_regressor(tr, va, var)
            if art is None:
                report.skipped_variables[var] = "regressor training returned None"
                continue
            artifacts[var] = art
            report.regressor_metrics[var] = art.metrics
            oof_tr.loc[tr["variable"] == var] = reg_mod.oof_predict(tr, var).to_numpy()
            for frame, store in ((va, val_pred), (te, test_pred)):
                mask = frame["variable"] == var
                if mask.any():
                    store.loc[mask] = reg_mod.predict_variable_error(art, frame[mask])
            if not te.empty and (te["variable"] == var).sum() >= 5:
                art.metrics["test"] = reg_mod._evaluate(
                    te.loc[te["variable"] == var, "abs_error"],
                    test_pred[te["variable"] == var],
                )

        report.modelled_variables = sorted(artifacts)
        if not artifacts:
            report.status = "failed"
            report.error = "no variable had enough paired rows to train a regressor"
            return report

        event_tr = pv.build_event_frame(tr, oof_tr, p90_error, bust_threshold, hbf)
        event_va = pv.build_event_frame(va, val_pred, p90_error, bust_threshold, hbf)
        event_te = (pv.build_event_frame(te, test_pred, p90_error, bust_threshold, hbf)
                    if not te.empty else pd.DataFrame())
        clf_art = clf_mod.train_bust_classifier(event_tr, event_va)
        report.classifier_metrics = dict(clf_art.metrics)
        if not event_te.empty and "y_bust" in event_te:
            proba_te = clf_mod.predict_bust_probability(clf_art, event_te)
            report.classifier_metrics["test"] = clf_mod._evaluate(event_te["y_bust"], proba_te)

        proba_va = (clf_mod.predict_bust_probability(clf_art, event_va)
                    if len(event_va) else np.array([]))
        risk_cuts = compute_risk_bands(proba_va)
        thresholds = Thresholds(
            bust_threshold=bust_threshold,
            p90_error=p90_error,
            risk_band_cuts=risk_cuts,
            notes=[
                f"bust_threshold = 90th pct of event-grain ensemble-mean abs error, "
                f"train split ({len(train_c)} cycles)",
                f"risk bands from {len(proba_va)} validation events",
            ],
        )
        report.thresholds = {
            "bust_threshold": bust_threshold,
            "p90_error": p90_error,
            "risk_band_cuts": risk_cuts,
        }

        if emit_eval:
            _emit_eval_events(run_id, clf_art,
                              {"train": event_tr, "val": event_va, "test": event_te})

        shap_frames = []
        for var, art in artifacts.items():
            sub = va[va["variable"] == var]
            if len(sub) >= 20:
                shap_frames.append(explain_mod.explain_model(
                    art.model, sub, art.feature_columns, _REG_CATEGORICAL,
                    model_name=f"regressor::{var}"))
        if len(event_va) >= 20:
            shap_frames.append(explain_mod.explain_model(
                clf_art.model, event_va, clf_art.feature_columns, _CLF_CATEGORICAL,
                model_name="classifier"))
        shap_summary = pd.concat([f for f in shap_frames if not f.empty], ignore_index=True) \
            if any(not f.empty for f in shap_frames) else pd.DataFrame()

        for var, art in artifacts.items():
            registry.save_regressor(run_id, var, art.model, art.feature_columns)
        registry.save_classifier(run_id, clf_art.model, clf_art.feature_columns)
        registry.save_thresholds(run_id, thresholds)
        registry.save_historical_bust_freq(run_id, hbf)
        if not shap_summary.empty:
            shap_summary.to_parquet(registry.run_dir(run_id) / "shap_summary.parquet", index=False)
        registry.save_metrics(run_id, {
            "regressors": report.regressor_metrics,
            "classifier": report.classifier_metrics,
        })
        registry.save_manifest(run_id, {
            "run_id": run_id,
            "triggered_by_batch_id": triggered_by_batch_id,
            "data_rows": report.data_rows,
            "paired_rows": report.paired_rows,
            "split_cycles": report.split_cycles,
            "modelled_variables": report.modelled_variables,
            "skipped_variables": report.skipped_variables,
            "shap_method": (shap_summary["method"].iloc[0] if not shap_summary.empty else "none"),
        })

        report.status = "success"
        report.seconds = time.time() - t0
        if make_current:
            registry.set_current(run_id)
            report.made_current = True
        return report

    except Exception as exc:  # noqa: BLE001
        report.status = "failed"
        report.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        report.seconds = time.time() - t0
        return report


def _emit_eval_events(run_id: str, clf_art, splits: dict) -> Path:
    out_dir = resolve_path(settings.data_dir) / "analysis" / "eval_events"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, ev in splits.items():
        if ev is None or ev.empty:
            continue
        f = ev.copy()
        f["split"] = name
        f["model_proba"] = clf_mod.predict_bust_probability(clf_art, ev)
        frames.append(f)
    out = out_dir / f"{run_id}.parquet"
    combined = pd.concat(frames, ignore_index=True)
    for col in combined.columns:
        if str(combined[col].dtype) == "category":
            combined[col] = combined[col].astype(str)
    combined.to_parquet(out, index=False)
    print(f"eval events -> {out}  ({len(combined):,} rows)")
    return out


def _event_mean_error(paired: pd.DataFrame) -> pd.DataFrame:
    em = (paired.groupby(fe.EVENT_KEYS + ["variable"], observed=True)
          .agg(fc_mean=("forecast_value", "mean"), obs=("observed_value", "mean"))
          .reset_index())
    em["abs_error"] = (em["fc_mean"] - em["obs"]).abs()
    return em[["variable", "abs_error"]]


def _print_report(r: TrainReport) -> None:
    print("\n" + "=" * 78)
    print(f"TRAIN RUN {r.run_id}   status={r.status}   made_current={r.made_current}   {r.seconds:.1f}s")
    print("=" * 78)
    if r.status != "success":
        print(f"  {r.error or r.status}")
        return
    print(f"canonical rows : {r.data_rows:,}    paired rows: {r.paired_rows:,}")
    print(f"cycles         : train={r.split_cycles['train']} val={r.split_cycles['val']} "
          f"test={r.split_cycles['test']}")
    print(f"modelled       : {', '.join(r.modelled_variables)}")
    if r.skipped_variables:
        for v, why in r.skipped_variables.items():
            print(f"  skipped {v}: {why}")

    print("\nREGRESSORS  — MAE / R2 by split (held-out test is the honest number):")
    print(f"  {'variable':26} {'MAE_val':>8} {'R2_val':>7} {'MAE_test':>9} {'R2_test':>8} {'MAE_base':>9}")
    for var, m in r.regressor_metrics.items():
        v = m.get("val") or {}
        t = m.get("test") or {}
        b = (m.get("test") or m.get("val") or m.get("train") or {}).get("baseline_mae_predict_mean", float("nan"))
        print(f"  {var:26} {v.get('mae', float('nan')):8.3f} {v.get('r2', float('nan')):7.3f} "
              f"{t.get('mae', float('nan')):9.3f} {t.get('r2', float('nan')):8.3f} {b:9.3f}")

    print("\nBUST CLASSIFIER:")
    for split in ("train", "val", "test"):
        m = r.classifier_metrics.get(split)
        if not m:
            continue
        print(f"  {split:6} n={m['n']:5d} bust_rate={m['bust_rate']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"ROC_AUC={m['roc_auc']:.3f} PR_AUC={m['pr_auc']:.3f} Brier={m['brier']:.3f}")
    if "best_iteration" in r.classifier_metrics:
        print(f"  (early-stopped at iteration {r.classifier_metrics['best_iteration']})")

    print("\nTHRESHOLDS (per-variable bust threshold = 90th pct event abs error):")
    for var, t in r.thresholds["bust_threshold"].items():
        print(f"  {var:26} {t:8.3f}")
    print(f"risk band cuts on P(bust): {r.thresholds['risk_band_cuts']}")
    n_cycles = sum(r.split_cycles.get(k, 0) for k in ("train", "val", "test"))
    n_test = (r.classifier_metrics.get("test") or {}).get("n", 0)
    scale = (f"{n_cycles} initialisations, {r.split_cycles.get('test', 0)} held out "
             f"({n_test:,} test events)")
    if n_cycles < 40:
        print(f"\nNOTE: small-sample metrics - {scale}. Treat as directional.")
    else:
        print(f"\nSample: {scale}.")


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="train + report, don't flip current.json")
    ap.add_argument("--json", action="store_true", help="also dump the report as JSON")
    ap.add_argument("--emit-eval", action="store_true",
                    help="also write the scored event frames to data/analysis/eval_events "
                         "(input for scripts/run_baselines.py)")
    args = ap.parse_args()

    from app.db.base import init_db
    init_db()

    r = full_retrain(make_current=not args.dry_run, emit_eval=args.emit_eval)
    _print_report(r)
    if args.json:
        from dataclasses import asdict
        print("\n" + json.dumps(asdict(r), indent=2, default=str))
    return 0 if r.status == "success" else 1


if __name__ == "__main__":
    sys.exit(_main())
