"""Why do training bust rates step up in 2014-15? Per-year event errors, then the answer.

The served run's training bust rate is 0.39-0.45 for 2000-2013 and 0.49 for 2014-2015, and
every year's observations came from the same ERA5 script. Before the retrain fits a bias
table on 2000-2015, this says which variable moved, whether it is a shift in the mean
(which the bias correction absorbs only if the key can see it) or a change in spread, and
which bias key holds up on a year it was not fitted on.

    python -m scripts.diagnose_label_drift build --years 2000-2019      # once; resumable
    python -m scripts.diagnose_label_drift report --train-years 2000-2013

`build` writes one file per year, `data/analysis/label_drift/errors_<year>.parquet`: one row
per event (district, init date, lead day) with the ensemble-mean error, forecast minus
observation, of every variable (wind direction as a signed circular difference). Read one
month of cycles at a time, so memory stays near one month's member rows.

`report` prints, per variable and year: the mean signed error, the RMSE and the share of
events above the training years' 90th percentile of |error| (the bust threshold's
definition, pooled over leads); the event bust rate (any label variable above its
threshold); and, leaving one year out at a time, the error left after removing a per
(district, variable, lead) bias keyed by nothing, by season or by month.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.features.engineering import _SEASONS  # noqa: E402

VARIABLES = ["temperature_c", "humidity_pct", "rainfall_mm", "pressure_hpa",
             "wind_speed_ms", "wind_direction_deg", "soil_moisture_pct",
             "atmospheric_moisture_kgm2"]
CIRCULAR = {"wind_direction_deg"}
OUT_DIR = BACKEND / "data" / "analysis" / "label_drift"
EVENT = ["region_id", "init_date", "lead_time_days"]


def event_errors(forecast: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Member forecasts + observations -> one row per event, one error column per variable."""
    fc = forecast.copy()
    for c in ("init_date", "valid_date"):
        fc[c] = pd.to_datetime(fc[c])
    keys = EVENT + ["valid_date", "variable"]
    lin = fc[~fc["variable"].isin(CIRCULAR)]
    means = [lin.groupby(keys, observed=True)["value"].mean()]
    circ = fc[fc["variable"].isin(CIRCULAR)]
    if not circ.empty:
        r = np.radians(circ["value"].to_numpy(dtype=float))
        c = circ.assign(_s=np.sin(r), _c=np.cos(r)).groupby(keys, observed=True)[["_s", "_c"]].mean()
        means.append(pd.Series(np.degrees(np.arctan2(c["_s"], c["_c"])) % 360.0,
                               index=c.index, name="value"))
    fcm = pd.concat(means).rename("fc").reset_index()

    ob = observed[["region_id", "variable", "valid_date", "value"]].copy()
    ob["valid_date"] = pd.to_datetime(ob["valid_date"])
    ob = ob.groupby(["region_id", "variable", "valid_date"], observed=True)["value"].mean()
    m = fcm.merge(ob.rename("ob").reset_index(), on=["region_id", "variable", "valid_date"],
                  how="inner")
    if m.empty:
        return pd.DataFrame(columns=EVENT + ["month"] + VARIABLES)
    err = m["fc"] - m["ob"]
    circ_rows = m["variable"].isin(CIRCULAR).to_numpy()
    err = np.where(circ_rows, (err + 180.0) % 360.0 - 180.0, err)
    m = m.assign(err=err.astype("float32"))
    wide = m.pivot_table(index=EVENT, columns="variable", values="err",
                         aggfunc="first", observed=True).reset_index()
    wide.columns.name = None
    for v in VARIABLES:
        if v not in wide:
            wide[v] = np.float32(np.nan)
    wide["region_id"] = wide["region_id"].astype(str)
    wide["month"] = pd.to_datetime(wide["init_date"]).dt.month.astype("int8")
    return wide[EVENT + ["month"] + VARIABLES]


def build_year(year: int, out_dir: Path = OUT_DIR) -> Path:
    from app.storage import parquet_store

    out = out_dir / f"errors_{year}.parquet"
    if out.exists():
        print(f"{year}: exists, skipped", flush=True)
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for month in range(1, 13):
        t0 = time.time()
        start = pd.Timestamp(year, month, 1)
        end = start + pd.offsets.MonthEnd(0)
        # Valid dates of these cycles run to end + 9 days (Day 10).
        fc = parquet_store.read_dataset(
            value_types=["forecast"], variables=VARIABLES,
            columns=["region_id", "variable", "value", "init_date", "valid_date",
                     "lead_time_days", "ensemble_member_id"],
            valid_date_min=start.date(), valid_date_max=(end + pd.Timedelta(days=9)).date(),
            exclude_provisional=True)
        fc["init_date"] = pd.to_datetime(fc["init_date"])
        fc = fc[(fc["init_date"] >= start) & (fc["init_date"] <= end)]
        ob = parquet_store.read_dataset(
            value_types=["observed"], variables=VARIABLES,
            columns=["region_id", "variable", "value", "valid_date"],
            valid_date_min=start.date(), valid_date_max=(end + pd.Timedelta(days=9)).date(),
            exclude_provisional=True)
        ev = event_errors(fc, ob)
        parts.append(ev)
        print(f"{year}-{month:02d}: {len(ev):,} events from {len(fc):,} member rows "
              f"in {time.time() - t0:.0f}s", flush=True)
        del fc, ob
    df = pd.concat(parts, ignore_index=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return out


# ------------------------------------------------------------------ the answers

def per_year_summary(frames: dict, train_years) -> pd.DataFrame:
    """Per (variable, year): n, mean signed error, RMSE, share above the training p90."""
    train = pd.concat([frames[y] for y in train_years if y in frames], ignore_index=True)
    rows = []
    for v in VARIABLES:
        if v not in train or train[v].notna().sum() == 0:
            continue
        thr = float(np.nanpercentile(np.abs(train[v].to_numpy(dtype=float)), 90))
        for y, f in sorted(frames.items()):
            e = f[v].to_numpy(dtype=float)
            e = e[~np.isnan(e)]
            if e.size == 0:
                continue
            rows.append({"variable": v, "year": y, "n": e.size, "mean_error": e.mean(),
                         "rmse": float(np.sqrt(np.mean(e ** 2))),
                         "above_p90": float(np.mean(np.abs(e) >= thr)), "p90_train": thr})
    return pd.DataFrame(rows)


def event_bust_rate(frames: dict, train_years, label_variables) -> pd.DataFrame:
    """Per year: share of events where any label variable is at or above its training p90."""
    train = pd.concat([frames[y] for y in train_years if y in frames], ignore_index=True)
    thr = {v: float(np.nanpercentile(np.abs(train[v].to_numpy(dtype=float)), 90))
           for v in label_variables if v in train and train[v].notna().any()}
    rows = []
    for y, f in sorted(frames.items()):
        hit = np.zeros(len(f), dtype=bool)
        seen = np.zeros(len(f), dtype=bool)
        for v, t in thr.items():
            a = np.abs(f[v].to_numpy(dtype=float))
            seen |= ~np.isnan(a)
            hit |= a >= t
        rows.append({"year": y, "events": int(seen.sum()),
                     "bust_rate": float(hit[seen].mean()) if seen.any() else np.nan})
    return pd.DataFrame(rows)


def _key_column(f: pd.DataFrame, key: str) -> pd.Series:
    if key == "none":
        return pd.Series("*", index=f.index)
    if key == "season":
        return f["month"].map(_SEASONS)
    if key == "month":
        return f["month"].astype(str)
    raise ValueError(key)


def held_out_mse(frames: dict, keys=("none", "season", "month")) -> pd.DataFrame:
    """Leave one year out: fit a per (district, variable, lead, key) mean error on the
    other years, remove it from the held-out year, and average the squared error left.
    Cells with no fit in the other years keep their error (nothing removed)."""
    rows = []
    years = sorted(frames)
    for key in keys:
        stats = {}
        for y in years:
            f = frames[y]
            k = _key_column(f, key)
            for v in VARIABLES:
                if v not in f:
                    continue
                g = pd.DataFrame({"region_id": f["region_id"].astype(str),
                                  "lead": f["lead_time_days"].astype(int), "k": k,
                                  "e": f[v].astype(float)}).dropna(subset=["e"])
                if g.empty:
                    continue
                s = g.groupby(["region_id", "lead", "k"])["e"].agg(["sum", "count"])
                stats[(y, v)] = (g, s)
        for v in VARIABLES:
            per = [(y, *stats[(y, v)]) for y in years if (y, v) in stats]
            if len(per) < 2:
                continue
            total = None
            for _, _, s in per:
                total = s if total is None else total.add(s, fill_value=0)
            sq, n = 0.0, 0
            for y, g, s in per:
                rest = total.sub(s, fill_value=0)
                b = (rest["sum"] / rest["count"]).where(rest["count"] > 0)
                idx = pd.MultiIndex.from_arrays([g["region_id"], g["lead"], g["k"]])
                bias = b.reindex(idx).to_numpy()
                e = g["e"].to_numpy() - np.nan_to_num(bias, nan=0.0)
                sq += float(np.sum(e ** 2))
                n += e.size
            rows.append({"variable": v, "key": key, "mse": sq / n, "n": n})
    return pd.DataFrame(rows)


def _parse_years(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--years", required=True)
    r = sub.add_parser("report")
    r.add_argument("--train-years", default="2000-2013")
    r.add_argument("--years", default="2000-2019")
    args = ap.parse_args()

    if args.cmd == "build":
        for y in _parse_years(args.years):
            build_year(y)
        return 0

    from app import contracts
    frames = {}
    for y in _parse_years(args.years):
        p = OUT_DIR / f"errors_{y}.parquet"
        if p.exists():
            frames[y] = pd.read_parquet(p)
    if not frames:
        print(f"no files under {OUT_DIR}; run build first", file=sys.stderr)
        return 1
    train = [y for y in _parse_years(args.train_years) if y in frames]
    label_vars = sorted(getattr(contracts, "LABEL_VARIABLES", VARIABLES))
    pd.set_option("display.width", 200)
    print(f"years {sorted(frames)}; thresholds from {train}\n")
    print("Event bust rate (any label variable at or above its training p90):")
    print(event_bust_rate(frames, train, label_vars).round(3).to_string(index=False))
    s = per_year_summary(frames, train)
    for col in ("mean_error", "rmse", "above_p90"):
        print(f"\n{col} by variable and year:")
        print(s.pivot(index="year", columns="variable", values=col).round(3).to_string())
    print("\nHeld-out MSE after removing a (district, variable, lead, key) bias:")
    print(held_out_mse(frames).pivot(index="variable", columns="key", values="mse")
          .round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
