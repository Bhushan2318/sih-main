"""District rainfall from IMD's gauge-based gridded product, merged into an already-fetched
ERA5/CDS district observation file.

Why this exists
----------------
CLAUDE.md's own stated plan: "IMD gauge-based gridded rainfall at native 0.25 deg for
precipitation, ERA5 ... for everything else." ERA5 precipitation is a reanalysis product
and is measurably weak over India - it is the hardest variable and the driver of most
busts. IMD's product is built from a dense in-situ rain-gauge network, interpolated onto
the same 0.25 deg grid the district weight table already uses.

Why merge rather than write a standalone observation file
-----------------------------------------------------------
IMD only publishes rain (and tmin/tmax, not used here). A precip_mm-only file would be
missing the other eight canonical variables `ingest_upload` expects, so this script reads
the best already-fetched ERA5-family file for the year (via
`ingest_backfill.observation_file(..., include_imd=False)`) and replaces only its precip_mm
column - everything else (temperature, humidity, wind, pressure, soil moisture, water
vapour) keeps coming from ERA5/CDS. The two products never blend inside one column: `source` is annotated so
it is always traceable which product produced a row's rainfall.

Which day IMD's rain belongs to - measured, not assumed
--------------------------------------------------------
IMD's gauge day accumulates 0830 IST to 0830 IST (0300 UTC to 0300 UTC). Sanket's day is
midnight to midnight UTC on both sides of a bust label - CLAUDE.md rule 4's
((k-1)*24, k*24] on the forecast side, `to_daily`'s (t-24h, t] on the ERA5 side.

IMD labels each day by the END of its window. The value IMD dates D is rain for 0830 IST
on D-1 to 0830 IST on D:

    IMD value dated D     (D-1 03:00 UTC, D   03:00 UTC]
    Sanket's day D-1      (D-1 00:00 UTC, D   00:00 UTC]   shares 21 of those 24 hours
    Sanket's day D        (D   00:00 UTC, D+1 00:00 UTC]   shares 3

So IMD's date D is filed under model date D-1 (IMD_DATE_TO_MODEL_DATE). This script used
to join D to D, following the project brief, which stated start-day attribution. That was
wrong: it put every IMD rainfall value one model day late against the forecast it
verified - the same class of bug rule 4 already records once.

IMD's own product page states no date convention at all. The end-day labelling was
established from the data, by correlating IMD district rainfall with two independent
hourly reanalyses (ERA5 and MERRA-2) summed to UTC days, over two years and both
monsoons; the figures are in docs/known-issues.md. `check_attribution` repeats that test
on every merge and refuses to write rain that lands on the wrong day, so a later change
in IMD's files cannot slip through either.

The residual three-hour mismatch - rain at 05:30-08:30 IST lands in the neighbouring model
day - is known and deliberately left: re-cutting the model day to 0300 UTC would push Day
10 to forecast hour 243, past the 240-hour end of the GEFSv12 reforecast archive.

Two consequences for `build`:
- Model date 31 Dec needs IMD's 1 Jan of the FOLLOWING year, so each year reads two IMD
  years. Without the second, the merge refuses rather than dropping the day.
- Output is named imd_aligned_*. Files named imd_merged_* come from the old join, and
  `ingest_backfill.observation_file` refuses them.

These windows live in this script because it is their only consumer. If a second IMD path
appears, move them into app/utils/ - do not copy them.

Access
------
Verified against the real archive 2026-09-13: a plain POST to
https://imdpune.gov.in/cmpg/Griddata/rainfall.php with {"rain": year} returns a real
binary .grd file, no account or key needed. `imdlib` (PyPI) wraps that request and the
binary parsing; this script uses it rather than re-implementing the format.

Grid: 129 lat points (6.5..38.5), 135 lon points (66.5..100.0), both 0.25 deg spacing -
a superset of the district weight table's own grid, so cell matching is a plain filter,
not an interpolation. Missing/sea cells carry the sentinel -999.0.

    pip install imdlib
    python -m scripts.fetch_imd_district_rainfall --years 2018
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.utils.district_observations import grid_cells, to_districts  # noqa: E402
from scripts import ingest_backfill as ib  # noqa: E402

OUT_DIR = BACKEND_DIR / "data" / "samples"
SOURCE_IMD = ("IMD gauge-based gridded rainfall 0.25 deg (imdpune.gov.in); "
              "IMD date D filed as model date D-1")
ALIGNED_STEM = ib.IMD_ALIGNED_STEM.replace("_{year}", "")
MISSING = -999.0

# --- Accumulation windows. See "the 0830 IST rain day" in the module docstring. ---
DAY = pd.Timedelta(hours=24)
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)   # India has no daylight saving
MODEL_DAY_START_UTC = pd.Timedelta(hours=0)      # 00:00 UTC, CLAUDE.md rule 4
IMD_DAY_START_IST = pd.Timedelta(hours=8, minutes=30)
IMD_DAY_START_UTC = IMD_DAY_START_IST - IST_OFFSET  # = 03:00 UTC
# IMD dates a day by the END of its 0830 IST window, so its date D is model date D-1.
IMD_DATE_TO_MODEL_DATE = pd.Timedelta(days=-1)


def _utc_midnight(day) -> pd.Timestamp:
    """00:00 UTC at the start of calendar date `day`.

    Refuses a timezone-aware timestamp. Which calendar day a moment belongs to is the
    whole question here, and an aware timestamp has no single answer: 02:00 IST on the
    15th is 20:30 UTC on the 14th. Converting it quietly would hand back the previous
    day's window - the off-by-one-day error this code exists to prevent.
    """
    ts = pd.Timestamp(day)
    if ts.tzinfo is not None:
        raise ValueError(
            f"expected a calendar date, got the timezone-aware {ts!r}, whose calendar "
            f"day depends on which timezone it is read in. Pass the date itself - e.g. "
            f"{ts.date().isoformat()!r} for the {ts.tzinfo} day.")
    return ts.normalize().tz_localize("UTC")


def model_day_window_utc(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Sanket's day `day` as a (start, end] window in UTC.

    Midnight to midnight UTC - the window the GEFS forecast side and the ERA5
    observation side both already use.
    """
    start = _utc_midnight(day) + MODEL_DAY_START_UTC
    return start, start + DAY


def imd_rain_day_window_utc(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The (start, end] UTC window of the rainfall IMD dates `day`.

    0830 IST on the previous day to 0830 IST on `day` - IMD labels a gauge day by when its
    window ENDS. In UTC that is 03:00 to 03:00, three hours after the model day it mostly
    belongs to, `model_day_window_utc(day + IMD_DATE_TO_MODEL_DATE)`.
    """
    end = _utc_midnight(day) + IMD_DAY_START_UTC
    return end - DAY, end


def window_overlap_hours(a: tuple[pd.Timestamp, pd.Timestamp],
                         b: tuple[pd.Timestamp, pd.Timestamp]) -> float:
    """Hours two (start, end] windows share; 0.0 when they do not touch.

    The join rule in one number: an IMD rain-day overlaps the model day of the same
    date by 21 hours and the next model day by 3.
    """
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    return max(pd.Timedelta(0), overlap).total_seconds() / 3600.0


# --- Checking which day the rain is on, against an independent reanalysis --------------
ATTRIBUTION_LAGS = (-1, 0, 1)
MIN_PAIRED_DAYS = 30  # per region; fewer and a rank correlation is mostly noise


def attribution_lag_correlations(rain_rows: pd.DataFrame, reference_rows: pd.DataFrame,
                                 lags=ATTRIBUTION_LAGS) -> dict[int, float]:
    """Mean over regions of the Spearman correlation between `rain_rows` precip_mm on date D
    and `reference_rows` precip_mm on date D+lag.

    Both are long (region_id, date, precip_mm) frames on the same calendar. Rain on the
    right day correlates best at lag 0; one day late peaks at lag -1, one day early at +1.
    Spearman because rainfall is zero-inflated and heavy-tailed - computed as Pearson on
    ranks, so it needs nothing beyond pandas. Regions with fewer than MIN_PAIRED_DAYS paired
    days, or no variation at all (an entirely dry spell), are left out.
    """
    def wide(rows):
        rows = rows.assign(date=pd.to_datetime(rows["date"]))
        return rows.pivot_table(index="date", columns="region_id", values="precip_mm",
                                aggfunc="first")

    rain, ref = wide(rain_rows), wide(reference_rows)
    regions = rain.columns.intersection(ref.columns)
    out = {}
    for lag in lags:
        moved = ref.copy()
        moved.index = moved.index - pd.Timedelta(days=lag)  # reference D+lag now sits at D
        per_region = []
        for r in regions:
            pair = pd.concat([rain[r], moved[r]], axis=1, join="inner").dropna()
            if len(pair) < MIN_PAIRED_DAYS:
                continue
            x, y = pair.iloc[:, 0].rank(), pair.iloc[:, 1].rank()
            if x.std() == 0 or y.std() == 0:
                continue
            per_region.append(x.corr(y))
        out[lag] = float(np.mean(per_region)) if per_region else float("nan")
    return out


def check_attribution(rain_rows: pd.DataFrame, reference_rows: pd.DataFrame) -> dict[int, float]:
    """`attribution_lag_correlations`, refusing unless the correlation peaks at lag 0."""
    corr = attribution_lag_correlations(rain_rows, reference_rows)
    finite = {k: v for k, v in corr.items() if np.isfinite(v)}
    if not finite:
        raise ValueError(f"cannot check which day the rain is on: no region had "
                         f"{MIN_PAIRED_DAYS}+ paired days with any variation")
    peak = max(finite, key=finite.get)
    if peak != 0:
        shown = {k: round(v, 3) for k, v in corr.items()}
        raise ValueError(
            f"rainfall correlates best with the reference at lag {peak:+d} day(s), not 0 "
            f"({shown}) - it is on the wrong model day. Refusing to write it; see "
            f"'Which day IMD's rain belongs to' in fetch_imd_district_rainfall.py.")
    return corr


def imd_to_long(imd_obj) -> pd.DataFrame:
    """An imdlib IMD object (or anything with the same `.data`/`.lat_array`/`.lon_array`/
    `.start_day` attributes) -> long rows of (lat, lon, date, precip_mm).

    `.data` is (days, lon, lat) - imdlib's own axis order, not this project's (lat, lon).
    The sentinel -999.0 becomes NaN: a sea/no-gauge cell is missing, never a real zero.
    """
    data = imd_obj.data
    n_days = data.shape[0]
    dates = pd.date_range(imd_obj.start_day, periods=n_days)
    lon_arr, lat_arr = imd_obj.lon_array, imd_obj.lat_array

    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr, indexing="ij")
    rows = []
    for i, date in enumerate(dates):
        day = data[i]
        vals = np.where(day == MISSING, np.nan, day)
        rows.append(pd.DataFrame({
            "lat": lat_grid.ravel(),
            "lon": lon_grid.ravel(),
            "date": date,
            "precip_mm": vals.ravel(),
        }))
    return pd.concat(rows, ignore_index=True)


def merge_precip(base: pd.DataFrame, imd_districts: pd.DataFrame) -> pd.DataFrame:
    """Replace `base`'s precip_mm with IMD's, filing IMD's date D under model date D-1.

    IMD dates a gauge day by the END of its 0830 IST window, so the rain IMD labels D fell
    mostly during model day D-1 - 21 of its 24 hours (see "Which day IMD's rain belongs
    to"). Joining D to D, as this function once did, verified every forecast against
    mostly the previous day's rain. Model date 31 Dec therefore needs IMD's 1 Jan of the
    following year.

    Refuses if IMD covers fewer (region_id, date) pairs than `base` - a real coverage gap,
    not a row to drop silently (CLAUDE.md rule 3: refuse rather than patch). A district-date
    IMD covers but has no valid gauge cell for stays NaN, which is different from IMD never
    having been asked about it at all.
    """
    base = base.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.date
    imd = imd_districts.copy()
    imd["date"] = (pd.to_datetime(imd["date"]) + IMD_DATE_TO_MODEL_DATE).dt.date

    base_keys = set(zip(base["region_id"], base["date"]))
    imd_keys = set(zip(imd["region_id"], imd["date"]))
    missing = base_keys - imd_keys
    if missing:
        sample = sorted(missing)[:5]
        hint = ""
        if any(d.month == 12 and d.day == 31 for _, d in missing):
            hint = (" Model date 31 Dec takes IMD's 1 Jan of the following year - read that "
                    "year's IMD file as well.")
        raise ValueError(
            f"IMD is missing {len(missing)} of {len(base_keys)} (region_id, date) pairs "
            f"the base file covers, e.g. {sample} - refusing a silently narrower merge.{hint}")

    imd_lookup = imd.set_index(["region_id", "date"])["precip_mm"]
    key = list(zip(base["region_id"], base["date"]))
    base["precip_mm"] = [imd_lookup.get(k, np.nan) for k in key]
    base["source"] = base["source"].astype(str) + " | precip_mm: " + SOURCE_IMD
    return base


def build(years: list[int]) -> None:
    import imdlib as imd

    cells = grid_cells()
    wanted = set(zip(np.round(cells.lat, 4), np.round(cells.lon, 4)))

    for year in years:
        # The ERA5 file to merge INTO - never an IMD file. The default lookup would return
        # one, and refuses a stale one, which would block the regeneration that fixes it.
        base_path = ib.observation_file(year, include_imd=False)
        if base_path is None:
            print(f"SKIP {year}: no existing ERA5-family observation file to merge into "
                  f"(run fetch_era5_cds_district_observations.py first)", file=sys.stderr)
            continue

        cache = OUT_DIR / "_imd"
        cache.mkdir(parents=True, exist_ok=True)
        # Two IMD years: model date 31 Dec is IMD's 1 Jan of the following year.
        print(f"downloading IMD rain {year}-{year + 1} ...")
        obj = imd.get_data("rain", year, year + 1, fn_format="yearwise", file_dir=str(cache))

        long = imd_to_long(obj)
        key = list(zip(np.round(long.lat, 4), np.round(long.lon, 4)))
        long = long[[k in wanted for k in key]]

        imd_districts = to_districts(long, cells, value_columns=["precip_mm"])

        base = (pd.read_parquet(base_path) if base_path.suffix == ".parquet"
                else pd.read_csv(base_path))
        merged = merge_precip(base, imd_districts)

        # Refuse to write rain that is not on the day ERA5 says it fell.
        cols = ["region_id", "date", "precip_mm"]
        corr = check_attribution(merged[cols], base[cols])
        print("  dating check against ERA5, mean Spearman by lag: " +
              "  ".join(f"{k:+d}:{v:.3f}" for k, v in sorted(corr.items())))

        out = OUT_DIR / f"{ib.IMD_ALIGNED_STEM.format(year=year)}.parquet"
        merged.to_parquet(out, index=False)
        delta = (merged["precip_mm"] - base["precip_mm"]).abs()
        print(f"  -> {out.name}  {len(merged):,} rows  "
              f"mean |IMD - ERA5| precip_mm = {delta.mean():.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", required=True, help="2018, or 2016-2019")
    args = ap.parse_args()

    years: set[int] = set()
    for chunk in args.years.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            years.update(range(int(a), int(b) + 1))
        elif chunk:
            years.add(int(chunk))

    build(sorted(years))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
