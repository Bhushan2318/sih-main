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
`ingest_backfill.observation_file`) and replaces only its precip_mm column - everything
else (temperature, humidity, wind, pressure, soil moisture, water vapour) keeps coming
from ERA5/CDS. The two products never blend inside one column: `source` is annotated so
it is always traceable which product produced a row's rainfall.

Accumulation windows - the 0830 IST rain day
---------------------------------------------
IMD and Sanket do not mean the same 24 hours by "a day". This is the one place that
difference gets stated rather than assumed.

    Sanket's day D     [D 00:00 UTC, D+1 00:00 UTC)   ==  05:30 IST -> 05:30 IST
    IMD's rain day D   [D 03:00 UTC, D+1 03:00 UTC)   ==  08:30 IST -> 08:30 IST

The model day is fixed by CLAUDE.md rule 4 - day k is forecast hours ((k-1)*24, k*24]
- and the ERA5 observation fetch matches it deliberately: `to_daily` in
fetch_era5_cds_district_observations.py shifts ERA5's end-of-hour accumulation stamp
so that a day is the half-open window (t-24h, t]. IMD's day is a rain-gauge
convention instead: the reading taken at 0830 IST covers the previous 24 hours and is
filed under the day that window STARTED.

India is UTC+05:30 all year, with no daylight saving, so the offset is exactly three
hours, every day.

THE JOIN: IMD's date D pairs with Sanket's date D. Nothing is shifted. That is not a
convenience - IMD's day D shares 21 of its 24 hours with the model's day D and only 3
with the model's day D+1, so the 21 hours outvote the 3. `merge_precip` therefore
joins on (region_id, date) with no offset, and test_fetch_imd_district_rainfall.py
pins that 21/3 ratio so a later tidy-up cannot quietly invert it. Getting this
backwards shifts every rainfall bust label by a day - the same class of bug rule 4
already records once.

WHAT IT COSTS: re-cutting the model day to start at 0300 UTC would match IMD exactly,
and would push Day 10 out to forecast hour 243 - past the 240-hour end of the GEFSv12
reforecast archive. That trades rainfall at the longest lead for three hours at the
edge of the window. The residual three-hour mismatch is a known limitation, written
down in docs/known-issues.md rather than left implicit.

NOT SETTLED HERE: that IMD attributes to the starting day is taken from IMD's own
documentation of the gridded product. Some IMD products file the 0830 reading under
the day it was taken instead, which is a whole day out and which no amount of
timestamp arithmetic can detect. The event tests in
test_fetch_imd_district_rainfall.py check it against real rainfall on dates we know
independently; they need a merged parquet on disk and skip without one.

These windows live in this script because it is their only consumer. If a second IMD
path appears, move them into app/utils/ - do not copy them.

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
SOURCE_IMD = "IMD gauge-based gridded rainfall 0.25 deg (imdpune.gov.in)"
MISSING = -999.0

# --- Accumulation windows. See "the 0830 IST rain day" in the module docstring. ---
DAY = pd.Timedelta(hours=24)
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)   # India has no daylight saving
MODEL_DAY_START_UTC = pd.Timedelta(hours=0)      # 00:00 UTC, CLAUDE.md rule 4
IMD_DAY_START_UTC = pd.Timedelta(hours=3)        # 08:30 IST


def _utc_midnight(day) -> pd.Timestamp:
    ts = pd.Timestamp(day)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.normalize()


def model_day_window_utc(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Sanket's day `day` as a half-open [start, end) window in UTC.

    Midnight to midnight UTC - the window the GEFS forecast side and the ERA5
    observation side both already use.
    """
    start = _utc_midnight(day) + MODEL_DAY_START_UTC
    return start, start + DAY


def imd_rain_day_window_utc(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    """IMD's gauge rain-day `day` as a half-open [start, end) window in UTC.

    0830 IST to 0830 IST, attributed to the day the window started, so in UTC it
    runs 03:00 to 03:00 - three hours behind `model_day_window_utc` for the same
    calendar date.
    """
    start = _utc_midnight(day) + IMD_DAY_START_UTC
    return start, start + DAY


def window_overlap_hours(a: tuple[pd.Timestamp, pd.Timestamp],
                         b: tuple[pd.Timestamp, pd.Timestamp]) -> float:
    """Hours two half-open [start, end) windows share; 0.0 when they do not touch.

    The join rule in one number: an IMD rain-day overlaps the model day of the same
    date by 21 hours and the next model day by 3.
    """
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    return max(pd.Timedelta(0), overlap).total_seconds() / 3600.0


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
    """Replace `base`'s precip_mm with IMD's, matched on (region_id, date).

    The dates are joined straight across, with NO shift, even though IMD's rain day
    runs 0830-0830 IST and `base`'s day runs midnight-to-midnight UTC. IMD's day D
    shares 21 of its 24 hours with the model's day D and only 3 with day D+1, so
    date D is the right partner for date D. See "the 0830 IST rain day" in the
    module docstring for the full window arithmetic, and
    test_imd_date_d_pairs_with_model_date_d_by_a_21_hour_majority for the assertion
    that stops this drifting.

    Refuses if IMD covers fewer (region_id, date) pairs than `base` - a real coverage gap,
    not a row to drop silently (CLAUDE.md rule 3: refuse rather than patch). A district-date
    IMD covers but has no valid gauge cell for stays NaN, which is different from IMD never
    having been asked about it at all.
    """
    base = base.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.date
    imd = imd_districts.copy()
    imd["date"] = pd.to_datetime(imd["date"]).dt.date

    base_keys = set(zip(base["region_id"], base["date"]))
    imd_keys = set(zip(imd["region_id"], imd["date"]))
    missing = base_keys - imd_keys
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"IMD is missing {len(missing)} of {len(base_keys)} (region_id, date) pairs "
            f"the base file covers, e.g. {sample} - refusing a silently narrower merge")

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
        base_path = ib.observation_file(year)
        if base_path is None:
            print(f"SKIP {year}: no existing ERA5-family observation file to merge into "
                  f"(run fetch_era5_cds_district_observations.py first)", file=sys.stderr)
            continue

        cache = OUT_DIR / "_imd"
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading IMD rain {year} ...")
        obj = imd.get_data("rain", year, year, fn_format="yearwise", file_dir=str(cache))

        long = imd_to_long(obj)
        key = list(zip(np.round(long.lat, 4), np.round(long.lon, 4)))
        long = long[[k in wanted for k in key]]

        imd_districts = to_districts(long, cells, value_columns=["precip_mm"])

        base = (pd.read_parquet(base_path) if base_path.suffix == ".parquet"
                else pd.read_csv(base_path))
        merged = merge_precip(base, imd_districts)

        out = OUT_DIR / f"imd_merged_district_observations_india_{year}.parquet"
        merged.to_parquet(out, index=False)
        delta = (merged["precip_mm"] - pd.read_parquet(base_path)["precip_mm"]).abs()
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
