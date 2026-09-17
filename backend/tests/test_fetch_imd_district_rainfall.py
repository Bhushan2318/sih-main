"""IMD gridded rainfall -> district precip_mm, merged into an existing ERA5-family
observation file.

Why merge rather than write a standalone file: IMD only publishes rain (and tmin/tmax,
which this project does not use it for - CLAUDE.md's own plan is "IMD ... for
precipitation, ERA5 ... for everything else"). A precip_mm-only file would either fail
`ingest_upload`'s completeness expectations or silently drop the other eight canonical
variables. Replacing just the precip_mm column of an already-fetched CDS/ERA5 file keeps
every other variable untouched and lets `ingest_backfill.py` treat this exactly like any
other observation source.

The grid arrays below are small hand-built stand-ins for IMD's real (365/366, 135, 129)
array - shape and masking logic only, never a value that reaches a metric.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fetch_imd_district_rainfall", BACKEND / "scripts" / "fetch_imd_district_rainfall.py")
fir = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fir
_spec.loader.exec_module(fir)


class _FakeIMD:
    """Stands in for an imdlib.IMD object: same attribute names, a tiny synthetic grid."""

    def __init__(self, data, lat_array, lon_array, start_day):
        self.data = data  # shape (days, lon, lat), IMD's own axis order
        self.lat_array = lat_array
        self.lon_array = lon_array
        self.start_day = start_day


def _tiny_imd(n_days=3):
    # 2 lon x 2 lat, one cell always missing (-999), one cell real rain.
    lon = np.array([70.0, 70.25])
    lat = np.array([20.0, 20.25])
    data = np.full((n_days, 2, 2), -999.0)
    data[:, 0, 0] = [0.0, 5.5, 12.3]  # (lon=70.0, lat=20.0): real values
    # (lon=70.25, lat=20.0) and both lat=20.25 cells stay -999 (sea / no data)
    return _FakeIMD(data, lat, lon, "2018-01-01")


def test_imd_grid_to_long_masks_the_missing_flag_and_keeps_real_values():
    obj = _tiny_imd(n_days=3)
    long = fir.imd_to_long(obj)
    assert set(long.columns) >= {"lat", "lon", "date", "precip_mm"}
    real = long[(long.lat == 20.0) & (long.lon == 70.0)].sort_values("date")
    assert list(real["precip_mm"]) == [0.0, 5.5, 12.3]
    missing = long[(long.lat == 20.25) & (long.lon == 70.25)]
    assert missing["precip_mm"].isna().all()


def test_imd_grid_to_long_dates_advance_from_start_day():
    obj = _tiny_imd(n_days=3)
    long = fir.imd_to_long(obj)
    got = sorted(long[(long.lat == 20.0) & (long.lon == 70.0)]["date"].unique())
    assert list(pd.to_datetime(got)) == list(pd.date_range("2018-01-01", periods=3))


def test_merge_precip_replaces_only_that_column():
    base = pd.DataFrame({
        "region_id": ["d1", "d1", "d2"],
        "region_name": ["D1", "D1", "D2"],
        "state_id": ["s1", "s1", "s2"],
        "state_name": ["S1", "S1", "S2"],
        "latitude": [20.0, 20.0, 21.0],
        "longitude": [70.0, 70.0, 71.0],
        "date": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-01"]).date,
        "t2m_c": [25.0, 26.0, 24.0],
        "precip_mm": [1.0, 2.0, 3.0],  # ERA5's own rainfall - must be overwritten
        "source": ["ERA5 hourly single levels via Copernicus CDS"] * 3,
    })
    imd_districts = pd.DataFrame({
        "region_id": ["d1", "d1", "d2"],
        "date": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-01"]).date,
        "precip_mm": [10.0, 20.0, 30.0],
    })
    merged = fir.merge_precip(base, imd_districts)
    assert list(merged["precip_mm"]) == [10.0, 20.0, 30.0]
    # Every other column is untouched.
    assert list(merged["t2m_c"]) == [25.0, 26.0, 24.0]
    assert (merged["source"] != base["source"]).any(), (
        "source column should record that precip_mm came from IMD, not silently keep "
        "the ERA5 attribution on a column ERA5 no longer supplied")


def test_merge_precip_leaves_a_row_nan_when_imd_has_no_value_for_it():
    """A district-date IMD could not cover must not fall back to ERA5's own number -
    that would silently blend two rainfall products under one column with no record of
    which one produced which row."""
    base = pd.DataFrame({
        "region_id": ["d1"], "date": [pd.Timestamp("2018-01-01").date()],
        "precip_mm": [1.0], "source": ["ERA5"],
    })
    imd_districts = pd.DataFrame({
        "region_id": ["d1"], "date": [pd.Timestamp("2018-01-01").date()],
        "precip_mm": [np.nan],
    })
    merged = fir.merge_precip(base, imd_districts)
    assert pd.isna(merged["precip_mm"].iloc[0])


def test_merge_precip_refuses_a_district_date_imd_does_not_cover_at_all():
    """IMD covering fewer district-dates than the base file is a real gap, not a row to
    drop silently - refuse rather than publish a file quietly narrower than it claims."""
    base = pd.DataFrame({
        "region_id": ["d1", "d2"], "date": [pd.Timestamp("2018-01-01").date()] * 2,
        "precip_mm": [1.0, 2.0], "source": ["ERA5", "ERA5"],
    })
    imd_districts = pd.DataFrame({
        "region_id": ["d1"], "date": [pd.Timestamp("2018-01-01").date()], "precip_mm": [10.0],
    })
    with pytest.raises(ValueError, match="d2"):
        fir.merge_precip(base, imd_districts)


# ---------------------------------------------------------------------------
# The 0830 IST accumulation window.
#
# IMD's gauge day and Sanket's forecast day are not the same 24 hours, and until
# now nothing said so. These tests pin the alignment so that a later "tidy-up"
# cannot silently shift every rainfall bust label by a day - the exact class of
# bug CLAUDE.md rule 4 already records once.
#
# Everything down to the event test is pure timestamp arithmetic: no rainfall
# values, no fixtures, nothing that could be mistaken for synthetic data.
# ---------------------------------------------------------------------------

IST_OFFSET_HOURS = 5.5


def test_imd_rain_day_starts_at_0830_ist_which_is_0300_utc():
    start, _ = fir.imd_rain_day_window_utc("2018-08-15")
    assert start == pd.Timestamp("2018-08-15 03:00", tz="UTC")
    # ...and 0300 UTC really is 0830 IST, rather than a number someone typed.
    assert (start + pd.Timedelta(hours=IST_OFFSET_HOURS)).strftime("%H:%M") == "08:30"


def test_model_day_starts_at_midnight_utc():
    """CLAUDE.md rule 4: day k is forecast hours ((k-1)*24, k*24], so the model day
    is midnight-to-midnight UTC. The ERA5 observation fetch matches it deliberately -
    see to_daily() in fetch_era5_cds_district_observations.py."""
    start, end = fir.model_day_window_utc("2018-08-15")
    assert start == pd.Timestamp("2018-08-15 00:00", tz="UTC")
    assert end == pd.Timestamp("2018-08-16 00:00", tz="UTC")


def test_both_windows_span_exactly_24_hours():
    for fn in (fir.model_day_window_utc, fir.imd_rain_day_window_utc):
        start, end = fn("2018-08-15")
        assert end - start == pd.Timedelta(hours=24), fn.__name__


def test_imd_date_d_pairs_with_model_date_d_by_a_21_hour_majority():
    """The whole join rests on this. IMD's day D shares 21 of its 24 hours with the
    model's day D and only 3 with the model's day D+1, so IMD[D] pairs with model[D].
    If this ratio ever inverts, merge_precip is joining the wrong days."""
    imd = fir.imd_rain_day_window_utc("2018-08-15")
    assert fir.window_overlap_hours(imd, fir.model_day_window_utc("2018-08-15")) == 21.0
    assert fir.window_overlap_hours(imd, fir.model_day_window_utc("2018-08-16")) == 3.0
    assert fir.window_overlap_hours(imd, fir.model_day_window_utc("2018-08-14")) == 0.0


def test_the_offset_is_exactly_three_hours_every_day_of_the_year():
    """India is UTC+05:30 all year - no daylight saving - so this never drifts.
    Includes a leap day, since 2016 is inside the project's scope."""
    for day in ("2016-01-01", "2016-02-29", "2016-06-21", "2018-12-31", "2019-07-04"):
        model_start, _ = fir.model_day_window_utc(day)
        imd_start, _ = fir.imd_rain_day_window_utc(day)
        assert imd_start - model_start == pd.Timedelta(hours=3), day


def test_windows_take_the_calendar_date_parquet_actually_stores():
    """merge_precip reads dates back as datetime.date, so that is what real callers pass."""
    import datetime as dt
    assert fir.imd_rain_day_window_utc(dt.date(2018, 8, 15)) == \
        fir.imd_rain_day_window_utc("2018-08-15")


def test_windows_refuse_a_timezone_aware_timestamp():
    """An aware timestamp has no single calendar day: 02:00 IST on the 15th is 20:30 UTC
    on the 14th. Converting it silently returned the 14th's window - the exact
    off-by-one-day error this code exists to prevent, triggered by precisely the input
    an Indian team would reach for. Refuse rather than guess (CLAUDE.md rule 3)."""
    ist = pd.Timestamp("2018-08-15 02:00", tz="Asia/Kolkata")
    for fn in (fir.model_day_window_utc, fir.imd_rain_day_window_utc):
        with pytest.raises(ValueError, match="calendar date"):
            fn(ist)
    with pytest.raises(ValueError, match="calendar date"):
        fir.imd_rain_day_window_utc(pd.Timestamp("2018-08-15", tz="UTC"))


def test_merge_precip_joins_imd_date_to_the_same_model_date():
    """The behavioural half of the join rule: merge_precip must not shift dates while
    swapping the column. A row dated D takes IMD's value for D - not D-1, not D+1.

    Distinct values per day, so an off-by-one shows up as a wrong value rather than
    an equal one. These are not rainfall measurements; they are position markers.
    """
    days = pd.to_datetime(["2018-08-14", "2018-08-15", "2018-08-16"]).date
    base = pd.DataFrame({
        "region_id": ["d1"] * 3,
        "date": days,
        "precip_mm": [0.0, 0.0, 0.0],
        "source": ["ERA5"] * 3,
    })
    imd_districts = pd.DataFrame({
        "region_id": ["d1"] * 3,
        "date": days,
        "precip_mm": [111.0, 222.0, 333.0],
    })
    merged = fir.merge_precip(base, imd_districts).sort_values("date")
    assert list(merged["precip_mm"]) == [111.0, 222.0, 333.0]


# --- The one that catches a whole-day error, not a three-hour one ------------
#
# Everything above assumes IMD attributes a 0830-to-0830 accumulation to the day it
# STARTS. Some IMD products instead file the 0830 reading under the day it was TAKEN,
# which is a full day out. No amount of timestamp arithmetic can tell the two apart -
# only real rainfall on a date we independently know can.
#
# Needs a merged parquet on disk. It is skipped on a fresh clone and in CI, and is
# NOT evidence until someone with the archive runs it and pastes the output.

IMD_EVENTS = [
    # (district name fragment, first documented day, last documented day)
    ("Idukki", "2018-08-15", "2018-08-17"),        # Kerala floods
    ("Wayanad", "2018-08-15", "2018-08-17"),       # Kerala floods
    ("Kanniyakumari", "2017-11-29", "2017-12-01"),  # Cyclone Ockhi landfall rains
]


def _merged_parquet(year: int):
    path = fir.OUT_DIR / f"imd_merged_district_observations_india_{year}.parquet"
    return path if path.exists() else None


@pytest.mark.parametrize("name_fragment,first_day,last_day", IMD_EVENTS)
def test_documented_extreme_rain_lands_on_its_documented_date(
        name_fragment, first_day, last_day, capsys):
    """A day-scale attribution error moves the peak outside the documented window.

    Searches a +/- 3 day margin around the event so the peak has somewhere wrong to
    land; asserts it landed inside the documented days.
    """
    year = int(first_day[:4])
    path = _merged_parquet(year)
    if path is None:
        pytest.skip(f"no merged IMD parquet for {year} on this machine - run "
                    f"scripts/fetch_imd_district_rainfall.py --years {year} first")

    from app.utils import india_districts as idist

    matches = [d for d in idist.load_registry()
               if name_fragment.lower() in d.region_name.lower()]
    assert matches, f"no district whose name contains {name_fragment!r}"
    region_ids = {d.region_id for d in matches}

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    lo = pd.Timestamp(first_day) - pd.Timedelta(days=3)
    hi = pd.Timestamp(last_day) + pd.Timedelta(days=3)
    window = df[df["region_id"].isin(region_ids)
                & df["date"].between(lo, hi)].dropna(subset=["precip_mm"])
    assert not window.empty, f"no IMD rainfall for {name_fragment} around {first_day}"

    peak = window.loc[window["precip_mm"].idxmax()]
    with capsys.disabled():
        print(f"\n  {name_fragment}: peak {peak['precip_mm']:.1f} mm on "
              f"{peak['date'].date()} (documented {first_day}..{last_day})")

    assert pd.Timestamp(first_day) <= peak["date"] <= pd.Timestamp(last_day), (
        f"{name_fragment} peak rainfall landed on {peak['date'].date()}, outside the "
        f"documented {first_day}..{last_day}. If it is consistently one day late, IMD "
        f"is attributing to the END of its 0830-0830 window and every rainfall bust "
        f"label is a day out.")
