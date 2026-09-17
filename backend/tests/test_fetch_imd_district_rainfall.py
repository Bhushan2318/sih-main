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


# ---------------------------------------------------------------------------
# Which day IMD files its rain under - and so which model day it joins.
#
# IMD's gauge day runs 0830 IST to 0830 IST, and IMD labels it by the day the window
# ENDS: the value dated D is rain for 0830 IST on D-1 -> 0830 IST on D. That was
# measured against two independent hourly reanalyses (docs/known-issues.md), and it
# contradicts the start-day convention the project brief stated. Joining IMD's D to
# model day D - which this script did - verified every rainfall forecast against
# mostly the previous day's rain: the same class of bug CLAUDE.md rule 4 records.
#
# merge_precip therefore files IMD's date D under model date D-1. The dates below are
# position markers, not rainfall: distinct values per day, so an off-by-one shows up
# as a wrong number rather than an equal one.
# ---------------------------------------------------------------------------

def _base(days, region="d1", **extra):
    n = len(days)
    return pd.DataFrame({"region_id": [region] * n, "date": pd.to_datetime(days).date,
                         "precip_mm": [0.0] * n, "source": ["ERA5"] * n, **extra})


def _imd(days, values, region="d1"):
    return pd.DataFrame({"region_id": [region] * len(days),
                         "date": pd.to_datetime(days).date, "precip_mm": values})


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
    # IMD dates one day later than the model dates they fill.
    imd_districts = pd.DataFrame({
        "region_id": ["d1", "d1", "d2"],
        "date": pd.to_datetime(["2018-01-02", "2018-01-03", "2018-01-02"]).date,
        "precip_mm": [10.0, 20.0, 30.0],
    })
    merged = fir.merge_precip(base, imd_districts)
    assert list(merged["precip_mm"]) == [10.0, 20.0, 30.0]
    # Every other column is untouched, dates included.
    assert list(merged["t2m_c"]) == [25.0, 26.0, 24.0]
    assert list(merged["date"]) == list(base["date"])
    assert (merged["source"] != base["source"]).any(), (
        "source column should record that precip_mm came from IMD, not silently keep "
        "the ERA5 attribution on a column ERA5 no longer supplied")


def test_merge_precip_leaves_a_row_nan_when_imd_has_no_value_for_it():
    """A district-date IMD could not cover must not fall back to ERA5's own number -
    that would silently blend two rainfall products under one column with no record of
    which one produced which row."""
    base = _base(["2018-01-01"]); base["precip_mm"] = [1.0]
    merged = fir.merge_precip(base, _imd(["2018-01-02"], [np.nan]))
    assert pd.isna(merged["precip_mm"].iloc[0])


def test_merge_precip_refuses_a_district_date_imd_does_not_cover_at_all():
    """IMD covering fewer district-dates than the base file is a real gap, not a row to
    drop silently - refuse rather than publish a file quietly narrower than it claims."""
    base = pd.concat([_base(["2018-01-01"], "d1"), _base(["2018-01-01"], "d2")])
    with pytest.raises(ValueError, match="d2"):
        fir.merge_precip(base, _imd(["2018-01-02"], [10.0], "d1"))


def test_merge_precip_files_imd_date_d_under_model_date_d_minus_1():
    base = _base(["2018-08-14", "2018-08-15"])
    imd = _imd(["2018-08-14", "2018-08-15", "2018-08-16"], [111.0, 222.0, 333.0])
    merged = fir.merge_precip(base, imd).sort_values("date")
    # model 14 Aug <- IMD's 15 Aug (0830 IST 14th -> 0830 IST 15th); model 15th <- IMD 16th
    assert list(merged["precip_mm"]) == [222.0, 333.0]


def test_merge_precip_takes_31_december_from_the_following_years_first_imd_day():
    base = _base(["2018-12-30", "2018-12-31"])
    imd = _imd(["2018-12-31", "2019-01-01"], [555.0, 666.0])
    merged = fir.merge_precip(base, imd).sort_values("date")
    assert list(merged["precip_mm"]) == [555.0, 666.0]


def test_merge_precip_refuses_31_december_without_the_following_years_imd():
    """A single year of IMD cannot fill its own last model day. Refuse, and say why -
    the generic coverage message would point nowhere near the year boundary."""
    base = _base(["2018-12-30", "2018-12-31"])
    with pytest.raises(ValueError, match="following year"):
        fir.merge_precip(base, _imd(["2018-12-31"], [555.0]))


def test_merge_precip_crosses_the_leap_day():
    base = _base(["2016-02-28", "2016-02-29"])
    merged = fir.merge_precip(base, _imd(["2016-02-29", "2016-03-01"], [777.0, 888.0]))
    assert list(merged.sort_values("date")["precip_mm"]) == [777.0, 888.0]


# --- The windows behind that one-day shift. Pure timestamp arithmetic. ----------------

IST_OFFSET_HOURS = 5.5


def test_imd_value_dated_d_ends_at_0830_ist_on_d():
    start, end = fir.imd_rain_day_window_utc("2018-08-15")
    assert start == pd.Timestamp("2018-08-14 03:00", tz="UTC")
    assert end == pd.Timestamp("2018-08-15 03:00", tz="UTC")
    # ...and 0300 UTC really is 0830 IST, rather than a number someone typed.
    assert (end + pd.Timedelta(hours=IST_OFFSET_HOURS)).strftime("%H:%M") == "08:30"


def test_model_day_is_midnight_to_midnight_utc():
    """CLAUDE.md rule 4: day k is forecast hours ((k-1)*24, k*24], so the model day is
    midnight-to-midnight UTC. The ERA5 fetch matches it - see to_daily()."""
    start, end = fir.model_day_window_utc("2018-08-15")
    assert start == pd.Timestamp("2018-08-15 00:00", tz="UTC")
    assert end == pd.Timestamp("2018-08-16 00:00", tz="UTC")


def test_both_windows_span_exactly_24_hours():
    for fn in (fir.model_day_window_utc, fir.imd_rain_day_window_utc):
        start, end = fn("2018-08-15")
        assert end - start == pd.Timedelta(hours=24), fn.__name__


def test_imd_date_d_belongs_to_model_date_d_minus_1_by_a_21_hour_majority():
    """The whole join rests on this ratio. If it ever inverts, merge_precip is joining
    the wrong days - and the constant it applies must agree with the arithmetic."""
    imd = fir.imd_rain_day_window_utc("2018-08-15")
    assert fir.window_overlap_hours(imd, fir.model_day_window_utc("2018-08-14")) == 21.0
    assert fir.window_overlap_hours(imd, fir.model_day_window_utc("2018-08-15")) == 3.0
    assert fir.window_overlap_hours(imd, fir.model_day_window_utc("2018-08-13")) == 0.0
    assert fir.IMD_DATE_TO_MODEL_DATE == pd.Timedelta(days=-1)


def test_the_residual_offset_is_exactly_three_hours_every_day_of_the_year():
    """India is UTC+05:30 all year - no daylight saving - so this never drifts."""
    for day in ("2016-01-01", "2016-02-29", "2016-03-01", "2018-12-31", "2019-07-04"):
        model_start, _ = fir.model_day_window_utc(pd.Timestamp(day) + fir.IMD_DATE_TO_MODEL_DATE)
        imd_start, _ = fir.imd_rain_day_window_utc(day)
        assert imd_start - model_start == pd.Timedelta(hours=3), day


def test_windows_take_the_calendar_date_parquet_actually_stores():
    """merge_precip reads dates back as datetime.date, so that is what real callers pass."""
    import datetime as dt
    assert fir.imd_rain_day_window_utc(dt.date(2018, 8, 15)) == \
        fir.imd_rain_day_window_utc("2018-08-15")


def test_windows_refuse_a_timezone_aware_timestamp():
    """An aware timestamp has no single calendar day: 02:00 IST on the 15th is 20:30 UTC
    on the 14th. Converting it silently returned the wrong day's window - the exact
    off-by-one error this code exists to prevent. Refuse rather than guess (rule 3)."""
    ist = pd.Timestamp("2018-08-15 02:00", tz="Asia/Kolkata")
    for fn in (fir.model_day_window_utc, fir.imd_rain_day_window_utc):
        with pytest.raises(ValueError, match="calendar date"):
            fn(ist)
    with pytest.raises(ValueError, match="calendar date"):
        fir.imd_rain_day_window_utc(pd.Timestamp("2018-08-15", tz="UTC"))


# --- The attribution check, which build() runs before writing anything ----------------
#
# Arithmetic cannot tell start-day from end-day dating; that is a fact about IMD's
# files. Rain timing can. Rainfall on the right model day correlates best with an
# independent reanalysis on the SAME day (lag 0); a file one day out peaks at lag -1 or
# +1. build() refuses to write a merge whose peak is not at lag 0.

def _real_era5_sample():
    """Real ERA5 rainfall (Open-Meteo, CC-BY 4.0) the repo already commits: 36 cities,
    ~380 days. Used here only to prove the check detects a one-day re-dating."""
    df = pd.read_parquet(BACKEND / "data" / "samples" / "era5_observations_india_2019.parquet")
    return df.rename(columns={"city": "region_id"})[["region_id", "date", "precip_mm"]]


def _redated(rows, days):
    out = rows.copy()
    out["date"] = (pd.to_datetime(out["date"]) + pd.Timedelta(days=days)).dt.date
    return out


def test_attribution_check_peaks_at_lag_0_for_correctly_dated_rain():
    rain = _real_era5_sample()
    corr = fir.attribution_lag_correlations(rain, rain)
    assert max(corr, key=corr.get) == 0, corr
    assert fir.check_attribution(rain, rain) == corr


@pytest.mark.parametrize("days,expected_peak", [(1, -1), (-1, 1)])
def test_attribution_check_detects_a_one_day_re_dating(days, expected_peak):
    """The mutation that matters: the same real rain, one day out, in either direction.
    A check that cannot fail here would prove nothing about the real merge."""
    rain = _real_era5_sample()
    corr = fir.attribution_lag_correlations(_redated(rain, days), rain)
    assert max(corr, key=corr.get) == expected_peak, corr
    with pytest.raises(ValueError, match="lag"):
        fir.check_attribution(_redated(rain, days), rain)


# --- Against the real IMD archive, where it exists on disk ---------------------------
#
# The check above proves the statistic works. This one applies it to real IMD rainfall,
# aligned by merge_precip, against the real ERA5 CDS file for the same year. Both are
# gitignored and large, so this skips on a fresh clone and in CI.

def _aligned_years():
    pairs = []
    for aligned in sorted(fir.OUT_DIR.glob(f"{fir.ALIGNED_STEM}_*.parquet")):
        year = aligned.stem.rsplit("_", 1)[-1]
        era5 = fir.OUT_DIR / f"era5_cds_district_observations_india_{year}.parquet"
        if era5.exists():
            pairs.append((year, aligned, era5))
    return pairs


def test_aligned_imd_rainfall_lands_on_the_same_day_as_era5(capsys):
    pairs = _aligned_years()
    if not pairs:
        pytest.skip("no aligned IMD file with a matching ERA5 CDS file on this machine - "
                    "run scripts/fetch_imd_district_rainfall.py --years <year> first")
    for year, aligned, era5 in pairs:
        cols = ["region_id", "date", "precip_mm"]
        imd_rows, era5_rows = pd.read_parquet(aligned)[cols], pd.read_parquet(era5)[cols]
        corr = fir.attribution_lag_correlations(imd_rows, era5_rows)
        with capsys.disabled():
            print(f"\n  {year}: mean Spearman by lag " +
                  "  ".join(f"{k:+d}:{v:.3f}" for k, v in sorted(corr.items())))
        assert max(corr, key=corr.get) == 0, (
            f"{aligned.name} peaks at lag {max(corr, key=corr.get):+d}, not 0 - its rainfall "
            f"is on the wrong day relative to ERA5: {corr}")
        # And the unshifted join this script used to make must be detectably worse.
        stale = _redated(imd_rows, 1)
        assert fir.attribution_lag_correlations(stale, era5_rows)[0] < corr[0]
