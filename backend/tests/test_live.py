"""Phase 6 (live ingestion) tests.

Deliberately offline: nothing here touches NOAA or Open-Meteo. The network-dependent
behaviour is exercised by the fetch scripts themselves; what matters to protect here is the
logic that decides *what* to fetch, *how* samples are grouped into days, and the guards
that keep live data consistent with what the models were trained on.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from app.config import settings
from app.ingestion.canonical_schema import CANONICAL_COLUMNS
from app.live import gefs
from app.live.orchestrator import FORECAST_MAPPINGS, OBSERVATION_MAPPINGS, cycle_target
from app.storage import parquet_store


# ---------------------------------------------------------------- cycle selection

def test_latest_expected_cycle_respects_publication_lag():
    """A cycle is only considered available once the publication lag has elapsed."""
    # 12:44 UTC with a 5.5 h lag: 06Z (issued 6.7 h ago) is ready, 12Z is not.
    now = datetime(2026, 8, 29, 12, 44, tzinfo=timezone.utc)
    assert gefs.latest_expected_cycle(now, lag_hours=5.5) == (date(2026, 8, 29), "06")

    # Just after midnight, the newest ready cycle is the previous day's 18Z.
    now = datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc)
    assert gefs.latest_expected_cycle(now, lag_hours=5.5) == (date(2026, 8, 28), "18")


def test_recent_cycles_walks_backwards_without_repeats():
    now = datetime(2026, 8, 29, 12, 44, tzinfo=timezone.utc)
    cycles = gefs.recent_cycles(5, now, lag_hours=5.5, stride_hours=24)
    assert len(cycles) == len(set(cycles)) == 5
    assert cycles[0] == (date(2026, 8, 29), "06")
    assert cycles[-1] == (date(2026, 8, 25), "06")


def test_cycle_target_is_hour_specific():
    """00Z and 06Z of one day are different cycles; a date alone cannot separate them."""
    assert cycle_target(date(2026, 8, 29), "00") != cycle_target(date(2026, 8, 29), "06")


# ---------------------------------------------------------------- day grouping

def test_00z_lead_days_reproduce_the_training_windows():
    """For a 00Z cycle, lead day k must be exactly forecast hours ((k-1)*24, k*24].

    This is the window the training data was built from, so any drift here would put live
    forecasts on a different footing from the data the regressors learned on.
    """
    init = datetime(2026, 8, 29, 0, tzinfo=timezone.utc)
    by_day = gefs._steps_by_day(init, gefs.STEP_HOURS)
    for k in range(1, 11):
        assert by_day[k] == list(range((k - 1) * 24 + 3, k * 24 + 1, 3)), f"lead {k}"


def test_valid_date_is_init_plus_lead_minus_one():
    """The convention fixed on 2026-08-29: lead day 1 is the init day itself for a 00Z run.

    Labelling it init+lead instead put every forecast a day after its own contents and
    verified it against the wrong day's observation.
    """
    init = datetime(2026, 8, 29, 0, tzinfo=timezone.utc)
    for fh in (3, 24):                     # first and last sample of lead day 1
        day, lead = gefs._lead_day_of(init, fh)
        assert lead == 1
        assert day == init.date() + timedelta(days=lead - 1) == date(2026, 8, 29)
    day, lead = gefs._lead_day_of(init, 240)
    assert (lead, day) == (10, date(2026, 9, 7))


def test_non_00z_cycles_group_by_calendar_day():
    """A 06Z run's lead days must still be whole calendar days.

    Observations are calendar-day aggregates, so a lead-hour window (which for a 06Z run
    straddles midnight) could never line up with the observation verifying it.
    """
    init = datetime(2026, 8, 29, 6, tzinfo=timezone.utc)
    by_day = gefs._steps_by_day(init, gefs.STEP_HOURS)
    complete = [k for k, hrs in by_day.items() if len(hrs) == 8 and 1 <= k <= 10]
    # Lead 1 is partial (the run starts 6 h into that day) and is dropped; 10 exceeds +240h.
    assert complete == list(range(2, 11))
    for k in complete:
        first, last = by_day[k][0], by_day[k][-1]
        assert (init + timedelta(hours=last)).date() - timedelta(days=0)
        # every sample of a lead day belongs to that same calendar day
        days = {gefs._lead_day_of(init, h)[0] for h in by_day[k]}
        assert len(days) == 1


def test_accumulation_uses_non_overlapping_buckets_only():
    """APCP buckets reset every 6 h, so the 3-hourly steps OVERLAP (0-3 and 0-6 both
    include hours 0-3). Summing every 3-hourly value would roughly double daily rainfall.
    A day must be tiled by exactly four 6-hourly buckets."""
    assert gefs.VAR_SPEC["apcp_mm"]["step_multiple"] == 6
    assert gefs.VAR_SPEC["apcp_mm"]["agg"] == "sum"
    init = datetime(2026, 8, 29, 0, tzinfo=timezone.utc)
    buckets = gefs._steps_by_day(init, 6)
    assert buckets[1] == [6, 12, 18, 24]
    for k in range(1, 11):
        assert len(buckets[k]) == gefs._expected_samples(6) == 4


# ---------------------------------------------------------------- consistency guards

def test_member_set_matches_what_the_models_were_trained_on():
    """Operational GEFS has 31 members; the reforecast the models learned from has 5.

    Ensemble spread grows with member count and spread_* are classifier inputs, so scoring
    a 31-member spread with a 5-member-trained model would bias its strongest features.
    """
    assert len(settings.gefs_member_list) == 5
    assert settings.gefs_member_list[0] == "gec00"


def test_only_mslp_becomes_canonical_pressure():
    """Both live files carry mslp_hpa and psfc_hpa; only mean-sea-level pressure is the
    canonical pressure variable. Surface pressure is dominated by elevation (Leh reads
    ~684 hPa), so mixing them would make the pressure model meaningless."""
    for mappings in (FORECAST_MAPPINGS, OBSERVATION_MAPPINGS):
        by_col = {m["source_column"]: m for m in mappings}
        assert by_col["mslp_hpa"]["variable"] == "pressure_hpa"
        assert by_col["psfc_hpa"].get("role") == "unmapped"
        assert "variable" not in by_col["psfc_hpa"]


def test_live_mappings_cover_every_measurement_column():
    """Live ingestion passes explicit mappings rather than trusting the schema mapper, so
    an unattended job cannot silently mis-map a column. Every canonical variable the
    models expect must therefore be present."""
    expected = {
        "temperature_c", "humidity_pct", "rainfall_mm", "pressure_hpa",
        "atmospheric_moisture_kgm2", "soil_moisture_pct",
        "wind_speed_ms", "wind_direction_deg",
    }
    for mappings, vt in ((FORECAST_MAPPINGS, "forecast"), (OBSERVATION_MAPPINGS, "observed")):
        mapped = {m["variable"] for m in mappings if m.get("variable")}
        assert mapped == expected
        assert all(m["value_type"] == vt for m in mappings if m.get("variable"))


# ---------------------------------------------------------------- storage behaviour

def test_verification_status_is_part_of_the_canonical_schema():
    assert "verification_status" in CANONICAL_COLUMNS
    assert "verification_status" in parquet_store.ARROW_SCHEMA.names


def _row(**kw):
    base = dict(
        record_id="r", upload_batch_id="b", source_file="f.csv", source_column="c",
        variable="temperature_c", value_type="observed", value=1.0,
        region_id="IN-MH", region_name="Maharashtra", lat=19.0, lon=72.0,
        init_date=None, valid_date=date(2026, 8, 1), lead_time_days=None,
        ensemble_member_id=None, mapping_confidence=1.0,
        ingested_at=pd.Timestamp("2026-08-29T00:00:00"),
        grain="native", region_resolution_method="name", verification_status=None,
    )
    base.update(kw)
    return base


def test_final_observation_supersedes_provisional(fresh_store):
    """The revision path: ERA5 landing later must replace the near-real-time value for the
    same reading, not sit alongside it - a duplicate observation would double every
    forecast row it merges with in feature engineering."""
    parquet_store.append_batch("prov", [_row(
        upload_batch_id="prov", value=30.0, verification_status="provisional",
        ingested_at=pd.Timestamp("2026-08-29T00:00:00"))])
    parquet_store.append_batch("fin", [_row(
        upload_batch_id="fin", value=28.5, verification_status="final",
        ingested_at=pd.Timestamp("2026-08-30T00:00:00"))])

    df = parquet_store.read_dataset()
    assert len(df) == 1
    assert df.iloc[0]["value"] == 28.5
    assert df.iloc[0]["verification_status"] == "final"
    # both partitions still exist on disk; deduplication happens on read
    assert len(parquet_store.read_dataset(dedupe=False)) == 2


def test_final_wins_even_when_it_arrives_first(fresh_store):
    """Precedence is by authority, not arrival order: a provisional refresh that runs
    after ERA5 has already landed must not overwrite the settled value."""
    parquet_store.append_batch("fin", [_row(
        upload_batch_id="fin", value=28.5, verification_status="final",
        ingested_at=pd.Timestamp("2026-08-29T00:00:00"))])
    parquet_store.append_batch("prov", [_row(
        upload_batch_id="prov", value=30.0, verification_status="provisional",
        ingested_at=pd.Timestamp("2026-08-30T00:00:00"))])

    df = parquet_store.read_dataset()
    assert len(df) == 1 and df.iloc[0]["value"] == 28.5


def test_exclude_provisional_keeps_legacy_null_rows(fresh_store):
    """Rows ingested before Phase 6 carry a NULL status and came from the ERA5 archive,
    so they are already final. Filtering with a bare `!= 'provisional'` would drop them:
    in Arrow, as in SQL, comparing a null yields null rather than true."""
    parquet_store.append_batch("legacy", [_row(
        upload_batch_id="legacy", verification_status=None,
        valid_date=date(2026, 8, 1))])
    parquet_store.append_batch("prov", [_row(
        upload_batch_id="prov", verification_status="provisional",
        valid_date=date(2026, 8, 2))])

    kept = parquet_store.read_dataset(exclude_provisional=True)
    assert len(kept) == 1
    assert kept.iloc[0]["upload_batch_id"] == "legacy"


def test_has_forecast_cycle_and_latest_init_date(fresh_store):
    assert parquet_store.latest_forecast_init_date() is None
    parquet_store.append_batch("fc", [
        _row(upload_batch_id="fc", value_type="forecast", init_date=date(2026, 8, 29),
             valid_date=date(2026, 8, 30), lead_time_days=2, ensemble_member_id="c00"),
    ])
    assert parquet_store.has_forecast_cycle(date(2026, 8, 29))
    assert not parquet_store.has_forecast_cycle(date(2026, 8, 28))
    assert parquet_store.latest_forecast_init_date() == date(2026, 8, 29)


def test_scheduler_is_off_unless_explicitly_enabled():
    """A fresh clone, a test run or CI must never start reaching out to NOAA on import."""
    from app.live.scheduler import scheduler
    assert settings.live_ingest_enabled is False
    scheduler.start()
    assert scheduler.running is False


# ---------------------------------------------------------------- feature engineering

def test_rate_of_change_survives_a_single_verified_lead_day():
    """A live cycle verifies one lead day at a time, so early on every group holds exactly
    one row. pandas' groupby.apply returns a DataFrame rather than a Series in that case,
    which used to crash feature engineering during precisely the window a fresh cycle is
    in for its first days."""
    from app.features import engineering as fe

    rows = []
    for region in ("IN-MH", "IN-KL"):
        for member in ("c00", "p01"):
            rows.append(dict(
                region_id=region, variable="pressure_hpa", value_type="forecast",
                value=1008.0, init_date=pd.Timestamp("2026-08-27"),
                valid_date=pd.Timestamp("2026-08-28"), lead_time_days=2,
                ensemble_member_id=member, verification_status=None,
            ))
        rows.append(dict(
            region_id=region, variable="pressure_hpa", value_type="observed",
            value=1007.0, init_date=pd.NaT, valid_date=pd.Timestamp("2026-08-28"),
            lead_time_days=None, ensemble_member_id=None,
            verification_status="provisional",
        ))
    frame = fe.build_training_frame(pd.DataFrame(rows), require_observed=True)

    assert not frame.empty
    assert sorted(frame["lead_time_days"].unique()) == [2]
    # No second lead to difference against, so the feature is absent - not an exception.
    assert frame["pressure_rate_of_change"].isna().all()


def test_verification_status_reaches_the_training_frame():
    """The UI badges provisional observations, so the status has to survive the
    forecast/observation merge. Both sides carry the column, and without dropping it from
    the forecast side pandas suffixes both to _x/_y and the real one disappears."""
    from app.features import engineering as fe

    rows = []
    for lead, status in ((2, "provisional"), (3, "final")):
        vd = pd.Timestamp("2026-08-26") + pd.Timedelta(days=lead)
        rows.append(dict(
            region_id="IN-MH", variable="temperature_c", value_type="forecast",
            value=30.0, init_date=pd.Timestamp("2026-08-26"), valid_date=vd,
            lead_time_days=lead, ensemble_member_id="c00", verification_status=None,
        ))
        rows.append(dict(
            region_id="IN-MH", variable="temperature_c", value_type="observed",
            value=29.0, init_date=pd.NaT, valid_date=vd, lead_time_days=None,
            ensemble_member_id=None, verification_status=status,
        ))
    frame = fe.build_training_frame(pd.DataFrame(rows), require_observed=True)

    assert "verification_status" in frame.columns
    got = dict(zip(frame["lead_time_days"], frame["verification_status"]))
    assert got == {2: "provisional", 3: "final"}


def test_legacy_null_status_is_treated_as_final():
    """Observations ingested before Phase 6 carry NULL and came from the ERA5 archive, so
    they are already settled. They must not be mistaken for provisional and withheld."""
    from app.features import engineering as fe

    rows = [
        dict(region_id="IN-MH", variable="temperature_c", value_type="forecast",
             value=30.0, init_date=pd.Timestamp("2019-12-11"),
             valid_date=pd.Timestamp("2019-12-12"), lead_time_days=2,
             ensemble_member_id="c00", verification_status=None),
        dict(region_id="IN-MH", variable="temperature_c", value_type="observed",
             value=29.0, init_date=pd.NaT, valid_date=pd.Timestamp("2019-12-12"),
             lead_time_days=None, ensemble_member_id=None, verification_status=None),
    ]
    frame = fe.build_training_frame(pd.DataFrame(rows), require_observed=True)
    assert frame["verification_status"].tolist() == ["final"]
