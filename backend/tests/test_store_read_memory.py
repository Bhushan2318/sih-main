"""What `read_dataset` costs to hand back a cycle.

Measured 2026-09-10 against the 661-district store: one cycle is 221,055 rows, which
Arrow holds in 18 MB and `to_pandas()` inflated to 84 MB by materialising every string
column as Python objects - one pointer per row. Serving peaked at 740 MB against a 512 MB
ceiling that kills rather than throttles, and this was the largest single term.

Converting the low-cardinality string columns to categoricals on the way out of Arrow
brings the same frame to 21 MB. It is the same failure this project already fixed once
for the paired frame, in a second place.

Deferred, deliberately, and these are xfail rather than red. The conversion was attempted
on 2026-09-10 and reverted: a pandas groupby over categorical columns returns the
cartesian product of the categories unless every call passes `observed=True`, which
turned build_training_frame's 24,928 rows into 90,032. Ten modules under app/ group by
these columns, so landing this means auditing all of them.

It was mandatory at Render's 512 MB ceiling, where /api/alerts measured 740 MB. Moving
serving to a 12 GB Oracle Ampere VM removes that ceiling, and docs/migration-oracle-vm.md
puts this fix on the other side of the finale. The tests stay because the measurement and
the trap are worth keeping executable - drop the xfail markers when the audit happens.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.storage import parquet_store


@pytest.mark.xfail(reason="deferred until after the finale - needs an observed=True "
                        "audit across app/; see docs/migration-oracle-vm.md",
                 strict=False)
def test_low_cardinality_columns_come_back_categorical(fresh_store):
    """value_type, variable and the member id have a handful of distinct values across
    millions of rows. As objects they dominate the frame."""
    df = _write_and_read(fresh_store)
    for col in ("value_type", "variable", "ensemble_member_id"):
        if col in df.columns:
            assert isinstance(df[col].dtype, pd.CategoricalDtype), (
                f"{col} came back as {df[col].dtype}, which costs one Python object "
                f"per row")


def test_values_survive_the_categorical_conversion(fresh_store):
    df = _write_and_read(fresh_store)
    assert set(df["variable"].astype(str)) == {"temperature_c", "rainfall_mm"}
    assert set(df["value_type"].astype(str)) == {"forecast"}
    assert df["value"].sum() == pytest.approx(6.0)


def test_filtering_still_works_on_categorical_columns(fresh_store):
    """Equality against a plain string must keep working - callers all over the app do
    `df["value_type"] == "forecast"`, and a categorical that failed that would break
    silently rather than raise."""
    df = _write_and_read(fresh_store)
    assert (df["value_type"] == "forecast").all()
    assert (df["variable"] == "temperature_c").sum() == 2


def _write_and_read(_fresh):
    # Distinct lead days on purpose: two rows identical on the dedupe key are one row
    # to the store, correctly, and an earlier version of this fixture tripped over that.
    rows = []
    for i, var in enumerate(["temperature_c", "temperature_c", "rainfall_mm"]):
        rows.append({
            "region_id": "IN-KL-IDUKKI", "variable": var, "value": float(i + 1),
            "value_type": "forecast", "init_date": pd.Timestamp("2017-12-31").date(),
            "valid_date": pd.Timestamp("2017-12-31").date(), "lead_time_days": i + 1,
            "ensemble_member_id": "gec00", "verification_status": None,
        })
    parquet_store.append_batch("test-batch", pd.DataFrame(rows))
    return parquet_store.read_dataset(
        value_types=["forecast"],
        columns=["region_id", "variable", "value", "value_type", "init_date",
                 "valid_date", "lead_time_days", "ensemble_member_id"],
    )
