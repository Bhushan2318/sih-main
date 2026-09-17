"""Checkpointing for the district observation fetch.

Two defects found by running it for real at scale, neither visible in a twelve-cell test:

**It held the whole year in memory and wrote once at the end.** 4,902 grid cells is ~245
batched requests; it died on batch 11 and lost all eleven. At this volume a failure is
certain, so anything not checkpointed is work that will be done twice against someone
else's API.

**It treated Open-Meteo's hourly cap as a fatal error.** The cap is on request *weight*,
not count - a batch of 20 cells x 379 days x 24 hours of water vapour is enormous - so it
trips within minutes and then resets on the hour. Failing is the wrong response to a limit
that clears itself.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import fetch_era5_district_observations as obs


def test_batches_already_on_disk_are_skipped(tmp_path):
    cells = pd.DataFrame({"lat": [10.0 + i * 0.25 for i in range(60)],
                          "lon": [70.0] * 60})
    done = obs.checkpoint_path(tmp_path, 2017, 0)
    done.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [1]}).to_parquet(done)

    todo = obs.batches_to_fetch(cells, tmp_path, 2017, batch_size=20)
    assert [i for i, _ in todo] == [20, 40], "batch 0 is on disk and must not be refetched"


def test_nothing_on_disk_means_everything_to_do(tmp_path):
    cells = pd.DataFrame({"lat": [10.0] * 40, "lon": [70.0] * 40})
    assert len(obs.batches_to_fetch(cells, tmp_path, 2017, batch_size=20)) == 2


def test_a_checkpoint_is_named_by_year_and_offset(tmp_path):
    a = obs.checkpoint_path(tmp_path, 2017, 0)
    b = obs.checkpoint_path(tmp_path, 2017, 20)
    c = obs.checkpoint_path(tmp_path, 2018, 0)
    assert a != b and a != c
    # the year is the parent directory, not part of the filename
    assert "2017" in str(a) and a.parent.name == "2017" and a.suffix == ".parquet"


def test_rate_limit_is_not_a_fatal_error():
    """A limit that resets on the hour is something to wait out, not to die on."""
    assert obs.is_rate_limited(429, '{"reason":"Hourly API request limit exceeded."}')
    assert not obs.is_rate_limited(500, "server error")
    assert not obs.is_rate_limited(200, "")


def test_seconds_until_the_next_hour_is_bounded():
    s = obs.seconds_until_next_hour()
    assert 0 < s <= 3600
