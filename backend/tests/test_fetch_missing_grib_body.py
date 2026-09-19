"""A missing .idx is handled; a missing .grib2 body was not, and it took the whole job
down with it.

Found on the real archive: cycle 2008-11-21 00Z, member p01. NOAA's bucket has
soilw_bgrnd_2008112100_p01.grib2.idx (5.3 KB) but not the .grib2 file it points to - a
404 on the data itself, not a transient blip. `pull_one_file` already survives a missing
.idx (returns an empty frame, which `cycle_is_complete` correctly turns into a refused
cycle). The byte-range GET of the file body had no such guard, so this one permanently
missing upstream file propagated a bare RuntimeError out of the ThreadPoolExecutor future
and killed the entire multi-year fetch - and because the crash happens before that
cycle's parquet part is written, `--resume` retries the same cycle and hits the same
dead file every time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fetch_gefs_body", BACKEND / "scripts" / "fetch_gefs_reforecast_sample.py")
fetch = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fetch
_spec.loader.exec_module(fetch)


class _FakeResponse:
    def __init__(self, text):
        self.text = text


def test_a_missing_grib_body_is_treated_like_a_missing_idx(monkeypatch):
    """The .idx exists and parses fine; the body 404s after retries. That must degrade to
    an empty result for this (variable, member), not crash the whole fetch."""
    idx_text = (
        "1:0:d=2008112100:SOILW:0-0.1 m below ground:24 hour fcst:\n"
        "2:1000:d=2008112100:SOILW:0-0.1 m below ground:48 hour fcst:\n"
    )

    def fake_get(url, headers=None):
        if url.endswith(".idx"):
            return _FakeResponse(idx_text)
        raise RuntimeError(f"GET failed after 5 tries: {url}\n  last error: 404 Not Found")

    monkeypatch.setattr(fetch, "_get", fake_get)

    df, grids = fetch.pull_one_file("soilw_bgrnd", "2008-11-21", "p01", prepared=None)

    assert df.empty
    assert grids == {}


def test_that_missing_file_folds_into_a_refused_cycle_not_a_crash(monkeypatch):
    """End to end at the level `cycle_is_complete` sees it: one dead file makes the
    cycle incomplete, not the process."""
    def fake_get(url, headers=None):
        if url.endswith(".idx"):
            return _FakeResponse("1:0:d=x:SOILW:0-0.1 m below ground:24 hour fcst:\n")
        raise RuntimeError("GET failed after 5 tries: ...\n  last error: 404 Not Found")

    monkeypatch.setattr(fetch, "_get", fake_get)

    df, grids = fetch.pull_one_file("soilw_bgrnd", "2008-11-21", "p01", prepared=None)
    # Every other (variable, member) present and healthy - only soilw_bgrnd/p01 is the
    # dead file, same as the real cycle.
    results = {(v, "p01"): (df if v == "soilw_bgrnd" else pd.DataFrame({"region_id": ["x"], v: [1.0]}))
               for v in fetch.VAR_SPEC}
    ok, why = fetch.cycle_is_complete(results, ["p01"])
    assert not ok
    assert "soilw_bgrnd" in why and "p01" in why
