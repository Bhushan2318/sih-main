"""What `_finalise` publishes for a year, pinned before its memory behaviour changes.

Fixtures are slices of REAL files: the district parts come from the 2018 fetch artifacts
(`data/samples/parts-2018/`), the city-keyed part from the pre-district cache
(`data/samples/_gefs_parts/_city_keyed_2018/`). Each is cut to a few rows so the test is
cheap; nothing here is generated. The tests skip when those files are not on disk.

What these cannot show is the reason for the change: concatenating 365 district parts in
pandas measures ~71 MB a part, ~26 GB for a year. A handful of rows passes either way. That
is checked by running it at full size, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
fetch = pytest.importorskip("scripts.fetch_gefs_reforecast_sample")

REAL_PARTS = BACKEND / "data/samples/parts-2018"
REAL_CITY = BACKEND / "data/samples/_gefs_parts/_city_keyed_2018"


def _real_slices(dst: Path, n_parts=3, n_rows=40) -> pd.DataFrame:
    src = sorted(REAL_PARTS.glob("2018-*.parquet"))[:n_parts]
    if len(src) < n_parts:
        pytest.skip("real 2018 parts not on disk")
    frames = []
    for f in src:
        df = pd.read_parquet(f).head(n_rows)
        df.to_parquet(dst / f.name, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_year_file_holds_every_part_row_in_order(tmp_path, monkeypatch):
    parts = tmp_path / "parts"; parts.mkdir()
    out = tmp_path / "out"; out.mkdir()
    expected = _real_slices(parts)
    monkeypatch.setattr(fetch, "OUT_DIR", out)
    monkeypatch.setattr(fetch, "BACKEND_DIR", tmp_path)

    fetch._finalise(parts, [2018])

    got = pd.read_parquet(out / "gefs_reforecast_india_2018.parquet")
    assert len(got) == len(expected)
    assert list(got.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected, check_dtype=False)
    assert len(pd.read_csv(out / "gefs_reforecast_india_2018.csv")) == len(expected)


def test_only_the_requested_year_is_written(tmp_path, monkeypatch):
    parts = tmp_path / "parts"; parts.mkdir()
    out = tmp_path / "out"; out.mkdir()
    _real_slices(parts)
    monkeypatch.setattr(fetch, "OUT_DIR", out)
    monkeypatch.setattr(fetch, "BACKEND_DIR", tmp_path)
    fetch._finalise(parts, [2019])
    assert not list(out.glob("*.parquet"))


def test_city_keyed_parts_are_refused_not_merged(tmp_path, monkeypatch):
    city = sorted(REAL_CITY.glob("2018-*.parquet"))[:1]
    if not city:
        pytest.skip("real city-keyed 2018 part not on disk")
    parts = tmp_path / "parts"; parts.mkdir()
    out = tmp_path / "out"; out.mkdir()
    _real_slices(parts, n_parts=1)
    pd.read_parquet(city[0]).head(10).to_parquet(parts / "2018-12-31.parquet", index=False)
    monkeypatch.setattr(fetch, "OUT_DIR", out)
    monkeypatch.setattr(fetch, "BACKEND_DIR", tmp_path)
    with pytest.raises(SystemExit, match="keyed by city"):
        fetch._finalise(parts, [2018])
    assert not list(out.glob("*.parquet"))


# --- surviving a transient Windows file lock --------------------------------------------
# Path.replace() raised PermissionError [WinError 32] renaming the finalised CSV into place
# on the 4060 laptop, twice in a row, with the write itself already complete - some other
# process (Defender's real-time scan, VS Code's file watcher, the indexer; all three were
# running) briefly had the destination open. POSIX rename has no such restriction, so this
# is Windows-only and transient - retrying clears it without touching data that was already
# written correctly.

def test_replace_with_retry_recovers_from_a_transient_lock(tmp_path, monkeypatch):
    tmp = tmp_path / "x.partial"; tmp.write_text("data")
    dest = tmp_path / "x"

    calls = {"n": 0}
    real_replace = Path.replace

    def flaky_replace(self, target):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(32, "The process cannot access the file")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    fetch._replace_with_retry(tmp, dest)
    assert dest.read_text() == "data"
    assert calls["n"] == 3


def test_replace_with_retry_gives_up_and_raises_eventually(tmp_path, monkeypatch):
    tmp = tmp_path / "x.partial"; tmp.write_text("data")
    dest = tmp_path / "x"

    def always_locked(self, target):
        raise PermissionError(32, "The process cannot access the file")

    monkeypatch.setattr(Path, "replace", always_locked)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        fetch._replace_with_retry(tmp, dest, tries=3)
