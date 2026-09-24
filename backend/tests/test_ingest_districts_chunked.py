"""scripts/ingest_districts_chunked.py imports `resource`, which is POSIX-only - importing
the module at all raised ImportError on Windows before this file's own peak-memory report
ever ran once. Fixed by measuring peak working set through psutil instead, which has a
wheel on every platform this script needs to run on.
"""
from __future__ import annotations

import sys

import pytest


def _psutil_or_skip_on_windows():
    """On Windows, peak RSS is read through psutil - a training extra
    (requirements-train.txt), not part of the core install CI's backend job uses."""
    if sys.platform == "win32":
        pytest.importorskip("psutil")


def test_module_imports_on_every_platform():
    """The historical bug: `import resource` at module level made the whole script
    unusable on Windows before a single line of its own logic executed.

    Neither platform-specific module may be imported at module level. The earlier
    platform-guarded version still put `resource` on the module on Mac and Linux - so
    this assertion could only ever pass on Windows - and put `psutil` there on Windows,
    where the core requirements do not install it."""
    import scripts.ingest_districts_chunked as m
    mod = sys.modules[m.__name__]
    assert not hasattr(mod, "resource"), (
        "resource is POSIX-only; importing it unconditionally breaks Windows")
    assert not hasattr(mod, "psutil"), (
        "psutil is a training extra; importing it at module level breaks the core install")


def test_peak_rss_mb_returns_a_positive_float():
    _psutil_or_skip_on_windows()
    from scripts.ingest_districts_chunked import peak_rss_mb
    peak = peak_rss_mb()
    assert isinstance(peak, float)
    assert peak > 0, "the process importing pytest has allocated more than 0 MB"


def test_peak_rss_mb_grows_after_a_real_allocation():
    """Not a tight bound - just proof this reads a real, moving number rather than a
    constant that would pass the test above by accident."""
    _psutil_or_skip_on_windows()
    from scripts.ingest_districts_chunked import peak_rss_mb

    before = peak_rss_mb()
    _hold = bytearray(200_000_000)  # 200 MB, comfortably above OS/allocator noise
    after = peak_rss_mb()
    assert after >= before, f"{after:.1f} MB is not >= {before:.1f} MB after a 200 MB alloc"
    del _hold


@pytest.mark.skipif(sys.platform != "win32", reason="only exercises the Windows path")
def test_peak_rss_mb_uses_psutil_on_windows():
    _psutil_or_skip_on_windows()
    from scripts.ingest_districts_chunked import peak_rss_mb
    peak_rss_mb()  # psutil is imported on first call now, not at module import
    assert "psutil" in sys.modules, (
        "expected the Windows path to go through psutil, not the POSIX resource module")


# --- --source override -------------------------------------------------------------------
# 2019 is the first year where a district-scale fetch collides with a filename the test
# suite already owns: tests/conftest.py hardcodes gefs_reforecast_india_2019.parquet as the
# small 36-city legacy sample. The real 2019 archive year has to live under a different
# name, so the ingest needs a way to be pointed at it explicitly.

def test_resolve_source_defaults_to_the_plain_year_filename():
    from scripts.ingest_districts_chunked import SAMPLES, resolve_source
    assert resolve_source(2018) == SAMPLES / "gefs_reforecast_india_2018.parquet"


def test_resolve_source_honours_an_explicit_filename():
    from scripts.ingest_districts_chunked import SAMPLES, resolve_source
    got = resolve_source(2019, "gefs_reforecast_india_2019_district.parquet")
    assert got == SAMPLES / "gefs_reforecast_india_2019_district.parquet"


def test_cycle_coverage_requires_all_core_dimensions():
    import pandas as pd
    from scripts.ingest_districts_chunked import cycle_has_expected_coverage

    rows = []
    variables = [f"v{i}" for i in range(8)]
    members = [f"m{i}" for i in range(5)]
    for i in range(666):
        rows.append({
            "region_id": f"IN-X-{i:04d}",
            "variable": variables[i % len(variables)],
            "ensemble_member_id": members[i % len(members)],
            "lead_time_days": i % 10 + 1,
        })
    frame = pd.DataFrame(rows)
    assert cycle_has_expected_coverage(frame)
    assert not cycle_has_expected_coverage(frame.iloc[:-1])
    assert not cycle_has_expected_coverage(frame[frame["lead_time_days"] != 10])


def test_resolve_source_honours_an_explicit_absolute_path(tmp_path):
    from scripts.ingest_districts_chunked import resolve_source
    p = tmp_path / "somewhere_else.parquet"
    assert resolve_source(2019, str(p)) == p
