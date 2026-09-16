"""scripts/ingest_districts_chunked.py imports `resource`, which is POSIX-only - importing
the module at all raised ImportError on Windows before this file's own peak-memory report
ever ran once. Fixed by measuring peak working set through psutil instead, which has a
wheel on every platform this script needs to run on.
"""
from __future__ import annotations

import sys

import pytest


def test_module_imports_on_every_platform():
    """The historical bug: `import resource` at module level made the whole script
    unusable on Windows before a single line of its own logic executed. `resource` is
    the correct, expected import on POSIX - only Windows must avoid it."""
    import scripts.ingest_districts_chunked as m
    has_resource = hasattr(sys.modules[m.__name__], "resource")
    if sys.platform == "win32":
        assert not has_resource, (
            "resource is POSIX-only; importing it unconditionally breaks Windows")
    else:
        assert has_resource, "POSIX platforms should take the resource-module path"


def test_peak_rss_mb_returns_a_positive_float():
    from scripts.ingest_districts_chunked import peak_rss_mb
    peak = peak_rss_mb()
    assert isinstance(peak, float)
    assert peak > 0, "the process importing pytest has allocated more than 0 MB"


def test_peak_rss_mb_grows_after_a_real_allocation():
    """Not a tight bound - just proof this reads a real, moving number rather than a
    constant that would pass the test above by accident."""
    from scripts.ingest_districts_chunked import peak_rss_mb

    before = peak_rss_mb()
    _hold = bytearray(200_000_000)  # 200 MB, comfortably above OS/allocator noise
    after = peak_rss_mb()
    assert after >= before, f"{after:.1f} MB is not >= {before:.1f} MB after a 200 MB alloc"
    del _hold


@pytest.mark.skipif(sys.platform != "win32", reason="only exercises the Windows path")
def test_peak_rss_mb_uses_psutil_on_windows():
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


def test_resolve_source_honours_an_explicit_absolute_path(tmp_path):
    from scripts.ingest_districts_chunked import resolve_source
    p = tmp_path / "somewhere_else.parquet"
    assert resolve_source(2019, str(p)) == p
