"""Finished responses are built in CI and served on the box as bytes.

Render's free instance is a 0.1-CPU share. Measured 2026-09-27 on this repo's live bundle:
the builders behind Replay and the map take 2.0 s and 0.9 s on a laptop, and 11-18 s and
6-10 s on the live site; the hero's first call also loads the model and took over 30 s.
`app/services/response_cache.py` moves that work into CI, where `package_for_deploy`
already scores the cycles.

The payloads here are plumbing stand-ins (tiny JSON literals) and are labelled as such -
they exercise the store, the keys and the wiring. Nothing here produces a metric. Parity
between a precomputed response and a live-built one is tested against a real trained model
in tests/test_api.py.
"""

from __future__ import annotations

import gzip
import json

import pytest

from app.config import settings
from app.services import response_cache


def _items(**bodies):
    return [(name, json.dumps(body).encode()) for name, body in bodies.items()]


def test_round_trip(tmp_path):
    response_cache.write_responses("run_A", _items(ensemble={"x": 1}), tmp_path)
    got = response_cache.read("ensemble", "run_A", tmp_path)
    assert got is not None
    assert json.loads(gzip.decompress(got)) == {"x": 1}


def test_an_absent_name_is_a_miss(tmp_path):
    response_cache.write_responses("run_A", _items(ensemble={"x": 1}), tmp_path)
    assert response_cache.read("regions_all", "run_A", tmp_path) is None


def test_a_different_run_does_not_match(tmp_path):
    """A newly promoted model must not serve the previous model's answers."""
    response_cache.write_responses("run_OLD", _items(ensemble={"x": 1}), tmp_path)
    assert response_cache.read("ensemble", "run_NEW", tmp_path) is None


def test_different_code_does_not_match(tmp_path, monkeypatch):
    """A response is a function of the code that built it. A box running other code than
    CI packaged with builds live - slow but right - rather than serving a shape its own
    schemas no longer describe."""
    response_cache.write_responses("run_A", _items(ensemble={"x": 1}), tmp_path)
    monkeypatch.setattr(response_cache, "code_fingerprint", lambda: "some-other-build")
    assert response_cache.read("ensemble", "run_A", tmp_path) is None


def test_a_half_written_set_is_ignored(tmp_path):
    """The manifest is written last. Without it, nothing in the directory is trusted -
    a set killed part-way through must not serve some screens from this cycle and others
    from nowhere."""
    response_cache.write_responses("run_A", _items(ensemble={"x": 1}), tmp_path)
    (tmp_path / "run_A" / response_cache.MANIFEST).unlink()
    assert response_cache.read("ensemble", "run_A", tmp_path) is None


def test_rewriting_drops_what_the_new_set_does_not_have(tmp_path):
    """CI restores the previous bundle before packaging, so a stale file from the last run
    (a Replay cycle that has aged out, a district that is gone) must not survive."""
    response_cache.write_responses("run_OLD", _items(ensemble={"x": 0}), tmp_path)
    response_cache.write_responses("run_A", _items(ensemble={"x": 1}, replay={"y": 1}),
                                   tmp_path)
    response_cache.write_responses("run_A", _items(ensemble={"x": 2}), tmp_path)
    assert response_cache.read("replay", "run_A", tmp_path) is None
    assert json.loads(gzip.decompress(response_cache.read("ensemble", "run_A", tmp_path))) \
        == {"x": 2}
    assert not (tmp_path / "run_OLD").exists()


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "x" * 300, "region__IN MH"])
def test_names_that_are_not_file_safe_are_never_looked_up(bad):
    assert response_cache.safe_name(bad) is None


def test_replay_names_only_for_plain_dates():
    assert response_cache.replay_name(None) == "replay"
    assert response_cache.replay_name("2018-08-13") == "replay__2018-08-13"
    assert response_cache.replay_name("2018-8-13") is None
    assert response_cache.replay_name("../../etc") is None


def test_precompute_builds_as_the_serving_box_would(tmp_path, monkeypatch):
    """CI must build exactly what the box would serve. The box refuses to score a cycle CI
    did not precompute (SERVING_READ_ONLY) and leaves it out of Replay's list; a CI build
    that scored it anyway would list a cycle the box cannot open."""
    seen = []

    def _build():
        seen.append(settings.serving_read_only)
        return {"ok": True}

    monkeypatch.setattr(settings, "serving_read_only", False)
    monkeypatch.setattr(response_cache, "catalogue", lambda: iter([("ensemble", _build)]))
    report = response_cache.precompute("run_A", tmp_path)

    assert seen == [True]
    assert settings.serving_read_only is False, "the switch must be put back afterwards"
    assert report["written"] == 1
    assert response_cache.read("ensemble", "run_A", tmp_path) is not None


def test_one_failing_response_does_not_lose_the_rest(tmp_path, monkeypatch):
    def _boom():
        raise RuntimeError("cannot build")

    monkeypatch.setattr(response_cache, "catalogue",
                        lambda: iter([("replay", _boom), ("ensemble", lambda: {"ok": 1})]))
    report = response_cache.precompute("run_A", tmp_path)
    assert report["written"] == 1
    assert [s[0] for s in report["skipped"]] == ["replay"]
    assert response_cache.read("ensemble", "run_A", tmp_path) is not None
    assert response_cache.read("replay", "run_A", tmp_path) is None


def test_a_catalogue_that_fails_part_way_keeps_what_was_built(tmp_path, monkeypatch):
    def _entries():
        yield "ensemble", lambda: {"ok": 1}
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(response_cache, "catalogue", _entries)
    report = response_cache.precompute("run_A", tmp_path)
    assert report["written"] == 1
    assert report["skipped"][0][0] == "<catalogue>"
    assert response_cache.read("ensemble", "run_A", tmp_path) is not None


def test_the_package_ships_the_responses():
    from scripts import package_for_deploy

    assert "data/analysis/responses" in package_for_deploy.EXTRA_PATHS
