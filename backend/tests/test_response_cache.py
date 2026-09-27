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


# ------------------------------------------------ is the box serving CI's files? (health)

def _box(monkeypatch, tmp_path, run_id="run_A"):
    from app.ml import registry

    monkeypatch.setattr(settings, "serving_read_only", True)
    monkeypatch.setattr(registry, "current_run_id", lambda: run_id)
    monkeypatch.setattr(response_cache, "default_dir", lambda: tmp_path)
    response_cache.invalidate()


def test_status_active_when_the_bundle_matches(tmp_path, monkeypatch):
    _box(monkeypatch, tmp_path)
    response_cache.write_responses("run_A", _items(ensemble={"x": 1}, replay={"y": 1}), tmp_path)
    st = response_cache.status()
    assert st["active"] is True
    assert st["files"] == 2
    assert st["built_at"]
    assert st["reason"] is None


def test_status_names_a_missing_bundle(tmp_path, monkeypatch):
    _box(monkeypatch, tmp_path)
    st = response_cache.status()
    assert st["active"] is False
    assert "run_A" in st["reason"]


def test_status_names_a_bundle_built_for_other_code(tmp_path, monkeypatch):
    """The 2026-09-27 13:00Z case: new code live on the old bundle."""
    _box(monkeypatch, tmp_path)
    response_cache.write_responses("run_A", _items(ensemble={"x": 1}), tmp_path)
    monkeypatch.setattr(response_cache, "code_fingerprint", lambda: "a-newer-build")
    response_cache.invalidate()
    st = response_cache.status()
    assert st["active"] is False
    assert "code" in st["reason"]


def test_status_off_the_box_says_so(tmp_path, monkeypatch):
    """A dev server builds every response by design; that is not a missing bundle."""
    _box(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "serving_read_only", False)
    response_cache.write_responses("run_A", _items(ensemble={"x": 1}), tmp_path)
    st = response_cache.status()
    assert st["active"] is False
    assert "SERVING_READ_ONLY" in st["reason"]


# ------------------------------------------- live builds on the box: one at a time, bounded

def _request():
    from starlette.requests import Request

    return Request({"type": "http", "headers": [(b"accept-encoding", b"gzip")]})


def test_live_builds_on_the_box_run_one_at_a_time(tmp_path, monkeypatch):
    """Without a matching bundle every screen is built live, and each build holds the model
    and its frames. Two at once is what the box could not hold on 2026-09-26 20:03Z and
    2026-09-27 13:03Z. The payload is a plumbing stand-in; the sleep only widens the window
    in which an overlap would show."""
    import threading
    import time

    _box(monkeypatch, tmp_path)
    active, peak = [0], [0]
    count = threading.Lock()

    def _build():
        with count:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.15)
        with count:
            active[0] -= 1
        return {"ok": True}

    threads = [threading.Thread(target=response_cache.respond,
                                args=(_request(), f"region__IN-XX-D{i}", _build))
               for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


def test_the_same_screen_asked_for_twice_at_once_is_built_once(tmp_path, monkeypatch):
    import threading
    import time

    _box(monkeypatch, tmp_path)
    calls = []

    def _build():
        calls.append(1)
        time.sleep(0.15)
        return {"ok": True}

    threads = [threading.Thread(target=response_cache.respond,
                                args=(_request(), "regions_all", _build)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


def test_a_prebuilt_file_never_waits_behind_a_live_build(tmp_path, monkeypatch):
    _box(monkeypatch, tmp_path)
    response_cache.write_responses("run_A", _items(ensemble={"x": 1}), tmp_path)
    assert response_cache._build_gate.acquire(timeout=1)
    try:
        r = response_cache.respond(_request(), "ensemble", lambda: pytest.fail("built"))
    finally:
        response_cache._build_gate.release()
    assert json.loads(gzip.decompress(r.body)) == {"x": 1}


def test_a_request_that_waits_too_long_is_told_to_retry(tmp_path, monkeypatch):
    """A 503 with Retry-After rather than a thread parked indefinitely behind a build that
    takes tens of seconds on a 0.1-CPU box. The dashboard retries 5xx on its own."""
    from fastapi import HTTPException

    _box(monkeypatch, tmp_path)
    monkeypatch.setattr(response_cache, "BUILD_WAIT_S", 0.05)
    assert response_cache._build_gate.acquire(timeout=1)
    try:
        with pytest.raises(HTTPException) as exc:
            response_cache.respond(_request(), "regions_all", lambda: {"ok": True})
    finally:
        response_cache._build_gate.release()
    assert exc.value.status_code == 503
    assert exc.value.headers.get("Retry-After")


def test_off_the_box_builds_are_not_gated(tmp_path, monkeypatch):
    """Dev servers, tests and CI build as they always have."""
    _box(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "serving_read_only", False)
    assert response_cache._build_gate.acquire(timeout=1)
    try:
        r = response_cache.respond(_request(), "regions_all", lambda: {"ok": True})
    finally:
        response_cache._build_gate.release()
    assert json.loads(gzip.decompress(r.body)) == {"ok": True}


# ----------------------------------- on the box: the catalogue only, refused before building
#
# A request outside what CI built loaded the model and built live: measured 2026-09-28 on a
# box-mode mirror of the live bundle, one such request each took a fresh server from ~160 MB
# to a 254-334 MB peak (a state panel, a lead-day map, the hero for another region, a Replay
# focus, an unlisted or malformed Replay date). The builders below fail the test if reached.

def _never_built():
    pytest.fail("built on request although it is outside the catalogue")


def test_on_the_box_an_uncatalogued_request_is_refused_before_building(tmp_path, monkeypatch):
    from fastapi import HTTPException

    _box(monkeypatch, tmp_path)
    response_cache.write_responses("run_A", _items(ensemble={"x": 1}), tmp_path)
    with pytest.raises(HTTPException) as exc:
        response_cache.respond(_request(), None, _never_built)
    assert exc.value.status_code == 409


def test_it_is_refused_without_a_matching_bundle_too(tmp_path, monkeypatch):
    """Where every screen builds live anyway (the bundle is for other code), a request the
    dashboard never sends is still not one more model load."""
    from fastapi import HTTPException

    _box(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as exc:
        response_cache.respond(_request(), None, _never_built)
    assert exc.value.status_code == 409


@pytest.mark.parametrize("name", ["region__IN-ZZ-NOWHERE", "replay__2020-01-01"])
def test_on_the_box_a_district_or_cycle_the_bundle_lacks_is_404(tmp_path, monkeypatch, name):
    """Districts and Replay cycles are enumerated from the store when CI builds the bundle,
    so one the bundle does not hold is not one the dashboard can ask for."""
    from fastapi import HTTPException

    _box(monkeypatch, tmp_path)
    response_cache.write_responses(
        "run_A", _items(**{"region__IN-MH-NAGPUR": {"x": 1}, "replay__2026-09-23": {"y": 1}}),
        tmp_path)
    with pytest.raises(HTTPException) as exc:
        response_cache.respond(_request(), name, _never_built)
    assert exc.value.status_code == 404


def test_a_district_the_bundle_holds_is_served(tmp_path, monkeypatch):
    _box(monkeypatch, tmp_path)
    response_cache.write_responses("run_A", _items(**{"region__IN-MH-NAGPUR": {"x": 1}}),
                                   tmp_path)
    r = response_cache.respond(_request(), "region__IN-MH-NAGPUR", _never_built)
    assert json.loads(gzip.decompress(r.body)) == {"x": 1}


def test_a_fixed_screen_ci_failed_to_build_still_builds_live(tmp_path, monkeypatch):
    """precompute skips a screen whose build failed rather than losing the rest; the box
    builds that one itself, as before (one at a time)."""
    _box(monkeypatch, tmp_path)
    response_cache.write_responses("run_A", _items(ensemble={"x": 1}), tmp_path)
    r = response_cache.respond(_request(), "model_status", lambda: {"built": True})
    assert json.loads(gzip.decompress(r.body)) == {"built": True}


def test_off_the_box_uncatalogued_requests_still_build(tmp_path, monkeypatch):
    _box(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "serving_read_only", False)
    r = response_cache.respond(_request(), None, lambda: {"built": True})
    assert json.loads(gzip.decompress(r.body)) == {"built": True}
