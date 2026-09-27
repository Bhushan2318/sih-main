"""Finished API responses: built in CI, served on the box as bytes.

WHY
`app/ml/precomputed.py` moved *scoring* off the serving box. Turning a scored cycle into
the JSON a screen asks for stayed there, and on Render's free instance - a 0.1-CPU share -
that step alone does not fit in a page load. Measured 2026-09-27 against the live bundle:

                          laptop     live site
  /api/replay (a cycle)    2.0 s     11-18 s, on every open
  /api/regions/all         0.9 s      6-10 s
  /api/ensemble (hero)     4.6 s     >30 s on its first call, which also loads the model
  loading the model        1.1-3.5 s  paid by whichever request comes first

So CI builds every response the dashboard opens with - the hero, the map, the model page,
the alerts, Replay's list and every cycle on it, and each district's panel - with the same
service code, in the same box mode, against the store it ships, and gzips each one. The
box reads a file. It does not load the model, score, or build, and the first visitor after
a restart waits no longer than the tenth.

KEYING
On the model run, exactly as precomputed.py and for the same reason: the store fingerprint
is built from mtimes that cannot agree between the runner and the box. Consistency with
the store is by construction - one CI run ingests, scores, builds and packages, and they
ship together or not at all.

Plus a fingerprint of the backend's own source. A response is a function of the code that
built it: a box running different code from the one CI packaged with (a code-only deploy
between two data refreshes) builds live - slow but right - rather than serving a shape its
own schemas no longer describe.

ONLY ON THE BOX
Files are read, and built responses kept, only under SERVING_READ_ONLY - the switch that
already means "this is the serving box" (set in the Dockerfile). Its store is the one CI
packaged and never changes. A dev server ingests and trains, so it builds every time, as
it always has.

WHAT IS NOT PRECOMPUTED IS NOT SERVED ON THE BOX
Requests with open-ended parameters the dashboard never sends - /api/ensemble?region_id=,
/api/replay?focus_region=, a state's panel, a Replay date it does not list - loaded the
model and built live: measured 2026-09-28 on a box-mode mirror of the live bundle, a
254-334 MB peak apiece from ~160 MB, on a box killed at 512. On the box they are refused
before anything is built: 409 for a parameter, 404 for a district or cycle the bundle does
not hold. /api/regions?lead_time_days= is served as one day of the all-days map.

A catalogued response the box had to build itself (CI failed to build it, or the bundle is
for other code) is kept in memory as compressed bytes, bounded, so it is paid for once per
process rather than on every open.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import logging
import re
import shutil
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import pydantic_core
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.db.base import resolve_path

log = logging.getLogger(__name__)

DIR_NAME = "responses"
MANIFEST = "manifest.json"

# The API's own ceiling on /api/alerts?limit=. Alerts are built once at this size and
# sliced to whatever limit is asked for, so the dashboard's page size is not a second
# constant here that has to agree with the frontend's.
ALERTS_MAX_LIMIT = 500
ALERT_BANDS = (None, "low", "medium", "high")

_SAFE = re.compile(r"^[A-Za-z0-9_.\-]{1,200}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Built-live responses kept on the box, as gzip. A Replay cycle is ~285 KB compressed.
_MEMO_MAX_BYTES = 16 * 1024 * 1024
_memo: "OrderedDict[tuple, bytes]" = OrderedDict()
_memo_bytes = 0
_lock = threading.Lock()
_manifest_cache: "dict[Path, tuple[int, Optional[dict], Optional[str]]]" = {}

# On the box, a response with no prebuilt file is built one at a time. Each build holds the
# model and its frames; two at once is what the 512 MB box could not hold on 2026-09-26
# 20:03Z and 2026-09-27 13:03Z, both times a deploy running without a matching bundle.
# A request that cannot start within BUILD_WAIT_S is told to retry (503 + Retry-After)
# rather than parked indefinitely behind a build that takes tens of seconds at 0.1 CPU.
BUILD_WAIT_S = 45.0
RETRY_AFTER_S = 30
_build_gate = threading.Lock()


def default_dir() -> Path:
    """A function, not a constant, so a test can point it somewhere else."""
    return Path(resolve_path(settings.data_dir)) / "analysis" / DIR_NAME


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """Hash of every Python source file in the backend package, by path and content."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:20]


# ------------------------------------------------------------------------------ names

def safe_name(name: Optional[str]) -> Optional[str]:
    """`name` if it can be a file name as it stands, else None (never looked up)."""
    if not name or not _SAFE.match(name) or ".." in name:
        return None
    return name


def replay_name(init_date: Optional[str]) -> Optional[str]:
    if init_date is None:
        return "replay"
    return f"replay__{init_date}" if _DATE.match(init_date) else None


def region_name(region_id: str) -> Optional[str]:
    return safe_name(f"region__{region_id}")


def alerts_name(risk_band: Optional[str]) -> str:
    return f"alerts__{risk_band or 'default'}"


# ------------------------------------------------------------------------ disk (CI -> box)

def write_responses(run_id: str, items: Iterable[tuple[str, bytes]],
                    base: Optional[Path] = None) -> Path:
    """Write one run's responses, replacing whatever was there. `items` are (name, JSON).

    The manifest is written LAST: a directory without one is never read, so a build killed
    part-way through cannot serve some screens from this cycle and others from nowhere.
    Everything else under `base` is removed first - CI restores the previous bundle before
    packaging, and a stale answer from the last run must not survive into this one.
    """
    base = Path(base or default_dir())
    if base.exists():
        shutil.rmtree(base)
    d = base / run_id
    d.mkdir(parents=True)
    files = {}
    for name, body in items:
        if safe_name(name) is None:
            raise ValueError(f"not a file-safe response name: {name!r}")
        fname = f"{name}.json.gz"
        (d / fname).write_bytes(gzip.compress(body, compresslevel=9, mtime=0))
        files[name] = fname
    (d / MANIFEST).write_text(json.dumps({
        "run_id": run_id,
        "code": code_fingerprint(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }))
    return d


def _manifest_checked(run_id: str, base: Path) -> "tuple[Optional[dict], Optional[str]]":
    """(manifest, None) when this run's bundle matches this code, else (None, why not)."""
    path = base / run_id / MANIFEST
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        held = sorted(p.name for p in base.glob("*") if p.is_dir()) if base.is_dir() else []
        return None, (f"no prebuilt responses for run {run_id}"
                      + (f" (the bundle holds {', '.join(held)})" if held else ""))
    hit = _manifest_cache.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1], hit[2]
    why = None
    try:
        m = json.loads(path.read_text())
    except (OSError, ValueError):
        m, why = None, f"the manifest in {path.parent.name} is unreadable"
    if m is not None and (m.get("run_id") != run_id or m.get("code") != code_fingerprint()):
        why = (f"the bundle was built for run {m.get('run_id')} / code {m.get('code')}; "
               f"this is run {run_id} / code {code_fingerprint()}")
        log.warning("precomputed responses in %s: %s; building live instead", path.parent, why)
        m = None
    _manifest_cache[path] = (mtime, m, why)
    return m, why


def _manifest(run_id: str, base: Path) -> Optional[dict]:
    return _manifest_checked(run_id, base)[0]


def status() -> dict:
    """Whether this process serves CI's prebuilt responses right now, and if not, why.

    Reported by /api/health. warm-on-push reads it before sending anything heavy: a box
    without a matching bundle builds every screen live, and warming one on top of a visitor
    is what OOM-killed it on 2026-09-26 20:03Z and 2026-09-27 13:03Z. Cheap by design - the
    platform's health check calls it: one stat, and a parse only when the manifest changes.
    """
    from app.ml import registry

    out = {"active": False, "files": 0, "built_at": None, "reason": None}
    if not settings.serving_read_only:
        out["reason"] = "SERVING_READ_ONLY is off: not the serving box, every response is built live"
        return out
    run_id = registry.current_run_id()
    if not run_id:
        out["reason"] = "no current model run"
        return out
    m, why = _manifest_checked(run_id, Path(default_dir()))
    if m is None:
        out["reason"] = why
        return out
    out.update(active=True, files=len(m.get("files", {})), built_at=m.get("built_at"))
    return out


def read(name: str, run_id: str, base: Optional[Path] = None) -> Optional[bytes]:
    """The gzipped response CI built for this run under `name`, or None."""
    base = Path(base or default_dir())
    m = _manifest(run_id, base)
    fname = (m or {}).get("files", {}).get(name)
    if not fname:
        return None
    try:
        return (base / run_id / fname).read_bytes()
    except OSError:
        return None


# --------------------------------------------------------------------------------- CI

def catalogue() -> Iterator[tuple[str, Callable]]:
    """Every (name, builder) the dashboard asks for. Lazy, because the Replay and district
    entries depend on what the store holds, which is only known in box mode."""
    from app.api.routers import model_status
    from app.ml import inference
    from app.services import (alert_service, ensemble_service, region_service,
                              replay_service)

    yield "ensemble", ensemble_service.get_divergence
    yield "regions_all", region_service.get_all_regions
    yield "model_status", model_status.build_status
    for band in ALERT_BANDS:
        yield alerts_name(band), (
            lambda band=band: alert_service.get_alerts(limit=ALERTS_MAX_LIMIT, risk_band=band))
    yield "replay_cycles", replay_service.list_cycles
    yield "replay", replay_service.get_replay
    for c in replay_service.list_cycles():
        name = replay_name(str(c.init_date))
        if name:
            yield name, (lambda d=str(c.init_date): replay_service.get_replay(d))

    scored = inference.score_latest_cycle()
    if scored is not None and not scored.events.empty:
        for rid in sorted(scored.events["region_id"].astype(str).unique()):
            name = region_name(rid)
            if name:
                yield name, (lambda rid=rid: region_service.get_region_detail(rid))


@contextlib.contextmanager
def _as_the_box():
    """Build in SERVING_READ_ONLY, exactly as the box would. Off it, a cycle CI did not
    precompute would be scored here and listed by Replay, and the box would then refuse to
    open it."""
    before = settings.serving_read_only
    settings.serving_read_only = True
    try:
        yield
    finally:
        settings.serving_read_only = before


def precompute(run_id: str, base: Optional[Path] = None) -> dict:
    """Build and write every catalogued response. One that fails is skipped, not fatal:
    it has no file and builds live, which is what happened before any of this."""
    from app.ml import inference

    items, skipped = [], []
    with _as_the_box():
        inference.invalidate_caches()
        entries = catalogue()
        while True:
            # Listing the catalogue reads the store too (Replay's cycles, the districts).
            # If that fails, keep what was built: a missing file builds live on the box, and
            # a packaging step that raised here would hold back the data refresh itself.
            try:
                name, build = next(entries)
            except StopIteration:
                break
            except Exception as exc:  # noqa: BLE001
                skipped.append(("<catalogue>", f"{type(exc).__name__}: {exc}"))
                break
            try:
                items.append((name, _to_json(build())))
            except Exception as exc:  # noqa: BLE001 - one bad screen must not lose the rest
                skipped.append((name, f"{type(exc).__name__}: {exc}"))
        inference.invalidate_caches()
    d = write_responses(run_id, items, base)
    size = sum(f.stat().st_size for f in d.glob("*.json.gz"))
    return {"written": len(items), "skipped": skipped, "bytes": size, "dir": d}


# -------------------------------------------------------------------------------- serving

def _to_json(obj) -> bytes:
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json().encode()
    return pydantic_core.to_json(obj)


def _memo_get(key: tuple) -> Optional[bytes]:
    with _lock:
        body = _memo.get(key)
        if body is not None:
            _memo.move_to_end(key)
        return body


def _memo_put(key: tuple, body: bytes) -> None:
    global _memo_bytes
    if len(body) > _MEMO_MAX_BYTES:
        return
    with _lock:
        old = _memo.pop(key, None)
        _memo_bytes -= len(old) if old else 0
        _memo[key] = body
        _memo_bytes += len(body)
        while _memo_bytes > _MEMO_MAX_BYTES and _memo:
            _, dropped = _memo.popitem(last=False)
            _memo_bytes -= len(dropped)


def invalidate() -> None:
    global _memo_bytes
    with _lock:
        _memo.clear()
        _memo_bytes = 0
        _manifest_cache.clear()


def _build_live(build: Callable, run_id: Optional[str], name: Optional[str],
                on_the_box: bool) -> bytes:
    """Build a response and gzip it - on the box, one build at a time (see _build_gate)."""
    if not on_the_box:
        return gzip.compress(_to_json(build()), compresslevel=6)
    if not _build_gate.acquire(timeout=BUILD_WAIT_S):
        raise HTTPException(
            status_code=503,
            detail=("This server is building a screen it has no prebuilt copy of, one at a "
                    "time to stay inside its memory. Try again in a moment."),
            headers={"Retry-After": str(RETRY_AFTER_S)},
        )
    try:
        # Whoever held the gate may have just built this very screen.
        if run_id:
            body = _memo_get((run_id, name))
            if body is not None:
                return body
        body = gzip.compress(_to_json(build()), compresslevel=6)
        if run_id:
            _memo_put((run_id, name), body)
        return body
    finally:
        _build_gate.release()


# Catalogue entries enumerated from the store when CI builds the bundle (districts of the
# latest cycle, the cycles Replay lists). Every other name is a fixed screen.
_ENUMERATED = ("region__", "replay__")

NOT_SERVED = ("This deployment serves the screens the dashboard opens, built ahead of time, "
              "and does not build others on request. Run the API locally for other parameters.")


def _refuse_outside_the_catalogue(name: Optional[str], run_id: Optional[str]) -> None:
    """On the box, raise before building anything the dashboard cannot have asked for.

    A fixed screen missing from a matching bundle (CI failed to build it) still builds live;
    so does everything when the bundle is for other code - there is no catalogue to go by,
    and _build_live runs one build at a time.
    """
    if name is None:
        raise HTTPException(status_code=409, detail=NOT_SERVED)
    if not run_id or not name.startswith(_ENUMERATED):
        return
    m = _manifest(run_id, Path(default_dir()))
    if m is not None and name not in m.get("files", {}):
        what = name.split("__", 1)[1]
        raise HTTPException(status_code=404, detail=(
            f"No panel for {what} in the latest cycle on this deployment."
            if name.startswith("region__") else
            f"Cycle {what} is not one Replay offers on this deployment (see /api/replay/cycles)."))


def respond(request: Request, name: Optional[str], build: Callable,
            patch: Optional[Callable[[object], object]] = None) -> Response:
    """Serve `name` from CI's file, else from this box's memory, else build it.

    `patch` adjusts the decoded body for this request - a slice, or a field that is live by
    nature (connected websocket clients) - and is applied however the body was obtained, so
    all three paths answer identically.
    """
    from app.ml import registry

    name = safe_name(name)
    # Only the serving box reads CI's files or keeps what it built: its store is the one CI
    # packaged and cannot change under it. A dev server ingests, so it always builds.
    on_the_box = settings.serving_read_only
    run_id = registry.current_run_id() if (name and on_the_box) else None
    if on_the_box:
        _refuse_outside_the_catalogue(name, run_id)
    body = None
    if run_id:
        body = read(name, run_id) or _memo_get((run_id, name))
    if body is None:
        body = _build_live(build, run_id, name, on_the_box)

    if patch is not None:
        return JSONResponse(patch(json.loads(gzip.decompress(body))))
    if "gzip" in request.headers.get("accept-encoding", "").lower():
        return Response(body, media_type="application/json",
                        headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})
    return Response(gzip.decompress(body), media_type="application/json",
                    headers={"Vary": "Accept-Encoding"})
