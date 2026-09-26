"""Call the API the way the dashboard does, in a process configured for one data root.

Invoked by `publish_serving_model.serving_check`. A separate process because settings are
read once at import, so a checkout that has already imported `app.config` cannot be
pointed at the downloaded serving bundle afterwards.

Prints one JSON line of what was observed. A non-zero exit means the model does not serve
what the dashboard needs - including `top_factors`, which is the SHAP explanation the
region panel shows, and which a run without shap_summary.parquet would quietly omit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))


def scored_regions(payload: dict) -> list:
    """The region rows in an /api/regions/all body.

    That endpoint answers for every lead day at once - `days[]`, each a RegionsResponse
    with its own `regions` - and carries no top-level `regions` key. Reading one gave an
    empty list and made a perfectly good model look like it scored nothing.
    """
    days = payload.get("days")
    if isinstance(days, list):
        return [r for day in days for r in (day.get("regions") or [])]
    return payload.get("regions") or []


def replay_event_problem(cycles: list, replay: dict, expect_events: bool) -> "str | None":
    """Why Replay's past events do not serve, or None if they do (or none are expected).

    A run that carries replay_cases/ must list them first, and /api/replay must open on
    the first one, all ten lead days, charting the district the event is about.
    """
    if not expect_events:
        return None
    events = [c for c in cycles if c.get("kind") == "event"]
    if not events:
        return "the run carries replay_cases/ but /api/replay/cycles lists no past event"
    first = events[0]
    if replay.get("init_date") != first.get("init_date"):
        return (f"/api/replay opened on {replay.get('init_date')}, not the first past event "
                f"{first.get('init_date')}")
    if len(replay.get("steps") or []) != 10:
        return f"the past event has {len(replay.get('steps') or [])} lead days, not 10"
    focus = (replay.get("focus") or {}).get("region_id")
    if focus != first.get("focus_region_id"):
        return f"the past event's focus is {focus}, not {first.get('focus_region_id')}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    from app.main import app

    observed: dict = {}
    with TestClient(app) as client:
        status = client.get("/api/model/status")
        status.raise_for_status()
        body = status.json()
        observed["current_run_id"] = body.get("current_run_id")
        observed["model_trained"] = body.get("model_trained")
        observed["explanation_method"] = body.get("explanation_method")
        if observed["current_run_id"] != args.run_id:
            print(json.dumps(observed))
            print(f"serving {observed['current_run_id']}, not {args.run_id}", file=sys.stderr)
            return 1

        regions = client.get("/api/regions/all")
        regions.raise_for_status()
        body = regions.json()
        rows = scored_regions(body)
        observed["lead_days"] = len(body.get("days") or [])
        observed["regions"] = len({r.get("region_id") for r in rows})
        observed["region_rows"] = len(rows)
        if not rows:
            print(json.dumps(observed))
            print("no regions scored", file=sys.stderr)
            return 1

        rid = rows[0].get("region_id")
        detail = client.get(f"/api/regions/{rid}")
        detail.raise_for_status()
        d = detail.json()
        factors = d.get("top_factors") or []
        observed["region"] = rid
        observed["top_factors"] = len(factors)
        observed["top_factors_method"] = d.get("top_factors_method")
        if not factors:
            print(json.dumps(observed))
            print(f"{rid} came back with no top_factors - the region panel would show no "
                  f"explanation", file=sys.stderr)
            return 1
        if observed["top_factors_method"] != "shap":
            print(json.dumps(observed))
            print(f"explanations are {observed['top_factors_method']}, not shap",
                  file=sys.stderr)
            return 1

        # An explanation that is the same everywhere explains nothing. A summary built by
        # grouping on a repeated index gave every district the national mean - byte for
        # byte identical - and still rendered, so "it returned factors" is not enough.
        # Different districts may honestly share a leading feature; they do not share its
        # value to the last decimal.
        sampled = [r.get("region_id") for r in rows][:40]
        seen = {}
        for one in dict.fromkeys(sampled):
            got = (client.get(f"/api/regions/{one}").json().get("top_factors") or [])
            if got:
                seen[one] = (got[0]["feature"], got[0]["importance"])
        observed["districts_sampled"] = len(seen)
        observed["distinct_leading_factors"] = len(set(seen.values()))
        if len(seen) > 2 and observed["distinct_leading_factors"] < 2:
            print(json.dumps(observed))
            print(f"every one of {len(seen)} districts reports the same leading factor and "
                  f"the same value - the explanation is not per-district", file=sys.stderr)
            return 1

        for path in ("/api/ensemble", "/api/alerts"):
            r = client.get(path)
            observed[path] = r.status_code
            if r.status_code != 200:
                print(json.dumps(observed))
                print(f"{path} returned {r.status_code}", file=sys.stderr)
                return 1

        from app.ml import registry
        from app.services.replay_cases import DIR_NAME

        expect_events = (registry.run_dir(args.run_id) / DIR_NAME).is_dir()
        cycles = client.get("/api/replay/cycles")
        replay = client.get("/api/replay")
        observed["/api/replay"] = replay.status_code
        observed["replay_events"] = sum(1 for c in (cycles.json() if cycles.status_code == 200
                                                    else []) if c.get("kind") == "event")
        problem = (f"/api/replay returned {replay.status_code}" if replay.status_code != 200
                   else replay_event_problem(cycles.json(), replay.json(), expect_events))
        if problem:
            print(json.dumps(observed))
            print(problem, file=sys.stderr)
            return 1

    print(json.dumps(observed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
