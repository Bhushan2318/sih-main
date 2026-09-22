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
        rows = regions.json().get("regions", [])
        observed["regions"] = len(rows)
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

        for path in ("/api/ensemble", "/api/alerts"):
            r = client.get(path)
            observed[path] = r.status_code
            if r.status_code != 200:
                print(json.dumps(observed))
                print(f"{path} returned {r.status_code}", file=sys.stderr)
                return 1

    print(json.dumps(observed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
