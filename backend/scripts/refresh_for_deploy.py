"""Pull the newest GEFS cycle, verify what can be verified, retrain, and report.

This is what CI runs on a schedule to keep the deployed site current. It exists because a
512 MB serving box cannot retrain - a full retrain peaks around 2 GB - so the work happens
on a 16 GB GitHub Actions runner and only the *result* (model + canonical store + metadata
db) is shipped to the host.

Run it from the backend directory, with the live extras installed:

    python -m scripts.refresh_for_deploy            # newest published cycle
    python -m scripts.refresh_for_deploy --skip-obs # forecast only, no verification pull

Exit status is what CI branches on: 0 means there is a trained model on disk worth
publishing, non-zero means do not touch the live site. A forecast pull that finds nothing
new is NOT a failure - the cycle may simply not have published yet - so it exits 0 and
leaves the previous artifact in place.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# The live pull reaches out to NOAA, which the scheduler gates behind this flag. CI is the
# one place it is deliberately on; set it before app.config is imported anywhere.
os.environ.setdefault("LIVE_INGEST_ENABLED", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("refresh")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-forecast", action="store_true", help="do not pull a GEFS cycle")
    ap.add_argument("--skip-obs", action="store_true", help="do not pull observations")
    ap.add_argument("--skip-train", action="store_true", help="do not retrain")
    ap.add_argument("--force-train", action="store_true",
                    help="retrain even when no new observations landed")
    args = ap.parse_args()

    from app.db.base import init_db
    from app.ml import registry

    init_db()

    if not args.skip_forecast:
        from app.live import orchestrator
        try:
            # None/None = whichever cycle is newest and past its publish lag.
            result = orchestrator.run_forecast_cycle(None, None, trigger="scheduled")
            log.info("forecast cycle: %s", result)
        except Exception:  # noqa: BLE001
            # A missing cycle is normal (GEFS runs ~5.5 h behind its init hour). Keep the
            # previous artifact rather than failing the run and blanking the live site.
            log.exception("forecast pull failed; continuing with what is already stored")

    new_rows = 0
    if not args.skip_obs:
        from app.live import orchestrator
        try:
            # retrain=False: this script decides when to train, not the orchestrator, so
            # the training run happens in *this* process where its exit code is visible.
            result = orchestrator.run_observation_refresh("final", None, trigger="scheduled",
                                                          retrain=False)
            new_rows = int(result.get("rows_ingested") or 0)
            log.info("observation refresh: %s", result)
        except Exception:  # noqa: BLE001
            log.exception("observation refresh failed; continuing")

    if not args.skip_train:
        if new_rows == 0 and not args.force_train and registry.current_run_id():
            log.info("no newly-verified rows and a model already exists; skipping retrain")
        else:
            from app.ml.train_pipeline import full_retrain
            report = full_retrain(triggered_by_batch_id=None, make_current=True)
            log.info("retrain: run_id=%s status=%s", report.run_id, report.status)
            if report.status != "success":
                log.error("retrain did not succeed; refusing to publish")
                return 1

    run_id = registry.current_run_id()
    if not run_id:
        log.error("no trained model on disk; nothing worth publishing")
        return 1

    log.info("ready to publish: run_id=%s", run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
