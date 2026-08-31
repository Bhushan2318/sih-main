"""Pull the newest GEFS cycle, verify what can be verified, retrain, and report.

This is what CI runs on a schedule to keep the deployed site current. It exists because a
512 MB serving box cannot retrain - a full retrain peaks around 2 GB - so the work happens
on a 16 GB GitHub Actions runner and only the *result* (model + canonical store + metadata
db) is shipped to the host.

Run it from the backend directory, with the live extras installed:

    python -m scripts.refresh_for_deploy            # newest published cycle
    python -m scripts.refresh_for_deploy --skip-obs # forecast only, no verification pull

Exit status is what CI branches on:

    0  a trained model is on disk and worth publishing
    1  the retrain failed, or nothing is trained
    2  the cycle arrived too incomplete to publish (see _reject_reason)

A forecast pull that finds nothing new is NOT a failure - the cycle may simply not have
published yet - so it exits 0 and leaves the previous artifact in place. Exit 2 is a
deliberate refusal rather than a crash: the run goes red so it is visible, and the live
site keeps serving the last cycle that was actually complete.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

# The live pull reaches out to NOAA, which the scheduler gates behind this flag. CI is the
# one place it is deliberately on; set it before app.config is imported anywhere.
os.environ.setdefault("LIVE_INGEST_ENABLED", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("refresh")


def _reject_reason(result: dict, args) -> Optional[str]:
    """Why this cycle is unfit to publish, or None if it is fine.

    A cycle arrives incomplete when NOAA is slow and some GRIB steps time out. The pipeline
    still produces values from what it got, and those values are real - but a daily
    aggregate built from fewer steps is not the quantity the models were trained on, and
    for accumulations it is not even close: rainfall summed from 2 of 4 steps is roughly
    half the true total, and rainfall drives most busts. Publishing that would understate
    risk on the exact variable the product exists to warn about.

    So the default is to refuse. The previous cycle stays live, which is a slightly older
    forecast rather than a subtly wrong one, and the next run re-attempts the same cycle.

    Only a completed pull is judged; "skipped" means nothing new was ingested at all.
    """
    if result.get("status") != "complete":
        return None

    # Both flags may be absent if the orchestrator is older than this script; an unknown
    # completeness is not evidence of a bad cycle, so the check is skipped rather than
    # guessed at.
    completeness = result.get("step_completeness")
    if completeness is None:
        fetched, expected = result.get("steps_fetched"), result.get("steps_expected")
        if fetched is not None and expected:
            completeness = fetched / expected
    if completeness is not None and args.min_steps > 0 and completeness < args.min_steps:
        return (f"only {completeness:.1%} of expected GRIB steps arrived "
                f"(need {args.min_steps:.0%}); {len(result.get('missing_steps') or [])} missing")

    short_sums = result.get("undersampled_sum_count") or 0
    if short_sums and not args.allow_short_accumulations:
        # The reported list is truncated, so it may hold no example even when the count is
        # non-zero; say so rather than printing an empty "e.g.".
        examples = [str(u) for u in (result.get("undersampled") or []) if "apcp" in str(u)]
        shown = ", ".join(examples[:3]) if examples else "not in the truncated sample"
        return (f"{short_sums} accumulated group(s) summed from too few steps, which "
                f"understates rainfall (e.g. {shown})")

    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-forecast", action="store_true", help="do not pull a GEFS cycle")
    ap.add_argument("--skip-obs", action="store_true", help="do not pull observations")
    ap.add_argument("--skip-train", action="store_true", help="do not retrain")
    ap.add_argument("--force-train", action="store_true",
                    help="retrain even when no new observations landed")
    ap.add_argument("--min-steps", type=float, default=0.98, metavar="FRACTION",
                    help="reject a cycle that fetched less than this fraction of its "
                         "expected GRIB steps (default 0.98; 0 disables the check)")
    ap.add_argument("--allow-short-accumulations", action="store_true",
                    help="publish even when a day's rainfall was summed from fewer steps "
                         "than it should have been. Understates precipitation; only for "
                         "getting *something* out when a partial cycle beats none.")
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
            reason = _reject_reason(result, args)
            if reason:
                # Stop before retraining. The runner's store is rebuilt from the published
                # artifact on every run and nothing here is persisted unless we publish, so
                # bailing now discards the thin cycle completely - no partial state, and
                # the next run re-attempts the same cycle from scratch.
                log.error("REJECTED %s: %s", result.get("target"), reason)
                log.error("nothing published; the site keeps the last good cycle and model")
                log.error("to publish anyway: --min-steps 0 --allow-short-accumulations")
                return 2
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
