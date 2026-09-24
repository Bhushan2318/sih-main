from __future__ import annotations

from fastapi import APIRouter

from app.api import schemas
from app.ml import inference, registry
from app.realtime.broadcaster import manager
from app.services import upload_service
from app.services.region_service import _last_trained_at
from app.storage import parquet_store

router = APIRouter(prefix="/api/model", tags=["model"])


def _training_data(manifest: dict) -> dict:
    splits = manifest.get("split_cycles") or {}
    counts = {k: splits.get(k) for k in ("train", "val", "test")}
    known = [v for v in counts.values() if isinstance(v, int)]
    # Two manifest shapes: the single-year pipeline records train_dates, a pooled run
    # records train_years + test_year. Read whichever the run wrote.
    train_dates = splits.get("train_dates") or []
    train_years = sorted(int(y) for y in splits.get("train_years") or [])
    if not train_years and train_dates:
        train_years = sorted({int(str(d)[:4]) for d in train_dates})
    test_year = splits.get("test_year", manifest.get("test_year"))
    return {
        "cycles": sum(known) if known else None,
        "train_cycles": counts["train"],
        "val_cycles": counts["val"],
        "held_out_cycles": counts["test"],
        "canonical_rows": manifest.get("data_rows"),
        "paired_rows": manifest.get("paired_rows"),
        "first_train_date": train_dates[0] if train_dates else None,
        "first_train_year": train_years[0] if train_years else None,
        "last_train_year": train_years[-1] if train_years else None,
        "test_year": int(test_year) if test_year is not None else None,
    }


@router.get("/status", response_model=schemas.ModelStatusResponse)
def model_status() -> schemas.ModelStatusResponse:
    data_volume = parquet_store.dataset_summary()
    state = inference.load_model_state()

    if state is None:
        return schemas.ModelStatusResponse(
            model_trained=False,
            training_in_progress=upload_service.training_in_progress(),
            last_training_error=upload_service.last_training_error(),
            data_volume=data_volume,
            websocket_clients=manager.connection_count,
            message=(
                "No trained model yet. Upload a dataset containing both forecasts and "
                "matching observations; training starts automatically."
            ),
        )

    manifest = state.manifest or {}
    return schemas.ModelStatusResponse(
        model_trained=True,
        current_run_id=state.run_id,
        training_data=_training_data(manifest),
        baselines=registry.load_baselines(state.run_id) or {},
        misses=registry.load_misses(state.run_id) or {},
        last_trained_at=_last_trained_at(),
        training_in_progress=upload_service.training_in_progress(),
        last_training_error=upload_service.last_training_error(),
        data_volume=data_volume,
        modelled_variables=manifest.get("modelled_variables", state.variables),
        skipped_variables=manifest.get("skipped_variables", {}),
        validation_metrics=inference.model_validation_metrics(state),
        thresholds={
            "bust_threshold": state.thresholds.bust_threshold,
            "p90_error": state.thresholds.p90_error,
            "risk_band_cuts": state.thresholds.risk_band_cuts,
            "threshold_percentile": state.thresholds.threshold_percentile,
        },
        explanation_method=manifest.get("shap_method"),
        websocket_clients=manager.connection_count,
    )


@router.get("/runs", tags=["model"])
def list_runs() -> dict:
    return {"current_run_id": registry.current_run_id(), "runs": registry.list_runs()}
