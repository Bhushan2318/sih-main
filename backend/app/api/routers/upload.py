from __future__ import annotations

import shutil

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api import schemas
from app.api.deps import get_db
from app.config import settings
from app.ingestion.parsers import ParseError
from app.ingestion.pipeline import MappingValidationError, safe_upload_filename
from app.services import upload_service

router = APIRouter(prefix="/api/upload", tags=["upload"])

MAX_BYTES = 200 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024


def _require_local_retrain() -> None:
    """Refuse mutating uploads on the read-only serving deployment."""
    if not settings.allow_local_retrain:
        raise HTTPException(
            409,
            "Uploads and retraining are disabled on this deployment. Models and data "
            "are refreshed by the deployment pipeline; this instance only serves them.",
        )


async def _stream_to_temp(file: UploadFile, destination) -> int:
    """Copy an UploadFile in bounded chunks and enforce the byte limit while copying."""
    total = 0
    with destination.open("wb") as out:
        while True:
            chunk = await file.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES:
                raise HTTPException(413, f"file exceeds the {MAX_BYTES // (1024*1024)} MB limit")
            out.write(chunk)
    return total


@router.post("", response_model=schemas.UploadResponse)
@router.post("/", response_model=schemas.UploadResponse, include_in_schema=False)
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> schemas.UploadResponse:
    _require_local_retrain()
    try:
        filename = safe_upload_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if file.size is not None and file.size > MAX_BYTES:
        raise HTTPException(413, f"file exceeds the {MAX_BYTES // (1024*1024)} MB limit")

    tmp = upload_service.new_temp_upload(filename)
    try:
        total = await _stream_to_temp(file, tmp)
        if total == 0:
            raise HTTPException(400, "uploaded file is empty")
        result = upload_service.handle_upload(db, tmp, filename)
    except HTTPException:
        raise
    except ParseError as exc:
        raise HTTPException(422, f"could not parse this file: {exc}") from exc
    except MappingValidationError as exc:
        raise HTTPException(422, f"invalid column mapping: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(422, f"invalid upload: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"ingestion failed: {type(exc).__name__}: {exc}") from exc
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)

    db.commit()
    if result.should_retrain:
        background.add_task(upload_service.run_retrain, result.batch_id)
    return upload_service.to_response(result)


@router.post("/{batch_id}/confirm-mapping", response_model=schemas.UploadResponse)
def confirm_mapping(
    batch_id: str,
    body: schemas.ConfirmMappingRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> schemas.UploadResponse:
    _require_local_retrain()
    mappings = [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in body.mappings]
    try:
        result = upload_service.handle_confirm(db, batch_id, mappings)
    except MappingValidationError as exc:
        raise HTTPException(422, f"invalid column mapping: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"canonicalisation failed: {type(exc).__name__}: {exc}") from exc

    db.commit()
    if result.should_retrain:
        background.add_task(upload_service.run_retrain, result.batch_id)
    return upload_service.to_response(result)
