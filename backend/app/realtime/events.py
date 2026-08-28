"""WebSocket event envelope and the event names the frontend reacts to.

The frontend does not merge these payloads into its state - it uses the event name to
invalidate the right TanStack Query keys and refetch over REST. That keeps the WS and
REST shapes decoupled (see the plan's API contract section).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # sent once to each client on connect; carries current state, NOT a state change.
    # It is deliberately its own name so a client can never mistake the opening frame
    # for a retrain that just finished.
    CONNECTED = "connected"
    UPLOAD_RECEIVED = "upload_received"
    MAPPING_PENDING = "mapping_pending"
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_FAILED = "training_failed"
    NEW_ALERT = "new_alert"


class Event(BaseModel):
    event: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict = Field(default_factory=dict)


def make_event(event: EventType, **payload: Any) -> Event:
    return Event(event=event, payload=payload)
