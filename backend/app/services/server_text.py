"""Sentences the backend writes, as translatable parts.

Replay's narration, the hero's national note and the risk-band definitions are composed
here from the scored cycle. Each response keeps its English string, and also carries the
same sentence as `{key, params}` parts, which the site renders through its own catalogue
(frontend/src/i18n/locales/en.json, translated offline like every other UI string).

The English below is the catalogue's English: tests/test_server_text.py fails if the two
differ by a character. Params are already formatted, so a translated screen shows the
same figures as the English one.
"""
from __future__ import annotations

import re

from app.api import schemas

TEMPLATES: dict[str, str] = {
    # Replay, one line per lead day: "Day 3 — part; part; part."
    "srv.rp.step": "Day {{lead}} — {{body}}.",
    "srv.rp.sep": "; ",
    "srv.rp.climbs": "bust risk over {{region}} climbs {{from}}→{{to}}",
    "srv.rp.opens": "{{region}} opens with the highest bust risk at {{p}}",
    "srv.rp.still": "{{region}} still carries the highest bust risk at {{p}}",
    "srv.rp.driven": "driven by {{var}} error",
    "srv.rp.nowDriven": "now driven by {{var}} error",
    "srv.rp.high1": "{{n}} region high-risk",
    "srv.rp.highN": "{{n}} regions high-risk",
    "srv.rp.nowHigh1": "{{n}} region now high-risk",
    "srv.rp.nowHighN": "{{n}} regions now high-risk",
    # Replay's summary of the whole cycle, parts concatenated as they are.
    "srv.rs.climbs": "Init {{date}}: mean bust risk climbs from {{first}} on Day {{d0}} to {{last}} on Day {{d1}}.",
    "srv.rs.eases": "Init {{date}}: mean bust risk eases from {{first}} on Day {{d0}} to {{last}} on Day {{d1}}.",
    "srv.rs.holds": "Init {{date}}: mean bust risk holds from {{first}} on Day {{d0}} to {{last}} on Day {{d1}}.",
    "srv.rs.peakLed": " It peaks at {{p}} over {{region}} on Day {{lead}}, led by {{var}} error.",
    "srv.rs.peak": " It peaks at {{p}} over {{region}} on Day {{lead}}.",
    "srv.rs.verified": " Its forecast-vs-observed track is charted below.",
    "srv.rs.unverified": " This cycle has not verified yet, so no observed track is drawn.",
    # The hero's national line.
    "srv.nat.climbs": "Mean bust probability climbs from {{first}} at Day {{d0}} to {{last}} at Day {{d1}}, across all {{n}} scored regions.",
    "srv.nat.falls": "Mean bust probability falls from {{first}} at Day {{d0}} to {{last}} at Day {{d1}}, across all {{n}} scored regions.",
    # Risk bands.
    "srv.band.low": "bust risk below {{p}}%",
    "srv.band.medium": "bust risk {{lo}}% to {{hi}}%",
    "srv.band.high": "bust risk {{p}}% or above",
    "srv.band.basis": "Band edges are set from how this model's own predictions were spread on its validation data, not chosen by hand.",
}

_PARAM = re.compile(r"\{\{(\w+)\}\}")


def part(key: str, **params) -> schemas.TextPart:
    if key not in TEMPLATES:
        raise KeyError(key)
    return schemas.TextPart(key=key, params={k: str(v) for k, v in params.items()})


def english(p: schemas.TextPart) -> str:
    return _PARAM.sub(lambda m: p.params.get(m.group(1), m.group(0)), TEMPLATES[p.key])
