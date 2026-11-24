"""Backend-written sentences ship as {key, params} parts beside their English.

The site translates the parts with the frontend catalogue, so every template here must
exist in frontend/src/i18n/locales/en.json with the very same English - otherwise a
translated screen and the English one would say different things.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.services import server_text

EN = Path(__file__).resolve().parents[2] / "frontend" / "src" / "i18n" / "locales" / "en.json"


def test_every_template_is_in_the_frontend_catalogue_word_for_word():
    en = json.loads(EN.read_text(encoding="utf-8"))
    missing = {k: v for k, v in server_text.TEMPLATES.items() if en.get(k) != v}
    assert missing == {}


def test_a_part_renders_to_its_english():
    p = server_text.part("srv.rp.climbs", region="Idukki, Kerala", **{"from": "0.41", "to": "0.77"})
    assert server_text.english(p) == "bust risk over Idukki, Kerala climbs 0.41→0.77"


def test_an_unknown_key_is_refused_rather_than_shipped():
    import pytest
    with pytest.raises(KeyError):
        server_text.part("srv.nope")
