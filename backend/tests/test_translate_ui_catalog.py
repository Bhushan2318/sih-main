"""Incremental-run and placeholder logic of the offline UI translator (no model loaded)."""

from __future__ import annotations

from scripts import translate_ui_catalog as tuc


def test_plan_picks_new_and_changed_keys_only():
    en = {"a": "Alerts", "b": "Day {{n}}", "c": "Model"}
    existing = {
        "a": "अलर्ट",
        "b": "दिन {{n}}",
        "_meta": {"a": {"src": tuc.digest("Alerts")}, "b": {"src": tuc.digest("Day {{n}} old")}},
    }
    assert tuc.plan(en, existing) == ["b", "c"]


def test_placeholders_compare_as_multisets():
    assert tuc.placeholders("{{b}} then {{a}}") == tuc.placeholders("{{a}} और {{b}}")
    assert tuc.placeholders("दिन") != tuc.placeholders("Day {{n}}")


def test_every_frontend_language_has_a_model_code():
    assert len(tuc.FLORES) == 22


def test_tidy_drops_only_added_full_stops():
    assert tuc.tidy("Alerts", "எச்சரிக்கை.") == "எச்சரிக்கை"
    assert tuc.tidy("Model", "मॉडल।") == "मॉडल"
    assert tuc.tidy("It is late.", "देर हो गई है।") == "देर हो गई है।"


def test_markup_tags_count_as_placeholders():
    assert tuc.placeholders("<1>{{n}}</1> busts") == tuc.placeholders("<1>{{n}}</1> बस्ट")
    assert tuc.placeholders("<1>{{n}}</1> busts") != tuc.placeholders("{{n}} बस्ट")


def test_mask_numbers_placeholders_and_unmask_restores_them_in_any_order():
    masked, names = tuc.mask("Day {{n}} of {{max}}, again {{n}}")
    assert masked == "Day {{1}} of {{2}}, again {{1}}"
    assert tuc.unmask("{{2}} में से {{1}} दिन, {{1}}", names) == "{{max}} में से {{n}} दिन, {{n}}"


def test_mask_protects_comparison_signs_and_unmask_repairs_a_dropped_brace():
    masked, toks = tuc.mask("bust ≥ {{b}}% →")
    assert masked == "bust {{1}} {{2}}% {{3}}"
    assert tuc.unmask("बस्ट {{1}} {2}% {{3}}", toks) == "बस्ट ≥ {{b}}% →"
    assert tuc.unmask("आरओसी-एयूसी {1}}", tuc.mask("ROC-AUC ↑")[1]) == "आरओसी-एयूसी ↑"
    assert tuc.unmask("{{{1}} बस्ट", tuc.mask("{{n}} bust")[1]) == "{{n}} बस्ट"


def test_file_names_and_urls_are_protected_and_debris_is_refused():
    masked, toks = tuc.mask("listed in <1>third-party-licenses.txt</1> and /api/alerts at https://open-meteo.com/")
    assert "third-party" not in masked and "/api" not in masked and "open-meteo" not in masked
    assert tuc.clean("<1>third-party-licenses.txt</1> में")
    assert not tuc.clean("<1><আই. ডি. 1></1>-এ")
    assert not tuc.clean("ROC-AUC {1}}")


def test_path_protection_leaves_closing_tags_and_word_pairs_alone():
    masked, toks = tuc.mask("<0>a</0> train/test, see /api/alerts and <1>/docs</1>")
    assert "</0>" in masked and "train/test" in masked
    assert toks == ["/api/alerts", "/docs"]


def test_a_sign_the_model_invents_or_drops_is_refused():
    assert tuc.placeholders("R² share") != tuc.placeholders("आर 2 हिस्सा")
    assert tuc.placeholders("plain") != tuc.placeholders("± plain")


def test_tidy_closes_tag_spacing_and_fixes_tamil_colon():
    assert tuc.tidy("Run <0>{{id}}</0>", "चलाएँ <0> {{id}} </0>") == "चलाएँ <0>{{id}}</0>"
    assert tuc.tidy("Worst: {{a}}", "மிக மோசமானதுஃ {{a}}", "ta") == "மிக மோசமானது: {{a}}"
    assert tuc.tidy("Worst {{a}}", "அஃது {{a}}", "ta") == "அஃது {{a}}"


def test_glossary_terms_are_masked_and_come_back_as_the_fixed_rendering():
    import re
    gloss = [(re.compile(r"\b[Bb]usts?\b"), "பஸ்ட்"), (re.compile(r"\bwatch\b"), "வாட்ச்")]
    masked, toks = tuc.mask("{{n}} bust · {{w}} watch · Busts caught · Watch it", gloss)
    assert masked == "{{1}} {{3}} · {{2}} {{4}} · {{3}} caught · Watch it"
    assert tuc.unmask("{{1}} {{3}} · {{2}} {{4}}", toks) == "{{n}} பஸ்ட் · {{w}} வாட்ச்"


def test_shipped_glossary_covers_every_translated_language():
    import json
    terms = json.loads(tuc.GLOSSARY.read_text())["terms"]
    shipped = {p.stem for p in tuc.LOCALES.glob("*.json")} - {"en"}
    for name, t in terms.items():
        assert shipped <= set(t), f"{name} has no rendering for {shipped - set(t)}"


# --- check_ui_translations: the reviewer's mistake finder -------------------------------

from scripts import check_ui_translations as cut  # noqa: E402


def test_checker_flags_each_failure_mode():
    en = {"a": "Alerts", "b": "Forecast bust detection", "c": "Model", "d": "Ranking (ROC-AUC)", "e": "Bust risk"}
    cat = {
        "a": "Alerts",                      # left in English
        "b": "முன்கணிப்பு குண்டுவெடிப்பு கண்டறிதல்",  # hazard: bomb explosion
        "d": "தரவரிசை (ROC-AUC)",           # fine: allowed Latin term
        "e": "ஆபத்து ஆபத்து ஆபத்து",          # decoder loop
    }
    got = {(k, kind) for k, kind, _ in cut.check("ta", en, cat)}
    assert ("a", "english") in got
    assert ("b", "hazard") in got
    assert ("c", "missing") in got
    assert ("e", "repeat") in got
    assert not any(k == "d" for k, _ in got)


def test_checker_flags_wrong_script():
    got = {(k, kind) for k, kind, _ in cut.check("ta", {"x": "Bust risk by day"}, {"x": "दिन के अनुसार जोखिम"})}
    assert ("x", "script") in got


def test_checker_flags_a_flipped_comparison_sign():
    got = {(k, kind) for k, kind, _ in cut.check("ta", {"x": "bust ≥ {{b}}%"}, {"x": "பஸ்ட் ≤ {{b}}%"})}
    assert ("x", "symbol") in got


def test_checker_does_not_count_dropped_markup_as_lost_symbols():
    got = cut.check("hi", {"x": "<0>Start here:</0> Replay"}, {"x": "यहाँ से शुरू करें: रीप्ले"})
    assert not any(kind == "symbol" for _, kind, _ in got)


def test_checker_flags_debris():
    got = {kind for _, kind, _ in cut.check("bn", {"x": "in <1>a.txt</1>", "y": "ROC-AUC ↑"},
                                              {"x": "<1><আই. ডি. 1></1>-এ", "y": "ROC-AUC {1}}"})}
    assert "debris" in got


def test_complete_catches_a_dropped_glossary_term():
    import re
    masked, toks = tuc.mask("Bust thresholds are fitted on training", [(re.compile(r"\b[Bb]usts?\b"), "बस्ट")])
    assert masked == "{{1}} thresholds are fitted on training"
    assert tuc.complete("{{1}} सीमाएँ प्रशिक्षण पर", toks)
    assert not tuc.complete("सीमाएँ प्रशिक्षण पर", toks)


def test_checker_ignores_placeholder_names_for_glossary():
    got = cut.check("hi", {"x": "Bust probability: {{pct}}% ({{band}})"}, {"x": "बस्ट संभाव्यता: {{pct}}% ({{band}})"})
    assert not any(kind == "glossary" for _, kind, _ in got)


def test_sentence_split_keeps_placeholders_and_abbreviations_together():
    parts = tuc.SENTENCE.split("Bust thresholds use {{n}} days. Whole runs are kept together; no day leaks. Rodwell et al. (2013) differs.")
    assert parts == [
        "Bust thresholds use {{n}} days.",
        "Whole runs are kept together; no day leaks.",
        "Rodwell et al. (2013) differs.",
    ]


def test_every_source_hint_keeps_the_englishs_placeholders_tags_and_signs():
    import json
    en = json.loads((tuc.LOCALES / "en.json").read_text())
    hints = json.loads(tuc.HINTS.read_text())
    for k, v in hints.items():
        if k.startswith("_"):
            continue
        assert k in en, k
        assert tuc.placeholders(v) == tuc.placeholders(en[k]), k


def test_peel_takes_edge_arrows_off_and_leaves_inner_ones():
    assert tuc.peel("Open Replay →") == ("", "Open Replay", " →")
    assert tuc.peel("← All India") == ("← ", "All India", "")
    assert tuc.peel("skill vs climatology ↑") == ("", "skill vs climatology", " ↑")
    assert tuc.peel("↓ lower is better · ↑ higher is better") == ("↓ ", "lower is better · ↑ higher is better", "")


def test_a_lone_brace_is_debris():
    assert not tuc.clean("A {பஸ்ட் என்பது")
    assert tuc.clean("{{n}} பஸ்ட்")
    assert any(k == "debris" for _, k, _ in cut.check("ta", {"x": "A bust is"}, {"x": "A {பஸ்ட் என்பது"}))


def test_tidy_restores_colons_written_as_visarga_but_keeps_real_visarga():
    assert tuc.tidy("Bust probability: {{p}}", "बस्ट संभाव्यताः {{p}}") == "बस्ट संभाव्यता: {{p}}"
    assert tuc.tidy("Start here: go", "এখানে শুরু করুনঃ যান") == "এখানে শুরু করুন: যান"
    assert tuc.tidy("Mostly late", "प्रायः देर") == "प्रायः देर"


def test_checker_flags_a_colon_written_as_visarga():
    assert any(k == "colon" for _, k, _ in cut.check("bn", {"x": "Start here: go"}, {"x": "এখানে শুরু করুনঃ যান"}))


def test_tidy_handles_every_scripts_visarga_and_the_odia_pipe():
    assert tuc.tidy("Bust probability: {{p}}", "બસ્ટ સંભાવનાઃ {{p}}", "gu") == "બસ્ટ સંભાવના: {{p}}"
    assert tuc.tidy("Bust probability: {{p}}", "ಬಸ್ಟ್ ಸಂಭವನೀಯತೆಃ {{p}}", "kn") == "ಬಸ್ಟ್ ಸಂಭವನೀಯತೆ: {{p}}"
    assert tuc.tidy("It goes wrong. Then", "ଭୁଲ୍ ହୋଇଯାଏ | ତାପରେ", "or") == "ଭୁଲ୍ ହୋଇଯାଏ। ତାପରେ"
    assert tuc.tidy("Where it goes wrong", "<0>ଭୁଲ୍ ହୋଇଯାଏ |</0>", "or") == "<0>ଭୁଲ୍ ହୋଇଯାଏ।</0>"


def test_glossary_mask_swallows_a_leading_article():
    import json
    gloss = tuc.load_glossary("or")
    masked, toks = tuc.mask("A bust is a forecast. The bust band.", gloss)
    assert "A " not in masked and "The " not in masked
    assert any(k == "article" for _, k, _ in cut.check("or", {"x": "A bust is"}, {"x": "A ବଷ୍ଟ ହେଉଛି"}))


def test_tidy_keeps_the_englishs_edge_spaces():
    assert tuc.tidy(" · ingested {{when}}", "· {{when}} अंतर्ग्रहण") == " · {{when}} अंतर्ग्रहण"
    assert tuc.tidy("valid {{date}} · ", "वैध {{date}} ·") == "वैध {{date}} · "
    assert tuc.tidy("Alerts", " चेतावनी ") == "चेतावनी"
