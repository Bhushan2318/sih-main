"""Flag likely mistakes in the machine-translated UI catalogues, for a human reviewer.

Reads frontend/src/i18n/locales/*.json against en.json and prints, per language, every
key that looks wrong and why. It cannot judge meaning - only a speaker can - but it
catches the failure modes IndicTrans2 is known for on short UI text:

  missing     no translation (refused for a lost placeholder, or never run)
  english     output identical to the English, or mostly Latin letters
  script      letters mostly outside the language's own script
  length      far shorter or longer than the English (dropped or hallucinated text)
  repeat      the same word three or more times in a row (decoder loop)
  debris      a mangled placeholder or a leftover <ID1>-style marker shown to the reader
  colon       the English has a colon and the translation has a visarga/aytham instead
  pipe        an ASCII "|" the English does not have (Odia's danda written wrongly)
  article     a stray English "a"/"an"/"the" left in the translation
  symbol      a comparison sign (≥ ≤ < > ±) lost or flipped - it reverses the meaning
  glossary    uses a glossary term in English but not its fixed rendering
  hazard      a rendering of a technical term known to be wrong (see HAZARDS)

    python backend/scripts/check_ui_translations.py [--langs hi ta] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

LOCALES = Path(__file__).resolve().parents[2] / "frontend" / "src" / "i18n" / "locales"

# Unicode script name (as unicodedata.name() spells it) per frontend language code.
SCRIPT = {
    "hi": "DEVANAGARI", "mr": "DEVANAGARI", "ne": "DEVANAGARI", "kok": "DEVANAGARI",
    "mai": "DEVANAGARI", "doi": "DEVANAGARI", "sa": "DEVANAGARI", "brx": "DEVANAGARI",
    "bn": "BENGALI", "as": "BENGALI", "ta": "TAMIL", "te": "TELUGU", "gu": "GUJARATI",
    "kn": "KANNADA", "ml": "MALAYALAM", "or": "ORIYA", "pa": "GURMUKHI",
    "ur": "ARABIC", "ks": "ARABIC", "sd": "ARABIC", "mni": "MEETEI MAYEK", "sat": "OL CHIKI",
}

# Renderings found in real output that change the meaning. Extend as reviewers find more.
HAZARDS = {
    "ta": {
        "குண்டுவெடிப்பு": "means 'bomb explosion', not a forecast bust",
        "மார்பளவு": "means 'chest measurement', not a forecast bust",
        "கடிகாரம்": "means 'clock', not the watch risk level",
        "இசைக்குழு": "means 'music band', not a risk band",
    },
    "bn": {
        "আবক্ষ": "means a sculpture bust / chest, not a forecast bust",
        "ঘড়ি": "means 'clock / wristwatch', not the watch risk level",
        "ধাওয়া": "means 'chase', not a forecast bust",
        "প্রতিস্থাপন": "means 'replacement', not replay",
    },
    "hi": {"छाती": "means 'chest', not a forecast bust"},
}

# Terms that are fine to leave in Latin script (metric and product names).
LATIN_OK = re.compile(
    r"ROC-AUC|PR-AUC|AUC|MAE|RMSE|R²|SHAP|F1|Brier|GEFS(v12)?|ERA5|NOAA|NCMRWF|IMD|CSV|TSV|XLSX|JSON|API|"
    r"HTTP|OpenAPI|Swagger|UI|FastAPI|XGBoost|React|TypeScript|Parquet|CI|GB|MB|UTC|IST|Z500|EMOS|IDR|"
    r"CORP|MJO|OMI|PSL|OLR|CGIAR-CSI|SRTM|CIAT|GADM|MIT|CC-BY|C3S|AWS|S3|SMS|Pydantic|Open-Meteo(\.com)?|"
    r"Rodwell|gadm\.org|Sanket|ID|ids?|p90|\{\{\w+\}\}|</?\d+>|/[\w./-]+|https?://\S+"
)


SYMBOLS = re.compile(r"[≥≤<>±↑↓→←]")
TAGS = re.compile(r"</?\d+>")
DEBRIS = re.compile(r"\{+\d+\}+|\{\{[^}]*[^\x00-\x7f][^}]*\}\}|<(?!/?\d+>)[^<>]{0,12}>")


def letters(text: str) -> list[str]:
    return [ch for ch in text if unicodedata.category(ch).startswith("L")]


def script_share(text: str, script: str) -> tuple[float, float]:
    """Share of letters in `script`, and share in Latin, after removing allowed Latin terms."""
    ls = letters(LATIN_OK.sub(" ", text))
    if not ls:
        return 1.0, 0.0
    own = sum(script in unicodedata.name(ch, "") for ch in ls)
    latin = sum("LATIN" in unicodedata.name(ch, "") for ch in ls)
    return own / len(ls), latin / len(ls)


def has_loop(text: str) -> bool:
    """Same word three times running. Split on spaces: regex word boundaries break inside
    Indic words at vowel signs, which are not \\w."""
    words = text.split()
    return any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2))


GLOSSARY = LOCALES.parent / "glossary.json"


def glossary_for(code: str) -> list[tuple[re.Pattern, str]]:
    if not GLOSSARY.exists():
        return []
    terms = json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]
    return [(re.compile(t["match"]), t[code]) for t in terms.values() if code in t]


def check(code: str, en: dict[str, str], cat: dict[str, str]) -> list[tuple[str, str, str]]:
    issues: list[tuple[str, str, str]] = []
    gloss = glossary_for(code)
    script = SCRIPT.get(code)
    for key, src in en.items():
        out = cat.get(key)
        if out is None:
            issues.append((key, "missing", ""))
            continue
        if out.strip() == src.strip() and letters(LATIN_OK.sub(" ", src)):
            issues.append((key, "english", out))
            continue
        if script:
            own, latin = script_share(out, script)
            if latin > 0.5:
                issues.append((key, "english", out))
            elif own < 0.6:
                issues.append((key, "script", out))
        ratio = len(out) / max(len(src), 1)
        if len(src) >= 12 and (ratio < 0.35 or ratio > 3.0):
            issues.append((key, "length", f"{ratio:.2f}x: {out}"))
        if DEBRIS.search(out) or re.search(r"[{}]", re.sub(r"\{\{\w+\}\}", "", out)):
            issues.append((key, "debris", out))
        # Markup tags are not comparison signs: strip them first (a sentence may ship
        # with its markup dropped whole, see translate_ui_catalog.py).
        if sorted(SYMBOLS.findall(TAGS.sub("", src))) != sorted(SYMBOLS.findall(TAGS.sub("", out))):
            issues.append((key, "symbol", out))
        if src.count(":") > out.count(":") and re.search(r"[ஃःঃઃಃഃଃਃః](?=\s|$)", out):
            issues.append((key, "colon", out))
        if "|" in out and "|" not in src:
            issues.append((key, "pipe", out))
        if re.search(r"(?<![A-Za-z.])(?:A|An|The|a|an|the)(?![A-Za-z.])", re.sub(r"\{\{\w+\}\}", "", out)):
            issues.append((key, "article", out))
        if has_loop(out):
            issues.append((key, "repeat", out))
        for pattern, rendering in gloss:
            # Placeholder names ({{band}}, {{bust}}) are not uses of the word.
            if pattern.search(re.sub(r"\{\{\w+\}\}", "", src)) and rendering not in out:
                issues.append((key, "glossary", f"expected {rendering}: {out}"))
        for bad, why in HAZARDS.get(code, {}).items():
            if bad in out:
                issues.append((key, "hazard", f"{why}: {out}"))
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--json", type=Path, help="also write every issue to this file")
    args = ap.parse_args()

    en = json.loads((LOCALES / "en.json").read_text(encoding="utf-8"))
    # Judge each translation against what the model was actually given.
    hints_path = LOCALES.parent / "source-hints.json"
    if hints_path.exists():
        hints = json.loads(hints_path.read_text(encoding="utf-8"))
        en = {k: hints.get(k, v) for k, v in en.items()}
    files = sorted(p for p in LOCALES.glob("*.json") if p.stem != "en")
    report: dict[str, list[dict]] = {}
    for path in files:
        code = path.stem
        if args.langs and code not in args.langs:
            continue
        cat = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if k != "_meta"}
        issues = check(code, en, cat)
        report[code] = [{"key": k, "kind": kind, "text": t} for k, kind, t in issues]
        kinds: dict[str, int] = {}
        for _, kind, _ in issues:
            kinds[kind] = kinds.get(kind, 0) + 1
        flagged = len({k for k, _, _ in issues})
        print(f"{code}: {len(cat)}/{len(en)} keys, {flagged} flagged {kinds}")
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
