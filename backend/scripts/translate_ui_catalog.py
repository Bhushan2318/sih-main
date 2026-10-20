"""Translate the frontend's English UI catalogue into the Eighth Schedule languages.

Runs offline, once per change to frontend/src/i18n/locales/en.json, on a machine with
a few GB to spare -- never on the serving box, and torch/transformers never enter the
serving dependency set. Output is plain JSON committed next to en.json; the site only
reads files.

Model: AI4Bharat IndicTrans2 En->Indic 1B (MIT). It is gated on Hugging Face: accept
the terms on the model page and `huggingface-cli login` first.

    pip install -r backend/scripts/requirements-translate.txt
    python backend/scripts/translate_ui_catalog.py --langs hi ta bn mr te

Incremental: a key is retranslated only when its English text changed (tracked by a
hash in `_meta`), so a human correction to a translation survives later runs. A
translation that loses a {{placeholder}} is refused and the key is left out, so the
site shows English for it instead of a broken sentence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

MODEL = "ai4bharat/indictrans2-en-indic-1B"
LOCALES = Path(__file__).resolve().parents[2] / "frontend" / "src" / "i18n" / "locales"

# Frontend code -> IndicTrans2 (FLORES-200) code.
FLORES = {
    "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym",
    "or": "ory_Orya", "pa": "pan_Guru", "as": "asm_Beng", "ur": "urd_Arab",
    "ks": "kas_Arab", "sd": "snd_Arab", "ne": "npi_Deva", "kok": "gom_Deva",
    "mai": "mai_Deva", "doi": "doi_Deva", "sa": "san_Deva", "brx": "brx_Deva",
    "mni": "mni_Mtei", "sat": "sat_Olck",
}

# {{name}} parameters and the <1>...</1> markup tags of i18n/rich.tsx.
PLACEHOLDER = re.compile(r"\{\{\w+\}\}|</?\d+>")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def placeholders(text: str) -> list[str]:
    """Parameters, tags and protected signs, as a multiset. Two strings with the same
    multiset can be shipped; a lost, added or flipped one cannot."""
    return sorted(PLACEHOLDER.findall(text) + SIGNS.findall(text))


SIGNS = re.compile(r"R²|[≥≤±÷−↑↓→←·‹›▶]|❚❚")


def clean(text: str) -> bool:
    """No mangled placeholder, no leftover marker, and no loose brace at all - a lone
    "{" reached the Tamil hero text ("A {பஸ்ட்") past the narrower check."""
    return not DEBRIS.search(text) and not re.search(r"[{}]", NAMED.sub("", text))


# Sentence-final marks the model adds to short UI labels ("Alerts" -> "...."), per script.
FINAL_MARKS = (".", "।", "॥", "۔", "|", "?", "؟")


NAMED = re.compile(r"\{\{(\w+)\}\}")
# Signs the model reverses or drops: "bust ≥ 57%" came back "≤" in Tamil, the arrows
# went missing, and "↓ … ↑" came back "Â … ^". Masked like a parameter, restored verbatim.
# File names, URLs and paths too: IndicTransToolkit swaps them for an <ID1> marker of its
# own and, in Bengali, failed to put "third-party-licenses.txt" back ("<আই. ডি. 1>").
# Acronyms stay in Latin script (standard in Indian-language technical text); the model
# otherwise spelled them out letter by letter ("एम. ए. ई.", "सी. एस. वी."). A paragraph
# dense with them ("Other inputs" has ten) overwhelms the model, so the last fallback
# translates with acronyms left to it - transliterated beats English.
ACRONYMS = (
    r"\b(?:ROC-AUC|PR-AUC|MAE|RMSE|CSV|TSV|XLSX|JSON|UTC|IST|HTTP|API|ERA5|GEFSv12|GEFS|NOAA|SHAP|CORP|MJO|"
    r"IMD|NCMRWF|EMOS|IDR|Z500|C3S|AWS|S3|SMS|GADM|SRTM|CIAT|CGIAR-CSI|OMI|PSL|OLR|MIT|CC-BY 4\.0|"
    r"OpenAPI|Swagger|FastAPI|XGBoost|React|TypeScript|Parquet|Pydantic|F1)\b|"
)
BASIC = (
    r"\{\{\w+\}\}|https?://\S+|(?:(?<=\s)|(?<=>)|^)/[a-z][\w./-]*\w|"
    r"[\w-]+\.(?:txt|json|md|csv|com|org)\b|R²|[≥≤±÷−↑↓→←·‹›▶]|❚❚"
)
PROTECTED = re.compile(ACRONYMS + BASIC)
PROTECTED_NO_ACRONYMS = re.compile(BASIC)
NUMBERED = re.compile(r"\{\{(\d+)\}\}")
# A lone-brace "{1}": the model sometimes drops one brace of a numbered placeholder.
BRACES = re.compile(r"\{+(\d+)\}+")
# Anything left over after unmasking means a placeholder was mangled past repair.
DEBRIS = re.compile(r"\{+\d+\}+|<(?!/?\d+>)[^<>]{0,12}>")


GLOSSARY = LOCALES.parent / "glossary.json"
HINTS = LOCALES.parent / "source-hints.json"


def load_sources(en: dict[str, str]) -> dict[str, str]:
    """What the model is given per key: the displayed English, or its plain-English
    rewording from source-hints.json where the English is idiomatic."""
    hints = json.loads(HINTS.read_text(encoding="utf-8")) if HINTS.exists() else {}
    out = {k: hints.get(k, v) for k, v in en.items()}
    return {k: reword(k, v) for k, v in out.items()}


# Terms the model reads in the wrong sense everywhere they appear. "Lead day" (how many
# days ahead a forecast is) came back as "leading / prominent / main day" (अग्रणी,
# प्रमुख, ಪ್ರಮುಖ, প্রধান) in most languages; "forecast day" says the same thing plainly.
REWRITES = [
    (re.compile(r"\bForecast lead day\b"), "Forecast day"),
    (re.compile(r"\bLead days\b"), "Forecast days"),
    (re.compile(r"\bLead day\b"), "Forecast day"),
    (re.compile(r"\blead days\b"), "forecast days"),
    (re.compile(r"\blead day\b"), "forecast day"),
    (re.compile(r"^Lead \+ "), "Forecast day + "),
    # "Run" came back as "race / running" (दौड़, ஓட்டம், పరుగు, …) in nine languages. It
    # means four things here; each is said plainly. "Long-run average" translated fine.
    # - a forecast:
    (re.compile(r"\bsince the last run\b"), "since the previous forecast"),
    (re.compile(r"\b([Rr])ecent runs\b"), r"\1ecent forecasts"),
    (re.compile(r"\bthis run's\b"), "this forecast's"),
    (re.compile(r"\bForecast runs are\b"), "Forecasts are"),
    (re.compile(r"\b(this|the current|this whole) forecast run\b"), r"\1 forecast"),
    (re.compile(r"\bpoints vs run\b"), "points compared with the forecast of"),
    (re.compile(r"\b(forecast-vs-observed chart|archived GEFS forecast) for this run\b"), r"\1 for this forecast"),
    (re.compile(r"\bwhole forecast runs are\b"), "all the days of one forecast are"),
    (re.compile(r"\bon a run it partly\b"), "on a forecast it partly"),
    # - a model version:
    (re.compile(r"\bThis run (was scored|recorded)\b"), r"This model version \1"),
    (re.compile(r"\bfor this run\b"), "for this model version"),
    (re.compile(r"\bof the current run\b"), "of the current model version"),
    (re.compile(r"^Run <0>"), "Model version <0>"),
    (re.compile(r"\bown training run\b"), "own training"),
    (re.compile(r"\bby the training run\b"), "during training"),
    # - a pipeline attempt:
    (re.compile(r"\bskipped runs\b"), "skipped attempts"),
    (re.compile(r"^Latest run's raw log"), "Raw log of the latest attempt"),
    # - operating / producing:
    (re.compile(r"\b([Ii]t|[Tt]his) runs on a free\b"), r"\1 is hosted on a free"),
    (re.compile(r"^How it runs$"), "How it works"),
    (re.compile(r"\bTraining runs on CI\b"), "Training is done on CI"),
    (re.compile(r"\bGEFS runs 31\b"), "GEFS produces 31"),
    (re.compile(r"\bhistorical re-runs for training\b"), "historical forecasts recomputed for training"),
    (re.compile(r"\bA re-run archive\b"), "An archive of recomputed forecasts"),
    (re.compile(r"\bthe forecast model re-run once a day\b"), "the forecast model recomputed once a day"),
    # "Store" (the data store) came back as "shop" (दुकान); "Parquet store" as "wood shop".
    (re.compile(r"\b(in the|this) store\b"), r"\1 database"),
    (re.compile(r"\bhive-partitioned Parquet store\b"), "database of Parquet files"),
    # Read in the wrong sense in a full Hindi read-through, 20 Oct:
    # "ensemble" as the verb "assemble" (असहमति इकट्ठा करें), "busted" as "exposed"
    # (भंडाफोड़), "held" as "organised" (आयोजित), "against" as "opposed to" (खिलाफ),
    # "drawn" as "made", "pull" as physically pulling, "scored" as "made".
    (re.compile(r"\bEnsemble disagreement\b"), "Disagreement between the forecast members"),
    (re.compile(r"\bensemble disagreement\b"), "disagreement between the forecast members"),
    (re.compile(r"\bensemble average\b"), "average of the forecast members"),
    (re.compile(r"\bEnsemble members available\b"), "Number of forecast members available"),
    (re.compile(r"^Ensemble spread$"), "Size of the disagreement between the forecast members"),
    (re.compile(r"\bensemble members\b"), "forecast members"),
    (re.compile(r"\b(compare|chart) its forecast against\b"), r"\1 its forecast with"),
    (re.compile(r"\bwe compare against\b"), "we compare with"),
    (re.compile(r"\bagainst\b"), "compared with"),
    (re.compile(r"— and it busted\b"), "— and the forecast did go badly wrong"),
    (re.compile(r"— and it held\b"), "— and the forecast stayed reliable"),
    (re.compile(r"\bhow often it actually busted\b"), "how often a bust actually happened"),
    (re.compile(r"^busted (\{\{p\}\})% of the time$"), r"a bust happened \1% of the time"),
    (re.compile(r"\bit actually busted (\{\{p\}\})% of the time\b"), r"a bust actually happened \1% of the time"),
    (re.compile(r"\b[Hh]eld-out\b"), lambda m: ("Set" if m.group(0)[0] == "H" else "set") + "-aside test"),
    (re.compile(r"\b[Hh]eld out\b"), lambda m: ("Kept" if m.group(0)[0] == "H" else "kept") + " aside for testing"),
    (re.compile(r"\bvalues are drawn\b"), "values are shown"),
    (re.compile(r"\bscored regions\b"), "regions with a risk score"),
    (re.compile(r"\bnothing scored yet\b"), "no risk scores yet"),
    (re.compile(r"\bNo scored risk bands yet\b"), "No risk bands calculated yet"),
    (re.compile(r"\bno region scored for\b"), "no region has a risk score for"),
    (re.compile(r"\bscored forecast days\b"), "forecast days with risk scores"),
    (re.compile(r"\b(district|districts) scored$"), r"\1 with a risk score"),
    (re.compile(r"\bis scored yet\b"), "has a risk score yet"),
    (re.compile(r"^Not scored - "), "No risk score - "),
    (re.compile(r"\bmid-pull\b"), "during a download"),
    (re.compile(r"\bLeading driver\b"), "Main cause"),
    (re.compile(r"^Driver: "), "Main cause: "),
    (re.compile(r"^driver$"), "main cause"),
    (re.compile(r"\b([Mm])ostly driven by\b"), r"\1ainly caused by"),
    (re.compile(r"\bbiggest driver\b"), "biggest cause"),
    # "Variable" came back as the adjective "changeable" (परिवर्तनशील); these are rainfall,
    # temperature and the like, so "weather measurement" says it.
    (re.compile(r"^Variable$"), "Weather measurement"),
    (re.compile(r"^Variables$"), "Weather measurements"),
    (re.compile(r"^Variables: "), "Weather measurements: "),
    (re.compile(r"\breforecast archive\b"), "archive of recomputed past forecasts"),
    (re.compile(r"\bNOAA’s reforecast\b"), "NOAA’s recomputed past forecasts"),
    (re.compile(r"\breforecast\b"), "recomputed past forecast"),
    (re.compile(r"\bingestion\b"), "data loading"),
    (re.compile(r"\bIngestion notes\b"), "Data loading notes"),
    (re.compile(r"\b[Ii]ngested\b"), lambda m: "Loaded" if m.group(0)[0] == "I" else "loaded"),
    (re.compile(r"\b[Ii]ngest\b"), lambda m: "Load" if m.group(0)[0] == "I" else "load"),
    (re.compile(r"^Mean\b"), "Average"),
    (re.compile(r"\bclimatology\b"), "the long-run average"),
    (re.compile(r"^Climatology$"), "Long-run average"),
    (re.compile(r"\blong-run\b"), "long-term"),
    (re.compile(r"\bLong-run\b"), "Long-term"),
    (re.compile(r"\b([Cc])onfidently wrong\b"), lambda m: ("W" if m.group(1) == "C" else "w") + "rong with the most certainty"),
    (re.compile(r"\bconfidence\b"), "certainty"),
    (re.compile(r"\bConfidence\b"), "Certainty"),
    (re.compile(r"\bOther inputs\b"), "Other input data"),
    (re.compile(r"\bhardest variable\b"), "hardest weather variable"),
    (re.compile(r"\ban em dash\b"), "a dash (—)"),
    # "Box" means the server.
    (re.compile(r"\b512 MB box\b"), "512 MB server"),
    (re.compile(r"\bfree-tier box\b"), "free server"),
]
WHOLE = {"al.colLead": "Forecast day"}


def reword(key: str, text: str) -> str:
    if key in WHOLE:
        return WHOLE[key]
    for pattern, repl in REWRITES:
        text = pattern.sub(repl, text)
    return text


def load_glossary(code: str) -> list[tuple[re.Pattern, str]]:
    """(pattern, rendering) for each term the glossary fixes in this language."""
    if not GLOSSARY.exists():
        return []
    terms = json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]
    # A leading "a"/"an"/"the" goes into the mask with the term: these languages have no
    # articles, and left outside it the model kept it verbatim ("A ବଷ୍ଟ ହେଉଛି").
    article = r"(?:\b(?:[Aa]n?|[Tt]he)\s+)?"
    return [(re.compile(article + t["match"]), t[code]) for t in terms.values() if code in t]


def mask(src: str, glossary: list[tuple[re.Pattern, str]] = (), acronyms: bool = True) -> tuple[str, list[str]]:
    """`Day {{n}} of {{max}} ≥` -> `Day {{1}} of {{2}} {{3}}`. The model translates the
    names inside braces ({{day}} came back as {{दिन}}, 77 of 535 Hindi strings refused) but
    leaves numbered ones alone (6 of 6 in a probe; #1# and the like broke every time)."""
    tokens: list[str] = []

    def number(token: str) -> str:
        if token not in tokens:
            tokens.append(token)
        return "{{%d}}" % (tokens.index(token) + 1)

    out = (PROTECTED if acronyms else PROTECTED_NO_ACRONYMS).sub(lambda m: number(m.group(0)), src)
    # Glossary terms become placeholders too, so the model places them in the sentence
    # but cannot translate them; unmask() writes the fixed rendering.
    for pattern, rendering in glossary:
        out = pattern.sub(lambda m: number(rendering), out)
    return out, tokens


def complete(raw: str, tokens: list[str]) -> bool:
    """Every numbered marker that went in came back. Needed beyond placeholders(): a
    glossary term is restored as plain text, so a dropped one ("bust" missing from a
    long About paragraph) was otherwise invisible."""
    got = {int(n) for n in BRACES.findall(raw)}
    return got == set(range(1, len(tokens) + 1))


def unmask(out: str, tokens: list[str]) -> str:
    # The model sometimes drops or adds braces: "{1}", "{1}}", "{{{1}}".
    out = BRACES.sub(r"{{\1}}", out)

    def sub(m: re.Match) -> str:
        i = int(m.group(1)) - 1
        return tokens[i] if 0 <= i < len(tokens) else m.group(0)

    return NUMBERED.sub(sub, out)


TAG = re.compile(r"</?\d+>")


def tidy(src: str, out: str, code: str = "") -> str:
    """Undo the model's and the processor's habits on short UI text: a full stop the
    English never had, spaces inside markup tags (`<0> x </0>`), and a colon turned into
    a letter (`மோசமானதுஃ`, `संभाव्यताः`, `শুরু করুনঃ`)."""
    out = out.strip()
    out = re.sub(r"<(\d+)>\s+", r"<\1>", out)
    out = re.sub(r"\s+</(\d+)>", r"</\1>", out)
    # The processor writes ":" as a script letter: Tamil aytham, Devanagari or Bengali
    # visarga. Visarga is also a real word-final letter (प्रायः), so only as many are
    # turned back as the English has colons, and only before a space or the end.
    need = src.count(":") - out.count(":")
    if need > 0:
        out = re.sub(r"[ஃःঃઃಃഃଃਃః](?=\s|$)", ":", out, count=need)
    # Odia: the processor writes the danda as an ASCII pipe ("ଭୁଲ୍ ହୋଇଯାଏ |").
    if code == "or":
        out = re.sub(r"\s*\|(?=\s|$|<)", "।", out)
    if not src.rstrip().endswith(FINAL_MARKS):
        while out.endswith(FINAL_MARKS):
            out = out[:-1].rstrip()
    # Keep the English's edge spaces: " · ingested {{when}}" is appended to the text
    # before it, and stripped it glued on ("…working.5 of these") - 348 strings.
    if out:
        out = src[: len(src) - len(src.lstrip())] + out + src[len(src.rstrip()):]
    return out


def plan(en: dict[str, str], existing: dict) -> list[str]:
    """Keys whose English is new or changed since the stored translation."""
    meta = existing.get("_meta", {})
    return [k for k, v in en.items() if k not in existing or meta.get(k, {}).get("src") != digest(v)]


def batches(keys: list[str], en: dict[str, str], size: int) -> list[list[str]]:
    """Shortest first, so one long paragraph does not pad a batch of labels to its length,
    and long strings go `size // 4` at a time: beam search holds every beam of every
    sentence, and 16 About-page paragraphs exhausted 6.7 GB of GPU memory."""
    ordered = sorted(keys, key=lambda k: len(en[k]))
    out: list[list[str]] = []
    i = 0
    while i < len(ordered):
        n = size if len(en[ordered[i]]) < 120 else max(1, size // 4)
        out.append(ordered[i : i + n])
        i += n
    return out


def translate(texts: list[str], tgt: str, model, tokenizer, ip, device: str) -> list[str]:
    import torch

    chunk = ip.preprocess_batch(texts, src_lang="eng_Latn", tgt_lang=tgt)
    enc = tokenizer(chunk, padding="longest", truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.inference_mode():
        gen = model.generate(**enc, num_beams=5, num_return_sequences=1, max_length=256)
    dec = tokenizer.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if device == "mps":
        torch.mps.empty_cache()
    return ip.postprocess_batch(dec, lang=tgt)


EDGE = re.compile(r"^((?:[←→↑↓‹▶·]|❚❚)\s*)?(.*?)(\s*[←→↑↓›·])?$", re.S)


def peel(src: str) -> tuple[str, str, str]:
    """Split off a leading or trailing arrow ("Open Replay →", "← All India", "Brier ↓"):
    left in, the model moved it mid-sentence ("बस्ट → की भविष्यवाणी")."""
    lead, core = src[: len(src) - len(src.lstrip())], src.strip()
    trail = src[len(src.rstrip()):]
    m = EDGE.match(core)
    return lead + (m.group(1) or ""), m.group(2), (m.group(3) or "") + trail


def translate_safely(texts: list[str], tgt: str, model, tokenizer, ip, device: str) -> list[str]:
    """translate(), but a GPU out-of-memory retries the batch one string at a time, and a
    single string that still does not fit comes back empty - refused, so the site shows
    English for it - instead of killing the run. Tamil and Marathi died three times each
    on the same long About-page paragraphs before this."""
    import torch

    try:
        return translate(texts, tgt, model, tokenizer, ip, device)
    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise
    out = []
    for t in texts:
        if device == "mps":
            torch.mps.empty_cache()
        try:
            out.extend(translate([t], tgt, model, tokenizer, ip, device))
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise
            out.append("")
    return out


# Not after "et al." / "e.g." / "i.e.", which end in a full stop mid-sentence.
SENTINEL = 7301
INDIC_DIGITS = str.maketrans("०१२३४५६७८९০১২৩৪৫৬৭৮৯੦੧੨੩੪੫੬੭੮੯૦૧૨૩૪૫૬૭૮૯୦୧୨୩୪୫୬୭୮୯௦௧௨௩௪௫௬௭௮௯౦౧౨౩౪౫౬౭౮౯೦೧೨೩೪೫೬೭೮೯൦൧൨൩൪൫൬൭൮൯",
                             "0123456789" * 9)


def by_number(src: str, tgt: str, glossary, model, tokenizer, ip, device: str, code: str) -> str:
    """Last-but-one resort: the model drops a placeholder that stands for a count ("changes
    {{1}} times" came back "changes repeatedly") but copies digits faithfully. So each
    {{param}} goes in as a distinctive number (7301, 7302, ...) and comes back as itself.
    Markup is dropped; glossary terms and signs stay masked as usual."""
    body = TAG.sub("", src)
    params = list(dict.fromkeys(NAMED.findall(body)))
    numbered = NAMED.sub(lambda m: str(SENTINEL + params.index(m.group(1))), body)
    masked, toks = mask(numbered, glossary)
    raw = translate_safely([masked], tgt, model, tokenizer, ip, device)[0]
    if not complete(raw, toks):
        return ""
    out = unmask(raw, toks).translate(INDIC_DIGITS)
    for i, name in enumerate(params):
        n = str(SENTINEL + i)
        if out.count(n) != numbered.count(n):
            return ""
        out = out.replace(n, "{{%s}}" % name)
    out = tidy(body, out, code)
    if not out or not clean(out) or placeholders(out) != placeholders(body):
        return ""
    return out


SENTENCE = re.compile(r"(?<!\bal\.)(?<!e\.g\.)(?<!i\.e\.)(?<=[.;?!])\s+(?=[A-Z“\"(])")


def by_sentence(src: str, tgt: str, glossary, model, tokenizer, ip, device: str, code: str) -> str:
    """Last resort for a long paragraph the model keeps losing a marker in: translate it
    one sentence at a time (markup dropped), each sentence held to the same checks.
    Returns "" if any sentence fails, so the paragraph is refused whole, never half-done."""
    parts = SENTENCE.split(TAG.sub("", src))
    if len(parts) < 2:
        return ""
    out = []
    masked = [mask(p, glossary) for p in parts]
    results = translate_safely([m for m, _ in masked], tgt, model, tokenizer, ip, device)
    for part, raw, (_, toks) in zip(parts, results, masked):
        v = tidy(part, unmask(raw, toks), code) if complete(raw, toks) else ""
        if not v or not clean(v) or placeholders(v) != placeholders(part):
            # One acronym-heavy sentence ("IMD–NCMRWF merged satellite-gauge rainfall")
            # sank whole paragraphs in ta/bn/ml: retry just that sentence unprotected.
            plain, ptoks = mask(part, glossary, acronyms=False)
            raw2 = translate_safely([plain], tgt, model, tokenizer, ip, device)[0]
            v = tidy(part, unmask(raw2, ptoks), code) if complete(raw2, ptoks) else ""
            if not v or not clean(v) or placeholders(v) != placeholders(part):
                return ""
        out.append(v)
    return " ".join(out)


def save(path: Path, en: dict[str, str], existing: dict) -> dict:
    """Write in English key order; drop keys English no longer has."""
    meta = existing.get("_meta", {})
    ordered = {k: existing[k] for k in en if k in existing}
    ordered["_meta"] = {k: meta[k] for k in en if k in meta}
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", nargs="+", default=list(FLORES), choices=list(FLORES))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--redo", nargs="*", default=[], help="keys to retranslate even if unchanged")
    ap.add_argument("--only", action="store_true", help="translate the --redo keys and nothing else")
    args = ap.parse_args()

    en: dict[str, str] = json.loads((LOCALES / "en.json").read_text(encoding="utf-8"))
    # From here on `en` is what gets translated; a hint changes the digest, so editing
    # one retranslates that key on the next run.
    en = load_sources(en)

    # torch 2.2 (the last Intel-Mac build) lacks a few ops on Apple's GPU backend;
    # those few run on CPU, everything else stays on the GPU.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).to(device).eval()
    ip = IndicProcessor(inference=True)

    for code in args.langs:
        path = LOCALES / f"{code}.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        glossary = load_glossary(code)
        todo = plan(en, existing) + [k for k in args.redo if k in en and k not in plan(en, existing)]
        if args.only:
            todo = [k for k in args.redo if k in en]
        t0 = time.monotonic()
        refused: list[str] = []
        meta = existing.setdefault("_meta", {})
        for group in batches(todo, en, args.batch):
            peeled = [peel(en[k]) for k in group]
            masked = [mask(body, glossary) for _, body, _ in peeled]
            results = translate_safely([m for m, _ in masked], FLORES[code], model, tokenizer, ip, device)
            for k, v, (_, names), (head, body, tail) in zip(group, results, masked, peeled):
                ok = complete(v, names)
                v = tidy(body, unmask(v, names), code)
                v = f"{head}{v}{tail}" if v else v
                if not ok or not clean(v):
                    v = ""
                if TAG.search(en[k]) and (not v or placeholders(v) != placeholders(en[k])):
                    # Emphasis is cosmetic; the sentence is not. Retry without the markup
                    # (rich() renders a tagless string as plain text) before refusing.
                    plain, plain_names = mask(TAG.sub("", en[k]), glossary)
                    v2 = translate_safely([plain], FLORES[code], model, tokenizer, ip, device)[0]
                    ok2 = complete(v2, plain_names)
                    v2 = tidy(en[k], unmask(v2, plain_names), code)
                    if ok2 and clean(v2) and placeholders(v2) == placeholders(TAG.sub("", en[k])) and v2.strip():
                        existing[k] = v2
                        meta[k] = {"src": digest(en[k]), "machine": True, "reviewed": False, "markup": "dropped"}
                        continue
                if placeholders(v) != placeholders(en[k]) or not v.strip():
                    v4 = by_number(en[k], FLORES[code], glossary, model, tokenizer, ip, device, code)
                    if v4:
                        existing[k] = v4
                        meta[k] = {"src": digest(en[k]), "machine": True, "reviewed": False,
                                   "markup": "dropped", "params": "as-numbers"}
                        continue
                    v3 = by_sentence(en[k], FLORES[code], glossary, model, tokenizer, ip, device, code)
                    if v3:
                        existing[k] = v3
                        meta[k] = {"src": digest(en[k]), "machine": True, "reviewed": False,
                                   "markup": "dropped", "split": "sentences"}
                        continue
                    plain, plain_toks = mask(TAG.sub("", en[k]), glossary, acronyms=False)
                    raw5 = translate_safely([plain], FLORES[code], model, tokenizer, ip, device)[0]
                    v5 = tidy(TAG.sub("", en[k]), unmask(raw5, plain_toks), code) if complete(raw5, plain_toks) else ""
                    if v5 and clean(v5) and placeholders(v5) == placeholders(TAG.sub("", en[k])):
                        existing[k] = v5
                        meta[k] = {"src": digest(en[k]), "machine": True, "reviewed": False,
                                   "markup": "dropped", "acronyms": "transliterated"}
                        continue
                    refused.append(k)
                    existing.pop(k, None)
                    meta.pop(k, None)
                    continue
                existing[k] = v
                meta[k] = {"src": digest(en[k]), "machine": True, "reviewed": False}
            # Saved after every batch: a crash costs one batch, not the language.
            save(path, en, existing)
        ordered = save(path, en, existing)
        print(f"{code}: {len(todo)} translated, {len(refused)} refused {refused}, "
              f"{len(ordered) - 1}/{len(en)} keys, {time.monotonic() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
