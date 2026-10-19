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


SIGNS = re.compile(r"R²|[≥≤±÷−↑↓→←]")


def clean(text: str) -> bool:
    """No mangled placeholder, no leftover marker, and no loose brace at all - a lone
    "{" reached the Tamil hero text ("A {பஸ்ட்") past the narrower check."""
    return not DEBRIS.search(text) and not re.search(r"[{}]", NAMED.sub("", text))


# Sentence-final marks the model adds to short UI labels ("Alerts" -> "...."), per script.
FINAL_MARKS = (".", "।", "॥", "۔")


NAMED = re.compile(r"\{\{(\w+)\}\}")
# Signs the model reverses or drops: "bust ≥ 57%" came back "≤" in Tamil, the arrows
# went missing, and "↓ … ↑" came back "Â … ^". Masked like a parameter, restored verbatim.
# File names, URLs and paths too: IndicTransToolkit swaps them for an <ID1> marker of its
# own and, in Bengali, failed to put "third-party-licenses.txt" back ("<আই. ডি. 1>").
PROTECTED = re.compile(
    r"\{\{\w+\}\}|https?://\S+|(?:(?<=\s)|(?<=>)|^)/[a-z][\w./-]*\w|[\w-]+\.(?:txt|json|md|csv|com|org)\b|R²|[≥≤±÷−↑↓→←]"
)
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
    hints = json.loads(HINTS.read_text()) if HINTS.exists() else {}
    return {k: hints.get(k, v) for k, v in en.items()}


def load_glossary(code: str) -> list[tuple[re.Pattern, str]]:
    """(pattern, rendering) for each term the glossary fixes in this language."""
    if not GLOSSARY.exists():
        return []
    terms = json.loads(GLOSSARY.read_text())["terms"]
    return [(re.compile(t["match"]), t[code]) for t in terms.values() if code in t]


def mask(src: str, glossary: list[tuple[re.Pattern, str]] = ()) -> tuple[str, list[str]]:
    """`Day {{n}} of {{max}} ≥` -> `Day {{1}} of {{2}} {{3}}`. The model translates the
    names inside braces ({{day}} came back as {{दिन}}, 77 of 535 Hindi strings refused) but
    leaves numbered ones alone (6 of 6 in a probe; #1# and the like broke every time)."""
    tokens: list[str] = []

    def number(token: str) -> str:
        if token not in tokens:
            tokens.append(token)
        return "{{%d}}" % (tokens.index(token) + 1)

    out = PROTECTED.sub(lambda m: number(m.group(0)), src)
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
        out = re.sub(r"[ஃःঃ](?=\s|$)", ":", out, count=need)
    if not src.rstrip().endswith(FINAL_MARKS):
        while out.endswith(FINAL_MARKS):
            out = out[:-1].rstrip()
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


EDGE = re.compile(r"^([←→↑↓]\s*)?(.*?)(\s*[←→↑↓])?$", re.S)


def peel(src: str) -> tuple[str, str, str]:
    """Split off a leading or trailing arrow ("Open Replay →", "← All India", "Brier ↓"):
    left in, the model moved it mid-sentence ("बस्ट → की भविष्यवाणी")."""
    m = EDGE.match(src)
    return m.group(1) or "", m.group(2), m.group(3) or ""


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
        if not complete(raw, toks):
            return ""
        v = tidy(part, unmask(raw, toks), code)
        if not v or not clean(v) or placeholders(v) != placeholders(part):
            return ""
        out.append(v)
    return " ".join(out)


def save(path: Path, en: dict[str, str], existing: dict) -> dict:
    """Write in English key order; drop keys English no longer has."""
    meta = existing.get("_meta", {})
    ordered = {k: existing[k] for k in en if k in existing}
    ordered["_meta"] = {k: meta[k] for k in en if k in meta}
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n")
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", nargs="+", default=list(FLORES), choices=list(FLORES))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--redo", nargs="*", default=[], help="keys to retranslate even if unchanged")
    args = ap.parse_args()

    en: dict[str, str] = json.loads((LOCALES / "en.json").read_text())
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
        existing = json.loads(path.read_text()) if path.exists() else {}
        glossary = load_glossary(code)
        todo = plan(en, existing) + [k for k in args.redo if k in en and k not in plan(en, existing)]
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
                    v3 = by_sentence(en[k], FLORES[code], glossary, model, tokenizer, ip, device, code)
                    if v3:
                        existing[k] = v3
                        meta[k] = {"src": digest(en[k]), "machine": True, "reviewed": False,
                                   "markup": "dropped", "split": "sentences"}
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
