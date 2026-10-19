import { create } from "zustand";
import en from "./locales/en.json";

/* UI language. English is the source catalogue and is bundled; every other language
 * is a JSON file produced offline (scripts/translate_catalog.py) and fetched only when
 * picked, so 22 languages do not cost the first paint anything. No translation happens
 * at runtime: the serving box has no room for a model, and none is needed. */

// Catalogues also carry a `_meta` object (source hash, reviewed flag); lookups only ever
// read string keys, so it is never displayed.
export type Catalog = Record<string, string>;
export type Key = keyof typeof en;

// English plus the 22 Eighth Schedule languages, each named in its own script.
// `code` is the IndicTrans2 / FLORES language as a short BCP-47 tag.
export const LANGUAGES: { code: string; name: string; rtl?: boolean }[] = [
  { code: "en", name: "English" },
  { code: "hi", name: "हिन्दी" },
  { code: "bn", name: "বাংলা" },
  { code: "ta", name: "தமிழ்" },
  { code: "te", name: "తెలుగు" },
  { code: "mr", name: "मराठी" },
  { code: "gu", name: "ગુજરાતી" },
  { code: "kn", name: "ಕನ್ನಡ" },
  { code: "ml", name: "മലയാളം" },
  { code: "or", name: "ଓଡ଼ିଆ" },
  { code: "pa", name: "ਪੰਜਾਬੀ" },
  { code: "as", name: "অসমীয়া" },
  { code: "ur", name: "اردو", rtl: true },
  { code: "ks", name: "کٲشُر", rtl: true },
  { code: "sd", name: "سنڌي", rtl: true },
  { code: "ne", name: "नेपाली" },
  { code: "kok", name: "कोंकणी" },
  { code: "mai", name: "मैथिली" },
  { code: "doi", name: "डोगरी" },
  { code: "sa", name: "संस्कृतम्" },
  { code: "brx", name: "बड़ो" },
  { code: "mni", name: "ꯃꯤꯇꯩꯂꯣꯟ" },
  { code: "sat", name: "ᱥᱟᱱᱛᱟᱲᱤ" },
];

const STORAGE_KEY = "sanket.lang";

// Vite turns each file into its own chunk; nothing but en.json lands in the main bundle.
const loaders = import.meta.glob<Catalog>(["./locales/*.json", "!./locales/en.json"], { import: "default" });

interface I18nState {
  lang: string;
  catalog: Catalog;
}

export const useI18n = create<I18nState>(() => ({ lang: "en", catalog: {} }));

export function format(text: string, params?: Record<string, string | number>): string {
  if (!params) return text;
  return text.replace(/\{\{(\w+)\}\}/g, (m, name: string) => (name in params ? String(params[name]) : m));
}

/** Non-hook form, for code outside components. Missing keys fall back to English. */
export function translate(key: Key, params?: Record<string, string | number>): string {
  const { catalog } = useI18n.getState();
  return format(catalog[key] ?? en[key] ?? key, params);
}

/** Subscribes the component to language changes. */
export function useT() {
  const catalog = useI18n((s) => s.catalog);
  return (key: Key, params?: Record<string, string | number>) => format(catalog[key] ?? en[key] ?? key, params);
}

// Noto family per script, fetched only when a language using it is picked. Most phones
// and desktops already carry Devanagari, Bengali and Tamil, but Ol Chiki (Santali) and
// Meetei Mayek (Manipuri) are commonly missing and render as empty boxes without one.
const SCRIPT_FONT: Record<string, string> = {
  hi: "Noto Sans Devanagari", mr: "Noto Sans Devanagari", ne: "Noto Sans Devanagari",
  kok: "Noto Sans Devanagari", mai: "Noto Sans Devanagari", doi: "Noto Sans Devanagari",
  sa: "Noto Sans Devanagari", brx: "Noto Sans Devanagari",
  bn: "Noto Sans Bengali", as: "Noto Sans Bengali", ta: "Noto Sans Tamil", te: "Noto Sans Telugu",
  gu: "Noto Sans Gujarati", kn: "Noto Sans Kannada", ml: "Noto Sans Malayalam", or: "Noto Sans Oriya",
  pa: "Noto Sans Gurmukhi", ur: "Noto Nastaliq Urdu", ks: "Noto Naskh Arabic", sd: "Noto Naskh Arabic",
  mni: "Noto Sans Meetei Mayek", sat: "Noto Sans Ol Chiki",
};

function loadScriptFont(code: string) {
  const family = SCRIPT_FONT[code];
  if (!family) return;
  const id = `font-${family.replace(/ /g, "-")}`;
  if (!document.getElementById(id)) {
    const link = document.createElement("link");
    link.id = id;
    link.rel = "stylesheet";
    link.href = `https://fonts.googleapis.com/css2?family=${family.replace(/ /g, "+")}:wght@400;600&display=swap`;
    document.head.appendChild(link);
  }
  // Listed after the Latin face, so numbers and Latin terms keep the site's typeface.
  document.documentElement.style.setProperty("--f-script", `"${family}"`);
}

function applyDocument(code: string) {
  const lang = LANGUAGES.find((l) => l.code === code);
  document.documentElement.lang = code;
  document.documentElement.dir = lang?.rtl ? "rtl" : "ltr";
  loadScriptFont(code);
}

export async function setLanguage(code: string): Promise<void> {
  if (!LANGUAGES.some((l) => l.code === code)) return;
  let catalog: Catalog = {};
  if (code !== "en") {
    const load = loaders[`./locales/${code}.json`];
    // A language with no catalogue yet still switches, and shows English, rather than
    // pretending to be translated.
    if (load) catalog = await load();
  }
  useI18n.setState({ lang: code, catalog });
  applyDocument(code);
  try {
    localStorage.setItem(STORAGE_KEY, code);
  } catch {
    /* private window: the choice just isn't remembered */
  }
}

/** Languages that have a shipped catalogue (English always does). */
export function availableLanguages() {
  return LANGUAGES.filter((l) => l.code === "en" || `./locales/${l.code}.json` in loaders);
}

export function restoreLanguage(): void {
  let saved: string | null = null;
  try {
    saved = localStorage.getItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
  if (saved && saved !== "en") void setLanguage(saved);
}

/** For keys built at runtime (`var.${name}`): undefined when English has no such key. */
export function translateKey(key: string, params?: Record<string, string | number>): string | undefined {
  if (!(key in en)) return undefined;
  return translate(key as Key, params);
}

/** Locale tag for Intl date and number formatting. English keeps en-GB, as before. */
export function locale(): string {
  const { lang } = useI18n.getState();
  return lang === "en" ? "en-GB" : `${lang}-IN`;
}
