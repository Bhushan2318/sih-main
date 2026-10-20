import { describe, expect, it, beforeEach } from "vitest";
import { act } from "@testing-library/react";
import en from "./locales/en.json";
import { LANGUAGES, format, setLanguage, translate, useI18n } from "./index";

describe("i18n core", () => {
  beforeEach(() => useI18n.setState({ lang: "en", catalog: {} }));

  it("lists English plus all 22 Eighth Schedule languages", () => {
    expect(LANGUAGES.map((l) => l.code)).toHaveLength(23);
    expect(LANGUAGES[0].code).toBe("en");
  });

  it("fills {{placeholders}} and leaves unknown ones visible", () => {
    expect(format("Day {{n}} of {{max}}", { n: 3, max: 10 })).toBe("Day 3 of 10");
    expect(format("Day {{n}}", {})).toBe("Day {{n}}");
  });

  it("falls back to English when the active catalogue lacks a key", () => {
    useI18n.setState({ lang: "hi", catalog: {} });
    expect(translate("nav.alerts")).toBe(en["nav.alerts"]);
  });

  it("uses the active catalogue when it has the key", () => {
    useI18n.setState({ lang: "hi", catalog: { "nav.alerts": "अलर्ट" } });
    expect(translate("nav.alerts")).toBe("अलर्ट");
  });

  it("sets <html lang> and dir, right-to-left for Urdu", async () => {
    await act(() => setLanguage("ur"));
    expect(document.documentElement.lang).toBe("ur");
    expect(document.documentElement.dir).toBe("rtl");
    await act(() => setLanguage("en"));
    expect(document.documentElement.dir).toBe("ltr");
  });

  it("fetches the script's Noto font once, only when that language is picked", async () => {
    expect(document.querySelector('link[href*="Ol+Chiki"]')).toBeNull();
    await act(() => setLanguage("sat"));
    await act(() => setLanguage("sat"));
    expect(document.querySelectorAll('link[href*="Noto+Sans+Ol+Chiki"]')).toHaveLength(1);
    expect(document.documentElement.style.getPropertyValue("--f-script")).toBe('"Noto Sans Ol Chiki"');
    await act(() => setLanguage("en"));
  });

  it("ignores a language code it does not know", async () => {
    await act(() => setLanguage("xx"));
    expect(useI18n.getState().lang).toBe("en");
  });
});

describe("shipped catalogues", () => {
  const files = import.meta.glob<Record<string, string>>("./locales/*.json", { eager: true, import: "default" });
  const params = (s: string) => (s.match(/\{\{\w+\}\}/g) ?? []).sort().join(",");
  const tags = (s: string) => (s.match(/<\/?\d+>/g) ?? []).sort().join(",");

  it("en.json has no empty strings", () => {
    for (const [k, v] of Object.entries(en)) expect(v, k).not.toBe("");
  });

  for (const [path, cat] of Object.entries(files)) {
    if (path.endsWith("/en.json")) continue;
    it(`${path}: only known keys, placeholders preserved`, () => {
      for (const [k, v] of Object.entries(cat)) {
        if (k.startsWith("_")) continue;
        expect(en, `unknown key ${k}`).toHaveProperty(k);
        const src = (en as Record<string, string>)[k];
        expect(params(v), k).toBe(params(src));
        // Edge spaces join a fragment to the text beside it; lose one and words glue.
        expect(v.startsWith(" "), `${k} leading space`).toBe(src.startsWith(" "));
        expect(v.endsWith(" "), `${k} trailing space`).toBe(src.endsWith(" "));
        // Markup is all or nothing: the translator drops it whole when the model loses a tag.
        if (tags(v)) expect(tags(v), k).toBe(tags(src));
      }
    });
  }
});
