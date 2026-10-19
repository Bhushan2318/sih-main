import { availableLanguages, setLanguage, useI18n, useT } from "./index";

/* Lists only languages that have a shipped catalogue, so nothing in the menu silently
 * shows English. With English alone there is nothing to pick, and it renders nothing. */
export function LanguagePicker() {
  const t = useT();
  const lang = useI18n((s) => s.lang);
  const langs = availableLanguages();
  if (langs.length < 2) return null;
  return (
    <select
      className="langpick"
      aria-label={t("lang.label")}
      title={lang === "en" ? undefined : t("lang.machineNote")}
      value={lang}
      onChange={(e) => void setLanguage(e.target.value)}
    >
      {langs.map((l) => (
        <option key={l.code} value={l.code} lang={l.code}>
          {l.name}
        </option>
      ))}
    </select>
  );
}
