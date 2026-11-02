import { translate, translateKey, useI18n } from "../i18n";
import type { TextPart } from "../api/types";
import { variableLabel } from "./displayNames";

/**
 * Sentences the backend writes (Replay's narration, the hero's national line, the risk
 * bands) arrive as English plus `{key, params}` parts - app/services/server_text.py. The
 * parts are rendered through this site's own catalogue, so they follow the language
 * picker; the English string is the fallback, and what an old response still carries.
 */
function one(p: TextPart): string | undefined {
  const params: Record<string, string> = { ...p.params };
  // The backend names the variable in English; in another language use the site's label,
  // and in English keep the backend's word so the sentence matches the API's string.
  if (params.var_id && params.var !== undefined && useI18n.getState().lang !== "en") {
    params.var = variableLabel(params.var_id).toLowerCase();
  }
  return translateKey(p.key, params);
}

/** Parts concatenated in order, or `fallback` if any part cannot be rendered. */
export function serverText(parts: TextPart[] | undefined, fallback: string, sep = ""): string {
  if (!parts?.length) return fallback;
  const out = parts.map(one);
  return out.every((s) => s !== undefined) ? out.join(sep) : fallback;
}

/** One Replay lead day: "Day 3 — part; part." */
export function stepNarration(lead: number, parts: TextPart[] | undefined, fallback: string): string {
  if (!parts?.length) return fallback;
  const body = serverText(parts, "", translate("srv.rp.sep"));
  return body ? translate("srv.rp.step", { lead, body }) : fallback;
}
