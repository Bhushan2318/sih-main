import type { ModelStatusResponse } from "../api/types";

type Baselines = NonNullable<ModelStatusResponse["baselines"]>;
export type BaselineModel = NonNullable<Baselines["models"]>[number];

/** The rung that guesses purely from lead time, with no weather input at all. */
export const LEAD_DAY_RUNG_NAME = "lead_day";

export function findRung(
  models: BaselineModel[] | undefined,
  name: string,
): BaselineModel | undefined {
  return models?.find((m) => m.name === name);
}

export function leadDayRung(models: BaselineModel[] | undefined): BaselineModel | undefined {
  return findRung(models, LEAD_DAY_RUNG_NAME);
}

/** Skill this close to zero is noise, not signal. The live run's lead_day rung scores
 * +0.0001 against the next rung up's +0.03: calling that "above zero, so lead time
 * carries some signal" read a rounding error as a finding. */
export const NEGLIGIBLE_SKILL = 0.005;

/**
 * The lead_day rung tests whether the model just learned "later lead days are
 * worse" rather than anything about the day's actual weather. A skill (bss)
 * at or within NEGLIGIBLE_SKILL of zero means guessing from lead time alone is no
 * better than climatology — proof the classifier isn't rediscovering that trivially
 * true pattern.
 */
export function leadDayIsUninformative(models: BaselineModel[] | undefined): boolean | null {
  const rung = leadDayRung(models);
  if (!rung || typeof rung.bss !== "number" || !Number.isFinite(rung.bss)) return null;
  return rung.bss <= NEGLIGIBLE_SKILL;
}

/**
 * The strongest baseline that is not the served model and not the climatology floor
 * itself - climatology is always shown in its own row, so naming it "the best baseline"
 * too would be circular. Ranked by bss (skill vs climatology), the metric every rung in
 * the ladder is already compared on. A rung with no measured bss is skipped rather than
 * treated as a skill of zero, which would let it win by default.
 */
export function bestBaseline(models: BaselineModel[] | undefined): BaselineModel | undefined {
  let best: BaselineModel | undefined;
  for (const m of models ?? []) {
    if (m.is_model || m.name === "climatology") continue;
    if (typeof m.bss !== "number" || !Number.isFinite(m.bss)) continue;
    if (!best || m.bss > (best.bss as number)) best = m;
  }
  return best;
}

/** The ladder's rungs by name, in words. The raw name stays available as a tooltip for
 * anyone checking against baselines.json. */
const RUNG_LABELS: Record<string, string> = {
  climatology: "Climatology",
  lead_day: "Lead day only",
  spread: "Ensemble spread",
  "lead+spread+season": "Lead + spread + season",
  analog: "Analogues",
  EMOS: "EMOS",
  IDR: "IDR",
};

export function rungLabel(name: string): string {
  return RUNG_LABELS[name] ?? name;
}
