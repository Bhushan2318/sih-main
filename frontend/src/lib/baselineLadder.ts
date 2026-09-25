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
