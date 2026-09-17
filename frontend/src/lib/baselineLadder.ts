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

/**
 * The lead_day rung tests whether the model just learned "later lead days are
 * worse" rather than anything about the day's actual weather. A skill (bss)
 * at or below zero means guessing from lead time alone is no better than
 * climatology — proof the classifier isn't rediscovering that trivially true
 * pattern.
 */
export function leadDayIsUninformative(models: BaselineModel[] | undefined): boolean | null {
  const rung = leadDayRung(models);
  if (!rung || typeof rung.bss !== "number" || !Number.isFinite(rung.bss)) return null;
  return rung.bss <= 0;
}
