import { translate } from "../../i18n";
import { rich } from "../../i18n/rich";
import type { ModelStatusResponse } from "../../api/types";
import { BaselineModel, bestBaseline, findRung, leadDayIsUninformative, leadDayRung, rungLabel } from "../../lib/baselineLadder";
import { formatMetric } from "../../lib/format";
import { BaselineLadderTable } from "../model/BaselineLadderTable";

/** The strongest evidence in the project, moved off the About tab and onto the
 * first screen: the model against a ladder of baselines, with the lead_day
 * rung called out because a negative skill there is what rules out "the
 * model just learned day 10 is worse than day 1".
 *
 * This is the compact read, not the full ladder: climatology (the floor), the single
 * best non-Sanket baseline, and the served classifier - three rows a duty forecaster
 * can take in without scrolling. The full comparison (every rung, including lead_day
 * and spread on their own) lives on the Model tab; `onSeeFull` sends the reader there. */
export function BaselineLadderCard({ data, onSeeFull }: {
  data?: ModelStatusResponse;
  /** Switches the app to the Model tab. Omitted, the "full comparison" link is hidden
   * rather than pointing nowhere. */
  onSeeFull?: () => void;
}) {
  if (!data?.model_trained) return null;
  const models = data.baselines?.models;
  if (!models?.length) return null;

  const compactModels = [
    findRung(models, "climatology"),
    bestBaseline(models),
    models.find((m) => m.is_model),
  ].filter((m): m is BaselineModel => Boolean(m));

  return (
    <section className="card ladder" aria-label={translate("ladder.aria")}>
      <header className="card__head">
        <h3>{translate("ladder.title")}</h3>
      </header>
      <BaselineLadderTable baselines={{ ...data.baselines, models: compactModels }} flagLeadDay />
      <LeadDayCallout models={models} />
      {onSeeFull ? (
        <p className="ladder__more">
          <button type="button" className="chip" onClick={onSeeFull}>
            {translate("ladder.more")}
          </button>
        </p>
      ) : null}
    </section>
  );
}

function LeadDayCallout({ models }: { models: BaselineModel[] }) {
  const leadDay = leadDayRung(models);
  const uninformative = leadDayIsUninformative(models);
  if (!leadDay || uninformative == null) return null;

  const served = models.find((m) => m.is_model);
  return (
    <p className="muted small">
      {rich("ladder.lead", { name: rungLabel(leadDay.name), bss: formatMetric(leadDay.bss, 4) },
        [<b title={leadDay.name} />, <b className="mono" />])}
      {uninformative
        ? rich("ladder.uninf", {
            verdict: translate((leadDay.bss as number) <= 0 ? "ladder.zero" : "ladder.effZero"),
            bss: formatMetric(served?.bss, 4),
          }, [<b />])
        : translate("ladder.signal")}
    </p>
  );
}
