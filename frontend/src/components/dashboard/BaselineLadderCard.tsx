import type { ModelStatusResponse } from "../../api/types";
import { BaselineModel, leadDayIsUninformative, leadDayRung } from "../../lib/baselineLadder";
import { formatMetric } from "../../lib/format";
import { BaselineLadderTable } from "../model/BaselineLadderTable";

/** The strongest evidence in the project, moved off the About tab and onto the
 * first screen: the model against a ladder of baselines, with the lead_day
 * rung called out because a negative skill there is what rules out "the
 * model just learned day 10 is worse than day 1". */
export function BaselineLadderCard({ data }: { data?: ModelStatusResponse }) {
  if (!data?.model_trained) return null;
  const models = data.baselines?.models;
  if (!models?.length) return null;

  return (
    <section className="card ladder" aria-label="Baseline comparison">
      <header className="card__head">
        <h3>Not just &ldquo;day 10 is worse than day 1&rdquo;</h3>
      </header>
      <BaselineLadderTable baselines={data.baselines} flagLeadDay />
      <LeadDayCallout models={models} />
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
      Guessing purely from lead time (<b>{leadDay.name}</b>, no weather input at all)
      scores <b className="mono">{formatMetric(leadDay.bss, 4)}</b> skill vs climatology —{" "}
      {uninformative ? (
        <>
          <b>at or below zero</b>, meaning it does no better than always guessing the
          long-run bust rate. The classifier&apos;s <b>{formatMetric(served?.bss, 4)}</b> is
          not that trick.
        </>
      ) : (
        <>above zero, so lead time alone carries some signal here.</>
      )}
    </p>
  );
}
