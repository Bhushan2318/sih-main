import type { ModelStatusResponse } from "../../api/types";
import { leadDayIsUninformative, leadDayRung } from "../../lib/baselineLadder";
import { BaselineLadderTable } from "../model/BaselineLadderTable";

const n3 = (v: unknown, dp = 4) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(dp) : "—";

/** The strongest evidence in the project, moved off the About tab and onto the
 * first screen: the model against a ladder of baselines, with the lead_day
 * rung called out because a negative skill there is what rules out "the
 * model just learned day 10 is worse than day 1". */
export function BaselineLadderCard({ data }: { data?: ModelStatusResponse }) {
  if (!data?.model_trained) return null;
  const models = data.baselines?.models;
  if (!models?.length) return null;

  const leadDay = leadDayRung(models);
  const uninformative = leadDayIsUninformative(models);
  const served = models.find((m) => m.is_model);

  return (
    <section className="card ladder" aria-label="Baseline comparison">
      <header className="card__head">
        <h3>Not just &ldquo;day 10 is worse than day 1&rdquo;</h3>
      </header>
      <BaselineLadderTable baselines={data.baselines} flagLeadDay />
      {leadDay && uninformative != null ? (
        <p className="muted small">
          Guessing purely from lead time (<b>{leadDay.name}</b>, no weather input at all)
          scores <b className="mono">{n3(leadDay.bss)}</b> skill vs climatology —{" "}
          {uninformative ? (
            <>
              <b>at or below zero</b>, meaning it does no better than always guessing the
              long-run bust rate. The classifier&apos;s <b>{n3(served?.bss)}</b> is not that
              trick.
            </>
          ) : (
            <>above zero, so lead time alone carries some signal here.</>
          )}
        </p>
      ) : null}
    </section>
  );
}
