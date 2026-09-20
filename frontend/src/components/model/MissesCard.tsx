import { useState } from "react";
import type { MissCase, ModelStatusResponse } from "../../api/types";
import { variableLabel, variableUnit } from "../../lib/displayNames";

type Mode = "missed" | "alarms";

/**
 * F6 - the held-out calls this model got most wrong.
 *
 * Every other panel here argues the model works. This one shows where it does not, and
 * it is on the Model tab next to the CORP diagram on purpose: CORP already says the
 * classifier is over-confident, and these are the same admission with names and dates
 * attached. A reviewer who wants to attack the model should find that we got there
 * first.
 *
 * The cases are picked by how confidently wrong the model was, on the held-out split
 * only - not a chosen sample. Both directions are shown, because a model with no false
 * alarms is usually one that never warns about anything.
 */
export function MissesCard({ data }: { data?: ModelStatusResponse }) {
  const [mode, setMode] = useState<Mode>("missed");

  if (!data?.model_trained) return null;
  const misses = data.misses;
  const missed = misses?.missed_busts ?? [];
  const alarms = misses?.false_alarms ?? [];

  if (!missed.length && !alarms.length) {
    return (
      <section className="card" aria-label="Where it was wrong">
        <header className="card__head"><h3>Where it was wrong</h3></header>
        <p className="muted small">
          This run was scored before these were recorded. They appear after the next
          retrain.
        </p>
      </section>
    );
  }

  const cases = mode === "missed" ? missed : alarms;

  return (
    <section className="card misses" aria-label="Where it was wrong">
      <header className="card__head">
        <h3>Where it was wrong</h3>
      </header>

      <div className="misses__switch" role="tablist" aria-label="Kind of mistake">
        <button
          type="button" role="tab" aria-selected={mode === "missed"}
          className={mode === "missed" ? "misses__tab is-on" : "misses__tab"}
          onClick={() => setMode("missed")}
        >
          Busts it missed
        </button>
        <button
          type="button" role="tab" aria-selected={mode === "alarms"}
          className={mode === "alarms" ? "misses__tab is-on" : "misses__tab"}
          onClick={() => setMode("alarms")}
        >
          False alarms
        </button>
      </div>

      <p className="muted small">
        {mode === "missed" ? (
          <>It busted, and this model said it probably would not. The costly kind — a
            forecaster reading this got no warning.</>
        ) : (
          <>It did not bust, and this model said it would. These cost trust rather than
            money, and a model with none of them is one that never warns about anything.</>
        )}
      </p>

      <ul className="misses__list">
        {cases.map((c) => <MissRow key={`${c.region_id}-${c.valid_date}-${c.lead_time_days}`} c={c} mode={mode} />)}
      </ul>

      <p className="muted small">
        The {cases.length} most confidently wrong calls on the{" "}
        <b>{misses?.split ?? "held-out"}</b> split — forecast cycles this model never
        trained on. Ranked by the model&apos;s own confidence, not chosen.
      </p>
    </section>
  );
}

function MissRow({ c, mode }: { c: MissCase; mode: Mode }) {
  const unit = variableUnit(c.variable);
  const pct = Math.round(c.bust_probability * 100);
  return (
    <li className="missrow">
      <div className="missrow__head">
        <b>{c.region_name ?? c.region_id}</b>
        <span className="muted small">
          {c.valid_date}
          {c.lead_time_days != null ? ` · day ${c.lead_time_days}` : ""}
        </span>
      </div>
      <p className="missrow__body">
        Said <b className={mode === "missed" ? "missrow__low" : "missrow__high"}>{pct}%</b>
        {mode === "missed" ? " — and it busted" : " — and it held"}
        {c.variable ? (
          <>
            {" "}on <b>{variableLabel(c.variable).toLowerCase()}</b>
            {c.actual_error != null && c.threshold != null ? (
              <>
                , off by <b className="mono">{c.actual_error.toFixed(2)}{unit ? ` ${unit}` : ""}</b>
                {" "}against a threshold of{" "}
                <span className="mono">{c.threshold.toFixed(2)}{unit ? ` ${unit}` : ""}</span>
              </>
            ) : null}
          </>
        ) : null}.
      </p>
    </li>
  );
}
