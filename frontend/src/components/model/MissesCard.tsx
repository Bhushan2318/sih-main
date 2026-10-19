import { translate } from "../../i18n";
import { useState } from "react";
import { rich } from "../../i18n/rich";

import type { MissCase, ModelStatusResponse } from "../../api/types";
import { variableLabel, variableUnit } from "../../lib/displayNames";
import { dayLabel } from "../../lib/format";

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
      <section className="card" aria-label={translate("miss.title")}>
        <header className="card__head"><h3>{translate("miss.title")}</h3></header>
        <p className="muted small">{translate("miss.notYet")}</p>
      </section>
    );
  }

  const cases = mode === "missed" ? missed : alarms;

  return (
    <section className="card misses" aria-label={translate("miss.title")}>
      <header className="card__head">
        <h3>{translate("miss.title")}</h3>
      </header>

      <div className="misses__switch" role="tablist" aria-label={translate("miss.kind")}>
        <button
          type="button" role="tab" aria-selected={mode === "missed"}
          className={mode === "missed" ? "misses__tab is-on" : "misses__tab"}
          onClick={() => setMode("missed")}
        >
          {translate("miss.missed")}
        </button>
        <button
          type="button" role="tab" aria-selected={mode === "alarms"}
          className={mode === "alarms" ? "misses__tab is-on" : "misses__tab"}
          onClick={() => setMode("alarms")}
        >
          {translate("miss.alarms")}
        </button>
      </div>

      <p className="muted small">
        {mode === "missed" ? translate("miss.missedBody") : translate("miss.alarmsBody")}
      </p>

      <ul className="misses__list">
        {cases.map((c) => <MissRow key={`${c.region_id}-${c.valid_date}-${c.lead_time_days}`} c={c} mode={mode} />)}
      </ul>

      <p className="muted small">
        {rich("miss.footer", { n: cases.length, split: misses?.split ?? translate("miss.heldOut") }, [<b />])}
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
          {c.lead_time_days != null ? ` · ${dayLabel(c.lead_time_days)}` : ""}
        </span>
      </div>
      <p className="missrow__body">
        {rich(mode === "missed" ? "miss.saidBusted" : "miss.saidHeld", { pct },
          [<b className={mode === "missed" ? "missrow__low" : "missrow__high"} />])}
        {c.variable ? (
          <>
            {rich("miss.on", { v: variableLabel(c.variable).toLowerCase() }, [<b />])}
            {c.actual_error != null && c.threshold != null
              ? rich("miss.off", {
                  err: `${c.actual_error.toFixed(2)}${unit ? ` ${unit}` : ""}`,
                  thr: `${c.threshold.toFixed(2)}${unit ? ` ${unit}` : ""}`,
                }, [<b className="mono" />, <span className="mono" />])
              : null}
          </>
        ) : null}.
      </p>
    </li>
  );
}
