import { translate } from "../../i18n";
import { useMemo } from "react";
import type { AllRegionsResponse, RegionsResponse } from "../../api/types";
import { variableLabel } from "../../lib/displayNames";
import { dayLabel, dayRange } from "../../lib/format";
import { isScoredRegion, riskBandForRegion, type RiskCuts } from "../../lib/riskBands";

export function LeadDayRail({ all, value, onChange, riskCuts }: {
  all?: AllRegionsResponse;
  value: number;
  onChange: (d: number) => void;
  riskCuts?: RiskCuts;
}) {
  const rows = useMemo(
    () => (all?.days ?? []).map((day) => summarise(day, riskCuts)).filter(Boolean) as Row[],
    [all, riskCuts],
  );
  const drivers = useMemo(() => driverRuns(rows), [rows]);

  if (rows.length < 2) return null;
  const peak = Math.max(...rows.map((r) => r.mean));

  return (
    <aside className="rail" aria-label={translate("rail.aria")}>
      <h3 className="rail__title">{translate("rail.title")}</h3>

      <div className="rail__rows" role="group" aria-label={translate("lead.label")}>
        {rows.map((r) => (
          <button
            key={r.lead}
            type="button"
            aria-pressed={r.lead === value}
            className={r.lead === value ? "railrow railrow--active" : "railrow"}
            onClick={() => onChange(r.lead)}
            title={translate("rail.rowTitle", { day: dayLabel(r.lead), bust: r.bust, watch: r.watch, low: r.low })}
          >
            <span className="railrow__day">{dayLabel(r.lead)}</span>

            <span className="railrow__bar" aria-hidden="true">
              {r.low ? <i className="railrow__seg railrow__seg--low" style={{ flexGrow: r.low }} /> : null}
              {r.watch ? <i className="railrow__seg railrow__seg--watch" style={{ flexGrow: r.watch }} /> : null}
              {r.bust ? <i className="railrow__seg railrow__seg--bust" style={{ flexGrow: r.bust }} /> : null}
            </span>

            <span className={r.mean >= peak - 1e-9 ? "railrow__mean railrow__mean--peak" : "railrow__mean"}>
              {(r.mean * 100).toFixed(0)}%
            </span>

            <span className="railrow__meta">
              {translate("rail.meta", { bust: r.bust, watch: r.watch })}{r.driver ? ` · ${variableLabel(r.driver)}` : ""}
            </span>
          </button>
        ))}
      </div>

      <p className="rail__key">
        <b><i className="railrow__seg--low" />{translate("band.low")}</b>
        <b><i className="railrow__seg--watch" />{translate("band.medium")}</b>
        <b><i className="railrow__seg--bust" />{translate("band.high")}</b>
      </p>

      <p className="rail__note">
        {translate("rail.note")}
      </p>

      {drivers ? <p className="rail__driver">{drivers}</p> : null}
    </aside>
  );
}

type Row = {
  lead: number;
  n: number;
  low: number;
  watch: number;
  bust: number;
  mean: number;
  driver: string | null;
};

function summarise(day: RegionsResponse, riskCuts?: RiskCuts): Row | null {
  const regions = (day.regions ?? []).filter(isScoredRegion);
  if (!regions.length) return null;

  const probs: number[] = [];
  const drivers = new Map<string, number>();
  let low = 0;
  let watch = 0;
  let bust = 0;

  for (const r of regions) {
    const band = riskBandForRegion(r, riskCuts);
    if (band === "high") bust += 1;
    else if (band === "medium") watch += 1;
    else if (band === "low") low += 1;
    probs.push(r.bust_probability as number);
    if (r.dominant_variable) drivers.set(r.dominant_variable, (drivers.get(r.dominant_variable) ?? 0) + 1);
  }
  if (!probs.length) return null;

  const top = [...drivers.entries()].sort((a, b) => b[1] - a[1])[0];
  return {
    lead: day.lead_time_days,
    n: regions.length,
    low,
    watch,
    bust,
    mean: probs.reduce((s, p) => s + p, 0) / probs.length,
    driver: top?.[0] ?? null,
  };
}

function driverRuns(rows: Row[]): string | null {
  const named = rows.filter((r) => r.driver);
  if (!named.length) return null;

  const runs: { driver: string; from: number; to: number }[] = [];
  for (const r of named) {
    const last = runs[runs.length - 1];
    if (last && last.driver === r.driver) last.to = r.lead;
    else runs.push({ driver: r.driver as string, from: r.lead, to: r.lead });
  }

  if (runs.length === 1) return translate("rail.oneCause", { v: variableLabel(runs[0].driver) });

  if (runs.length > 3) return translate("rail.manyCauses", { n: runs.length - 1 });
  return translate("rail.mostly", { list: runs.map((r) => translate("rail.item", { v: pretty(r.driver), span: span(r) })).join(translate("rail.then")) });
}

const span = (r: { from: number; to: number }) => (r.from === r.to ? translate("rail.on", { day: dayLabel(r.from) }) : dayRange(r.from, r.to));

const pretty = (v: string) => variableLabel(v).toLowerCase();
