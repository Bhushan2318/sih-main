import { translate } from "../../i18n";
import { useMemo } from "react";
import { variableLabel } from "../../lib/displayNames";
import { rich } from "../../i18n/rich";

import type { AllRegionsResponse, RegionsResponse } from "../../api/types";
import {
  isScoredRegion,
  riskBandForProbability,
  riskBandForRegion,
  type RiskCuts,
} from "../../lib/riskBands";
import { dayLabel } from "../../lib/format";

export function KpiStrip({ all, day, riskCuts, onSelectRegion }: {
  all?: AllRegionsResponse;
  day?: RegionsResponse;
  riskCuts?: RiskCuts;
  onSelectRegion?: (regionId: string) => void;
}) {
  const stats = useMemo(() => derive(all, day, riskCuts), [all, day, riskCuts]);
  if (!stats) return null;

  return (
    <section className="kpis" aria-label={translate("kpi.aria")}>
      <Kpi
        cap={stats.meanCap}
        label={translate("kpi.mean", { day: dayLabel(stats.lead) })}
        value={stats.mean != null ? `${(stats.mean * 100).toFixed(0)}%` : "—"}
        note={
          stats.scored
            ? rich("kpi.meanNote", { n: stats.scored, day: dayLabel(stats.lead) }, [<b />])
            : translate("kpi.noScored")
        }
      />
      <Kpi
        cap="bust"
        label={translate("kpi.bustRegions", { day: dayLabel(stats.lead) })}
        value={
          stats.scored ? (
            <>
              {stats.high}
              <small>/ {stats.scored}</small>
            </>
          ) : (
            "—"
          )
        }
        note={stats.scored ? translate("kpi.bustNote", { day: dayLabel(stats.lead) }) : translate("kpi.nothing")}
      />
      <Kpi
        cap="blue"
        label={translate("kpi.conf")}
        value={
          stats.decayPct != null ? (
            <>
              {stats.decayPct > 0 ? "+" : ""}
              {stats.decayPct.toFixed(0)}
              <small>%</small>
            </>
          ) : (
            "—"
          )
        }
        note={
          stats.decayPct != null && stats.decayFrom != null && stats.decayTo != null ? (
            translate("kpi.confNote", { from: dayLabel(stats.decayFrom), to: dayLabel(stats.decayTo) })
          ) : (
            translate("kpi.confNeeds")
          )
        }
      />
      <Kpi
        cap="watch"
        label={translate("kpi.peak", { day: dayLabel(stats.lead) })}
        value={stats.peak ? `${(stats.peak.value * 100).toFixed(0)}%` : "—"}
        onOpen={stats.peak && onSelectRegion ? () => onSelectRegion(stats.peak!.id) : undefined}
        note={
          stats.peak ? (
            <>
              <b>{stats.peak.name}</b>
              {stats.peak.driver ? <> · {label(stats.peak.driver)}</> : null}
            </>
          ) : (
            translate("kpi.noPeak")
          )
        }
      />
    </section>
  );
}

function Kpi({ cap, label, value, note, onOpen }: {
  cap: string;
  label: string;
  value: React.ReactNode;
  note: React.ReactNode;
  onOpen?: () => void;
}) {
  const open = onOpen
    ? {
        role: "button",
        tabIndex: 0,
        onClick: onOpen,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpen(); }
        },
      }
    : {};
  return (
    <article className={onOpen ? "kpi kpi--open rise" : "kpi rise"} {...open}>
      <div className={`kpi__cap kpi__cap--${cap}`} aria-hidden="true" />
      <div className="kpi__label">{label}</div>
      <div className="kpi__value">{value}</div>
      <p className="kpi__note">{note}</p>
    </article>
  );
}

function derive(all?: AllRegionsResponse, day?: RegionsResponse, riskCuts?: RiskCuts) {
  if (!all?.model_trained || !day) return null;

  const scoredRegions = day.regions.filter(isScoredRegion);
  if (!scoredRegions.length) return null;

  const probs = scoredRegions.map((r) => r.bust_probability as number);
  const mean = probs.reduce((a, b) => a + b, 0) / probs.length;
  const high = scoredRegions.filter((r) => riskBandForRegion(r, riskCuts) === "high").length;

  const peakRow = scoredRegions.reduce<(typeof scoredRegions)[number] | null>(
    (best, r) => (best == null || (r.bust_probability as number) > (best.bust_probability as number) ? r : best),
    null,
  );

  const withConfidence = all.days
    .map((d) => ({ lead: d.lead_time_days, mean: meanConfidence(d) }))
    .filter((d): d is { lead: number; mean: number } => d.mean != null)
    .sort((a, b) => a.lead - b.lead);

  const first = withConfidence[0];
  const last = withConfidence[withConfidence.length - 1];
  const canDecay = first && last && first.lead !== last.lead && first.mean > 0;
  const meanBand = riskBandForProbability(mean, riskCuts);

  return {
    lead: day.lead_time_days,
    scored: scoredRegions.length,
    mean,
    meanCap: meanBand === "high" ? "bust" : meanBand === "medium" ? "watch" : meanBand === "low" ? "calm" : "blue",
    high,
    peak: peakRow
      ? {
          id: peakRow.region_id,
          name: peakRow.region_name ?? peakRow.region_id,
          value: peakRow.bust_probability as number,
          driver: peakRow.dominant_variable,
        }
      : null,
    decayPct: canDecay ? ((last.mean - first.mean) / first.mean) * 100 : null,
    decayFrom: canDecay ? first.lead : null,
    decayTo: canDecay ? last.lead : null,
  };
}

function meanConfidence(day: RegionsResponse): number | null {
  const vals = day.regions
    .filter(isScoredRegion)
    .map((r) => r.confidence)
    .filter((c): c is number => c != null);
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
}

function label(variable: string): string {
  return variableLabel(variable).toLowerCase();
}
