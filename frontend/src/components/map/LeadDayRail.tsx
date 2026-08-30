import { useMemo } from "react";
import type { AllRegionsResponse, RegionsResponse } from "../../api/types";

/**
 * The whole 10-day horizon, down the side of the map.
 *
 * The lead-day pills could only say which day you were looking at. The same payload
 * already carries every day's scored regions, so the rail shows how the country's risk is
 * composed at each lead - and replaces the pills rather than sitting beside them.
 *
 * The driver is deliberately NOT repeated on every row. It is the same variable for runs
 * of consecutive days, and printing "rainfall_mm" five times down a narrow column is
 * noise; where it changes is the only part that carries information, so that gets one
 * sentence underneath instead.
 */
export function LeadDayRail({ all, value, onChange }: {
  all?: AllRegionsResponse;
  value: number;
  onChange: (d: number) => void;
}) {
  const rows = useMemo(() => (all?.days ?? []).map(summarise).filter(Boolean) as Row[], [all]);
  const drivers = useMemo(() => driverRuns(rows), [rows]);

  if (rows.length < 2) return null;
  const peak = Math.max(...rows.map((r) => r.mean));

  return (
    <aside className="rail" aria-label="Forecast horizon">
      <h3 className="rail__title">Horizon</h3>

      <div className="rail__rows" role="group" aria-label="Lead day">
        {rows.map((r) => (
          <button
            key={r.lead}
            type="button"
            aria-pressed={r.lead === value}
            className={r.lead === value ? "railrow railrow--active" : "railrow"}
            onClick={() => onChange(r.lead)}
            title={`Lead day ${r.lead}: ${r.bust} bust, ${r.watch} watch, ${r.low} low`}
          >
            <span className="railrow__day">D{r.lead}</span>

            {/* one bar per lead day: the country's composition, not a value on a scale */}
            <span className="railrow__bar" aria-hidden="true">
              {r.low ? <i className="railrow__seg railrow__seg--low" style={{ flexGrow: r.low }} /> : null}
              {r.watch ? <i className="railrow__seg railrow__seg--watch" style={{ flexGrow: r.watch }} /> : null}
              {r.bust ? <i className="railrow__seg railrow__seg--bust" style={{ flexGrow: r.bust }} /> : null}
            </span>

            <span className={r.mean >= peak - 1e-9 ? "railrow__mean railrow__mean--peak" : "railrow__mean"}>
              {r.mean.toFixed(2)}
            </span>
          </button>
        ))}
      </div>

      <p className="rail__key">
        <b><i className="railrow__seg--low" />low</b>
        <b><i className="railrow__seg--watch" />watch</b>
        <b><i className="railrow__seg--bust" />bust</b>
      </p>

      <p className="rail__note">
        Regions per band at each lead day, and the mean P(bust) across them.
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

/** One lead day, counted from its own scored regions. Days with nothing scored drop out. */
function summarise(day: RegionsResponse): Row | null {
  const regions = day.regions ?? [];
  if (!regions.length) return null;

  const probs: number[] = [];
  const drivers = new Map<string, number>();
  let low = 0;
  let watch = 0;
  let bust = 0;

  for (const r of regions) {
    if (r.risk_band === "high") bust += 1;
    else if (r.risk_band === "medium") watch += 1;
    else if (r.risk_band === "low") low += 1;
    if (r.bust_probability != null) probs.push(r.bust_probability);
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

/**
 * One sentence about which variable is breaking the forecast, and where that changes.
 *
 * Collapsing the per-day drivers into runs is the whole point: "rainfall to D6, then
 * atmospheric moisture" is a claim about the atmosphere; ten repeated labels is a list.
 */
function driverRuns(rows: Row[]): string | null {
  const named = rows.filter((r) => r.driver);
  if (!named.length) return null;

  const runs: { driver: string; from: number; to: number }[] = [];
  for (const r of named) {
    const last = runs[runs.length - 1];
    if (last && last.driver === r.driver) last.to = r.lead;
    else runs.push({ driver: r.driver as string, from: r.lead, to: r.lead });
  }

  if (runs.length === 1) return `${pretty(runs[0].driver)} is the dominant driver at every lead day.`;
  // More than a couple of switches is churn rather than a story - say so plainly.
  if (runs.length > 3) return `The dominant driver changes ${runs.length - 1} times across the horizon.`;
  return `Dominant driver: ${runs.map((r) => `${pretty(r.driver)} ${span(r)}`).join(", then ")}.`;
}

const span = (r: { from: number; to: number }) => (r.from === r.to ? `at D${r.from}` : `D${r.from}–D${r.to}`);

/** Column names are snake_case on the wire; they read as prose here. */
const pretty = (v: string) => v.replace(/_(pct|mm|c|hpa|ms|deg|kgm2)$/, "").replace(/_/g, " ");
