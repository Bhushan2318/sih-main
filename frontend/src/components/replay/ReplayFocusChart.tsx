import {
  Area, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { ReplayFocusSeries } from "../../api/types";
import { CHART } from "../../theme";

/**
 * The one variable that diverged most over the replayed cycle: forecast vs what was
 * actually observed, across lead days. A vertical marker tracks the current replay step.
 * Observed is drawn only where the forecast verified - gaps stay gaps.
 */
export function ReplayFocusChart({
  focus,
  currentLead,
}: {
  focus: ReplayFocusSeries;
  currentLead: number;
}) {
  // The band is the bust criterion drawn in the units already on screen: a forecast
  // inside it is within tolerance, one outside it is a bust. Without it, two lines far
  // apart tell a reader nothing about whether "far" is far enough to matter - which is
  // the only question this chart exists to answer.
  const thr = focus.bust_threshold;
  const data = focus.points.map((p) => ({
    lead: p.lead_time_days,
    forecast: p.predicted_value,
    observed: p.observed_value,
    spread: p.ensemble_spread,
    band:
      thr != null && p.observed_value != null
        ? [p.observed_value - thr, p.observed_value + thr]
        : null,
  }));
  const anyObserved = data.some((d) => d.observed != null);

  return (
    <div className="replay-focus">
      <div className="replay-focus__head">
        <strong>{focus.region_name ?? focus.region_id}</strong>
        <span className="muted small">
          {focus.variable.replace(/_/g, " ")}
          {focus.unit ? ` · ${focus.unit}` : ""}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 20, right: 14, bottom: 4, left: -6 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="lead" tickFormatter={(d) => `D${d}`} />
          {/* Scale to the data, not to zero. Recharts defaults a numeric axis to a
              zero-based domain, which is fatal here: pressure lives near 1020 hPa, so a
              0-1200 axis compresses the entire real variation into 3% of the plot and
              draws the +-3.3 hPa bust threshold thinner than a pixel. A 19.7 hPa error -
              a badly blown forecast - rendered as ~4px of separation, so the chart showed
              two identical lines beside a 98% bust call and read as the model being
              wrong. The band is a data series, so an auto domain always contains it. */}
          <YAxis width={58} domain={["auto", "auto"]} tickFormatter={(v: number) => fmt(v)} />
          <Tooltip
            labelFormatter={(l) => `Lead day ${l}`}
            formatter={(v: unknown) => {
              // A range series hands this an array, and anything that throws in here
              // takes the entire chart down with it - which is exactly what happened:
              // hovering blanked both charts with "v.toFixed is not a function".
              const one = (x: unknown) =>
                typeof x === "number" && Number.isFinite(x) ? fmt(x) : "—";
              const body = Array.isArray(v) ? v.map(one).join(" – ") : one(v);
              return body === "—" ? body : `${body}${focus.unit ? ` ${focus.unit}` : ""}`;
            }}
          />
          <Legend />
          <ReferenceLine x={currentLead} stroke={CHART.marker} strokeWidth={2}
            label={{ value: `Day ${currentLead}`, position: "top", fontSize: 10, fill: CHART.marker }} />
          {thr != null ? (
            <Area type="monotone" dataKey="band" name={`Close enough — not a bust (±${fmt(thr)})`}
              stroke="none" fill={CHART.observed} fillOpacity={0.07} connectNulls={false}
              activeDot={false} isAnimationActive={false} legendType="rect"
              tooltipType="none" />
          ) : null}
          <Line type="monotone" dataKey="forecast" name="Forecast (ensemble average)"
            stroke={CHART.forecast} strokeWidth={2} dot={{ r: 2 }} />
          {anyObserved ? (
            <Line type="monotone" dataKey="observed" name="Observed"
              stroke={CHART.observed} strokeWidth={2} strokeDasharray="6 4" dot={{ r: 2 }} connectNulls={false} />
          ) : null}
        </ComposedChart>
      </ResponsiveContainer>
      {!anyObserved ? (
        <p className="muted small">This cycle has not verified, so no observed values are drawn.</p>
      ) : null}
    </div>
  );
}

function fmt(v: number): string {
  if (v == null || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
