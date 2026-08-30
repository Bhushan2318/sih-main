import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
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
  const data = focus.points.map((p) => ({
    lead: p.lead_time_days,
    forecast: p.predicted_value,
    observed: p.observed_value,
    spread: p.ensemble_spread,
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
        <LineChart data={data} margin={{ top: 8, right: 14, bottom: 4, left: -6 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="lead" tickFormatter={(d) => `D${d}`} />
          <YAxis width={58} tickFormatter={(v: number) => fmt(v)} />
          <Tooltip
            labelFormatter={(l) => `Lead day ${l}`}
            formatter={(v: number) => (v == null ? "—" : `${fmt(v)}${focus.unit ? ` ${focus.unit}` : ""}`)}
          />
          <Legend />
          <ReferenceLine x={currentLead} stroke={CHART.marker} strokeWidth={2}
            label={{ value: `Day ${currentLead}`, position: "top", fontSize: 10, fill: CHART.marker }} />
          <Line type="monotone" dataKey="forecast" name="Forecast (ens. mean)"
            stroke={CHART.forecast} strokeWidth={2} dot={{ r: 2 }} />
          {anyObserved ? (
            <Line type="monotone" dataKey="observed" name="Observed"
              stroke={CHART.observed} strokeWidth={2} strokeDasharray="6 4" dot={{ r: 2 }} connectNulls={false} />
          ) : null}
        </LineChart>
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
