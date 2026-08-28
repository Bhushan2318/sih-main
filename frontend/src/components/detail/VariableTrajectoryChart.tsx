import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { VariableSeries } from "../../api/types";

/**
 * Forecast vs observed across lead days for one variable. `observed` is only drawn where
 * the forecast has actually verified - unverified leads leave a gap rather than a
 * guessed value.
 */
export function VariableTrajectoryChart({ series }: { series: VariableSeries }) {
  if (!series.available || !series.points.length) {
    return (
      <p className="muted">
        No model for {series.variable} — too few paired forecast/observation rows in the data.
      </p>
    );
  }
  const data = series.points.map((p) => ({
    lead: p.lead_time_days,
    forecast: p.predicted_value,
    observed: p.observed_value,
    error: p.predicted_error,
    confidence: p.confidence,
  }));
  const anyObserved = data.some((d) => d.observed != null);

  return (
    <>
      <ResponsiveContainer width="100%" height={210}>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -4 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="lead" tickFormatter={(d) => `D${d}`} />
          {/* the unit belongs on the axis once, not appended to every tick */}
          <YAxis width={58} tickFormatter={(v: number) => formatTick(v)} />
          <Tooltip
            labelFormatter={(l) => `Lead day ${l}`}
            formatter={(v: number) => (v == null ? "—" : `${formatTick(v)}${series.unit ? ` ${series.unit}` : ""}`)}
          />
          <Legend />
          <Line type="monotone" dataKey="forecast" name="Forecast (ens. mean)" strokeWidth={2} dot={{ r: 2 }} />
          {anyObserved ? (
            <Line type="monotone" dataKey="observed" name="Observed" strokeWidth={2}
                  strokeDasharray="5 4" dot={{ r: 2 }} connectNulls={false} />
          ) : null}
          <Line type="monotone" dataKey="error" name="Predicted |error|" strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
      {series.unit ? <p className="muted small axis-unit">Values in {series.unit}</p> : null}
      {!anyObserved ? (
        <p className="muted small">
          This cycle has not verified yet, so no observed values are drawn.
        </p>
      ) : null}
    </>
  );
}

/** Compact tick labels: pressures need no decimals, small values need a couple. */
function formatTick(v: number): string {
  if (v == null || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
