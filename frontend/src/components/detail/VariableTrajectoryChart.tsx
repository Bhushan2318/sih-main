import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { VariableSeries } from "../../api/types";
import { variableLabel, variableUnit } from "../../lib/displayNames";
import { dayLabel } from "../../lib/format";
import { CHART } from "../../theme";

export function VariableTrajectoryChart({ series }: { series: VariableSeries }) {
  if (!series.available || !series.points.length) {
    return (
      <p className="muted">
        No model for {variableLabel(series.variable)} — too few matched forecast–observation
        pairs in the data.
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

  const provisional = series.points.filter((p) => p.observed_status === "provisional");

  const shortUnit = variableUnit(series.variable);

  return (
    <>
      <ResponsiveContainer width="100%" height={210}>
        <LineChart data={data} margin={{ top: 8, right: 22, bottom: 4, left: -4 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="lead" tickFormatter={dayLabel}
            label={{ value: "Lead day", position: "insideBottom", offset: -2, fontSize: 10 }}
            height={34}
          />
          {/* The unit belongs on the axis, not only in a caption under the chart: the
            * caption sat bottom-right, far from the numbers it explains, so the y scale
            * read as bare figures. Short form here (kg/m2), full string still below,
            * because the API's unit carries detail worth keeping (TCWV, from-direction). */}
          <YAxis
            width={58} domain={["auto", "auto"]} tickFormatter={(v: number) => formatTick(v)}
            label={shortUnit ? {
              value: shortUnit, angle: -90, position: "insideLeft",
              offset: 14, fontSize: 10, style: { textAnchor: "middle" },
            } : undefined}
          />
          <Tooltip
            labelFormatter={(l) => `Lead day ${l}`}
            formatter={(v: number) => (v == null ? "—" : `${formatTick(v)}${series.unit ? ` ${series.unit}` : ""}`)}
          />
          <Legend />
          <Line type="monotone" dataKey="forecast" name="Forecast (ensemble average)"
                stroke={CHART.forecast} strokeWidth={2} dot={{ r: 2 }} />
          {anyObserved ? (
            <Line type="monotone" dataKey="observed" name="Observed"
                  stroke={CHART.observed} strokeWidth={2} strokeDasharray="6 4" dot={{ r: 2 }} connectNulls={false} />
          ) : null}
          <Line type="monotone" dataKey="error" name="Predicted error size"
                stroke={CHART.error} strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
      {series.unit ? <p className="muted small axis-unit">Values in {series.unit}</p> : null}
      {provisional.length ? (
        <p className="muted small provisional-note">
          <span className="badge--provisional">provisional</span>{" "}
          {provisional.length === 1 ? "Day" : "Days"}{" "}
          {provisional.map((p) => p.lead_time_days).join(", ")} checked against a fast
          preliminary record rather than the final ERA5 one — these values may be revised.
        </p>
      ) : null}
      {!anyObserved ? (
        <p className="muted small">
          This cycle has not verified yet, so no observed values are drawn.
        </p>
      ) : null}
    </>
  );
}

function formatTick(v: number): string {
  if (v == null || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
