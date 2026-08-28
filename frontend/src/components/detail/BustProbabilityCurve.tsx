import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { BustProbabilityPoint } from "../../api/types";

export function BustProbabilityCurve({ points, cuts }: {
  points: BustProbabilityPoint[];
  cuts?: { medium: number; high: number };
}) {
  if (!points.length) return <p className="muted">No bust-probability curve for this region.</p>;
  const data = points.map((p) => ({
    lead: p.lead_time_days,
    probability: Number((p.bust_probability * 100).toFixed(2)),
    band: p.risk_band,
    driver: p.dominant_variable,
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="lead" tickFormatter={(d) => `D${d}`} />
        <YAxis domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} width={54} />
        <Tooltip
          formatter={(v: number, _n, item) =>
            [`${v}%  (${item.payload.band})`, item.payload.driver ? `driver: ${item.payload.driver}` : "P(bust)"]
          }
          labelFormatter={(l) => `Lead day ${l}`}
        />
        {cuts ? (
          <>
            <ReferenceLine y={cuts.medium * 100} strokeDasharray="4 4" label={{ value: "medium", position: "right", fontSize: 10 }} />
            <ReferenceLine y={cuts.high * 100} strokeDasharray="4 4" label={{ value: "high", position: "right", fontSize: 10 }} />
          </>
        ) : null}
        <Line type="monotone" dataKey="probability" strokeWidth={2} dot={{ r: 3 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}
