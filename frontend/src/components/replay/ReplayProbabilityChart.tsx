import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { CHART } from "../../theme";

/**
 * What the model *said*, on the same lead-day axis as what actually happened.
 *
 * The chart above shows the forecast diverging from the observation - which proves only
 * that the forecast was wrong. The claim this project makes is the prediction, so the
 * predicted probability has to be visible beside the outcome; otherwise a reader has to
 * correlate a chart on one side with a number on the other and take the link on trust.
 *
 * Deliberately a second chart rather than a second y-axis on the first. Probability is
 * 0-1 and the forecast is in degrees or millimetres; putting them on one plot with two
 * scales lets the author choose where the lines appear to cross, which is the single
 * most common way a chart like this misleads.
 *
 * The band cuts are drawn with visible text labels, not colour alone: the "watch" amber
 * sits at 2.5:1 against the surface, which is below the 3:1 floor, so the label carries
 * the meaning and the colour only reinforces it.
 */
export function ReplayProbabilityChart({
  points,
  currentLead,
  cuts,
}: {
  points: { lead: number; p: number | null }[];
  currentLead: number;
  cuts?: { medium?: number; high?: number };
}) {
  const any = points.some((d) => d.p != null);
  if (!any) return null;
  const pct = (v: number) => `${Math.round(v * 100)}%`;

  return (
    <div className="replay-focus">
      <div className="replay-focus__head">
        <strong>Predicted P(bust)</strong>
        <span className="muted small">what the model said, before the outcome</span>
      </div>
      <ResponsiveContainer width="100%" height={132}>
        <LineChart data={points} margin={{ top: 8, right: 40, bottom: 4, left: -6 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
          <XAxis dataKey="lead" tickFormatter={(d) => `D${d}`} stroke={CHART.axis} />
          <YAxis width={58} domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]}
                 tickFormatter={pct} stroke={CHART.axis} />
          <Tooltip
            labelFormatter={(l) => `Lead day ${l}`}
            formatter={(v: number) => (v == null ? "—" : pct(v))}
          />
          {typeof cuts?.medium === "number" ? (
            <ReferenceLine y={cuts.medium} stroke={CHART.medium} strokeDasharray="4 4"
              label={{ value: "watch", position: "right", fontSize: 9, fill: CHART.axis, dy: -4, dx: -34 }} />
          ) : null}
          {typeof cuts?.high === "number" ? (
            <ReferenceLine y={cuts.high} stroke={CHART.high} strokeDasharray="4 4"
              label={{ value: "bust", position: "right", fontSize: 9, fill: CHART.axis, dy: -4, dx: -30 }} />
          ) : null}
          <ReferenceLine x={currentLead} stroke={CHART.marker} strokeWidth={2} />
          <Line type="monotone" dataKey="p" name="P(bust)" stroke={CHART.forecast}
            strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>
      <p className="muted small">
        Scored by the deployed model from this cycle&apos;s forecast alone — no observation
        is used to produce it.
      </p>
    </div>
  );
}
