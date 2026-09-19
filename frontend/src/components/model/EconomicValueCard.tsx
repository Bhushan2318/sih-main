import { useMemo, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceDot, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { ModelStatusResponse } from "../../api/types";
import { finiteCurve, peakValue, valueAtRatio } from "../../lib/economicValue";
import { formatMetric } from "../../lib/format";
import { CHART } from "../../theme";

/** Where the slider starts: protecting costs a tenth of what the damage costs.
 * A common enough ratio in disaster response to be a fair default, and far enough
 * from the peak that the curve's shape is visible rather than flat. */
const DEFAULT_ALPHA = 0.1;

/**
 * F3 - relative economic value against cost-loss ratio (Richardson 2000).
 *
 * Answers the question a district officer actually has, which no ROC-AUC answers: "is
 * acting on this worth it for me?" Value 1.0 means the forecast captures everything a
 * perfect forecast would be worth; 0 means it is worth no more than always doing the
 * same thing regardless of the forecast. The curve peaks near the base rate, so a model
 * can be genuinely valuable to one user and worthless to another - which is why this is
 * a slider rather than a single number.
 */
export function EconomicValueCard({ data }: { data?: ModelStatusResponse }) {
  const [alpha, setAlpha] = useState(DEFAULT_ALPHA);

  const served = data?.baselines?.models?.find((m) => m.is_model);
  const curve = useMemo(() => finiteCurve(served?.economic_value), [served]);
  const here = valueAtRatio(curve, alpha);
  const peak = useMemo(() => peakValue(curve), [curve]);

  if (!data?.model_trained) return null;
  // A run trained before Workstream D carries no economic_value. Say so rather than
  // render an empty chart that looks like a value of zero.
  if (!served) return null;
  if (!curve.length) {
    return (
      <section className="card" aria-label="Relative economic value">
        <header className="card__head"><h3>What it is worth to act on</h3></header>
        <p className="muted small">
          This run was scored before the economic-value curve existed. It appears after
          the next retrain.
        </p>
      </section>
    );
  }

  const rows = curve.map((p) => ({
    alpha: Number((p.cost_loss_ratio * 100).toFixed(0)),
    value: Number(p.value.toFixed(4)),
  }));

  return (
    <section className="card evcard" aria-label="Relative economic value">
      <header className="card__head">
        <h3>What it is worth to act on</h3>
      </header>

      <p className="muted small">
        Relative economic value: <b>1.0</b> captures everything a perfect forecast would
        be worth, <b>0</b> is no better than doing the same thing every day whatever the
        forecast. Where you sit depends on what precaution costs you against what the
        bust costs you.
      </p>

      <div className="evcard__slider">
        <label htmlFor="ev-alpha">
          Precaution costs <b className="mono">{(alpha * 100).toFixed(0)}%</b> of the loss
          it prevents
        </label>
        <input
          id="ev-alpha"
          type="range"
          min={1}
          max={99}
          step={1}
          value={Math.round(alpha * 100)}
          onChange={(e) => setAlpha(Number(e.target.value) / 100)}
          aria-valuetext={`cost-loss ratio ${alpha.toFixed(2)}`}
        />
      </div>

      <dl className="evcard__read">
        <div>
          <dt>Value at this ratio</dt>
          <dd className="mono">{formatMetric(here?.value, 3)}</dd>
        </div>
        <div>
          <dt>Best served ratio</dt>
          <dd className="mono">
            {peak ? `${(peak.cost_loss_ratio * 100).toFixed(0)}%` : "—"}
          </dd>
        </div>
        <div>
          <dt>Value there</dt>
          <dd className="mono">{formatMetric(peak?.value, 3)}</dd>
        </div>
      </dl>

      <div className="evcard__chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 14, bottom: 4, left: -12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
            <XAxis
              dataKey="alpha" type="number" domain={[0, 100]} stroke={CHART.axis}
              tickLine={false} tickFormatter={(v: number) => `${v}%`}
            />
            <YAxis
              type="number" domain={[0, 1]} width={44} stroke={CHART.axis}
              tickLine={false} tickFormatter={(v: number) => v.toFixed(1)}
            />
            <Tooltip content={<ValueTip />} cursor={{ strokeDasharray: "3 3" }} />
            <ReferenceLine y={0} stroke={CHART.axis} strokeWidth={1} />
            <Line
              type="monotone" dataKey="value" stroke={CHART.forecast}
              strokeWidth={2.4} dot={false} isAnimationActive={false}
            />
            {here ? (
              <ReferenceDot
                x={Number((here.cost_loss_ratio * 100).toFixed(0))}
                y={Number(here.value.toFixed(4))}
                r={5} fill={CHART.marker} stroke="none" isFront
              />
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="muted small">
        Swept over every probability threshold in the held-out data, reporting the best
        each ratio could achieve — a decision-maker picks the cutoff that suits them,
        not one fixed cutoff for everyone.
      </p>
    </section>
  );
}

function ValueTip({ active, payload }: {
  active?: boolean;
  payload?: { payload: { alpha: number; value: number } }[];
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="charttip">
      <b>{p.alpha}% cost-loss ratio</b>
      <span>value {p.value.toFixed(3)}</span>
    </div>
  );
}
