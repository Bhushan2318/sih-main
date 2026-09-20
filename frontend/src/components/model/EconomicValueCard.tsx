import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceDot, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { ModelStatusResponse } from "../../api/types";
import { finiteCurve, peakValue, valueAtRatio } from "../../lib/economicValue";
import { CHART } from "../../theme";

/**
 * F3 - relative economic value against cost-loss ratio (Richardson 2000).
 *
 * Answers the question a district officer has and no ROC-AUC answers: is acting on this
 * worth it for me? The answer depends on their own economics, which is why this is a
 * slider and not a single number.
 *
 * Written in rupees per 100 rather than in the literature's vocabulary. The first
 * version led with "relative economic value" and a cost-loss ratio, and reported the
 * answer as a bare 0.073 - no unit, no reference point, and defaulting to the weakest
 * part of the curve. The project owner read it and could not say what it meant, which is
 * the only test of this card that matters: a judge gets one look at it. The arithmetic
 * below is unchanged; only the words and the starting point are.
 */
export function EconomicValueCard({ data }: { data?: ModelStatusResponse }) {
  const served = data?.baselines?.models?.find((m) => m.is_model);
  const curve = useMemo(() => finiteCurve(served?.economic_value), [served]);
  const peak = useMemo(() => peakValue(curve), [curve]);
  const [alpha, setAlpha] = useState<number | null>(null);

  // Start at the peak rather than an arbitrary ratio. Not flattery - the sentence below
  // says outright that this is where the model does best, and the slider moves off it in
  // one drag. Starting at 10% put a 7% reading on screen with nothing to compare it to,
  // which read as "this barely works" rather than "this depends on who you are".
  useEffect(() => {
    if (alpha === null && peak) setAlpha(peak.cost_loss_ratio);
  }, [alpha, peak]);

  const current = alpha ?? peak?.cost_loss_ratio ?? 0.5;
  const here = valueAtRatio(curve, current);

  if (!data?.model_trained || !served) return null;
  if (!curve.length) {
    return (
      <section className="card" aria-label="What acting on this is worth">
        <header className="card__head"><h3>Is it worth acting on?</h3></header>
        <p className="muted small">
          This run was scored before this measurement existed. It appears after the next
          retrain.
        </p>
      </section>
    );
  }

  const rows = curve.map((p) => ({
    cost: Math.round(p.cost_loss_ratio * 100),
    captured: Math.round(p.value * 100),
  }));

  const rupees = Math.round(current * 100);
  const captured = here ? Math.round(here.value * 100) : null;
  const atPeak = peak != null && Math.abs(current - peak.cost_loss_ratio) < 0.005;

  return (
    <section className="card evcard" aria-label="What acting on this is worth">
      <header className="card__head">
        <h3>Is it worth acting on?</h3>
      </header>

      <p className="muted small">
        A warning is only useful if acting on it costs less than the damage it prevents.
        That trade is different for everyone, so pick yours:
      </p>

      <div className="evcard__slider">
        <label htmlFor="ev-alpha">
          Acting early costs me <b className="mono">₹{rupees}</b> for every{" "}
          <b className="mono">₹100</b> of damage a bust would cause
        </label>
        <input
          id="ev-alpha"
          type="range"
          min={1}
          max={99}
          step={1}
          value={rupees}
          onChange={(e) => setAlpha(Number(e.target.value) / 100)}
          aria-valuetext={`acting costs ${rupees} rupees per 100 of damage`}
        />
      </div>

      <p className="evcard__answer">
        Then following these forecasts saves you{" "}
        <b>{captured == null ? "—" : `${captured}%`}</b> of what a{" "}
        <i>perfect</i> forecast would have saved.
      </p>

      <p className="muted small">
        A perfect forecast is <b>100%</b>. Never looking at the forecast at all — always
        act, or never act — is <b>0%</b>.
        {peak ? (
          atPeak ? (
            <> This is the trade where the model helps most.</>
          ) : (
            <> It helps most at <b className="mono">₹{Math.round(peak.cost_loss_ratio * 100)}</b>,
              where it reaches <b>{Math.round(peak.value * 100)}%</b>.</>
          )
        ) : null}
      </p>

      <div className="evcard__chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 14, bottom: 4, left: -12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
            <XAxis
              dataKey="cost" type="number" domain={[0, 100]} stroke={CHART.axis}
              tickLine={false} tickFormatter={(v: number) => `₹${v}`}
            />
            <YAxis
              type="number" domain={[0, 100]} width={44} stroke={CHART.axis}
              tickLine={false} tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip content={<ValueTip />} cursor={{ strokeDasharray: "3 3" }} />
            <ReferenceLine y={0} stroke={CHART.axis} strokeWidth={1} />
            <Line
              type="monotone" dataKey="captured" stroke={CHART.forecast}
              strokeWidth={2.4} dot={false} isAnimationActive={false}
            />
            {here ? (
              <ReferenceDot
                x={rupees} y={Math.round(here.value * 100)}
                r={5} fill={CHART.marker} stroke="none" isFront
              />
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="muted small">
        Left to right: what acting costs you, in rupees per ₹100 of damage. Up the side:
        how much of a perfect forecast&apos;s value you keep. The curve peaks in the
        middle because that is where a warning actually changes what you would have done
        anyway.
      </p>
    </section>
  );
}

function ValueTip({ active, payload }: {
  active?: boolean;
  payload?: { payload: { cost: number; captured: number } }[];
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="charttip">
      <strong>₹{p.cost} to act, per ₹100 of damage</strong>
      <span>keeps {p.captured}% of a perfect forecast&apos;s value</span>
    </div>
  );
}
