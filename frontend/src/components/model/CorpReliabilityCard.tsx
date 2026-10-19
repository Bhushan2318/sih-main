import { translate } from "../../i18n";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { ModelStatusResponse } from "../../api/types";
import { CHART } from "../../theme";

/**
 * F4 - CORP reliability diagram (Dimitriadis, Gneiting & Jordan 2021, PNAS).
 *
 * The hero already shows a reliability scatter, but that one bins probabilities into
 * fixed-width buckets, and its shape depends on a bin count nobody can justify. CORP
 * has no bins to choose: it fits the optimal monotone recalibration by pool-adjacent-
 * violators, and points are pooled only where the data itself forces it. That makes it
 * the version a reviewer can argue with, which is why it belongs on the model page next
 * to the ladder rather than replacing the friendlier hero plot.
 *
 * Each block's width is real information - a long flat run means the model cannot
 * separate those forecasts - so blocks are drawn as a step line, not a smooth curve.
 */
export function CorpReliabilityCard({ data }: { data?: ModelStatusResponse }) {
  if (!data?.model_trained) return null;
  const served = data.baselines?.models?.find((m) => m.is_model);
  if (!served) return null;

  const blocks = (served.corp_reliability ?? []).filter(
    (b) => Number.isFinite(b.predicted_mean) && Number.isFinite(b.observed_rate),
  );

  if (!blocks.length) {
    return (
      <section className="card" aria-label={translate("corp.aria")}>
        <header className="card__head"><h3>{translate("corp.title")}</h3></header>
        <p className="muted small">{translate("corp.notYet")}</p>
      </section>
    );
  }

  const rows = blocks.map((b) => ({
    predicted: Number((b.predicted_mean * 100).toFixed(2)),
    observed: Number((b.observed_rate * 100).toFixed(2)),
    n: b.n,
  }));

  const total = blocks.reduce((s, b) => s + b.n, 0);

  return (
    <section className="card corpcard" aria-label={translate("corp.aria")}>
      <header className="card__head">
        <h3>{translate("corp.title")}</h3>
      </header>

      <p className="muted small">
        {translate("corp.intro")}
      </p>

      <div className="corpcard__chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 14, bottom: 4, left: -12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
            <XAxis
              dataKey="predicted" type="number" domain={[0, 100]} stroke={CHART.axis}
              tickLine={false} tickFormatter={(v: number) => `${v}%`}
            />
            <YAxis
              type="number" domain={[0, 100]} width={44} stroke={CHART.axis}
              tickLine={false} tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip content={<CorpTip />} cursor={{ strokeDasharray: "3 3" }} />
            <ReferenceLine
              segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
              stroke={CHART.axis} strokeDasharray="5 5" strokeWidth={1.5}
            />
            <Line
              type="stepAfter" dataKey="observed" stroke={CHART.forecast}
              strokeWidth={2.4} dot={{ r: 2.2 }} isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="muted small">
        {translate("corp.foot", { blocks: blocks.length.toLocaleString(), n: total.toLocaleString() })}
      </p>
    </section>
  );
}

function CorpTip({ active, payload }: {
  active?: boolean;
  payload?: { payload: { predicted: number; observed: number; n: number } }[];
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="charttip">
      <strong>{translate("corp.said", { p: p.predicted.toFixed(0) })}</strong>
      <span>{translate("corp.busted", { p: p.observed.toFixed(0) })}</span>
      <span className="charttip__meta">{translate("corp.block", { n: p.n.toLocaleString() })}</span>
    </div>
  );
}
