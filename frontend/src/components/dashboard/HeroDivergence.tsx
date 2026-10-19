import { rich } from "../../i18n/rich";
import { translate } from "../../i18n";
import { useState } from "react";
import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer,
  Scatter, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import type { EnsembleDivergenceResponse } from "../../api/types";
import { dayLabel } from "../../lib/format";
import { isScoredProbability, riskBandForProbability, type RiskCuts } from "../../lib/riskBands";
import { CHART } from "../../theme";

type Tab = "national" | "skill";

export function HeroDivergence({ data, riskCuts }: {
  data?: EnsembleDivergenceResponse;
  riskCuts?: RiskCuts;
}) {
  const [tab, setTab] = useState<Tab>("national");

  if (!data?.model_trained) return null;

  const mean = data.mean_bust_probability;
  const hasScore = data.n_scored_regions > 0
    && isScoredProbability(mean)
    && (data.national?.length ?? 0) > 0;
  if (!hasScore) {
    return (
      <section className="hero hero--empty" aria-live="polite">
        <div className="hero__grid">
          <div className="hero__lead">
            <div className="hero__kick"><i aria-hidden="true" /><span>{translate("hero.noScoreKick")}</span></div>
            <h1 className="hero__head">{rich("hero.noScoreHead", undefined, [<em />])}</h1>
            <p className="hero__sub">
              {data.message ?? translate("hero.noScoreSub")}
            </p>
            <p className="hero__why">
              {translate("hero.noScoreWhy")}
            </p>
          </div>
        </div>
      </section>
    );
  }

  const prior = data.prior_mean_bust_probability;
  const delta = mean != null && prior != null ? mean - prior : null;
  const apart = data.n_high_regions > 0;

  return (
    <section className="hero">
      <div className="hero__grid">
        <div className="hero__lead">
          <div className="hero__kick">
            <i aria-hidden="true" />
            <span>{data.eyebrow}</span>
          </div>

          <h1 className="hero__head">
            {rich(apart ? "hero.apart" : "hero.holds", undefined, [<em />])}
          </h1>

          <p className="hero__sub">{data.national_note ?? data.headline_note}</p>

          <p className="hero__why">
            {translate("hero.why")}
          </p>

          {mean != null ? (
            <div className="hero__gauge">
              <Ring value={mean} cuts={riskCuts} />
              <div className="hero__gaugemeta">
                <span className="hero__gaugelabel">{translate("hero.gauge")}</span>
                <strong className="hero__gaugebig">
                  {translate("hero.reach", { high: data.n_high_regions, n: data.n_scored_regions })}
                </strong>
                {delta != null ? (
                  <span className={deltaClass(delta)}>
                    {translate("hero.delta", {
                      arrow: delta >= 0 ? "▲" : "▼",
                      sign: delta >= 0 ? "+" : "",
                      pts: (delta * 100).toFixed(1),
                      date: data.prior_init_date ?? "",
                    })}
                  </span>
                ) : (
                  <span className="hero__delta hero__delta--none">{data.prior_note}</span>
                )}
              </div>
            </div>
          ) : null}
        </div>

        <figure className="hero__panel">
          <figcaption className="hero__panelhead">
            <div className="hero__tabs" role="tablist" aria-label={translate("hero.tabsAria")}>
              <button
                type="button" role="tab" aria-selected={tab === "national"}
                className={tab === "national" ? "hero__tab is-active" : "hero__tab"}
                onClick={() => setTab("national")}
              >
                {translate("hero.tabNational")}
              </button>
              <button
                type="button" role="tab" aria-selected={tab === "skill"}
                className={tab === "skill" ? "hero__tab is-active" : "hero__tab"}
                onClick={() => setTab("skill")}
              >
                {translate("hero.tabSkill")}
              </button>
            </div>
          </figcaption>

          <div className="hero__chart">
            {tab === "national" ? <NationalChart data={data} /> : <SkillChart data={data} />}
          </div>

          {tab === "national" ? (
            <div className="hero__key">
              <b><i style={{ background: CHART.forecast }} />{translate("hero.keyMean")}</b>
              <b><i style={{ background: CHART.member }} />{translate("hero.keyRange")}</b>
            </div>
          ) : (
            <div className="hero__key">
              <b><i style={{ background: CHART.forecast }} />{translate("hero.keyActual")}</b>
              <b><i style={{ background: CHART.axis }} />{translate("hero.keyIdeal")}</b>
            </div>
          )}

          <p className="hero__source">
            {tab === "national"
              ? translate("hero.source", { source: data.source ?? "", date: data.init_date ?? "" })
              : skillCaption(data)}
          </p>
        </figure>
      </div>
    </section>
  );
}

function NationalChart({ data }: { data: EnsembleDivergenceResponse }) {
  const rows = data.national.map((n) => ({
    lead: n.lead_time_days,
    mean: Number((n.mean_bust_probability * 100).toFixed(1)),

    band: [n.min_bust_probability * 100, n.max_bust_probability * 100] as [number, number],
    high: n.n_high_regions,
    n: n.n_regions,
  }));
  if (!rows.length) return <Empty>{translate("hero.noRegions")}</Empty>;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={rows} margin={{ top: 10, right: 22, bottom: 18, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} vertical={false} />
        <XAxis
          dataKey="lead" tickFormatter={dayLabel} stroke={CHART.axis} tickLine={false}
          label={{ value: translate("lead.label"), position: "insideBottom", offset: -12, fontSize: 11, fill: CHART.axis }}
        />
        <YAxis
          domain={[0, 100]} width={46} stroke={CHART.axis} tickLine={false}
          tickFormatter={(v: number) => `${v}%`}
          label={{ value: translate("hero.axisRisk"), angle: -90, position: "insideLeft", offset: -2, fontSize: 11, fill: CHART.axis }}
        />
        <Tooltip
          labelFormatter={(l) => translate("lead.day", { d: l })}
          formatter={(v: number | number[], _n: string, item) => {
            if (Array.isArray(v)) {
              return [`${v[0].toFixed(0)}% – ${v[1].toFixed(0)}%`, translate("hero.keyRange")];
            }
            return [
              translate("hero.tipTop", { pct: v.toFixed(1), high: item.payload.high, n: item.payload.n }),
              translate("hero.meanRisk"),
            ];
          }}
        />
        <Area
          dataKey="band" stroke="none" fill={CHART.member} fillOpacity={0.4}
          isAnimationActive={false} name={translate("hero.keyRange")}
        />
        <Line
          type="monotone" dataKey="mean" name={translate("hero.meanRisk")}
          stroke={CHART.forecast} strokeWidth={2.6} dot={{ r: 2.5 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function SkillChart({ data }: { data: EnsembleDivergenceResponse }) {
  const skill = data.skill;
  if (!skill) return <Empty>{translate("hero.noSkill")}</Empty>;

  if (!skill.calibration.length) {
    return (
      <div className="hero__skillfallback">
        <SkillNumbers skill={skill} />
        {skill.note ? <p className="muted small">{skill.note}</p> : null}
      </div>
    );
  }

  const rows = skill.calibration.map((b) => ({
    predicted: Number((b.predicted_mean * 100).toFixed(1)),
    observed: Number((b.observed_rate * 100).toFixed(1)),
    n: b.n,
  }));

  return (
    <>
      <SkillNumbers skill={skill} />
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 6, right: 14, bottom: 6, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
          <XAxis
            type="number" dataKey="predicted" domain={[0, 100]} stroke={CHART.axis}
            tickLine={false} tickFormatter={(v: number) => `${v}%`}
          />
          <YAxis
            type="number" dataKey="observed" domain={[0, 100]} width={46} stroke={CHART.axis}
            tickLine={false} tickFormatter={(v: number) => `${v}%`}
          />
          <ZAxis type="number" dataKey="n" range={[60, 400]} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<CalibrationTip />} />
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
            stroke={CHART.axis} strokeDasharray="5 5" strokeWidth={1.5}
          />
          <Scatter dataKey="observed" fill={CHART.forecast} fillOpacity={0.75} />
        </ComposedChart>
      </ResponsiveContainer>
    </>
  );
}

function SkillNumbers({ skill }: { skill: NonNullable<EnsembleDivergenceResponse["skill"]> }) {
  return (
    <dl className="hero__skill">
      <div><dt>{translate("hero.auc")}</dt><dd>{num(skill.roc_auc, 3)}</dd></div>
      <div><dt>{translate("hero.precision")}</dt><dd>{num(skill.precision, 2)}</dd></div>
      <div><dt>{translate("hero.recall")}</dt><dd>{num(skill.recall, 2)}</dd></div>
      <div><dt>{translate("hero.brier")}</dt><dd>{num(skill.brier, 3)}</dd></div>
    </dl>
  );
}

function CalibrationTip({ active, payload }: {
  active?: boolean;
  payload?: { payload: { predicted: number; observed: number; n: number } }[];
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="hero__tip">
      <strong>{translate("hero.tipSaid", { p: p.predicted.toFixed(0) })}</strong>
      <span>{translate("hero.tipBusted", { p: p.observed.toFixed(0) })}</span>
      <span className="hero__tipmeta">{translate("hero.tipAcross", { n: p.n.toLocaleString() })}</span>
    </div>
  );
}

function Ring({ value, cuts }: { value: number; cuts?: RiskCuts }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(1, value));
  const band = riskBandForProbability(pct, cuts);
  const colour = band === "high"
    ? CHART.high
    : band === "medium"
      ? CHART.medium
      : band === "low"
        ? CHART.low
        : CHART.forecast;
  return (
    <div className="hero__ring">
      <svg viewBox="0 0 128 128" aria-hidden="true">
        <circle cx="64" cy="64" r={r} fill="none" stroke="#dfe4ec" strokeWidth="11" />
        <circle
          cx="64" cy="64" r={r} fill="none" stroke={colour} strokeWidth="11"
          strokeLinecap="round" strokeDasharray={`${c * pct} ${c}`}
          transform="rotate(-90 64 64)"
        />
      </svg>
      <span className="hero__ringvalue">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="muted small hero__empty">{children}</p>;
}

function skillCaption(data: EnsembleDivergenceResponse): string {
  const s = data.skill;
  if (!s?.n) return translate("hero.noMetrics");
  return translate("hero.caption", { n: s.n.toLocaleString(), split: s.split ?? "" });
}

function deltaClass(delta: number): string {
  if (Math.abs(delta) < 0.005) return "hero__delta hero__delta--none";
  return delta > 0 ? "hero__delta hero__delta--up" : "hero__delta hero__delta--down";
}

function num(v: number | null | undefined, digits: number): string {
  return v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);
}

