import { rich } from "../../i18n/rich";
import { locale, translate } from "../../i18n";
import type { ModelStatusResponse } from "../../api/types";
import { stamp } from "../../format";
import { useModelStatus } from "../../hooks/useDashboardData";
import { variableLabel, variableUnit } from "../../lib/displayNames";
import { useLiveStore } from "../../store/liveStore";
import { ErrorState, LoadingState } from "../common/States";
import { retryingHint } from "../../lib/retryHint";
import { UploadPanel } from "../upload/UploadPanel";
import { BaselineLadderTable } from "./BaselineLadderTable";
import { CorpReliabilityCard } from "./CorpReliabilityCard";
import { EconomicValueCard } from "./EconomicValueCard";
import { MissesCard } from "./MissesCard";
import { PipelineLog } from "./PipelineLog";

const UPLOAD_ENABLED = import.meta.env.VITE_ENABLE_UPLOAD !== "false";

export function ModelPage() {
  const { data, isLoading, error, failureCount } = useModelStatus();
  const connection = useLiveStore((s) => s.connectionStatus);
  const training = useLiveStore((s) => s.trainingInProgress);

  if (isLoading) return <main className="page page--wide"><LoadingState label={translate("mp.loading")} hint={retryingHint(failureCount)} /></main>;
  if (error) return <main className="page page--wide"><ErrorState error={error} /></main>;
  if (!data) return null;

  const clf = data.validation_metrics?.classifier ?? {};
  const regressors = data.validation_metrics?.regressors ?? {};
  const cuts = data.thresholds?.risk_band_cuts;
  const vol = data.data_volume ?? {};
  const td = data.training_data ?? {};
  // Pooled runs record years; single-year runs record dates. The API fills both shapes.
  const firstYear = td.first_train_year
    ?? (td.first_train_date ? Number(String(td.first_train_date).slice(0, 4)) : null);
  const lastYear = td.last_train_year ?? firstYear;
  const trainYears = firstYear != null && lastYear != null ? lastYear - firstYear + 1 : null;
  // Train and validation cycles both come from firstYear..lastYear; the held-out year is
  // after it. Dividing the train count alone by the span read "about 343" for a daily archive.
  const cyclesPerYear = trainYears && typeof td.train_cycles === "number"
    ? Math.round((td.train_cycles + (td.val_cycles ?? 0)) / trainYears) : null;
  const bustVars = Object.keys(data.thresholds?.bust_threshold ?? {}).length;
  const pctile = data.thresholds?.threshold_percentile;
  const q = typeof pctile === "number" ? (pctile <= 1 ? pctile : pctile / 100) : null;

  return (
    <main className="page page--wide">
      <header className="pagehead">
        <div>
          <h1 className="pagehead__title">{translate("nav.model")}</h1>
          <p className="pagehead__sub">
            {data.model_trained
              ? <>{rich("mp.run", { id: data.current_run_id ?? "" }, [<b className="mono" />])}{data.last_trained_at ? translate("mp.trained", { when: stamp(data.last_trained_at) }) : null}</>
              : data.message}
          </p>
        </div>
        <span className={`badge badge--${connection === "open" ? "live" : "offline"}`}>
          {connection === "open" ? translate("mp.live") : connection}
        </span>
      </header>

      {training || data.training_in_progress ? <p className="notice">{translate("mp.retraining")}</p> : null}
      {data.last_training_error ? (
        <p className="notice notice--error">{translate("mp.lastErr", { err: data.last_training_error })}</p>
      ) : null}

      {!data.model_trained ? (
        UPLOAD_ENABLED ? <UploadPanel /> : (
          <p className="muted">
            {translate("mp.noUpload")}
          </p>
        )
      ) : (
        <>
          <div className="kpis kpis--flush">
            <Stat cap="blue" label="ROC-AUC" value={num(clf.roc_auc, 3)}
              note={translate("mp.aucNote")} />
            <Stat cap="blue" label="PR-AUC" value={num(clf.pr_auc, 3)}
              note={rich("mp.prNote", { rate: pctOf(clf.bust_rate) }, [<b />])} />
            <Stat cap="watch" label={translate("mp.brier")} value={num(clf.brier, 3)}
              note={translate("mp.brierNote")} />
            <Stat cap="bust" label="F1" value={num(clf.f1, 3)}
              note={rich("mp.f1Note", { p: num(clf.precision, 2), r: num(clf.recall, 2) }, [<b />])} />
          </div>

          <p className="pagenote">
            {rich("mp.heldOut", { split: String(clf.split ?? translate("map.unknown")) }, [<b />])}
            {typeof clf.n === "number" ? rich("mp.neverTrained", { n: clf.n.toLocaleString() }, [<b />]) : null}
            {data.explanation_method ? rich("mp.explain", { m: data.explanation_method }, [<b />]) : null}
          </p>

          <div className="page--split page--even">
            <section className="card">
              <header className="card__head"><h3>{translate("mp.trainData")}</h3></header>
              {firstYear != null ? (
                <dl className="metrics">
                  <div className="metrics__wide">
                    <dt>{translate("mp.span")}</dt>
                    <dd className="mono">
                      {firstYear} → {vol.valid_date_max ?? lastYear ?? "—"}
                    </dd>
                  </div>
                  <div className="metrics__wide">
                    <dt>{translate("mp.archive")}</dt>
                    <dd className="mono small">
                      {firstYear}–{lastYear ?? firstYear}
                      {td.test_year != null ? translate("mp.heldOutYear", { y: td.test_year }) : null}
                    </dd>
                  </div>
                </dl>
              ) : null}
              <dl className="metrics metrics--compact">
                <div>
                  <dt>{translate("mp.cycles")}</dt>
                  <dd>{typeof td.cycles === "number" ? td.cycles.toLocaleString() : "—"}</dd>
                </div>
                <div>
                  <dt>{translate("mp.heldOutCol")}</dt>
                  <dd>{typeof td.held_out_cycles === "number" ? td.held_out_cycles.toLocaleString() : "—"}</dd>
                </div>
                <div>
                  <dt>{translate("mp.pairs")}</dt>
                  <dd>{typeof td.paired_rows === "number" ? td.paired_rows.toLocaleString() : "—"}</dd>
                </div>
                <div><dt>{translate("mp.vars")}</dt><dd>{data.modelled_variables.length}</dd></div>
              </dl>
              {typeof td.cycles === "number" && td.cycles > 0 ? (
                <p className="muted small">
                  {/* It read "Trained on 6,558 forecast cycles from 2000 to 2016": that count
                    * includes the validation and held-out cycles, and a pooled run fits its
                    * models on a sample of the training ones. */}
                  {rich("mp.cyclesTotal", { n: td.cycles.toLocaleString() }, [<b />])}
                  {typeof td.train_cycles === "number" && typeof td.val_cycles === "number" ? (
                    <>{translate("mp.cyclesSplit", { train: td.train_cycles.toLocaleString(), val: td.val_cycles.toLocaleString() })}
                      {firstYear != null ? translate("mp.cyclesFrom", { from: firstYear, to: lastYear ?? firstYear }) : null}
                      {typeof td.held_out_cycles === "number" ? (
                        <>{translate("mp.cyclesHeld", { n: td.held_out_cycles.toLocaleString() })}
                          {td.test_year != null ? translate("mp.cyclesHeldFrom", { y: td.test_year }) : null}</>
                      ) : null}</>
                  ) : null}.
                  {typeof td.fit_cycles === "number" && typeof td.train_cycles === "number" ? (
                    <>{rich("mp.fit", { fit: td.fit_cycles.toLocaleString(), train: td.train_cycles.toLocaleString() }, [<b />])}
                      {typeof td.classifier_cycles === "number" && td.classifier_cycles !== td.fit_cycles
                        ? translate("mp.fitClf", { n: td.classifier_cycles.toLocaleString() }) : null}.</>
                  ) : null}
                  {cyclesPerYear != null ? rich("mp.perYear", { n: cyclesPerYear }, [<b />]) : null}
                </p>
              ) : (
                <p className="muted small">
                  {translate("mp.noProv")}
                </p>
              )}
              <div className="panel__section">
                <header className="card__head"><h4>{translate("mp.server")}</h4></header>
                <dl className="metrics metrics--compact">
                  <div><dt>{translate("mp.rows")}</dt><dd>{(vol.total_rows ?? 0).toLocaleString()}</dd></div>
                  <div><dt>{translate("mp.regionIds")}</dt><dd>{vol.regions ?? "—"}</dd></div>
                  <div><dt>{translate("mp.cyclesShort")}</dt><dd>{vol.forecast_cycles ?? "—"}</dd></div>
                  <div><dt>{translate("mp.batches")}</dt><dd>{vol.batches ?? "—"}</dd></div>
                </dl>
                <dl className="metrics">
                  <div className="metrics__wide">
                    <dt>{translate("mp.validRange")}</dt>
                    <dd className="mono small">
                      {vol.valid_date_min ?? "—"} → {vol.valid_date_max ?? "—"}
                    </dd>
                  </div>
                </dl>
                {vol.unavailable_reason ? (
                  <p className="notice">{vol.unavailable_reason}</p>
                ) : null}
                <p className="muted small">
                  {translate("mp.serverNote")}
                  {firstYear != null ? rich("mp.serverSince", { y: firstYear }, [<b />]) : null}
                </p>
              </div>
              <ul className="taglist">
                {data.modelled_variables.map((v) => (
                  <li key={v} className="tag" title={v}>{variableLabel(v)}</li>
                ))}
              </ul>
              {Object.keys(data.skipped_variables).length ? (
                <p className="muted small">
                  {translate("det.notModelled", { list: Object.entries(data.skipped_variables).map(([v, why]) => `${variableLabel(v)} (${why})`).join("; ") })}
                </p>
              ) : (
                <p className="muted small">{translate("mp.allModelled")}</p>
              )}
            </section>

            <section className="card">
              <header className="card__head"><h3>{translate("mp.bustTitle")}</h3></header>
              <p className="muted small">
                {rich("mp.bustBody", { pct: pct(data.thresholds?.threshold_percentile) }, [<b />])}
              </p>
              <div className="tablewrap">
                <table className="dtable dtable--tight">
                  <thead>
                    <tr>
                      <th>{translate("mp.colVar")}</th>
                      <th className="dtable__num">{translate("mp.colBustAbove")}</th>
                      <th className="dtable__num">{translate("mp.colP90")}</th>
                      <th>{translate("mp.colUnit")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.thresholds?.bust_threshold ?? {}).map(([v, t]) => (
                      <tr key={v}>
                        <td className="mono" title={v}>{variableLabel(v)}</td>
                        <td className="dtable__num mono dtable__strong">{t.toFixed(2)}</td>
                        <td className="dtable__num mono muted">
                          {num(data.thresholds?.p90_error?.[v], 2)}
                        </td>
                        <td className="muted">{variableUnit(v) ?? ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {bustVars > 0 && q != null && typeof clf.bust_rate === "number" ? (
                <p className="muted small">
                  {/* Why "the worst 10%" makes about half of all forecasts busts. */}
                  {rich("mp.anyVar", {
                    k: bustVars,
                    each: Math.round((1 - q) * 100),
                    indep: Math.round((1 - q ** bustVars) * 100),
                    actual: Math.round(clf.bust_rate * 100),
                  }, [<b />])}
                </p>
              ) : null}
              {cuts ? (
                <p className="muted small">
                  {rich("mp.flagged", { w: (cuts.medium * 100).toFixed(0), b: (cuts.high * 100).toFixed(0) },
                    [<b />, <b className="mono" />])}
                </p>
              ) : null}
            </section>
          </div>

          {Object.keys(regressors).length ? (
            <section className="card card--table">
              <header className="card__head">
                <h3>{translate("mp.errTitle")}</h3>
              </header>
              <div className="tablewrap">
                <table className="dtable">
                  <thead>
                    <tr>
                      <th>{translate("mp.colVar")}</th>
                      <th className="dtable__num">MAE</th>
                      <th className="dtable__num">{translate("mp.colBaseMae")}</th>
                      <th className="dtable__num">{translate("mp.colSkill")}</th>
                      {/* Dropped on a phone. Skill is the column this table exists to
                        * show, and at 390px it was the one pushed off the right edge
                        * behind a scroll with no affordance. See .dtable__opt. */}
                      <th className="dtable__num dtable__opt">RMSE</th>
                      <th className="dtable__num dtable__opt">R²</th>
                      <th className="dtable__num dtable__opt">{translate("mp.colForecasts")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(regressors).map(([v, m]) => {
                      const skill = skillScore(m.mae, m.baseline_mae);
                      return (
                        <tr key={v}>
                          <td className="mono dtable__strong" title={v}>{variableLabel(v)}</td>
                          <td className="dtable__num mono">{num(m.mae, 3)}</td>
                          <td className="dtable__num mono muted">{num(m.baseline_mae, 3)}</td>
                          <td className="dtable__num mono">
                            {skill == null ? "—" : <SkillCell skill={skill} />}
                          </td>
                          <td className="dtable__num dtable__opt mono muted">{num(m.rmse, 3)}</td>
                          <td className="dtable__num dtable__opt mono muted">{num(m.r2, 3)}</td>
                          <td className="dtable__num dtable__opt mono muted">{m.n?.toLocaleString() ?? "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="muted small">
                {translate("mp.errNote")}
              </p>
            </section>
          ) : null}

          {/* The full ladder: every rung, not just the compact climatology / best baseline
            * / classifier read on Operations. That card links here for exactly this. */}
          {data.baselines?.models?.length ? (
            <section className="card card--table">
              <header className="card__head"><h3>{translate("mp.ladderTitle")}</h3></header>
              <p className="muted small">
                {translate("mp.ladderNote")}
              </p>
              <BaselineLadderTable baselines={data.baselines} flagLeadDay />
              <p className="muted small">{translate("mp.arrows")}</p>
            </section>
          ) : null}

          {/* Workstream D's evidence, which had no screen until now: what the forecast is
            * worth to act on (F3) and whether its probabilities mean what they say (F4). */}
          {/* The two charts share a row because they are the same shape of thing. The
            * misses go full width underneath rather than stacked in one column: stacking
            * them made that column 338px taller than the other, which read as a hole in
            * the page, and a list of cases uses the width better than a chart would. */}
          <div className="page--split page--even">
            <EconomicValueCard data={data} />
            <CorpReliabilityCard data={data} />
          </div>
          <MissesCard data={data} />

          {UPLOAD_ENABLED ? <UploadPanel /> : null}
        </>
      )}
          <PipelineLog />
</main>
  );
}

function SkillCell({ skill }: { skill: number }) {
  const cls = skill > 0 ? "skill skill--up" : skill < 0 ? "skill skill--down" : "skill";
  return <span className={cls}>{skill > 0 ? "+" : ""}{(skill * 100).toFixed(1)}%</span>;
}

function Stat({ cap, label, value, note }: {
  cap: string; label: string; value: string; note: React.ReactNode;
}) {
  return (
    <div className="kpi">
      <div className={`kpi__cap kpi__cap--${cap}`} />
      <span className="kpi__label">{label}</span>
      <p className="kpi__value">{value}</p>
      <p className="kpi__note">{note}</p>
    </div>
  );
}

function skillScore(mae: number | null, baseline: number | null): number | null {
  if (mae == null || baseline == null || !Number.isFinite(mae) || !baseline) return null;
  return 1 - mae / baseline;
}

function pctOf(v: unknown): string {
  return typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(0)}%` : "—";
}

function num(v: unknown, digits: number): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
}

function pct(v: ModelStatusResponse["thresholds"]["threshold_percentile"]): string {
  if (typeof v !== "number") return translate("mp.configured");
  const n = (v <= 1 ? v * 100 : v).toFixed(0);
  // "90th" is an English ordinal; the Indian-language sentences read "90 percentile".
  return locale() === "en-GB" ? `${n}th` : n;
}
