import type { ModelStatusResponse } from "../../api/types";
import { stamp } from "../../format";
import { useModelStatus } from "../../hooks/useDashboardData";
import { variableLabel } from "../../lib/displayNames";
import { useLiveStore } from "../../store/liveStore";
import { ErrorState, LoadingState } from "../common/States";
import { retryingHint } from "../../lib/retryHint";
import { UploadPanel } from "../upload/UploadPanel";
import { CorpReliabilityCard } from "./CorpReliabilityCard";
import { EconomicValueCard } from "./EconomicValueCard";
import { MissesCard } from "./MissesCard";
import { PipelineLog } from "./PipelineLog";

const UPLOAD_ENABLED = import.meta.env.VITE_ENABLE_UPLOAD !== "false";

export function ModelPage() {
  const { data, isLoading, error, failureCount } = useModelStatus();
  const connection = useLiveStore((s) => s.connectionStatus);
  const training = useLiveStore((s) => s.trainingInProgress);

  if (isLoading) return <main className="page page--wide"><LoadingState label="Loading model status…" hint={retryingHint(failureCount)} /></main>;
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
  const cyclesPerYear = trainYears && typeof td.train_cycles === "number"
    ? Math.round(td.train_cycles / trainYears) : null;

  return (
    <main className="page page--wide">
      <header className="pagehead">
        <div>
          <h1 className="pagehead__title">Model</h1>
          <p className="pagehead__sub">
            {data.model_trained
              ? <>Run <b className="mono">{data.current_run_id}</b>{data.last_trained_at ? <> · trained {stamp(data.last_trained_at)}</> : null}</>
              : data.message}
          </p>
        </div>
        <span className={`badge badge--${connection === "open" ? "live" : "offline"}`}>
          {connection === "open" ? "live" : connection}
        </span>
      </header>

      {training || data.training_in_progress ? <p className="notice">Retraining in progress…</p> : null}
      {data.last_training_error ? (
        <p className="notice notice--error">Last training error: {data.last_training_error}</p>
      ) : null}

      {!data.model_trained ? (
        UPLOAD_ENABLED ? <UploadPanel /> : (
          <p className="muted">
            This deployment serves a model trained elsewhere and cannot retrain, so there
            is nothing to upload here. No model is currently published.
          </p>
        )
      ) : (
        <>
          <div className="kpis kpis--flush">
            <Stat cap="blue" label="ROC-AUC" value={num(clf.roc_auc, 3)}
              note={<>ranking skill · 0.5 is a coin flip</>} />
            <Stat cap="blue" label="PR-AUC" value={num(clf.pr_auc, 3)}
              note={<>busts are <b>{pctOf(clf.bust_rate)}</b> of all forecasts, so that is the score to beat</>} />
            <Stat cap="watch" label="Brier score" value={num(clf.brier, 3)}
              note={<>how far off the probabilities are · lower is better</>} />
            <Stat cap="bust" label="F1" value={num(clf.f1, 3)}
              note={<>precision <b>{num(clf.precision, 2)}</b> · recall <b>{num(clf.recall, 2)}</b></>} />
          </div>

          <p className="pagenote">
            Bust classifier on the held-out <b>{String(clf.split ?? "unknown")}</b> split
            {typeof clf.n === "number" ? <> · <b>{clf.n.toLocaleString()}</b> forecasts the model never trained on</> : null}
            {data.explanation_method ? <> · explanations via <b>{data.explanation_method}</b></> : null}
          </p>

          <div className="page--split">
            <section className="card">
              <header className="card__head"><h3>Training data</h3></header>
              {firstYear != null ? (
                <dl className="metrics">
                  <div className="metrics__wide">
                    <dt>Data span</dt>
                    <dd className="mono">
                      {firstYear} → {vol.valid_date_max ?? lastYear ?? "—"}
                    </dd>
                  </div>
                  <div className="metrics__wide">
                    <dt>Training archive</dt>
                    <dd className="mono small">
                      {firstYear}–{lastYear ?? firstYear}
                      {td.test_year != null ? <> · held out {td.test_year}</> : null}
                    </dd>
                  </div>
                </dl>
              ) : null}
              <dl className="metrics metrics--compact">
                <div>
                  <dt>Forecast cycles</dt>
                  <dd>{typeof td.cycles === "number" ? td.cycles.toLocaleString() : "—"}</dd>
                </div>
                <div>
                  <dt>Held out</dt>
                  <dd>{typeof td.held_out_cycles === "number" ? td.held_out_cycles.toLocaleString() : "—"}</dd>
                </div>
                <div>
                  <dt>Forecast–observation pairs</dt>
                  <dd>{typeof td.paired_rows === "number" ? td.paired_rows.toLocaleString() : "—"}</dd>
                </div>
                <div><dt>Variables</dt><dd>{data.modelled_variables.length}</dd></div>
              </dl>
              {typeof td.cycles === "number" && td.cycles > 0 ? (
                <p className="muted small">
                  Trained on <b>{td.cycles.toLocaleString()}</b> forecast cycles
                  {firstYear != null ? <> from {firstYear}{lastYear != null && lastYear !== firstYear ? <> to {lastYear}</> : null}</> : null}
                  {typeof td.train_cycles === "number" && typeof td.val_cycles === "number" ? (
                    <>{" "}— {td.train_cycles.toLocaleString()} train · {td.val_cycles.toLocaleString()} validation ·{" "}
                      {td.held_out_cycles?.toLocaleString()} held out</>
                  ) : null}
                  {cyclesPerYear != null ? (
                    <>. That is about <b>{cyclesPerYear}</b> initialisations per training year
                      of the GEFSv12 reforecast (one 00 UTC cycle per day at most).</>
                  ) : "."}
                </p>
              ) : (
                <p className="muted small">
                  Training provenance unavailable — this model predates the field; it is
                  written on the next retrain.
                </p>
              )}
              <div className="panel__section">
                <header className="card__head"><h4>On this server</h4></header>
                <dl className="metrics metrics--compact">
                  <div><dt>Rows</dt><dd>{(vol.total_rows ?? 0).toLocaleString()}</dd></div>
                  <div><dt>Regions</dt><dd>{vol.regions ?? "—"}</dd></div>
                  <div><dt>Cycles</dt><dd>{vol.forecast_cycles ?? "—"}</dd></div>
                  <div><dt>Batches</dt><dd>{vol.batches ?? "—"}</dd></div>
                </dl>
                <dl className="metrics">
                  <div className="metrics__wide">
                    <dt>Valid-date range on this server</dt>
                    <dd className="mono small">
                      {vol.valid_date_min ?? "—"} → {vol.valid_date_max ?? "—"}
                    </dd>
                  </div>
                </dl>
                {vol.unavailable_reason ? (
                  <p className="notice">{vol.unavailable_reason}</p>
                ) : null}
                <p className="muted small">
                  The serving copy carries the cycles needed to score today and to replay
                  recent ones — not the full training archive, which lives where the model
                  is trained.
                  {firstYear != null ? (
                    <> The model itself was trained on data going back to{" "}
                      <b>{firstYear}</b>; this range is only what this 512&nbsp;MB box carries.</>
                  ) : null}
                </p>
              </div>
              <ul className="taglist">
                {data.modelled_variables.map((v) => (
                  <li key={v} className="tag" title={v}>{variableLabel(v)}</li>
                ))}
              </ul>
              {Object.keys(data.skipped_variables).length ? (
                <p className="muted small">
                  Not modelled (too few matched forecast–observation pairs):{" "}
                  {Object.entries(data.skipped_variables).map(([v, why]) => `${variableLabel(v)} (${why})`).join("; ")}
                </p>
              ) : (
                <p className="muted small">Every ingested variable is modelled.</p>
              )}
            </section>

            <section className="card">
              <header className="card__head"><h3>What counts as a bust</h3></header>
              <p className="muted small">
                A forecast busts when its error exceeds this variable&apos;s own threshold, set at
                the <b>{pct(data.thresholds?.threshold_percentile)}</b> percentile of historical
                error. The p90 column is the error only one forecast in ten exceeds.
              </p>
              <div className="tablewrap">
                <table className="dtable dtable--tight">
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th className="dtable__num">Bust above</th>
                      <th className="dtable__num">p90 error</th>
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
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {cuts ? (
                <p className="muted small">
                  A region is flagged <b>watch</b> above{" "}
                  <b className="mono">{(cuts.medium * 100).toFixed(0)}%</b> and <b>bust</b> above{" "}
                  <b className="mono">{(cuts.high * 100).toFixed(0)}%</b>.
                </p>
              ) : null}
            </section>
          </div>

          {Object.keys(regressors).length ? (
            <section className="card card--table">
              <header className="card__head">
                <h3>Error per variable, against a baseline that just says “tomorrow is like today”</h3>
              </header>
              <div className="tablewrap">
                <table className="dtable">
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th className="dtable__num">MAE</th>
                      <th className="dtable__num">Baseline MAE</th>
                      <th className="dtable__num">Skill</th>
                      {/* Dropped on a phone. Skill is the column this table exists to
                        * show, and at 390px it was the one pushed off the right edge
                        * behind a scroll with no affordance. See .dtable__opt. */}
                      <th className="dtable__num dtable__opt">RMSE</th>
                      <th className="dtable__num dtable__opt">R²</th>
                      <th className="dtable__num dtable__opt">forecasts</th>
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
                Skill = 1 − MAE ÷ baseline MAE: the share of the naive forecast&apos;s error the
                model removes. Positive is better than the baseline. MAE is the average miss,
                RMSE the same thing but weighted towards the big misses, and R² the share of
                the variation the model accounts for.
              </p>
            </section>
          ) : null}

          {/* Workstream D's evidence, which had no screen until now: what the forecast is
            * worth to act on (F3) and whether its probabilities mean what they say (F4). */}
          {/* The two charts share a row because they are the same shape of thing. The
            * misses go full width underneath rather than stacked in one column: stacking
            * them made that column 338px taller than the other, which read as a hole in
            * the page, and a list of cases uses the width better than a chart would. */}
          <div className="page--split">
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
  return typeof v === "number" ? `${(v <= 1 ? v * 100 : v).toFixed(0)}th` : "configured";
}
