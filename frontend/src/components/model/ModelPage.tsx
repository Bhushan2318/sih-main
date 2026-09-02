import type { ModelStatusResponse } from "../../api/types";
import { stamp } from "../../format";
import { useModelStatus } from "../../hooks/useDashboardData";
import { useLiveStore } from "../../store/liveStore";
import { ErrorState, LoadingState } from "../common/States";
import { UploadPanel } from "../upload/UploadPanel";

/**
 * Upload drives a retrain, which needs ~2 GB. On a 512 MB serving host that would OOM
 * the container mid-demo, so the host sets VITE_ENABLE_UPLOAD=false and the panel is not
 * built at all. Unset (local dev) means on, where the laptop can actually do the work.
 */
const UPLOAD_ENABLED = import.meta.env.VITE_ENABLE_UPLOAD !== "false";

/**
 * The model, in full: what it was trained on, how well it scores, and what it treats as a
 * bust for each variable.
 *
 * The side-card version showed four classifier numbers. The registry actually records a
 * per-variable regressor split with a persistence baseline beside it, and the thresholds
 * every bust label was derived from - the numbers that answer "is this thing any good?"
 * and "what does bust even mean here?". Those are the case for the project, so they get
 * the page rather than being computed and thrown away.
 *
 * Skill is the one derived figure and it is labelled as such: 1 - MAE/baseline MAE, the
 * fraction of the naive forecast's error the model removes. Everything else is served.
 */
export function ModelPage() {
  const { data, isLoading, error } = useModelStatus();
  const connection = useLiveStore((s) => s.connectionStatus);
  const training = useLiveStore((s) => s.trainingInProgress);

  if (isLoading) return <main className="page page--wide"><LoadingState label="Loading model status…" /></main>;
  if (error) return <main className="page page--wide"><ErrorState error={error} /></main>;
  if (!data) return null;

  const clf = data.validation_metrics?.classifier ?? {};
  const regressors = data.validation_metrics?.regressors ?? {};
  const cuts = data.thresholds?.risk_band_cuts;
  const vol = data.data_volume ?? {};
  const td = data.training_data ?? {};

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
          {/* ── the headline question: does the bust classifier actually work? */}
          <div className="kpis kpis--flush">
            <Stat cap="blue" label="ROC-AUC" value={num(clf.roc_auc, 3)}
              note={<>ranking skill · 0.5 is a coin flip</>} />
            <Stat cap="blue" label="PR-AUC" value={num(clf.pr_auc, 3)}
              note={<>base rate <b>{num(clf.bust_rate, 3)}</b></>} />
            <Stat cap="watch" label="Brier score" value={num(clf.brier, 3)}
              note={<>calibration error · lower is better</>} />
            <Stat cap="bust" label="F1" value={num(clf.f1, 3)}
              note={<>P <b>{num(clf.precision, 2)}</b> · R <b>{num(clf.recall, 2)}</b></>} />
          </div>

          <p className="pagenote">
            Bust classifier on the held-out <b>{String(clf.split ?? "unknown")}</b> split
            {typeof clf.n === "number" ? <> · <b>{clf.n.toLocaleString()}</b> forecasts the model never trained on</> : null}
            {data.explanation_method ? <> · explanations via <b>{data.explanation_method}</b></> : null}
          </p>

          <div className="page--split">
            {/* ── what it learned from */}
            <section className="card">
              <header className="card__head"><h3>Training data</h3></header>
              {/* What the model learned from, read from the run's own manifest - not from
                  this box's copy of the store. Training runs on a 16 GB CI runner against
                  the full reforecast archive; the 512 MB serving box carries only the
                  cycles it needs to answer a request, and never trains. Counting rows
                  here would report a smaller evidence base than the model actually has. */}
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
                  <dt>Paired rows</dt>
                  <dd>{typeof td.paired_rows === "number" ? td.paired_rows.toLocaleString() : "—"}</dd>
                </div>
                <div><dt>Variables</dt><dd>{data.modelled_variables.length}</dd></div>
              </dl>
              {typeof td.cycles === "number" && td.cycles > 0 ? (
                <p className="muted small">
                  Trained on <b>{td.cycles.toLocaleString()}</b> forecast cycles
                  {td.first_train_date ? <> from {String(td.first_train_date).slice(0, 10)}</> : null}
                  {typeof td.train_cycles === "number" && typeof td.val_cycles === "number" ? (
                    <>{" "}— {td.train_cycles} train · {td.val_cycles} validation ·{" "}
                      {td.held_out_cycles} held out</>
                  ) : null}
                  . Sampled, not continuous: a reforecast archive of a handful of
                  initialisations a year, plus the live cycles since deployment.
                </p>
              ) : (
                <p className="muted small">
                  Training provenance unavailable — this model predates the field; it is
                  written on the next retrain.
                </p>
              )}
              {/* Everything below describes THIS box's copy of the store, which is
                  deliberately smaller than what the model trained on: it is a 512 MB
                  instance that never trains and only needs the cycles it serves. Labelled
                  so the two counts above and below cannot be read as the same number. */}
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
                    <dt>Valid-date range</dt>
                    <dd className="mono small">
                      {vol.valid_date_min ?? "—"} → {vol.valid_date_max ?? "—"}
                    </dd>
                  </div>
                </dl>
                <p className="muted small">
                  The serving copy carries the cycles needed to score today and to replay
                  recent ones — not the full training archive, which lives where the model
                  is trained.
                </p>
              </div>
              <ul className="taglist">
                {data.modelled_variables.map((v) => <li key={v} className="tag">{v}</li>)}
              </ul>
              {Object.keys(data.skipped_variables).length ? (
                <p className="muted small">
                  Not modelled:{" "}
                  {Object.entries(data.skipped_variables).map(([v, why]) => `${v} (${why})`).join("; ")}
                </p>
              ) : (
                <p className="muted small">Every ingested variable is modelled.</p>
              )}
            </section>

            {/* ── what "bust" means, per variable, and where the bands cut */}
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
                        <td className="mono">{v}</td>
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
                  Risk bands cut at <b className="mono">{cuts.medium.toFixed(2)}</b> (watch) and{" "}
                  <b className="mono">{cuts.high.toFixed(2)}</b> (bust).
                </p>
              ) : null}
            </section>
          </div>

          {/* ── the per-variable case: beating persistence, variable by variable */}
          {Object.keys(regressors).length ? (
            <section className="card card--table">
              <header className="card__head">
                <h3>Per-variable error, against a persistence baseline</h3>
              </header>
              <div className="tablewrap">
                <table className="dtable">
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th className="dtable__num">MAE</th>
                      <th className="dtable__num">Baseline MAE</th>
                      <th className="dtable__num">Skill</th>
                      <th className="dtable__num">RMSE</th>
                      <th className="dtable__num">R²</th>
                      <th className="dtable__num">n</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(regressors).map(([v, m]) => {
                      const skill = skillScore(m.mae, m.baseline_mae);
                      return (
                        <tr key={v}>
                          <td className="mono dtable__strong">{v}</td>
                          <td className="dtable__num mono">{num(m.mae, 3)}</td>
                          <td className="dtable__num mono muted">{num(m.baseline_mae, 3)}</td>
                          <td className="dtable__num mono">
                            {skill == null ? "—" : <SkillCell skill={skill} />}
                          </td>
                          <td className="dtable__num mono muted">{num(m.rmse, 3)}</td>
                          <td className="dtable__num mono muted">{num(m.r2, 3)}</td>
                          <td className="dtable__num mono muted">{m.n?.toLocaleString() ?? "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="muted small">
                Skill = 1 − MAE ÷ baseline MAE: the share of the naive forecast&apos;s error the
                model removes. Positive is better than the baseline.
              </p>
            </section>
          ) : null}

          {UPLOAD_ENABLED ? <UploadPanel /> : null}
        </>
      )}
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

/** Fraction of the baseline's error removed. Null unless both figures are real. */
function skillScore(mae: number | null, baseline: number | null): number | null {
  if (mae == null || baseline == null || !Number.isFinite(mae) || !baseline) return null;
  return 1 - mae / baseline;
}

function num(v: unknown, digits: number): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
}

function pct(v: ModelStatusResponse["thresholds"]["threshold_percentile"]): string {
  return typeof v === "number" ? `${(v <= 1 ? v * 100 : v).toFixed(0)}th` : "configured";
}
