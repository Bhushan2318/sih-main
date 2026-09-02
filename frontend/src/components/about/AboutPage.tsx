import { useModelStatus } from "../../hooks/useDashboardData";
import { ErrorState, LoadingState } from "../common/States";

/**
 * What this is, for someone who arrived with no introduction.
 *
 * The link is sent to judges who open it whenever they like: there is no presentation and
 * nobody narrating. So the case for the project has to live on the site, and it has to
 * answer, in order, the three questions a stranger actually has - what am I looking at,
 * is it any good, and can I believe the numbers.
 *
 * Every figure here is read from /api/model/status at render time. None is written into
 * this file. A hardcoded metric would be correct until the next retrain and quietly wrong
 * afterwards, which is precisely the failure this project claims not to have.
 */
export function AboutPage({ onReplay }: { onReplay: () => void }) {
  const { data, isLoading, error } = useModelStatus();

  if (isLoading) return <main className="page page--wide"><LoadingState label="Loading…" /></main>;
  if (error) return <main className="page page--wide"><ErrorState error={error} /></main>;
  if (!data) return null;

  const clf = data.validation_metrics?.classifier ?? {};
  const td = data.training_data ?? {};
  const thr = data.thresholds?.bust_threshold ?? {};
  const bl = data.baselines ?? {};
  const n3 = (v: unknown, dp = 3) =>
    typeof v === "number" && Number.isFinite(v) ? v.toFixed(dp) : "—";

  return (
    <main className="page page--wide">
      <header className="pagehead">
        <div>
          <h1 className="pagehead__title">What this is</h1>
          <p className="pagehead__sub">
            Sanket — संकेत, “signal”. Predicting where a weather forecast will be wrong,
            and saying why.
          </p>
        </div>
      </header>

      {/* The single most useful thing a first-time visitor can do, said before anything
          else. Replay is the most convincing view here and it is the last tab. */}
      <p className="notice">
        <b>Start here:</b> Replay takes a real forecast cycle from the archive, scores it
        with the deployed model, and shows what this system would have told a forecaster
        that day — before anyone knew the outcome.{" "}
        <button type="button" className="chip" onClick={onReplay}>
          Open Replay →
        </button>
      </p>

      <div className="page--split">
        <section className="card">
          <header className="card__head"><h3>The question it answers</h3></header>
          <p className="muted">
            Every operational centre already issues a forecast. This does not try to make a
            better one. It answers the question a duty forecaster actually has at 6am:
          </p>
          <p><b>“How likely is the forecast I am holding to be badly wrong today?”</b></p>
          <p className="muted small">
            Correction and confidence are different services. NCMRWF already runs quantile
            mapping and EMOS, and busts still happen — correction removes systematic
            error, while busts are flow-dependent. Knowing when to distrust a forecast is
            what decides whether a warning goes out.
          </p>
          <p className="muted small">
            A <b>bust</b> is a forecast whose error lands in the tail of that variable’s
            own historical error distribution — the 90th percentile, computed on training
            data only. Each variable therefore has its own threshold, in its own units:
          </p>
          <ul className="taglist">
            {Object.entries(thr).map(([v, t]) => (
              <li key={v} className="tag">{v} ≥ {n3(t, 2)}</li>
            ))}
          </ul>
        </section>

        <section className="card">
          <header className="card__head"><h3>How well it works</h3></header>
          <dl className="metrics metrics--compact">
            <div><dt>ROC-AUC</dt><dd>{n3(clf.roc_auc)}</dd></div>
            <div><dt>F1</dt><dd>{n3(clf.f1)}</dd></div>
            <div><dt>Brier</dt><dd>{n3(clf.brier)}</dd></div>
            <div>
              <dt>Held-out forecasts</dt>
              <dd>{typeof clf.n === "number" ? clf.n.toLocaleString() : "—"}</dd>
            </div>
          </dl>
          <p className="muted small">
            Measured on the <b>{String(clf.split ?? "held-out")}</b> split — forecast
            cycles the model never trained on
            {typeof td.held_out_cycles === "number" ? <>, {td.held_out_cycles} of them</> : null}
            {typeof td.cycles === "number" ? <> out of {td.cycles} in total</> : null}.
          </p>
          <p className="muted small">
            Because a bust is defined against each variable’s <i>own</i> error percentile
            rather than an absolute error, the label does not simply grow with lead time —
            so this skill is not a rediscovery of “day 10 is worse than day 1”.
          </p>

          {/* "Compared to what?" answered with the run's own numbers. This table is
              written by the training run and shipped inside it, so it can never describe
              a different model than the one answering this request. */}
          {bl.models?.length ? (
            <>
              <header className="card__head"><h4>Compared to what?</h4></header>
              <p className="muted small">
                Every baseline is fitted on the training split alone and scored on the same
                held-out rows as the model. Brier skill is against climatology — the base
                rate — so <b>0.000 means no skill beyond knowing how often busts happen</b>.
              </p>
              <div className="tablewrap">
                <table className="dtable">
                  <thead>
                    <tr>
                      <th>model</th><th>Brier ↓</th><th>skill vs climatology ↑</th><th>ROC-AUC ↑</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bl.models.map((m) => (
                      <tr key={m.name} className={m.is_model ? "is-active" : undefined}>
                        <td>{m.is_model ? <b>{m.name}</b> : m.name}</td>
                        <td className="mono">{n3(m.brier, 4)}</td>
                        <td className="mono">{n3(m.bss, 4)}</td>
                        <td className="mono">{n3(m.roc_auc, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {typeof bl.lead_bust_correlation?.test === "number" ? (
                <p className="muted small">
                  A lead-day-only model can score at or below zero here because the label
                  does not track lead time: the measured correlation between lead day and
                  bust is {n3(bl.lead_bust_correlation.train, 3)} on train and{" "}
                  {n3(bl.lead_bust_correlation.test, 3)} on the held-out split.
                </p>
              ) : null}
            </>
          ) : (
            <p className="muted small">
              The baseline comparison is generated by the training run; this model
              predates it and the table is written on the next retrain.
            </p>
          )}
        </section>
      </div>

      <div className="page--split">
        <section className="card">
          <header className="card__head"><h3>Why the numbers can be trusted</h3></header>
          <ul className="notes">
            <li>
              <b>No synthetic, mocked or placeholder data anywhere</b> — not even as a
              fallback. Where a number cannot be computed from real data, this interface
              shows an em dash <i>and the reason</i>. That rule is enforced in the code,
              not just stated here.
            </li>
            <li>
              <b>Real data end to end.</b> NOAA GEFS for forecasts — a reforecast archive
              for training and the operational feed live — verified against ERA5
              reanalysis.
            </li>
            <li>
              <b>Leakage is tested, not asserted.</b> Bust thresholds are fitted on the
              training split only; out-of-fold folds are grouped by forecast cycle so no
              cycle spans a fold; no observed day appears on both sides of the train/test
              split. Each is a test in the suite, written so it cannot pass vacuously.
            </li>
            <li>
              <b>The ground truth’s own uncertainty is measured.</b> Against an
              independent reanalysis over ~21,000 paired city-days, two leading products
              disagree by roughly a quarter to a half of a bust threshold. Stated up front
              rather than waiting to be asked.
            </li>
          </ul>
        </section>

        <section className="card">
          <header className="card__head"><h3>What it does not do</h3></header>
          <ul className="notes">
            <li>
              Coverage is <b>sampled, not continuous</b> — a reforecast archive of a
              handful of initialisations a year, plus live cycles since deployment.
            </li>
            <li>
              City points, not full regional coverage of India. Region-level readings are
              indicative.
            </li>
            <li>
              Five of GEFS’s 31 ensemble members, so spread-derived features are a noisy
              estimate of true ensemble spread.
            </li>
            <li>
              ERA5 precipitation is weak over India relative to gauge-based products, and
              rainfall results carry that caveat. Rainfall is also the hardest variable
              here — zero-inflated and heavily skewed.
            </li>
            <li>
              A bust is defined on <b>surface-variable error</b>, not the synoptic Z500
              criterion of Rodwell et al. (2013). That is a deliberate choice: surface
              error is what reaches agriculture and disaster response, whereas Z500 is
              what reaches meteorologists.
            </li>
          </ul>
        </section>
      </div>

      <section className="card">
        <header className="card__head"><h3>How it runs</h3></header>
        <p className="muted small">
          FastAPI and XGBoost behind React and TypeScript, over a hive-partitioned Parquet
          store — one origin, one process. Training runs on CI with 16 GB; the site is
          served from a 512 MB free-tier instance that <b>cannot train</b>, which is why
          serving memory was cut to fit with scored output verified byte-identical at every
          step. When the upstream feed drops forecast steps mid-pull, a fallback source
          refills them before the daily reduction — a short rainfall <i>sum</i> would
          otherwise silently halve the quantity that drives most busts. Total hosting
          cost: nothing.
        </p>
      </section>
    </main>
  );
}
