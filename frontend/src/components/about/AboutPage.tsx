import { API_BASE } from "../../api/client";
import { useModelStatus } from "../../hooks/useDashboardData";
import { variableLabel, variableUnit } from "../../lib/displayNames";
import { formatMetric } from "../../lib/format";
import { ErrorState, LoadingState } from "../common/States";
import { retryingHint } from "../../lib/retryHint";
import { BaselineLadderTable } from "../model/BaselineLadderTable";

export function AboutPage({ onReplay }: { onReplay: () => void }) {
  const { data, isLoading, error, failureCount } = useModelStatus();

  if (isLoading) return <main className="page page--wide"><LoadingState label="Loading…" hint={retryingHint(failureCount)} /></main>;
  if (error) return <main className="page page--wide"><ErrorState error={error} /></main>;
  if (!data) return null;

  const clf = data.validation_metrics?.classifier ?? {};
  const td = data.training_data ?? {};
  const thr = data.thresholds?.bust_threshold ?? {};
  const bl = data.baselines ?? {};

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

      <div className="notice">
        <p style={{ margin: 0 }}>
          <b>Start here:</b> Replay takes a real forecast cycle from the archive, scores it
          with the deployed model, and shows what this system would have told a forecaster
          that day — before anyone knew the outcome.
        </p>
        <p style={{ margin: "0.6rem 0 0" }}>
          <button type="button" className="chip" onClick={onReplay}>
            Open Replay →
          </button>
        </p>
      </div>

      <div className="page--cols">
        <div className="page__col">
          <section className="card">
            <header className="card__head"><h3>The question it answers</h3></header>
            <p className="muted">
              Every operational centre already issues a forecast. This does not try to make a
              better one. It answers the question a duty forecaster actually has at 6am:
            </p>
            <p><b>“How likely is the forecast I am holding to be badly wrong today?”</b></p>
            <p className="muted small">
              Correction and confidence are different services. NCMRWF already corrects its
              forecasts statistically, and busts still happen. Correction removes the errors a
              model makes <i>consistently</i>; a bust comes from the particular weather pattern
              of that day, which is not consistent and so is not corrected away. Knowing when
              to distrust a forecast is what decides whether a warning goes out.
            </p>
            <p className="muted small">
              A <b>bust</b> is a forecast whose error lands in the tail of that variable’s
              own historical error distribution — the 90th percentile, computed on training
              data only. Each variable therefore has its own threshold, in its own units:
            </p>
            <ul className="taglist">
              {Object.entries(thr).map(([v, t]) => {
                const unit = variableUnit(v);
                return (
                  <li key={v} className="tag" title={v}>
                    {variableLabel(v)} ≥ {formatMetric(t, 2)}{unit ? ` ${unit}` : ""}
                  </li>
                );
              })}
            </ul>
          </section>

          <section className="card">
            <header className="card__head"><h3>How well it works</h3></header>
            <dl className="metrics metrics--compact metrics--hero">
              <div><dt>ROC-AUC</dt><dd>{formatMetric(clf.roc_auc, 3)}</dd></div>
              <div><dt>F1</dt><dd>{formatMetric(clf.f1, 3)}</dd></div>
              <div><dt>Brier</dt><dd>{formatMetric(clf.brier, 3)}</dd></div>
              <div>
                <dt>Held-out forecasts</dt>
                <dd>{typeof clf.n === "number" ? clf.n.toLocaleString() : "—"}</dd>
              </div>
            </dl>
            <p className="muted small">
              <b>Reading these:</b> ROC-AUC is the chance the model ranks a real bust above a
              non-bust — 0.5 is a coin flip, 1.0 is perfect. F1 balances how often its warnings
              are right against how many busts it catches. Brier is the average error in the
              probability itself, so lower is better.
            </p>
            <p className="muted small">
              Measured on the <b>{String(clf.split ?? "held-out")}</b> split — forecast
              cycles the model never trained on{" "}
              {typeof td.held_out_cycles === "number" ? <>— {td.held_out_cycles} of them</> : null}
              {typeof td.cycles === "number" ? <> out of {td.cycles} in total</> : null}.
            </p>
            <p className="muted small">
              One caution, found in our own data and being fixed rather than hidden: for
              temperature, humidity and soil moisture most large errors are the same
              districts being off in the same direction every day — a steady bias that
              ordinary bias correction already removes — rather than a forecast failing on
              a particular day. The next model defines a bust on bias-corrected error, so a
              warning means “today is unusual”, not “this district is always off”.
            </p>

            {bl.models?.length ? (
              <>
                <header className="card__head"><h4>Compared to what?</h4></header>
                <p className="muted small">
                  Every baseline is fitted on the training split alone and scored on the same
                  held-out rows as the model. Each is compared against climatology — simply
                  guessing the long-run bust rate every time. <b>0.000 means it does no better
                  than that guess</b>; higher is better.
                </p>
                <BaselineLadderTable baselines={data.baselines} />
                <p className="muted small">↓ lower is better · ↑ higher is better</p>
                {typeof bl.lead_bust_correlation?.test === "number" ? (
                  <p className="muted small">
                    Guessing from the lead day alone scores about zero skill, because busts do
                    not simply become more likely further out. The measured link between lead
                    day and bust is only {formatMetric(bl.lead_bust_correlation.train, 3)} on
                    training data and {formatMetric(bl.lead_bust_correlation.test, 3)} on
                    held-out data.
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
          {/* The API has been public and documented since the first deploy, and nothing on
            * the site said so. For a problem statement set by a forecasting centre, "you
            * could integrate with this" is a different claim from "look at this dashboard",
            * and it costs one link to make. */}
          <section className="card">
            <header className="card__head"><h3>Use it as a service, not just a page</h3></header>
            <p className="muted small">
              Every number on this site is read from a public, documented HTTP API — the same
              one this page calls. There is no private back channel and no figure baked into
              the frontend. <b>17 endpoints</b>, schema-checked with Pydantic and described by
              an auto-generated OpenAPI document, so a forecast desk could pull bust risk
              straight into its own tooling.
            </p>
            <ul className="notes">
              <li>
                <b>Interactive docs</b> — <a href={`${API_BASE}/docs`} target="_blank" rel="noopener noreferrer">/docs</a>{" "}
                (Swagger UI, every endpoint callable from the browser), and the raw schema at{" "}
                <a href={`${API_BASE}/openapi.json`} target="_blank" rel="noopener noreferrer">/openapi.json</a>.
              </li>
              <li>
                <b>The ones worth starting with</b> —{" "}
                <a className="mono" href={`${API_BASE}/api/regions/all`} target="_blank" rel="noopener noreferrer">/api/regions/all</a>{" "}
                for every district at every lead day,{" "}
                <a className="mono" href={`${API_BASE}/api/alerts`} target="_blank" rel="noopener noreferrer">/api/alerts</a>{" "}
                for what is above the watch level right now, and{" "}
                <a className="mono" href={`${API_BASE}/api/model/status`} target="_blank" rel="noopener noreferrer">/api/model/status</a>{" "}
                for the scores on this page, straight from the served model.
              </li>
              <li>
                <b>Fair warning</b> — this is a free-tier box that sleeps after 15 minutes
                idle, so a first call can take 30–50 seconds to wake it. That is the cost of
                the $0 hosting, not a fault.
              </li>
            </ul>
          </section>
        </div>

        <div className="page__col">
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
                <b>Real data end to end.</b> NOAA’s GEFS forecasts — historical re-runs for
                training, the live feed for today — checked against ERA5, a reconstructed
                record of what the weather actually did.
              </li>
              <li>
                <b>Leakage is tested for, and a leak we found is written down.</b> Bust
                thresholds are fitted on the training split only; whole forecast runs are kept
                together so the model is never tested on a run it partly trained on; and no
                observed day appears on both sides of the train/test split — each a test in the
                suite. A later audit still found one input that looked at an observation from
                after the forecast was issued; it is documented in the project’s known issues
                and removed in the next retrain, and scores are re-measured then.
              </li>
              <li>
                <b>Even the “truth” we score against is uncertain.</b> Two leading global
                weather records disagree with each other by roughly a quarter to a half of a
                bust threshold, measured over ~21,000 city-days. Stated up front rather than
                waiting to be asked.
              </li>
            </ul>
          </section>

          <section className="card">
            <header className="card__head"><h3>What it does not do</h3></header>
            <ul className="notes">
              <li>
                <b>A re-run archive, not the live model’s own history.</b> The model learns
                from NOAA’s reforecast — the forecast model re-run once a day for past years
                with 5 ensemble members — and some variables stop early there: 10 m wind at Day
                5, soil moisture at Day 3. Those are not scored beyond that, even though the
                live feed carries all ten days.
              </li>
              <li>
                <b>One geography now, not two.</b> The live feed used to sample the nearest
                grid point at 36 station locations, one per state and union territory, while
                the training archive covered all 666 districts as an area-weighted mean of
                every 0.25° cell each polygon overlaps. Since 2026-09-23 the live feed uses
                the same area-weighted method over the same 666 districts, so a live cycle
                and the archive behind the model describe geography the same way.
              </li>
              <li>
                GEFS runs 31 parallel forecasts; this uses 5 of them. How much those disagree
                is one of the model’s inputs, so that input is noisier here than it would be
                with all 31.
              </li>
              <li>
                ERA5 precipitation is weak over India relative to gauge-based products, and
                rainfall results carry that caveat. Rainfall is also the hardest variable
                here: most days have none at all, and the rest are dominated by a few extreme
                ones.
              </li>
              <li>
                <b>It does not send alerts to anyone.</b> Warnings appear in this interface
                and nowhere else — there is no SMS, email or push delivery, and nothing is
                dispatched to a duty desk. This is a decision-support view a forecaster
                reads, not a warning system that acts on its own.
              </li>
              <li>
                A bust is defined on <b>surface-variable error</b>, not the synoptic Z500
                criterion of Rodwell et al. (2013). That is a deliberate choice: surface
                error is what reaches agriculture and disaster response, whereas Z500 is
                what reaches meteorologists.
              </li>
            </ul>
          </section>

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

        <section className="card">
          <header className="card__head"><h3>Data &amp; attribution</h3></header>
          <ul className="notes">
            <li>
              <b>Forecasts</b> — NOAA GEFSv12, the reforecast archive (2000–2019, one 00 UTC
              cycle a day, 5 ensemble members) for training, the live operational feed for
              today. Public domain, via NOAA&apos;s Open Data bucket on AWS S3.
            </li>
            <li>
              <b>Observations</b> — ERA5 reanalysis from the Copernicus Climate Change Service
              (C3S), licensed CC-BY 4.0: from the Climate Data Store for training, and via the
              Open-Meteo Historical Weather API for recent days. Some training years also
              carry IMD–NCMRWF merged satellite-gauge rainfall (India Meteorological
              Department).
            </li>
            <li>
              <b>Other inputs</b> — the MJO index is NOAA PSL’s OLR-based MJO Index (OMI);
              district elevation is CGIAR-CSI SRTM 250 m, via the Open-Elevation API.
            </li>
            <li>
              <b>District boundaries</b> — GADM 4.1, India admin-2 (gadm.org). Two corrections
              are applied before use: Ladakh is reassigned out of Jammu &amp; Kashmir, since
              GADM predates the 2019 reorganisation; and disputed-territory features are kept
              and dissolved into their parent district rather than filtered out, so Jammu
              &amp; Kashmir, Ladakh and Arunachal Pradesh still appear on the map. Used here
              for a non-commercial hackathon prototype — formal licence review for any use
              beyond that is not yet done.
            </li>
            <li>
              <b>The project.</b> Built for Smart India Hackathon 2026, Problem Statement
              26079, set by NCMRWF (National Centre for Medium Range Weather Forecasting),
              Ministry of Earth Sciences.
            </li>
          </ul>
        </section>
        </div>
      </div>
    </main>
  );
}
