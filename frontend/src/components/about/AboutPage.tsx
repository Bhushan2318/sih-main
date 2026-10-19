import { rich } from "../../i18n/rich";
import { translate } from "../../i18n";
import { API_BASE } from "../../api/client";
import { useModelStatus } from "../../hooks/useDashboardData";
import { variableLabel, variableUnit } from "../../lib/displayNames";
import { formatMetric } from "../../lib/format";
import { ErrorState, LoadingState } from "../common/States";
import { retryingHint } from "../../lib/retryHint";
import { BaselineLadderTable } from "../model/BaselineLadderTable";

export function AboutPage({ onReplay }: { onReplay: () => void }) {
  const { data, isLoading, error, failureCount } = useModelStatus();

  if (isLoading) return <main className="page page--wide"><LoadingState label={translate("state.loading")} hint={retryingHint(failureCount)} /></main>;
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
          <h1 className="pagehead__title">{translate("ab.title")}</h1>
          <p className="pagehead__sub">
            {translate("ab.sub")}
          </p>
        </div>
      </header>

      <div className="notice">
        <p style={{ margin: 0 }}>
          {rich("ab.start", undefined, [<b />])}
        </p>
        <p style={{ margin: "0.6rem 0 0" }}>
          <button type="button" className="chip" onClick={onReplay}>
            {translate("ab.openReplay")}
          </button>
        </p>
      </div>

      <div className="page--cols">
        <div className="page__col">
          <section className="card">
            <header className="card__head"><h3>{translate("ab.qTitle")}</h3></header>
            <p className="muted">
              {translate("ab.q1")}
            </p>
            <p><b>{translate("ab.q2")}</b></p>
            <p className="muted small">
              {rich("ab.q3", undefined, [<i />])}
            </p>
            <p className="muted small">
              {rich("ab.q4", undefined, [<b />])}
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
            <header className="card__head"><h3>{translate("ab.howTitle")}</h3></header>
            <dl className="metrics metrics--compact metrics--hero">
              <div><dt>ROC-AUC</dt><dd>{formatMetric(clf.roc_auc, 3)}</dd></div>
              <div><dt>F1</dt><dd>{formatMetric(clf.f1, 3)}</dd></div>
              <div><dt>Brier</dt><dd>{formatMetric(clf.brier, 3)}</dd></div>
              <div>
                <dt>{translate("ab.heldOutN")}</dt>
                <dd>{typeof clf.n === "number" ? clf.n.toLocaleString() : "—"}</dd>
              </div>
            </dl>
            <p className="muted small">
              {rich("ab.reading", undefined, [<b />])}
            </p>
            <p className="muted small">
              {rich("ab.measured", { split: String(clf.split ?? translate("miss.heldOut")) }, [<b />])}{" "}
              {typeof td.held_out_cycles === "number" ? translate("ab.measuredN", { n: td.held_out_cycles }) : null}
              {typeof td.cycles === "number" ? translate("ab.measuredOf", { n: td.cycles }) : null}.
            </p>
            <p className="muted small">
              {translate("ab.caution")}
            </p>

            {bl.models?.length ? (
              <>
                <header className="card__head"><h4>{translate("ab.cmpTitle")}</h4></header>
                <p className="muted small">
                  {rich("ab.cmp", undefined, [<b />])}
                </p>
                <BaselineLadderTable baselines={data.baselines} />
                <p className="muted small">{translate("mp.arrows")}</p>
                {typeof bl.lead_bust_correlation?.test === "number" ? (
                  <p className="muted small">
                    {translate("ab.leadCorr", { train: formatMetric(bl.lead_bust_correlation.train, 3), test: formatMetric(bl.lead_bust_correlation.test, 3) })}
                  </p>
                ) : null}
              </>
            ) : (
              <p className="muted small">
                {translate("ab.noLadder")}
              </p>
            )}
          </section>
          {/* The API has been public and documented since the first deploy, and nothing on
            * the site said so. For a problem statement set by a forecasting centre, "you
            * could integrate with this" is a different claim from "look at this dashboard",
            * and it costs one link to make. */}
          <section className="card">
            <header className="card__head"><h3>{translate("ab.svcTitle")}</h3></header>
            <p className="muted small">
              {rich("ab.svc", undefined, [<b />])}
            </p>
            <ul className="notes">
              <li>
                {rich("ab.docs", undefined, [<b />, <a href={`${API_BASE}/docs`} target="_blank" rel="noopener noreferrer" />, <a href={`${API_BASE}/openapi.json`} target="_blank" rel="noopener noreferrer" />])}
              </li>
              <li>
                {rich("ab.endpoints", undefined, [<b />, <a className="mono" href={`${API_BASE}/api/regions/all`} target="_blank" rel="noopener noreferrer" />, <a className="mono" href={`${API_BASE}/api/alerts`} target="_blank" rel="noopener noreferrer" />, <a className="mono" href={`${API_BASE}/api/model/status`} target="_blank" rel="noopener noreferrer" />])}
              </li>
              <li>
                {rich("ab.fair", undefined, [<b />])}
              </li>
            </ul>
          </section>
        </div>

        <div className="page__col">
          <section className="card">
            <header className="card__head"><h3>{translate("ab.trustTitle")}</h3></header>
            <ul className="notes">
              <li>
                {rich("ab.t1", undefined, [<b />, <i />])}
              </li>
              <li>
                {rich("ab.t2", undefined, [<b />])}
              </li>
              <li>
                {rich("ab.t3", undefined, [<b />])}
              </li>
              <li>
                {rich("ab.t4", undefined, [<b />])}
              </li>
            </ul>
          </section>

          <section className="card">
            <header className="card__head"><h3>{translate("ab.notTitle")}</h3></header>
            <ul className="notes">
              <li>
                {rich("ab.n1", undefined, [<b />])}
              </li>
              <li>
                {rich("ab.n2", undefined, [<b />])}
              </li>
              <li>
                {translate("ab.n3")}
              </li>
              <li>
                {translate("ab.n4")}
              </li>
              <li>
                {rich("ab.n5", undefined, [<b />])}
              </li>
              <li>
                {rich("ab.n6", undefined, [<b />])}
              </li>
            </ul>
          </section>

        <section className="card">
          <header className="card__head"><h3>{translate("ab.runsTitle")}</h3></header>
          <p className="muted small">
            {rich("ab.runs", undefined, [<b />, <i />])}
          </p>
        </section>

        <section className="card">
          <header className="card__head"><h3>{translate("ab.dataTitle")}</h3></header>
          <ul className="notes">
            <li>
              {rich("ab.d1", undefined, [<b />])}
            </li>
            <li>
              {rich("ab.d2", undefined, [<b />, <a href="https://open-meteo.com/" target="_blank" rel="noopener noreferrer" />])}
            </li>
            <li>
              {rich("ab.d3", undefined, [<b />])}
            </li>
            <li>
              {rich("ab.d4", undefined, [<b />])}
            </li>
            <li>
              {rich("ab.d5", undefined, [<b />, <a href="/third-party-licenses.txt" target="_blank" rel="noopener noreferrer" />, <a href="https://github.com/Bhushan2318/sih-main/blob/main/THIRD_PARTY_NOTICES.md" target="_blank" rel="noopener noreferrer" />])}
            </li>
            <li>
              {rich("ab.d6", undefined, [<b />])}
            </li>
          </ul>
        </section>
        </div>
      </div>
    </main>
  );
}
