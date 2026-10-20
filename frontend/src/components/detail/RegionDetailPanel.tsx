import { rich } from "../../i18n/rich";
import { translate } from "../../i18n";
import { useState } from "react";
import { useRegionDetail } from "../../hooks/useDashboardData";
import { featureLabel, variableLabel } from "../../lib/displayNames";
import { EmptyState, ErrorState, LoadingState, RiskBadge } from "../common/States";
import { CopyLinkButton } from "../common/CopyLinkButton";
import { retryingHint } from "../../lib/retryHint";
import { BustProbabilityCurve } from "./BustProbabilityCurve";
import { ShapFactorsList } from "./ShapFactorsList";
import { VariableTrajectoryChart } from "./VariableTrajectoryChart";

export function RegionDetailPanel({
  regionId,
  onClose,
  riskCuts,
}: {
  regionId: string | null;
  onClose: () => void;
  riskCuts?: { medium: number; high: number };
}) {
  const { data, isLoading, error, failureCount } = useRegionDetail(regionId);
  // Keyed by region so a newly opened district starts on its own leading driver, not on
  // whichever tab the previous district was left on.
  const [picked, setPicked] = useState<{ region: string; variable: string } | null>(null);
  const activeVariable = picked && picked.region === regionId ? picked.variable : null;

  if (!regionId) {
    return (
      <aside className="panel">
        <EmptyState title={translate("worst.emptyTitle")} message={translate("worst.emptyMsg")} />
      </aside>
    );
  }
  if (isLoading) return <aside className="panel"><LoadingState label="Loading region…" hint={retryingHint(failureCount)} /></aside>;
  if (error) return <aside className="panel"><ErrorState error={error} /></aside>;
  if (!data) return null;

  if (!data.model_trained) {
    return (
      <aside className="panel">
        <EmptyState title={data.region_name ?? regionId} message={data.message} />
      </aside>
    );
  }

  const available = data.variables.filter((v) => v.available);
  const unavailable = data.variables.filter((v) => !v.available);
  const worst = [...data.bust_probability_curve].sort((a, b) => b.bust_probability - a.bust_probability)[0];
  const driver = available.find((v) => v.variable === worst?.dominant_variable) ?? null;
  const current = available.find((v) => v.variable === activeVariable) ?? driver ?? available[0];
  const showingDriver = driver != null && current?.variable === driver.variable;

  return (
    <aside className="panel">
      <header className="panel__head">
        <div>
          <h2>{data.region_name ?? data.region_id}</h2>
          <p className="muted small">
            {data.region_id}
            {data.init_date ? translate("det.cycle", { date: data.init_date }) : ""}
          </p>
        </div>
        <div className="panel__actions">
          {/* The URL carries this district and the lead day, so the panel is the one
            * place a shareable link is genuinely worth offering. */}
          <CopyLinkButton />
          <button type="button" className="btn btn--ghost" onClick={onClose} aria-label={translate("det.close")}>×</button>
        </div>
      </header>

      {data.message ? <p className="muted">{data.message}</p> : null}

      {worst ? (
        <section className="panel__section">
          <h3>{translate("det.peak")}</h3>
          <div className="peak">
            <span className="peak__value">{(worst.bust_probability * 100).toFixed(1)}%</span>
            <div className="peak__meta">
              <RiskBadge band={worst.risk_band} />
              <span className="muted small">{translate("det.atLead", { d: worst.lead_time_days })}</span>
            </div>
          </div>
          {worst.dominant_variable ? (
            <p className="muted small">{translate("curve.driven", { v: variableLabel(worst.dominant_variable).toLowerCase() })}</p>
          ) : null}
        </section>
      ) : null}

      <section className="panel__section">
        <h3>{translate("det.shapTitle")}</h3>
        {data.top_factors.length ? (
          <p className="panel__why">
            {translate("det.leans", { list: data.top_factors.slice(0, 3).map((f) => featureLabel(f.feature)).join(", ") })}
          </p>
        ) : null}
        <ShapFactorsList factors={data.top_factors} method={data.top_factors_method} />
      </section>

      <section className="panel__section">
        <h3>
          {showingDriver
            ? translate("det.leading", { v: variableLabel(current.variable).toLowerCase() })
            : translate("det.fcVsObs")}
        </h3>
        {showingDriver ? (
          <p className="muted small">
            {translate("det.pushing", { d: worst.lead_time_days })}
          </p>
        ) : null}
        {available.length ? (
          <>
            <div className="tabs" role="tablist">
              {available.map((v) => (
                <button
                  key={v.variable}
                  role="tab"
                  aria-selected={current?.variable === v.variable}
                  className={current?.variable === v.variable ? "tab tab--active" : "tab"}
                  onClick={() => setPicked({ region: regionId, variable: v.variable })}
                  title={v.variable}
                >
                  {variableLabel(v.variable)}
                  {driver?.variable === v.variable ? <span className="tab__chip">{translate("det.driverChip")}</span> : null}
                </button>
              ))}
            </div>
            {current ? (
              <>
                <VariableTrajectoryChart series={current} />
                {current.max_lead_day ? (
                  <p className="muted small">
                    {translate("det.maxLead", { max: current.max_lead_day, v: variableLabel(current.variable).toLowerCase() })}
                  </p>
                ) : null}
                <dl className="metrics">
                  <div><dt>{translate("det.mae")}</dt><dd>{fmt(current.model_mae)} {current.unit ?? ""}</dd></div>
                  <div><dt>{translate("det.rmse")}</dt><dd>{fmt(current.model_rmse)}</dd></div>
                  <div><dt>{translate("det.r2")}</dt><dd>{fmt(current.model_r2)}</dd></div>
                  <div><dt>{translate("det.thr")}</dt><dd>{fmt(current.bust_threshold)} {current.unit ?? ""}</dd></div>
                </dl>
                <p className="muted small">
                  {current.unit ? translate("det.maeExplainUnit", { u: current.unit }) : translate("det.maeExplain")}
                </p>
                <p className="muted small">
                  {rich("det.measured", { split: current.metrics_split ?? translate("map.unknown") }, [<b />])}
                </p>
              </>
            ) : null}
          </>
        ) : (
          <p className="muted">{translate("det.noVar")}</p>
        )}
        {unavailable.length ? (
          <p className="muted small">
            {translate("det.notModelled", { list: unavailable.map((v) => variableLabel(v.variable)).join(", ") })}
          </p>
        ) : null}
      </section>

      <section className="panel__section">
        <h3>{translate("det.curve")}</h3>
        <BustProbabilityCurve points={data.bust_probability_curve} cuts={riskCuts} />
      </section>

      <section className="panel__section">
        <h3>{translate("det.similar")}</h3>
        <p className="muted small">
          {translate("det.similarBody")}
        </p>
      </section>
    </aside>
  );
}

const fmt = (v: number | null | undefined) =>
  v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(3);
