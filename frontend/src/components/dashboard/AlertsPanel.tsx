import { useAlerts } from "../../hooks/useDashboardData";
import type { RiskBand } from "../../api/types";
import { EmptyState, ErrorState, LoadingState, RiskBadge } from "../common/States";

export function AlertsPanel({ onSelect, filter, onFilter }: {
  onSelect: (regionId: string, lead: number) => void;
  filter: RiskBand | undefined;
  onFilter: (b: RiskBand | undefined) => void;
}) {
  const { data, isLoading, error } = useAlerts(25, filter);

  return (
    <section className="card">
      <header className="card__head">
        <h3>Alerts</h3>
        <div className="filters">
          {(["high", "medium"] as RiskBand[]).map((b) => (
            <button
              key={b}
              type="button"
              className={filter === b ? "chip chip--active" : "chip"}
              onClick={() => onFilter(filter === b ? undefined : b)}
            >
              {b}
            </button>
          ))}
        </div>
      </header>

      {isLoading ? <LoadingState /> : null}
      {error ? <ErrorState error={error} /> : null}
      {data && !data.model_trained ? <EmptyState title="No alerts yet" message={data.message} /> : null}
      {data?.model_trained && !data.alerts.length ? (
        <EmptyState title="No medium or high risk regions" message="Every region in the current cycle scored low risk." />
      ) : null}

      {data?.alerts.length ? (
        <ul className="alerts">
          {data.alerts.map((a) => (
            <li key={a.alert_id}>
              <button type="button" className="alert" onClick={() => onSelect(a.region_id, a.lead_time_days)}>
                <span className="alert__region">{a.region_name ?? a.region_id}</span>
                <span className="alert__lead">D{a.lead_time_days}</span>
                <span className="alert__prob">{(a.bust_probability * 100).toFixed(0)}%</span>
                <RiskBadge band={a.risk_band} />
                {a.dominant_variable ? <span className="alert__driver">{a.dominant_variable}</span> : null}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
