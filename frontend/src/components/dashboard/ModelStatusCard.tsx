import { useModelStatus } from "../../hooks/useDashboardData";
import { useLiveStore } from "../../store/liveStore";
import { ErrorState, LoadingState } from "../common/States";

export function ModelStatusCard() {
  const { data, isLoading, error } = useModelStatus();
  const connection = useLiveStore((s) => s.connectionStatus);
  const training = useLiveStore((s) => s.trainingInProgress);

  if (isLoading) return <section className="card"><LoadingState /></section>;
  if (error) return <section className="card"><ErrorState error={error} /></section>;
  if (!data) return null;

  const clf = data.validation_metrics?.classifier ?? {};
  const num = (v: unknown) => (typeof v === "number" ? v.toFixed(3) : "—");

  return (
    <section className="card">
      <header className="card__head">
        <h3>Model</h3>
        <span className={`badge badge--${connection === "open" ? "live" : "offline"}`}>
          {connection === "open" ? "live" : connection}
        </span>
      </header>

      {!data.model_trained ? (
        <p className="muted">{data.message}</p>
      ) : (
        <>
          <dl className="metrics">
            <div className="metrics__wide">
              <dt>Run</dt>
              <dd className="mono small" title={data.current_run_id ?? undefined}>
                {data.current_run_id}
              </dd>
            </div>
          </dl>
          <dl className="metrics metrics--compact">
            <div><dt>Variables</dt><dd>{data.modelled_variables.length}</dd></div>
            <div><dt>Rows</dt><dd>{(data.data_volume.total_rows ?? 0).toLocaleString()}</dd></div>
            <div><dt>Batches</dt><dd>{data.data_volume.batches ?? 0}</dd></div>
          </dl>
          <dl className="metrics metrics--compact">
            <div><dt>ROC-AUC</dt><dd>{num(clf.roc_auc)}</dd></div>
            <div><dt>F1</dt><dd>{num(clf.f1)}</dd></div>
            <div><dt>Precision</dt><dd>{num(clf.precision)}</dd></div>
            <div><dt>Recall</dt><dd>{num(clf.recall)}</dd></div>
          </dl>
          <p className="muted small">
            Bust classifier on the <b>{String(clf.split ?? "unknown")}</b> split
            {data.explanation_method ? ` · explanations: ${data.explanation_method}` : ""}
          </p>
          {Object.keys(data.skipped_variables).length ? (
            <p className="muted small">
              Not modelled: {Object.entries(data.skipped_variables).map(([v, why]) => `${v} (${why})`).join("; ")}
            </p>
          ) : null}
        </>
      )}

      {training || data.training_in_progress ? <p className="notice">Retraining in progress…</p> : null}
      {data.last_training_error ? <p className="notice notice--error">Last training error: {data.last_training_error}</p> : null}
    </section>
  );
}
