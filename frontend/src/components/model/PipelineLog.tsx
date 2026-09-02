import type { IngestRunRow } from "../../api/types";
import { useIngestRuns } from "../../hooks/useDashboardData";

/**
 * What the pipeline actually did, newest first.
 *
 * Every attempt is listed, including the ones that were skipped or refused. A cycle NOAA
 * served too incompletely to trust is rejected rather than published — a short rainfall
 * *sum* is roughly half the real accumulation, and rainfall drives most busts — so a
 * refusal is the system working, not failing. Showing only the successes would be the
 * same convenient fiction this project exists to avoid.
 *
 * Every row is a database record written when the run happened. Nothing here is
 * generated for display.
 */
const LABEL: Record<string, string> = {
  forecast: "GEFS cycle",
  observations_provisional: "observations (provisional)",
  observations_final: "observations (final)",
};

const BAND: Record<string, string> = {
  complete: "low",       // green — it worked
  skipped: "medium",     // amber — nothing to do, or already present
  failed: "high",        // red — refused or errored, and shown as such
  running: "medium",
};

function when(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime())
    ? "—"
    : d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

export function PipelineLog() {
  const { data, error, isLoading } = useIngestRuns(25);
  if (isLoading) return null;
  // Optional by design: a deployment without this endpoint shows the rest of the page.
  if (error || !data?.runs?.length) return null;

  const runs: IngestRunRow[] = data.runs;
  const refused = runs.filter((r) => r.status === "failed" || r.status === "skipped").length;

  return (
    <section className="card">
      <header className="card__head"><h3>Pipeline activity</h3></header>
      <p className="muted small">
        The last {runs.length} attempts by the ingestion pipeline, newest first — each one a
        record written when it ran. <b>Refused and skipped runs are shown too.</b> A cycle
        delivered too incompletely to trust is rejected rather than published, because a
        short rainfall total would understate the quantity that drives most busts. A
        refusal here is the guard working.
        {refused > 0 ? <> {refused} of these {runs.length} did not ingest.</> : null}
      </p>
      <div className="tablewrap">
        <table className="dtable">
          <thead>
            <tr>
              <th>started</th><th>what</th><th>target</th><th>status</th>
              <th>rows</th><th>took</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id}>
                <td className="mono small">{when(r.started_at)}</td>
                <td>{LABEL[r.kind] ?? r.kind}</td>
                <td className="mono small">{r.target ?? "—"}</td>
                <td>
                  <span className={`dot dot--${BAND[r.status] ?? "medium"}`} />{" "}
                  {r.status}
                  {r.trigger && r.trigger !== "schedule" ? (
                    <span className="muted small"> · {r.trigger}</span>
                  ) : null}
                </td>
                <td className="mono">{r.rows_ingested ? r.rows_ingested.toLocaleString() : "—"}</td>
                <td className="mono small">{r.seconds != null ? `${r.seconds}s` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* The detail line carries the evidence: transport used, steps retrieved of steps
          expected, bytes, duration. It is what distinguishes a real pull from a claim. */}
      {runs[0]?.detail ? (
        <p className="muted small mono" style={{ wordBreak: "break-word" }}>
          latest: {runs[0].detail}
        </p>
      ) : null}
      {runs.find((r) => r.error) ? (
        <p className="muted small">
          Most recent error: {runs.find((r) => r.error)?.error?.slice(0, 240)}
        </p>
      ) : null}
    </section>
  );
}
