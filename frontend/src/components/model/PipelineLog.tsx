import { translate } from "../../i18n";
import { useState } from "react";
import { rich } from "../../i18n/rich";

import type { IngestRunRow } from "../../api/types";
import { stampShort } from "../../format";
import { useIngestRuns } from "../../hooks/useDashboardData";

/** Rows shown before the reader has to ask for more. A healthy pipeline's log is the
 * same line over and over - five is enough to see that pattern (or a break in it)
 * without the tail of near-identical successes pushing the rest of the Model tab down. */
const VISIBLE = 5;

const LABEL: Record<string, () => string> = {
  forecast: () => translate("pipe.forecast"),
  observations_provisional: () => translate("pipe.obsProv"),
  observations_final: () => translate("pipe.obsFinal"),
};

const BAND: Record<string, string> = {
  complete: "low",
  skipped: "medium",
  failed: "high",
  running: "medium",
};

function when(iso: string | null): string {
  return stampShort(iso);
}

export function PipelineLog() {
  const { data, error, isLoading } = useIngestRuns(25);
  const [showAll, setShowAll] = useState(false);
  if (isLoading) return null;

  if (error || !data?.runs?.length) return null;

  const runs: IngestRunRow[] = data.runs;
  const refused = runs.filter((r) => r.status === "failed" || r.status === "skipped").length;
  const visible = showAll ? runs : runs.slice(0, VISIBLE);

  return (
    <section className="card">
      <header className="card__head"><h3>{translate("pipe.title")}</h3></header>
      <p className="muted small">
        {rich("pipe.intro", { n: runs.length }, [<b />])}
        {refused > 0 ? translate("pipe.refused", { refused, n: runs.length }) : null}
      </p>
      <div className="tablewrap">
        <table className="dtable">
          <thead>
            <tr>
              <th>{translate("pipe.colStarted")}</th><th>{translate("pipe.colWhat")}</th>
              <th>{translate("pipe.colTarget")}</th><th>{translate("pipe.colStatus")}</th>
              <th>{translate("pipe.colRows")}</th><th>{translate("pipe.colTook")}</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <tr key={r.id}>
                <td className="mono small">{when(r.started_at)}</td>
                <td>{LABEL[r.kind]?.() ?? r.kind}</td>
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
      {runs.length > VISIBLE ? (
        <div className="tablemore">
          <button type="button" className="chip" onClick={() => setShowAll((s) => !s)}>
            {showAll ? translate("pipe.fewer") : translate("pipe.all", { n: runs.length })}
          </button>
        </div>
      ) : null}
      {/* The raw detail line is for whoever is debugging the feed, and it is long - a
        * page of HTTP 429 bodies when the observation API rate-limits. One click away,
        * not gone: partial pulls are a documented known issue, not something to hide. */}
      {runs[0]?.detail ? (
        <details className="pipelog__detail">
          <summary className="muted small">
            {translate("pipe.rawLog")}{/fail/i.test(runs[0].detail) ? translate("pipe.rawFail") : ""}
          </summary>
          <p className="muted small mono" style={{ wordBreak: "break-word" }}>{runs[0].detail}</p>
        </details>
      ) : null}
      {runs.find((r) => r.error) ? (
        <p className="muted small">
          {translate("pipe.lastError", { err: runs.find((r) => r.error)?.error?.slice(0, 240) ?? "" })}
        </p>
      ) : null}
    </section>
  );
}
