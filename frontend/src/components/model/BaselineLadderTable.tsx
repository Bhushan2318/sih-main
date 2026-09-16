import type { ModelStatusResponse } from "../../api/types";
import { LEAD_DAY_RUNG_NAME } from "../../lib/baselineLadder";

type Baselines = ModelStatusResponse["baselines"];

const n3 = (v: unknown, dp = 4) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(dp) : "—";

/** The ladder of baselines every rung is scored against, shared by the About
 * page's full writeup and the dashboard's compact callout. */
export function BaselineLadderTable({ baselines, flagLeadDay = false }: {
  baselines: Baselines;
  /** Outline the lead_day row instead of (or alongside) the served model's is-active row. */
  flagLeadDay?: boolean;
}) {
  const models = baselines?.models;
  if (!models?.length) return null;

  return (
    <div className="tablewrap">
      <table className="dtable">
        <thead>
          <tr>
            <th>model</th><th>Brier ↓</th><th>skill vs climatology ↑</th><th>ROC-AUC ↑</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const isLeadDay = flagLeadDay && m.name === LEAD_DAY_RUNG_NAME;
            const cls = [m.is_model && "is-active", isLeadDay && "is-flagged"]
              .filter(Boolean)
              .join(" ") || undefined;
            return (
              <tr key={m.name} className={cls}>
                <td>{m.is_model ? <b>{m.name}</b> : m.name}</td>
                <td className="mono">{n3(m.brier)}</td>
                <td className="mono">{n3(m.bss)}</td>
                <td className="mono">{n3(m.roc_auc)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
