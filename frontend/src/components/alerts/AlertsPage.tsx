import { useEffect, useMemo, useState } from "react";
import type { Alert, RiskBand } from "../../api/types";
import { stamp } from "../../format";
import { useAlerts, useAllRegions } from "../../hooks/useDashboardData";
import { EmptyState, ErrorState, LoadingState, RiskBadge } from "../common/States";
import { retryingHint } from "../../lib/retryHint";
import { bandLabel } from "../../theme";
import { variableLabel } from "../../lib/displayNames";
import { dayLabel } from "../../lib/format";
import { alertsToCsv, csvFilename } from "../../lib/alertsCsv";
import { alertTotals } from "../../lib/alertTotals";
import { groupAlertsByDistrict, type GroupedAlert } from "../../lib/alertGroups";
import { stateIdOf } from "../../lib/stateNames";

const LIMIT = 200;

/**
 * How many rows to show before asking. The request still fetches all 200 and the summary
 * above still counts all 200 - this is only how much of the table is on screen at once.
 *
 * 200 rows measured 8,348px tall on a desktop and 9,123px on a phone, which is eleven
 * screens of near-identical rows below a summary that has already said what they add up
 * to. Nobody scrolls that, and on a bad forecast day every one of them is a real alert,
 * so it is not a problem the next retrain removes.
 */
const PAGE = 25;

/** A district row and a grouped-district row share every column the table draws, so one
 * render path handles both - only "days_on_alert" distinguishes them. */
type AlertRow = Alert | GroupedAlert;

const NO_STATE_NAMES = new Map<string, string>();

export function AlertsPage({ onSelect, filter, onFilter, stateNames }: {
  onSelect: (regionId: string, lead: number) => void;
  filter: RiskBand | undefined;
  onFilter: (b: RiskBand | undefined) => void;
  /**
   * state_id -> state_name, read from the map topology the dashboard already loads (see
   * lib/stateNames.ts). Passed down rather than fetched again here, so the state filter
   * costs this page nothing extra - and its labels cannot drift from the map's. Optional,
   * and falls back to the raw id (e.g. "IN-UT") until the topology has loaded.
   */
  stateNames?: Map<string, string>;
}) {
  const { data, isLoading, error, failureCount } = useAlerts(LIMIT, filter);
  const alerts = data?.alerts;
  const [shown, setShown] = useState(PAGE);
  const [search, setSearch] = useState("");
  const [stateFilter, setStateFilter] = useState("");
  const [groupByDistrict, setGroupByDistrict] = useState(false);

  // Back to the top of the list whenever a filter changes, so refining the search or
  // switching state does not silently keep an expansion made while looking at a wider list.
  useEffect(() => setShown(PAGE), [filter, search, stateFilter, groupByDistrict]);

  const stats = useMemo(() => summarise(alerts), [alerts]);
  // The summary counts every district-day, not the capped list (see lib/alertTotals). The
  // dashboard has already fetched this, so it comes from the query cache.
  const allRegions = useAllRegions().data;
  const totals = useMemo(() => alertTotals(allRegions), [allRegions]);

  const names = stateNames ?? NO_STATE_NAMES;
  const stateOptions = useMemo(() => {
    if (!alerts?.length) return [];
    const ids = new Set<string>();
    for (const a of alerts) ids.add(stateIdOf(a.region_id));
    return [...ids]
      .map((id) => ({ id, name: names.get(id) ?? id }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [alerts, names]);

  const filtered = useMemo(() => {
    if (!alerts) return alerts;
    const q = search.trim().toLowerCase();
    if (!q && !stateFilter) return alerts;
    return alerts.filter((a) => {
      if (stateFilter && stateIdOf(a.region_id) !== stateFilter) return false;
      if (q && !`${a.region_name ?? ""} ${a.region_id}`.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [alerts, search, stateFilter]);

  const rows: AlertRow[] | undefined = useMemo(() => {
    if (!filtered) return filtered;
    return groupByDistrict ? groupAlertsByDistrict(filtered) : filtered;
  }, [filtered, groupByDistrict]);

  const visible = rows?.slice(0, shown) ?? [];
  const remaining = (rows?.length ?? 0) - visible.length;
  const isFiltered = Boolean(search.trim() || stateFilter);

  const clearFilters = () => {
    setSearch("");
    setStateFilter("");
  };

  /**
   * Exports every alert fetched, not just the rows currently expanded or filtered on
   * screen - "Show 25 more" and the search box are both about reading, and someone
   * asking for the file wants the whole list the API sent.
   */
  const downloadCsv = () => {
    if (!alerts?.length) return;
    const blob = new Blob([alertsToCsv(alerts)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = csvFilename(data?.generated_at);
    a.click();
    // Freed on the next tick: revoking synchronously can cancel the download in Safari.
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  return (
    <main className="page page--wide">
      <header className="pagehead">
        <div>
          <h1 className="pagehead__title">Alerts</h1>
          <p className="pagehead__sub">
            Every region above the watch level in the current forecast run, worst first.
            {stats && stats.total >= LIMIT ? ` Capped at the ${LIMIT} most severe.` : ""}
          </p>
        </div>
        <div className="filters">
          {alerts?.length ? (
            <button type="button" className="chip" onClick={downloadCsv}>
              Download CSV
            </button>
          ) : null}
          {(["high", "medium"] as RiskBand[]).map((b) => (
            <button
              key={b}
              type="button"
              className={filter === b ? "chip chip--active" : "chip"}
              onClick={() => onFilter(filter === b ? undefined : b)}
            >
              {bandLabel(b)}
            </button>
          ))}
        </div>
      </header>

      {isLoading ? <LoadingState label="Loading alerts…" hint={retryingHint(failureCount)} /> : null}
      {error ? <ErrorState error={error} /> : null}
      {data && !data.model_trained ? <EmptyState title="No alerts yet" message={data.message} /> : null}
      {data?.model_trained && !alerts?.length ? (
        <EmptyState
          title="No watch or bust regions"
          message="Every region in the current cycle scored low."
        />
      ) : null}

      {stats ? (
        <div className="kpis kpis--flush">
          {totals ? (
            <>
              <Stat cap="bust" label="In the bust band" value={totals.bust.toLocaleString("en-IN")}
                note={<>district-days across all <b>{totals.days}</b> lead days</>} />
              <Stat cap="watch" label="In the watch band" value={totals.watch.toLocaleString("en-IN")}
                note={<><b>{totals.districts}</b> districts on alert at least once</>} />
            </>
          ) : (
            <>
              <Stat cap="bust" label="In the bust band" value={String(stats.bust)}
                note={<>of <b>{stats.total}</b> alerts shown</>} />
              <Stat cap="watch" label="In the watch band" value={String(stats.watch)}
                note={<>across <b>{stats.regions}</b> distinct regions</>} />
            </>
          )}
          <Stat cap="blue" label="Peak bust risk" value={`${(stats.peak.bust_probability * 100).toFixed(0)}%`}
            note={<><b>{stats.peak.region_name ?? stats.peak.region_id}</b> · {dayLabel(stats.peak.lead_time_days)}</>} />
          {totals ? (
            <Stat cap="blue" label="Most common cause" value={variableLabel(totals.topDriver?.[0]) || "—"}
              note={totals.topDriver
                ? <>the main cause in <b>{totals.topDriver[1].toLocaleString("en-IN")}</b> of {(totals.bust + totals.watch).toLocaleString("en-IN")} alert district-days</>
                : <>no dominant variable recorded</>} />
          ) : (
            <Stat cap="blue" label="Most common cause" value={variableLabel(stats.topDriver?.[0]) || "—"}
              note={stats.topDriver
                ? <>the main cause in <b>{stats.topDriver[1]}</b> of {stats.total}</>
                : <>no dominant variable recorded</>} />
          )}
        </div>
      ) : null}

      {alerts?.length ? (
        <section className="card card--table">
          <div className="alerts-toolbar">
            <div className="alerts-search">
              <input
                type="search"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Find a district or state…"
                aria-label="Find a district or state"
              />
            </div>
            <select
              className="alerts-statefilter"
              value={stateFilter}
              onChange={(e) => setStateFilter(e.target.value)}
              aria-label="Filter by state"
            >
              <option value="">All states</option>
              {stateOptions.map((s) => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </select>
            <button
              type="button"
              className={groupByDistrict ? "chip chip--active" : "chip"}
              aria-pressed={groupByDistrict}
              onClick={() => setGroupByDistrict((g) => !g)}
            >
              One row per district
            </button>
          </div>

          {rows && rows.length === 0 ? (
            <EmptyState
              title="No matches"
              message="No alerts match this search or state filter."
              action={
                <button type="button" className="chip" onClick={clearFilters}>
                  Clear filters
                </button>
              }
            />
          ) : (
            <>
              <div className="tablewrap">
                <table className="dtable dtable--spread">
                  <thead>
                    <tr>
                      <th>Region</th>
                      <th className="dtable__num">Lead</th>
                      {/* Dropped on a phone: the date is lead + cycle, and the band is a
                        * restatement of the risk column beside it. See .dtable__opt. */}
                      <th className="dtable__opt">Valid date</th>
                      <th className="dtable__num">Bust risk</th>
                      <th className="dtable__opt">Band</th>
                      <th>Main cause</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visible.map((a) => (
                      <tr
                        key={`${a.region_id}::${a.lead_time_days}`}
                        className="dtable__row"
                        tabIndex={0}
                        role="button"
                        onClick={() => onSelect(a.region_id, a.lead_time_days)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            onSelect(a.region_id, a.lead_time_days);
                          }
                        }}
                      >
                        <td className="dtable__strong">
                          {a.region_name ?? a.region_id}
                          {"days_on_alert" in a ? (
                            <span className="muted small">
                              {" "}· on alert {a.days_on_alert} day{a.days_on_alert === 1 ? "" : "s"}
                            </span>
                          ) : null}
                        </td>
                        <td className="dtable__num mono">{dayLabel(a.lead_time_days)}</td>
                        <td className="dtable__opt mono muted">{a.valid_date ?? "—"}</td>
                        <td className="dtable__num mono dtable__strong">
                          {(a.bust_probability * 100).toFixed(0)}%
                        </td>
                        <td className="dtable__opt"><RiskBadge band={a.risk_band} /></td>
                        <td className="muted">{variableLabel(a.dominant_variable) || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {remaining > 0 ? (
                <div className="tablemore">
                  <button type="button" className="chip" onClick={() => setShown((n) => n + PAGE)}>
                    Show {Math.min(PAGE, remaining)} more
                  </button>
                  <button type="button" className="chip" onClick={() => setShown(rows!.length)}>
                    Show all {rows!.length}
                  </button>
                  <span className="muted small">
                    Showing <b>{visible.length}</b> of <b>{rows!.length}</b>, worst first
                    {isFiltered ? <> · filtered from {alerts.length}</> : null}
                  </span>
                </div>
              ) : isFiltered && rows?.length ? (
                <div className="tablemore">
                  <span className="muted small">
                    Showing <b>{visible.length}</b> of <b>{rows.length}</b>, filtered from {alerts.length}
                  </span>
                </div>
              ) : null}
            </>
          )}
        </section>
      ) : null}

      {data?.generated_at ? (
        <p className="pagefoot">Generated {stamp(data.generated_at)}</p>
      ) : null}
    </main>
  );
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

function summarise(alerts: Alert[] | undefined) {
  if (!alerts?.length) return null;

  const drivers = new Map<string, number>();
  let bust = 0;
  let watch = 0;
  const regions = new Set<string>();
  let peak = alerts[0];

  for (const a of alerts) {
    if (a.risk_band === "high") bust += 1;
    else if (a.risk_band === "medium") watch += 1;
    regions.add(a.region_id);
    if (a.bust_probability > peak.bust_probability) peak = a;
    if (a.dominant_variable) {
      drivers.set(a.dominant_variable, (drivers.get(a.dominant_variable) ?? 0) + 1);
    }
  }

  const topDriver = [...drivers.entries()].sort((a, b) => b[1] - a[1])[0] ?? null;
  return { total: alerts.length, bust, watch, regions: regions.size, peak, topDriver };
}
