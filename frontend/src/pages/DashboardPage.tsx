import { useEffect, useRef, useState } from "react";
import type { Topology } from "topojson-specification";
import type { RiskBand } from "../api/types";
import { AlertsPanel } from "../components/dashboard/AlertsPanel";
import { BustSummaryChart } from "../components/dashboard/BustSummaryChart";
import { FeedFreshness } from "../components/dashboard/FeedFreshness";
import { ModelStatusCard } from "../components/dashboard/ModelStatusCard";
import { RegionDetailPanel } from "../components/detail/RegionDetailPanel";
import { EmptyState, ErrorState, LoadingState } from "../components/common/States";
import { IndiaChoroplethMap, loadTopology } from "../components/map/IndiaChoroplethMap";
import { LeadDaySelector } from "../components/map/LeadDaySelector";
import { MapLegend } from "../components/map/MapLegend";
import { ReplayView } from "../components/replay/ReplayView";
import { UploadPanel } from "../components/upload/UploadPanel";
import { useAllRegions, useModelStatus } from "../hooks/useDashboardData";
import { useLiveSocket } from "../hooks/useLiveSocket";

export function DashboardPage() {
  useLiveSocket();

  const [view, setView] = useState<"live" | "replay">("live");
  const [leadDay, setLeadDay] = useState(1);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [alertFilter, setAlertFilter] = useState<RiskBand | undefined>(undefined);
  const [topology, setTopology] = useState<Topology | null>(null);
  const [topoError, setTopoError] = useState<unknown>(null);

  const regionsQuery = useAllRegions();
  const statusQuery = useModelStatus();

  useEffect(() => {
    loadTopology().then(setTopology).catch(setTopoError);
  }, []);

  // All 10 lead days arrive in one payload; picking the current one is a local lookup, so
  // moving the lead-day selector never triggers a request or a loading state.
  const allRegions = regionsQuery.data;
  const regions =
    allRegions?.days.find((d) => d.lead_time_days === leadDay) ?? allRegions?.days[0];
  const riskCuts = statusQuery.data?.thresholds?.risk_band_cuts;

  // A 06/12/18 UTC cycle has no day-1 forecast, so the default lead day can point at
  // nothing. Move once, to the first day this cycle actually covers, rather than showing
  // an empty map. The ref keeps this from fighting a deliberate choice afterwards.
  const autoLeadPicked = useRef(false);
  const available = regions?.available_lead_days;
  useEffect(() => {
    if (autoLeadPicked.current || !available?.length) return;
    if (!available.includes(leadDay)) setLeadDay(available[0]);
    autoLeadPicked.current = true;
  }, [available, leadDay]);

  return (
    <div className="app">
      <header className="app__head">
        <div>
          <h1>ForecastGuard AI</h1>
          <p className="muted small">
            Medium-range forecast bust detection · SIH 2026 PS 26079 (NCMRWF)
          </p>
        </div>
        <div className="app__headright">
          {view === "live" && regions?.model_trained && regions.init_date ? (
            <p className="muted small">
              Forecast cycle <b>{regions.init_date}</b>
              {regions.valid_date ? ` → valid ${regions.valid_date}` : ""}
            </p>
          ) : null}
          <div className="viewtabs" role="tablist" aria-label="View">
            <button
              type="button"
              role="tab"
              aria-selected={view === "live"}
              className={view === "live" ? "viewtab is-active" : "viewtab"}
              onClick={() => setView("live")}
            >
              Live
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={view === "replay"}
              className={view === "replay" ? "viewtab is-active" : "viewtab"}
              onClick={() => setView("replay")}
            >
              Replay a real bust
            </button>
          </div>
        </div>
      </header>

      {view === "live" ? <FeedFreshness /> : null}

      {view === "replay" ? (
        <main className="app__body app__body--replay">
          <ReplayView topology={topology} />
        </main>
      ) : (
      <main className="app__body">
        <section className="map-column">
          <div className="map-toolbar">
            <LeadDaySelector
              value={leadDay}
              onChange={setLeadDay}
              disabled={!regions?.model_trained}
              available={regions?.available_lead_days}
            />
            {regions ? <MapLegend definitions={regions.risk_band_definitions} /> : null}
          </div>

          {regionsQuery.isLoading ? <LoadingState label="Scoring the current cycle…" /> : null}
          {regionsQuery.error ? <ErrorState error={regionsQuery.error} /> : null}
          {topoError ? <ErrorState error={topoError} /> : null}

          {regions && !regions.model_trained ? (
            <EmptyState
              title="No trained model yet"
              message={regions.message}
              action={<p className="muted small">Upload a dataset with forecasts and matching observations to begin.</p>}
            />
          ) : null}

          {regions?.model_trained ? (
            regions.regions.length ? (
              <IndiaChoroplethMap
                regions={regions.regions}
                selectedRegionId={selectedRegion}
                onSelect={setSelectedRegion}
                topology={topology}
              />
            ) : (
              <EmptyState title="Nothing to show for this lead day" message={regions.message} />
            )
          ) : null}

          {regions?.regions.length ? (
            <BustSummaryChart regions={regions.regions} onSelect={setSelectedRegion} />
          ) : null}
        </section>

        <section className="side-column">
          <ModelStatusCard />
          <AlertsPanel
            filter={alertFilter}
            onFilter={setAlertFilter}
            onSelect={(regionId, lead) => {
              setSelectedRegion(regionId);
              setLeadDay(lead);
            }}
          />
          <UploadPanel />
        </section>

        <RegionDetailPanel
          regionId={selectedRegion}
          onClose={() => setSelectedRegion(null)}
          riskCuts={riskCuts}
        />
      </main>
      )}
    </div>
  );
}
