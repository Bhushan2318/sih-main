import { useEffect, useRef, useState } from "react";
import type { Topology } from "topojson-specification";
import type { RiskBand } from "../api/types";
import { AlertsPanel } from "../components/dashboard/AlertsPanel";
import { BustSummaryChart } from "../components/dashboard/BustSummaryChart";
import { FeedFreshness } from "../components/dashboard/FeedFreshness";
import { HeroDivergence } from "../components/dashboard/HeroDivergence";
import { KpiStrip } from "../components/dashboard/KpiStrip";
import { ModelStatusCard } from "../components/dashboard/ModelStatusCard";
import { RiskTicker } from "../components/dashboard/RiskTicker";
import { RegionDetailPanel } from "../components/detail/RegionDetailPanel";
import { EmptyState, ErrorState, LoadingState } from "../components/common/States";
import { IndiaChoroplethMap, loadTopology } from "../components/map/IndiaChoroplethMap";
import { LeadDaySelector } from "../components/map/LeadDaySelector";
import { MapLegend } from "../components/map/MapLegend";
import { ReplayView } from "../components/replay/ReplayView";
import { UploadPanel } from "../components/upload/UploadPanel";
import { useAllRegions, useEnsembleDivergence, useModelStatus } from "../hooks/useDashboardData";
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
  // The hero is the national frame for the whole cycle and deliberately does not follow
  // the map selection - region detail has its own panel.
  const ensembleQuery = useEnsembleDivergence();

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

  const highCount = regions?.regions.filter((r) => r.risk_band === "high").length ?? 0;

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__inner">
          <div className="brand">
            <SanketGlyph />
            <span className="brand__name">Sanket</span>
            <span className="brand__div" aria-hidden="true" />
            <span className="brand__sub">Forecast Bust Detection</span>
          </div>

          <nav className="viewtabs" role="tablist" aria-label="View">
            <button
              type="button"
              role="tab"
              aria-selected={view === "live"}
              className={view === "live" ? "viewtab is-active" : "viewtab"}
              onClick={() => setView("live")}
            >
              Operations
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
          </nav>

          <div className="topbar__right">
            {view === "live" && regions?.model_trained && regions.init_date ? (
              <span className="pill pill--quiet" title="The forecast cycle currently in the store">
                <i aria-hidden="true" />
                {regions.init_date}
                {regions.valid_date ? ` → ${regions.valid_date}` : ""}
              </span>
            ) : null}
            {view === "live" && highCount > 0 ? (
              <span className="pill pill--alarm" title={`Regions in the bust band at lead day ${leadDay}`}>
                <i aria-hidden="true" />
                {highCount} bust · D{leadDay}
              </span>
            ) : null}
          </div>
        </div>
      </header>

      {view === "replay" ? (
        <main className="app__body app__body--replay">
          <ReplayView topology={topology} />
        </main>
      ) : (
        <>
          <HeroDivergence data={ensembleQuery.data} />
          {regions?.regions.length ? (
            <RiskTicker
              regions={regions.regions}
              leadDay={regions.lead_time_days}
              onSelect={setSelectedRegion}
            />
          ) : null}
          <FeedFreshness />
          <KpiStrip all={allRegions} day={regions} />

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
        </>
      )}
    </div>
  );
}

/**
 * The mark: two traces leaving one origin and coming apart. That divergence between the
 * forecast and what actually happened is the whole product, so the logo states it rather
 * than decorating around it.
 */
function SanketGlyph() {
  return (
    <svg className="brand__glyph" viewBox="0 0 32 32" role="img" aria-label="Sanket">
      <circle cx="16" cy="16" r="15" fill="#0b1220" />
      <path d="M6 16h8" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" fill="none" />
      <path d="M14 16l12-6" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" fill="none" />
      <path d="M14 16l12 7" stroke="#f5254a" strokeWidth="2.4" strokeLinecap="round" fill="none" />
      <circle cx="14" cy="16" r="2.6" fill="#2b4eff" />
    </svg>
  );
}
