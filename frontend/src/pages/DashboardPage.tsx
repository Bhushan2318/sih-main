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
import { UploadPanel } from "../components/upload/UploadPanel";
import { useModelStatus, useRegions } from "../hooks/useDashboardData";
import { useLiveSocket } from "../hooks/useLiveSocket";

export function DashboardPage() {
  useLiveSocket();

  const [leadDay, setLeadDay] = useState(1);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [alertFilter, setAlertFilter] = useState<RiskBand | undefined>(undefined);
  const [topology, setTopology] = useState<Topology | null>(null);
  const [topoError, setTopoError] = useState<unknown>(null);

  const regionsQuery = useRegions(leadDay);
  const statusQuery = useModelStatus();

  useEffect(() => {
    loadTopology().then(setTopology).catch(setTopoError);
  }, []);

  const regions = regionsQuery.data;
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
        {regions?.model_trained && regions.init_date ? (
          <p className="muted small">
            Forecast cycle <b>{regions.init_date}</b>
            {regions.valid_date ? ` → valid ${regions.valid_date}` : ""}
          </p>
        ) : null}
      </header>

      <FeedFreshness />

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
    </div>
  );
}
