import { useEffect, useRef, useState } from "react";
import type { Topology } from "topojson-specification";
import type { RiskBand } from "../api/types";
import { AlertsPage } from "../components/alerts/AlertsPage";
import { BustSummaryChart } from "../components/dashboard/BustSummaryChart";
import { FeedFreshness } from "../components/dashboard/FeedFreshness";
import { HeroDivergence } from "../components/dashboard/HeroDivergence";
import { KpiStrip } from "../components/dashboard/KpiStrip";
import { ModelPage } from "../components/model/ModelPage";
import { RiskTicker } from "../components/dashboard/RiskTicker";
import { RegionDetailPanel } from "../components/detail/RegionDetailPanel";
import { EmptyState, ErrorState, LoadingState } from "../components/common/States";
import { IndiaChoroplethMap, loadTopology } from "../components/map/IndiaChoroplethMap";
import { LeadDayRail } from "../components/map/LeadDayRail";
import { LeadDaySelector } from "../components/map/LeadDaySelector";
import { MapLegend } from "../components/map/MapLegend";
import { AboutPage } from "../components/about/AboutPage";
import { ReplayView } from "../components/replay/ReplayView";
import { useAllRegions, useEnsembleDivergence, useModelStatus } from "../hooks/useDashboardData";
import { useLiveSocket } from "../hooks/useLiveSocket";
import { useMediaQuery } from "../hooks/useMediaQuery";

/**
 * Operations is the map and the region behind it, and nothing else. Alerts and Model each
 * get their own tab so the screen a judge stares at is not sharing room with a run id, a
 * metrics table and a file dropzone.
 */
type View = "live" | "alerts" | "model" | "replay" | "about";

const TABS: { id: View; label: string }[] = [
  { id: "live", label: "Operations" },
  { id: "alerts", label: "Alerts" },
  { id: "model", label: "Model" },
  { id: "replay", label: "Replay a real bust" },
  // The link goes to judges who open it unattended, with nobody to explain what they are
  // looking at. The case for the project therefore has to be reachable from the page.
  { id: "about", label: "About" },
];

export function DashboardPage() {
  useLiveSocket();

  const [view, setView] = useState<View>("live");
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

  // The opening screen only claims the viewport when there is a hero to fill it; with no
  // trained model the hero renders nothing and a full-height empty band would be absurd.
  const heroFills = Boolean(ensembleQuery.data?.model_trained);

  // The horizon rail only earns its place when the map still has room beside it; below
  // this the map would be squeezed to buy space for a column, so the pills come back.
  const showRail = useMediaQuery("(min-width: 1440px) and (min-height: 700px)");

  // Picking a region from the ticker has to carry you to it. The ticker sits on the
  // opening screen but the detail panel it fills is a whole viewport below, so selecting
  // without scrolling looks like the click did nothing.
  const selectAndReveal = (regionId: string) => {
    setSelectedRegion(regionId);
    scrollToOperations();
  };

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
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={view === t.id}
                className={view === t.id ? "viewtab is-active" : "viewtab"}
                onClick={() => setView(t.id)}
              >
                {t.label}
              </button>
            ))}
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

      {view === "about" ? (
        <AboutPage onReplay={() => setView("replay")} />
      ) : view === "replay" ? (
        <main className="app__body app__body--replay">
          <ReplayView topology={topology} />
        </main>
      ) : view === "alerts" ? (
        <AlertsPage
          filter={alertFilter}
          onFilter={setAlertFilter}
          // An alert is a pointer at a place and a day, so acting on one belongs on the
          // map rather than in the list you clicked it from.
          onSelect={(regionId, lead) => {
            setSelectedRegion(regionId);
            setLeadDay(lead);
            setView("live");
            // Operations opens on the hero, so land the map in view rather than making
            // someone scroll a viewport to find what they just clicked. One frame later,
            // once the live view has actually mounted #operations.
            requestAnimationFrame(scrollToOperations);
          }}
        />
      ) : view === "model" ? (
        <ModelPage />
      ) : (
        <>
          {/* The opening screen: the national argument, held for one full viewport, with
              the ticker as its base rail. Everything operational starts below the fold. */}
          <section className={heroFills ? "screen1" : undefined}>
            <HeroDivergence data={ensembleQuery.data} />
            <KpiStrip all={allRegions} day={regions} />
            {heroFills ? <OpeningCues onReplay={() => setView("replay")} /> : null}
            {regions?.regions.length ? (
              <RiskTicker
                regions={regions.regions}
                leadDay={regions.lead_time_days}
                onSelect={selectAndReveal}
              />
            ) : null}
          </section>

          <FeedFreshness />

          {/* Operations holds exactly one viewport: the map and the region behind it, both
              whole, with no scrolling. The summary chart is the next screen down. */}
          <main
            className={showRail ? "app__body app__body--ops app__body--rail" : "app__body app__body--ops"}
            id="operations"
          >
            <section className="map-column">
              <div className="map-toolbar">
                {/* The rail carries the lead day when there is room for it; the pills are
                    the same control for narrower screens, and only ever one exists. */}
                {showRail ? null : (
                  <LeadDaySelector
                    value={leadDay}
                    onChange={setLeadDay}
                    disabled={!regions?.model_trained}
                    available={regions?.available_lead_days}
                  />
                )}
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
            </section>

            {showRail ? (
              <LeadDayRail all={allRegions} value={leadDay} onChange={setLeadDay} />
            ) : null}

            <RegionDetailPanel
              regionId={selectedRegion}
              onClose={() => setSelectedRegion(null)}
              riskCuts={riskCuts}
            />
          </main>

          {regions?.regions.length ? (
            <section className="app__below">
              <BustSummaryChart regions={regions.regions} onSelect={setSelectedRegion} />
            </section>
          ) : null}
        </>
      )}
    </div>
  );
}

/**
 * Bring the map and the region detail panel into view. `behavior` is chosen here rather
 * than left to CSS because the reduced-motion override in styles.css only reaches the
 * `scroll-behavior` property, not the option passed to scrollIntoView.
 */
function scrollToOperations() {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document
    .getElementById("operations")
    ?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
}

/**
 * The only affordance saying the page continues. A full-height opening screen hides the
 * map completely, so something has to name what is below it - a bare chevron would leave
 * a judge guessing whether scrolling is worth it.
 */
function OpeningCues({ onReplay }: { onReplay: () => void }) {
  return (
    <div className="cuerow">
      {/* Named for the payoff, not the feature. "Replay" is a label; this is a reason. */}
      <button type="button" className="cuerow__primary" onClick={onReplay}>
        Watch it call a real bust →
      </button>
      <ScrollCue />
    </div>
  );
}

function ScrollCue() {
  return (
    <button type="button" className="scrollcue" onClick={scrollToOperations}>
      <span>The national map</span>
      <svg viewBox="0 0 16 10" aria-hidden="true" width="16" height="10">
        <path d="M1 1l7 7 7-7" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    </button>
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
