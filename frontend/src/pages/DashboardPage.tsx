import { useEffect, useMemo, useRef, useState } from "react";
import type { Topology } from "topojson-specification";
import type { RiskBand } from "../api/types";
import { AlertsPage } from "../components/alerts/AlertsPage";
import { BaselineLadderCard } from "../components/dashboard/BaselineLadderCard";
import { WorstDistrictsPanel } from "../components/dashboard/WorstDistrictsPanel";
import { FeedFreshness } from "../components/dashboard/FeedFreshness";
import { HeroDivergence } from "../components/dashboard/HeroDivergence";
import { KpiStrip } from "../components/dashboard/KpiStrip";
import { ModelPage } from "../components/model/ModelPage";
import { RiskTicker } from "../components/dashboard/RiskTicker";
import { RegionDetailPanel } from "../components/detail/RegionDetailPanel";
import { EmptyState, ErrorState, LoadingState } from "../components/common/States";
import { retryingHint } from "../lib/retryHint";
import { IndiaChoroplethMap, loadTopology } from "../components/map/IndiaChoroplethMap";
import { LeadDayRail } from "../components/map/LeadDayRail";
import { LeadDaySelector } from "../components/map/LeadDaySelector";
import { MapLegend } from "../components/map/MapLegend";
import { AboutPage } from "../components/about/AboutPage";
import { ReplayView } from "../components/replay/ReplayView";
import { useAllRegions, useEnsembleDivergence, useModelStatus } from "../hooks/useDashboardData";
import { useLiveSocket } from "../hooks/useLiveSocket";
import { stateNamesFrom } from "../lib/stateNames";
import { parseAppState, toSearch, type View } from "../lib/urlState";
import { useMediaQuery } from "../hooks/useMediaQuery";
import { dayLabel } from "../lib/format";
import { inferRiskCuts, isScoredProbability, resolveRiskCuts, riskBandForRegion } from "../lib/riskBands";

// View lives in lib/urlState: it is the set of values the ?view= parameter accepts, so
// the tab list and the URL parser must not be able to disagree about it.

const TABS: { id: View; label: string; tail?: string }[] = [
  { id: "live", label: "Operations" },
  { id: "alerts", label: "Alerts" },
  { id: "model", label: "Model" },
  { id: "replay", label: "Replay", tail: " a real bust" },

  { id: "about", label: "About" },
];

export function DashboardPage() {
  useLiveSocket();

  // Read once, on mount: after this the URL follows the state rather than driving it,
  // except when the back button fires popstate below.
  const opened = useMemo(() => parseAppState(window.location.search), []);

  const [view, setView] = useState<View>(opened.view);
  const [leadDay, setLeadDay] = useState(opened.leadDay);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(opened.region);
  const [alertFilter, setAlertFilter] = useState<RiskBand | undefined>(opened.band);
  const [topology, setTopology] = useState<Topology | null>(null);
  const [topoError, setTopoError] = useState<unknown>(null);

  const regionsQuery = useAllRegions();
  const statusQuery = useModelStatus();

  const ensembleQuery = useEnsembleDivergence();

  useEffect(() => {
    loadTopology().then(setTopology).catch(setTopoError);
  }, []);

  const stateNames = useMemo(() => stateNamesFrom(topology), [topology]);

  const allRegions = regionsQuery.data;
  const regions =
    allRegions?.days.find((d) => d.lead_time_days === leadDay) ?? allRegions?.days[0];
  const riskCuts = resolveRiskCuts(
    statusQuery.data?.thresholds?.risk_band_cuts,
    regions?.risk_band_definitions,
  ) ?? inferRiskCuts(regions?.regions ?? []);

  const autoLeadPicked = useRef(false);
  const available = regions?.available_lead_days;
  useEffect(() => {
    if (autoLeadPicked.current || !available?.length) return;
    if (!available.includes(leadDay)) setLeadDay(available[0]);
    autoLeadPicked.current = true;
  }, [available, leadDay]);

  /**
   * Keep the address bar in step with the screen.
   *
   * Changing tab or opening a district is a navigation and gets a history entry, so the
   * back button does the obvious thing - out of a district, back to the previous tab.
   * Changing the lead day or the alert filter is a refinement of the view you are already
   * on, and pushing those would bury the real steps under a stack of near-identical
   * entries: ten lead-day clicks would need ten presses of back to escape.
   */
  const lastNav = useRef({ view, region: selectedRegion });
  useEffect(() => {
    const search = toSearch({ view, leadDay, region: selectedRegion, band: alertFilter });
    if (search === window.location.search) return;
    const navigated = view !== lastNav.current.view || selectedRegion !== lastNav.current.region;
    lastNav.current = { view, region: selectedRegion };
    const url = `${window.location.pathname}${search}`;
    window.history[navigated ? "pushState" : "replaceState"](null, "", url);
  }, [view, leadDay, selectedRegion, alertFilter]);

  // Back and forward put the URL back; the screen has to follow it.
  useEffect(() => {
    const onPop = () => {
      const s = parseAppState(window.location.search);
      lastNav.current = { view: s.view, region: s.region };
      setView(s.view);
      setLeadDay(s.leadDay);
      setSelectedRegion(s.region);
      setAlertFilter(s.band);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const highCount = regions?.regions.filter(
    (r) => r.data_available !== false
      && isScoredProbability(r.bust_probability)
      && riskBandForRegion(r, riskCuts) === "high",
  ).length ?? 0;

  const ensemble = ensembleQuery.data;
  const heroFills = Boolean(
    ensemble
      && ensemble.model_trained
      && isScoredProbability(ensemble.mean_bust_probability)
      && ensemble.n_scored_regions > 0
      && (ensemble.national?.length ?? 0) > 0,
  );

  const showRail = useMediaQuery("(min-width: 1440px) and (min-height: 700px)");

  const selectAndReveal = (regionId: string) => {
    setSelectedRegion(regionId);
    scrollToOperations();
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__inner">
          <button
            type="button"
            className="brand"
            aria-label="Sanket - back to the opening screen"
            onClick={() => {
              setView("live");
              window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
            }}
          >
            <img className="brand__glyph" src="/logo.png" alt="" width={40} height={32} />
            <img className="brand__wordmark" src="/wordmark.png" alt="Sanket" />
            <span className="brand__div" aria-hidden="true" />
            <span className="brand__sub">Forecast Bust Detection</span>
          </button>

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
                {t.tail ? <span className="viewtab__tail">{t.tail}</span> : null}
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
                {highCount} bust · {dayLabel(leadDay)}
              </span>
            ) : null}
          </div>
        </div>
      </header>

      {view === "about" ? (
        <AboutPage onReplay={() => setView("replay")} />
      ) : view === "replay" ? (
        <main className="app__body app__body--replay">
          {/* Every other view titles itself through .pagehead; this one opens straight
            * into the cycle picker and had no h1 at all. It lives here rather than inside
            * ReplayView because that returns early while a cycle is scoring, which is
            * exactly when a screen reader most needs to know what the page is. Hidden
            * rather than drawn: the missing thing is structure, not a visible heading. */}
          <h1 className="visually-hidden">Replay a real forecast bust</h1>
          <ReplayView topology={topology} />
        </main>
      ) : view === "alerts" ? (
        <AlertsPage
          filter={alertFilter}
          onFilter={setAlertFilter}

          onSelect={(regionId, lead) => {
            setSelectedRegion(regionId);
            setLeadDay(lead);
            setView("live");

            requestAnimationFrame(scrollToOperations);
          }}
        />
      ) : view === "model" ? (
        <ModelPage />
      ) : (
        <>
          <section className={heroFills ? "screen1" : undefined}>
            <HeroDivergence data={ensembleQuery.data} riskCuts={riskCuts} />
            <KpiStrip all={allRegions} day={regions} riskCuts={riskCuts} />
            {heroFills ? <OpeningCues onReplay={() => setView("replay")} /> : null}
            {regions?.regions.length ? (
              <RiskTicker
                regions={regions.regions}
                leadDay={regions.lead_time_days}
                onSelect={selectAndReveal}
                stateNames={stateNames}
                riskCuts={riskCuts}
              />
            ) : null}
          </section>

          <FeedFreshness />

          <main
            className={showRail ? "app__body app__body--ops app__body--rail" : "app__body app__body--ops"}
            id="operations"
          >
            <section className="map-column">
              <div className="map-toolbar">
                {showRail ? null : (
                  <LeadDaySelector
                    value={leadDay}
                    onChange={setLeadDay}
                    disabled={!regions || !regions.model_trained || regions.regions.length === 0}
                    available={regions?.available_lead_days}
                  />
                )}
                {regions ? <MapLegend definitions={regions.risk_band_definitions} /> : null}
              </div>

              {regionsQuery.isLoading ? <LoadingState label="Scoring the current cycle…" hint={retryingHint(regionsQuery.failureCount)} /> : null}
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
                    riskCuts={riskCuts}
                  />
                ) : (
                  <EmptyState title="Nothing to show for this lead day" message={regions.message} />
                )
              ) : null}
            </section>

            {showRail ? (
              <LeadDayRail
                all={allRegions}
                value={leadDay}
                onChange={setLeadDay}
                riskCuts={riskCuts}
              />
            ) : null}

            {selectedRegion ? (
              <RegionDetailPanel
                regionId={selectedRegion}
                onClose={() => setSelectedRegion(null)}
                riskCuts={riskCuts}
              />
            ) : (
              <WorstDistrictsPanel day={regions} riskCuts={riskCuts} onSelect={setSelectedRegion} />
            )}
          </main>

          {/* The ladder sits directly under the map: you look at today's risk, then
            * immediately at the evidence that the risk is worth believing. Guarded so this
            * never renders as bare padding when there are no baselines. The worst-districts
            * chart that used to sit beside it is now the district panel's resting state. */}
          {statusQuery.data?.baselines?.models?.length ? (
            <section className="app__below">
              <BaselineLadderCard data={statusQuery.data} />
            </section>
          ) : null}
        </>
      )}
    </div>
  );
}

function scrollToOperations() {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document
    .getElementById("operations")
    ?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
}

function OpeningCues({ onReplay }: { onReplay: () => void }) {
  return (
    <div className="cuerow">
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

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
