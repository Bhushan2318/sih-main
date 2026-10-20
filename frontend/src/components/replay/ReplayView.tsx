import { translate } from "../../i18n";
import { rich } from "../../i18n/rich";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Topology } from "topojson-specification";
import type { RegionSummary, ReplayRegionStep } from "../../api/types";
import {
  useModelStatus, usePrefetchReplay, useReplay, useReplayCycles,
} from "../../hooks/useDashboardData";
import { initialReplayStepIndex } from "../../lib/eventLeadDay";
import { resolveRiskCuts } from "../../lib/riskBands";
import { EmptyState, ErrorState, LoadingState } from "../common/States";
import { IndiaChoroplethMap } from "../map/IndiaChoroplethMap";
import { MapLegend } from "../map/MapLegend";
import { ReplayCyclePicker } from "./ReplayCyclePicker";
import { ReplayEventHeader } from "./ReplayEventHeader";
import { ReplayFocusChart } from "./ReplayFocusChart";
import { ReplayProbabilityChart } from "./ReplayProbabilityChart";
import { TopDistrictsList } from "../common/TopDistrictsList";

const STEP_MS = 2200;

export function ReplayView({ topology }: { topology: Topology | null }) {
  const [selectedInit, setSelectedInit] = useState<string | undefined>(undefined);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  const [focusRegionId, setFocusRegionId] = useState<string | null>(null);

  const cyclesQuery = useReplayCycles(true);
  const statusQuery = useModelStatus();
  const replayQuery = useReplay(selectedInit, true);
  const replay = replayQuery.data;
  const riskCuts = resolveRiskCuts(
    statusQuery.data?.thresholds?.risk_band_cuts,
    replay?.risk_band_definitions,
  );
  const steps = replay?.steps ?? [];
  // With one cycle on screen, fetch the rest of the list behind it, so switching is instant.
  usePrefetchReplay(replay ? "all" : null);

  // The cycle this replay belongs to, as listed by /api/replay/cycles - the only place
  // peak_valid_date and kind live. Derived before the reset effect below so a newly
  // opened event can pick its own opening day instead of always Day 1.
  const cycles = cyclesQuery.data ?? replay?.available_cycles ?? [];
  const shownCycle = cycles.find((c) => c.init_date === replay?.init_date);
  const event = shownCycle?.kind === "event" ? shownCycle : null;

  useEffect(() => {
    const leadDays = steps.map((s) => s.lead_time_days);
    const opening = initialReplayStepIndex(replay?.init_date, event?.peak_valid_date, leadDays);
    setStepIdx(opening ?? 0);
    setPlaying(false);
    setFocusRegionId(null);
    // Also keyed on the peak date: the cycle list can land after the replay itself, and
    // an event must still open on its peak day rather than stay on Day 1.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replay?.init_date, event?.peak_valid_date]);

  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!playing || steps.length === 0) return;
    timer.current = window.setInterval(() => {
      setStepIdx((i) => {
        if (i >= steps.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, STEP_MS);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [playing, steps.length]);

  const step = steps[Math.min(stepIdx, Math.max(steps.length - 1, 0))];
  const mapRegions: RegionSummary[] = useMemo(
    () => (step?.regions ?? []).map(toRegionSummary),
    [step],
  );
  const focusOptions = replay?.focus_options ?? [];
  const shownFocus =
    focusOptions.find((o) => o.region_id === focusRegionId) ?? replay?.focus ?? null;
  const chartableRegionIds = new Set(focusOptions.map((o) => o.region_id));

  if (replayQuery.isLoading || cyclesQuery.isLoading) {
    return (
      <LoadingState
        label={translate("rv.loading")}
        hint={translate("rv.loadingHint")}
      />
    );
  }
  if (replayQuery.error) return <ErrorState error={replayQuery.error} />;
  if (replay && !replay.model_trained) {
    return <EmptyState title={translate("rv.noModel")} message={replay.message} />;
  }
  if (!replay || !steps.length) {
    return <EmptyState title={translate("rv.nothing")} message={replay?.message ?? translate("rv.noCycle")} />;
  }

  const scrub = (i: number) => {
    setPlaying(false);
    setStepIdx(i);
  };

  return (
    <div className="replay">
      {event ? (
        <ReplayEventHeader cycle={event} steps={steps} focusOptions={focusOptions} />
      ) : null}
      <div className="replay__intro">
        <ReplayCyclePicker
          cycles={cycles}
          value={selectedInit ?? replay.init_date ?? ""}
          onChange={setSelectedInit}
        />
        {replay.summary_narration ? (
          <p className="replay__summary">{replay.summary_narration}</p>
        ) : null}
      </div>

      <div className="replay__controls">
        <button
          type="button"
          className="replay__play"
          onClick={() => {
            if (stepIdx >= steps.length - 1) setStepIdx(0);
            setPlaying((p) => !p);
          }}
        >
          {playing ? translate("rv.pause") : translate("rv.play")}
        </button>
        <button type="button" onClick={() => scrub(Math.max(0, stepIdx - 1))} disabled={stepIdx === 0}>
          {translate("rv.prev")}
        </button>
        <button
          type="button"
          onClick={() => scrub(Math.min(steps.length - 1, stepIdx + 1))}
          disabled={stepIdx >= steps.length - 1}
        >
          {translate("rv.next")}
        </button>
        <div className="replay__ticks">
          {steps.map((s, i) => (
            <button
              key={s.lead_time_days}
              type="button"
              className={`replay__tick ${i === stepIdx ? "is-active" : ""} replay__tick--${bandOfStep(s)}`}
              aria-label={translate("lead.day", { d: s.lead_time_days })}
              onClick={() => scrub(i)}
            >
              {s.lead_time_days}
            </button>
          ))}
        </div>
      </div>

      <div className="replay__body">
        <div className="replay__mapcol">
          {replay.risk_band_definitions ? (
            <MapLegend definitions={replay.risk_band_definitions} />
          ) : null}
          <IndiaChoroplethMap
            regions={mapRegions}
            selectedRegionId={shownFocus?.region_id ?? null}
            onSelect={(rid) => chartableRegionIds.has(rid) && setFocusRegionId(rid)}
            topology={topology}
            riskCuts={riskCuts}
          />
          <p className="muted small">{translate("rv.clickHint")}</p>
        </div>

        <div className="replay__sidecol">
          <div className="replay__step">
            <div className="replay__stephead">
              <span className="replay__day">{translate("day.one", { n: step.lead_time_days })}</span>
              {step.valid_date ? <span className="muted small">{translate("rv.valid", { date: step.valid_date })}</span> : null}
            </div>
            <p className="replay__narration">{step.narration}</p>
            <div className="replay__stats">
              <span>{rich("rv.avg", { pct: pct(step.mean_bust_probability) }, [<b />])}</span>
              <span className="chip chip--high">{translate("rv.nBust", { n: step.n_high })}</span>
              <span className="chip chip--medium">{translate("rv.nWatch", { n: step.n_medium })}</span>
            </div>
            <TopDistrictsList
              items={step.regions.slice(0, 5).map((r) => ({ ...r, band: r.risk_band }))}
              onSelect={setFocusRegionId}
              activeId={shownFocus?.region_id}
              isSelectable={(id) => chartableRegionIds.has(id)}
              selectTitle={translate("rv.selectTitle")}
            />
          </div>

          {shownFocus ? (
            <>
              <ReplayFocusChart
                focus={shownFocus}
                currentLead={step.lead_time_days}
                why={
                  event &&
                  shownFocus.region_id === event.focus_region_id &&
                  shownFocus.variable === event.focus_variable
                    ? translate("rv.remembered")
                    : undefined
                }
              />
              <ReplayProbabilityChart
                currentLead={step.lead_time_days}
                cuts={riskCuts}
                variable={shownFocus.variable}
                points={(replayQuery.data?.steps ?? []).map((st) => {

                  const fp = shownFocus.points.find(
                    (x) => x.lead_time_days === st.lead_time_days);
                  const thr = shownFocus.bust_threshold;
                  const busted =
                    fp && thr != null && fp.predicted_value != null && fp.observed_value != null
                      ? Math.abs(fp.predicted_value - fp.observed_value) >= thr
                      : null;
                  return {
                    lead: st.lead_time_days,
                    p: st.regions.find((r) => r.region_id === shownFocus.region_id)
                         ?.bust_probability ?? null,
                    busted,
                  };
                })}
              />
            </>
          ) : (

            <div className="replay-focus">
              <div className="replay-focus__head">
                <strong>{translate("rv.noChart")}</strong>
              </div>
              <p className="muted small">
                {translate("rv.noChartBody")}
              </p>
              <p className="muted small">
                {rich("rv.pickPast", undefined, [<b />])}
              </p>
            </div>
          )}
        </div>
      </div>

      <p className="muted small replay__foot">
        {translate("rv.foot")}
      </p>
    </div>
  );
}

function toRegionSummary(r: ReplayRegionStep): RegionSummary {
  return {
    region_id: r.region_id,
    region_name: r.region_name,
    bust_probability: r.bust_probability,
    risk_band: r.risk_band,
    confidence: r.confidence,
    dominant_variable: r.dominant_variable,
    data_available: true,
  };
}

function bandOfStep(s: { n_high: number; n_medium: number }): string {
  if (s.n_high >= 3) return "high";
  if (s.n_high + s.n_medium >= 3) return "medium";
  return "low";
}

function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(0)}%`;
}
