import { useEffect, useMemo, useRef, useState } from "react";
import type { Topology } from "topojson-specification";
import type { RegionSummary, ReplayRegionStep } from "../../api/types";
import { useReplay, useReplayCycles } from "../../hooks/useDashboardData";
import { EmptyState, ErrorState, LoadingState } from "../common/States";
import { IndiaChoroplethMap } from "../map/IndiaChoroplethMap";
import { MapLegend } from "../map/MapLegend";
import { ReplayFocusChart } from "./ReplayFocusChart";

const STEP_MS = 2200;

/**
 * Guided replay: pick a real historical forecast cycle and step through its lead days,
 * watching the map recolour while the narration - generated from the cycle's own scored
 * numbers - explains how the bust developed. Nothing here is scripted or synthetic.
 */
export function ReplayView({ topology }: { topology: Topology | null }) {
  const [selectedInit, setSelectedInit] = useState<string | undefined>(undefined);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  // which region the focus chart shows; null = the cycle's peak region (server default)
  const [focusRegionId, setFocusRegionId] = useState<string | null>(null);

  const cyclesQuery = useReplayCycles(true);
  const replayQuery = useReplay(selectedInit, true);
  const replay = replayQuery.data;
  const steps = replay?.steps ?? [];

  // Reset to the first lead day (and the default focus region) whenever the cycle changes.
  useEffect(() => {
    setStepIdx(0);
    setPlaying(false);
    setFocusRegionId(null);
  }, [replay?.init_date]);

  // Autoplay.
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
    return <LoadingState label="Scoring the historical cycle…" />;
  }
  if (replayQuery.error) return <ErrorState error={replayQuery.error} />;
  if (replay && !replay.model_trained) {
    return <EmptyState title="No trained model yet" message={replay.message} />;
  }
  if (!replay || !steps.length) {
    return <EmptyState title="Nothing to replay" message={replay?.message ?? "No scoreable cycle in the store."} />;
  }

  const cycles = cyclesQuery.data ?? replay.available_cycles;
  const scrub = (i: number) => {
    setPlaying(false);
    setStepIdx(i);
  };

  return (
    <div className="replay">
      <div className="replay__intro">
        <div className="replay__pick">
          <label htmlFor="replay-cycle" className="muted small">Forecast cycle</label>
          <select
            id="replay-cycle"
            value={selectedInit ?? replay.init_date ?? ""}
            onChange={(e) => setSelectedInit(e.target.value || undefined)}
          >
            {cycles.map((c) => (
              <option key={c.init_date} value={c.init_date}>
                {c.init_date}
                {c.verified ? ` · ${c.verified_lead_days}d verified` : " · unverified"}
                {c.peak_bust_probability != null
                  ? ` · peak ${(c.peak_bust_probability * 100).toFixed(0)}%`
                  : ""}
              </option>
            ))}
          </select>
        </div>
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
          {playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <button type="button" onClick={() => scrub(Math.max(0, stepIdx - 1))} disabled={stepIdx === 0}>
          ‹ Prev
        </button>
        <button
          type="button"
          onClick={() => scrub(Math.min(steps.length - 1, stepIdx + 1))}
          disabled={stepIdx >= steps.length - 1}
        >
          Next ›
        </button>
        <div className="replay__ticks">
          {steps.map((s, i) => (
            <button
              key={s.lead_time_days}
              type="button"
              className={`replay__tick ${i === stepIdx ? "is-active" : ""} replay__tick--${bandOfStep(s)}`}
              aria-label={`Lead day ${s.lead_time_days}`}
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
          />
          <p className="muted small">Click a state to chart its forecast vs what was observed.</p>
        </div>

        <div className="replay__sidecol">
          <div className="replay__step">
            <div className="replay__stephead">
              <span className="replay__day">Day {step.lead_time_days}</span>
              {step.valid_date ? <span className="muted small">valid {step.valid_date}</span> : null}
            </div>
            <p className="replay__narration">{step.narration}</p>
            <div className="replay__stats">
              <span><b>{pct(step.mean_bust_probability)}</b> mean P(bust)</span>
              <span className="chip chip--high">{step.n_high} high</span>
              <span className="chip chip--medium">{step.n_medium} medium</span>
            </div>
            <ol className="replay__toplist">
              {step.regions.slice(0, 5).map((r) => {
                const chartable = chartableRegionIds.has(r.region_id);
                const active = shownFocus?.region_id === r.region_id;
                return (
                  <li key={r.region_id}>
                    <button
                      type="button"
                      className={`replay__toprow ${active ? "is-active" : ""}`}
                      disabled={!chartable}
                      title={chartable ? "Show this region's forecast vs observed" : undefined}
                      onClick={() => setFocusRegionId(r.region_id)}
                    >
                      <span className={`dot dot--${r.risk_band}`} />
                      {r.region_name ?? r.region_id}
                      <b>{pct(r.bust_probability)}</b>
                      {r.dominant_variable ? (
                        <span className="muted small">{r.dominant_variable.replace(/_/g, " ")}</span>
                      ) : null}
                    </button>
                  </li>
                );
              })}
            </ol>
          </div>

          {shownFocus ? (
            <ReplayFocusChart focus={shownFocus} currentLead={step.lead_time_days} />
          ) : null}
        </div>
      </div>

      <p className="muted small replay__foot">
        Every value is scored from real GEFS reforecast + ERA5 for this cycle; every
        sentence is generated from those numbers. No scripted or synthetic content.
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
