import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { RegionSummary } from "../../api/types";

/** How fast the ticker reads, in CSS pixels per second.
 *
 * The duration used to be a fixed 60s in the stylesheet, which fixes the *time* for one
 * loop rather than the *speed*. That was fine at ~35 states; at 666 districts the track
 * is 484,898 px wide (measured), so a 60s loop ran at ~4,040 px/s - roughly fifty times
 * faster than anything legible. Fixing the speed and deriving the duration keeps it
 * readable whatever the region count does next. */
const SPEED_PX_PER_SEC = 70;

/** `IN-MH-NAGPUR` -> `IN-MH`. The region_id scheme is `IN-<state>-<district>`, and the
 * API sends no state field, so the prefix is what there is to group on. */
function stateIdOf(regionId: string): string {
  return regionId.split("-").slice(0, 2).join("-");
}

export function RiskTicker({ regions, leadDay, onSelect, stateNames }: {
  regions: RegionSummary[];
  leadDay: number;
  onSelect: (regionId: string) => void;
  /** state_id -> state_name, from the map topology: the same table the choropleth
   * labels states with, so the ticker cannot drift from the map. */
  stateNames: Map<string, string>;
}) {
  const [paused, setPaused] = useState(false);
  const trackRef = useRef<HTMLDivElement>(null);
  const [duration, setDuration] = useState<number | null>(null);

  // One entry per state, not per district. 666 districts made the track 484,898 px wide,
  // so at a readable speed a full loop took ~58 minutes and any given district came round
  // about once an hour - a ticker nobody could actually follow. Rolling up to states
  // gives ~36 entries and a loop of a couple of minutes.
  //
  // Each state carries its *worst* district, matching the map's default "Worst district"
  // colouring, and clicking opens that district so the ticker still leads somewhere real.
  const items = useMemo(() => {
    const worst = new Map<string, RegionSummary>();
    for (const r of regions) {
      if (r.bust_probability == null) continue;
      const sid = stateIdOf(r.region_id);
      const held = worst.get(sid);
      if (!held || (held.bust_probability as number) < r.bust_probability) worst.set(sid, r);
    }
    return [...worst.entries()]
      .map(([stateId, district]) => ({ stateId, district }))
      .sort((a, b) =>
        (b.district.bust_probability as number) - (a.district.bust_probability as number));
  }, [regions]);

  // Measured rather than estimated from the item count: names vary in width, and the
  // keyframes move the track by -50%, so one loop is exactly half its scroll width.
  // useLayoutEffect, not useEffect - this runs before paint, so the ticker is never
  // briefly visible at the stylesheet's fallback speed.
  useLayoutEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const loopPx = el.scrollWidth / 2;
    setDuration(loopPx > 0 ? loopPx / SPEED_PX_PER_SEC : null);
  }, [items, leadDay]);

  if (items.length < 2) return null;

  const half = (hidden: boolean) => (
    <div className="ticker__half" aria-hidden={hidden || undefined}>
      {items.map(({ stateId, district }) => (
        <button
          key={stateId}
          type="button"
          className="ticker__item"
          onClick={() => onSelect(district.region_id)}
          tabIndex={hidden ? -1 : 0}
          title={`Worst district: ${district.region_name ?? district.region_id}`}
        >
          <i className={`ticker__dot ticker__dot--${district.risk_band ?? "none"}`} aria-hidden="true" />
          <span className="ticker__name">{stateNames.get(stateId) ?? stateId}</span>
          <span className="ticker__value">
            {((district.bust_probability as number) * 100).toFixed(0)}%
          </span>
          <span className="ticker__unit">worst district · day {leadDay}</span>
        </button>
      ))}
    </div>
  );

  return (
    <div
      className="ticker"
      role="region"
      aria-label={`Worst district per state, lead day ${leadDay}`}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}

      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <div
        className="ticker__track"
        ref={trackRef}
        style={{
          animationPlayState: paused ? "paused" : "running",
          ...(duration ? { animationDuration: `${duration}s` } : {}),
        }}
      >
        {half(false)}
        {half(true)}
      </div>
    </div>
  );
}
