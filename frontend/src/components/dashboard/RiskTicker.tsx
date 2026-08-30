import { useMemo, useRef, useState } from "react";
import type { RegionSummary } from "../../api/types";

/**
 * Every scored region for the current lead day, worst first, on a slow rail.
 *
 * It exists so the dashboard states its whole hand at a glance rather than only the five
 * regions a top-N list has room for. Hovering pauses it, and it holds still entirely under
 * `prefers-reduced-motion`.
 */
export function RiskTicker({ regions, leadDay, onSelect }: {
  regions: RegionSummary[];
  leadDay: number;
  onSelect: (regionId: string) => void;
}) {
  const [paused, setPaused] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);
  const items = useMemo(
    () =>
      regions
        .filter((r) => r.bust_probability != null)
        .sort((a, b) => (b.bust_probability as number) - (a.bust_probability as number)),
    [regions],
  );

  if (items.length < 2) return null;

  // The rail is rendered twice and translated by exactly half its width, so the loop has
  // no visible seam.
  const rail = [...items, ...items];

  return (
    <div
      className="ticker"
      ref={boxRef}
      role="region"
      aria-label={`Bust probability by region, lead day ${leadDay}`}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
      // Focusing a button inside an overflow container makes the browser scroll it into
      // view, which yanks the rail back to its start and looks like the ticker resetting.
      // The rail is positioned purely by its animation, so any scroll here is unwanted.
      onScroll={(e) => { e.currentTarget.scrollLeft = 0; }}
    >
      <div
        className="ticker__rail"
        // Pausing from React rather than :hover keeps the running animation exactly where
        // it is - toggling only play-state never restarts it.
        style={{ animationPlayState: paused ? "paused" : "running" }}
      >
        {rail.map((r, i) => (
          <button
            key={`${r.region_id}-${i}`}
            type="button"
            className="ticker__item"
            // Keep the click without taking focus, so the container is never scrolled.
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => onSelect(r.region_id)}
            tabIndex={i < items.length ? 0 : -1}
            aria-hidden={i >= items.length}
          >
            <i className={`ticker__dot ticker__dot--${r.risk_band ?? "none"}`} aria-hidden="true" />
            <span className="ticker__name">{r.region_name ?? r.region_id}</span>
            <span className="ticker__value">{(r.bust_probability as number).toFixed(2)}</span>
            <span className="ticker__unit">P(bust) · D{leadDay}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
