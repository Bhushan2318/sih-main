import { useMemo, useState } from "react";
import type { RegionSummary } from "../../api/types";

/**
 * Every scored region for the current lead day, worst first, on a slow rail.
 *
 * Two identical halves are laid side by side and the track is translated by exactly one
 * half-width (-50%); when the first half has fully left, the second is where it started,
 * so the loop has no seam. `translate3d` + `backface-visibility: hidden` keep the moving
 * text on its own GPU layer and crisp (a plain `%` translate on a text layer renders
 * blurry and snaps a pixel at the loop). `overflow: clip` means the frame is not a scroll
 * container, so tabbing to or clicking an item can never scroll the rail and look like a
 * reset. Pause toggles only `animation-play-state`, freezing it in place, and is wired to
 * focus as well as hover so keyboard users can reach an item without chasing it.
 *
 * On a touch device - and under reduced motion - the CSS stops the animation entirely and
 * makes the frame swipeable instead, because there is no hover to pause with. The second
 * half is hidden there; it exists only to make the animated loop seamless.
 */
export function RiskTicker({ regions, leadDay, onSelect }: {
  regions: RegionSummary[];
  leadDay: number;
  onSelect: (regionId: string) => void;
}) {
  const [paused, setPaused] = useState(false);

  const items = useMemo(
    () =>
      regions
        .filter((r) => r.bust_probability != null)
        .sort((a, b) => (b.bust_probability as number) - (a.bust_probability as number)),
    [regions],
  );

  if (items.length < 2) return null;

  const half = (hidden: boolean) => (
    <div className="ticker__half" aria-hidden={hidden || undefined}>
      {items.map((r) => (
        <button
          key={r.region_id}
          type="button"
          className="ticker__item"
          onClick={() => onSelect(r.region_id)}
          tabIndex={hidden ? -1 : 0}
        >
          <i className={`ticker__dot ticker__dot--${r.risk_band ?? "none"}`} aria-hidden="true" />
          <span className="ticker__name">{r.region_name ?? r.region_id}</span>
          <span className="ticker__value">{(r.bust_probability as number).toFixed(2)}</span>
          <span className="ticker__unit">P(bust) · D{leadDay}</span>
        </button>
      ))}
    </div>
  );

  return (
    <div
      className="ticker"
      role="region"
      aria-label={`Bust probability by region, lead day ${leadDay}`}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      // Keyboard too. Pausing was wired to hover alone, so tabbing into an item left the
      // rail moving and the focus ring rode off the edge of the screen. React maps these
      // onto focusin/focusout, which bubble, so focus on any child button is caught here.
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <div className="ticker__track" style={{ animationPlayState: paused ? "paused" : "running" }}>
        {half(false)}
        {half(true)}
      </div>
    </div>
  );
}
