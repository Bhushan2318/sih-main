import type { RiskBand } from "../../api/types";
import { variableLabel } from "../../lib/displayNames";

export interface TopDistrict {
  region_id: string;
  region_name: string | null;
  bust_probability: number;
  band: RiskBand | null;
  dominant_variable: string | null;
}

/** A ranked, clickable list of districts: band dot, name, bust risk, main cause.
 *
 * One component for both places a list like this appears - the operations panel's worst
 * districts for the chosen day, and Replay's top five for a step - so the two cannot
 * drift apart again. Replay's used to print the raw column name ("soil moisture pct"). */
export function TopDistrictsList({ items, onSelect, activeId, isSelectable, selectTitle }: {
  items: TopDistrict[];
  onSelect: (regionId: string) => void;
  activeId?: string | null;
  /** Rows this returns false for are shown but cannot be clicked. */
  isSelectable?: (regionId: string) => boolean;
  selectTitle?: string;
}) {
  return (
    <ol className="toplist">
      {items.map((r) => {
        const selectable = isSelectable ? isSelectable(r.region_id) : true;
        const name = r.region_name ?? r.region_id;
        return (
          <li key={r.region_id}>
            <button
              type="button"
              className={r.region_id === activeId ? "toplist__row is-active" : "toplist__row"}
              disabled={!selectable}
              title={selectable ? selectTitle : undefined}
              onClick={() => onSelect(r.region_id)}
            >
              <span className={`dot dot--${r.band ?? "nodata"}`} aria-hidden="true" />
              <span className="toplist__name">{name}</span>
              <b className="toplist__pct">{(r.bust_probability * 100).toFixed(0)}%</b>
              <span className="toplist__cause">
                {r.dominant_variable ? variableLabel(r.dominant_variable) : ""}
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
