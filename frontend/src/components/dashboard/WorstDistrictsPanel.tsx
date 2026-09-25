import type { RegionsResponse } from "../../api/types";
import { dayLabel } from "../../lib/format";
import { isScoredRegion, riskBandForRegion, type RiskCuts } from "../../lib/riskBands";
import { EmptyState } from "../common/States";
import { TopDistrictsList } from "../common/TopDistrictsList";

/** How many districts the panel lists. Twenty fills the panel on a 1080p screen; on a
 * shorter one the panel scrolls, as it already does for a district. */
const LIMIT = 20;

/** What the district panel shows before a district is picked: the chosen day's worst
 * districts, worst first, each one click from its full view. It used to be a two-line
 * "No region selected" in a card as tall as the screen, while the only worst-district view
 * sat below the fold as ten near-identical red bars. */
export function WorstDistrictsPanel({ day, riskCuts, onSelect }: {
  day?: RegionsResponse;
  riskCuts?: RiskCuts;
  onSelect: (regionId: string) => void;
}) {
  const scored = (day?.regions ?? [])
    .filter(isScoredRegion)
    .sort((a, b) => b.bust_probability - a.bust_probability);

  if (!day || !scored.length) {
    return (
      <aside className="panel">
        <EmptyState title="No region selected" message="Open a state on the map and pick a district — or search for one directly." />
      </aside>
    );
  }

  let bust = 0;
  let watch = 0;
  for (const r of scored) {
    const band = riskBandForRegion(r, riskCuts);
    if (band === "high") bust += 1;
    else if (band === "medium") watch += 1;
  }

  const items = scored.slice(0, LIMIT).map((r) => ({
    region_id: r.region_id,
    region_name: r.region_name,
    bust_probability: r.bust_probability,
    band: riskBandForRegion(r, riskCuts),
    dominant_variable: r.dominant_variable,
  }));

  return (
    <aside className="panel" aria-label="Worst districts">
      <header className="panel__head">
        <div>
          <h2>Worst districts · {dayLabel(day.lead_time_days)}</h2>
          <p className="muted small">
            {day.valid_date ? `valid ${day.valid_date} · ` : ""}
            <b>{bust}</b> in the bust band · <b>{watch}</b> on watch · of {scored.length} districts
          </p>
        </div>
      </header>
      <TopDistrictsList items={items} onSelect={onSelect} selectTitle="Open this district" />
      <p className="muted small worst__hint">
        Click one to open it, or pick any district on the map or search for it.
      </p>
    </aside>
  );
}
