import type { GeometryCollection, Topology } from "topojson-specification";

interface DistrictProps {
  state_id: string;
  state_name: string;
}

/**
 * `state_id` -> `state_name`, read from the district topology's own properties.
 *
 * The regions API sends no state field, only `region_id` (`IN-<state>-<district>`), so
 * anything grouping districts by state needs names from somewhere. Taking them from the
 * topology the map already loads means the labels cannot drift from the ones the
 * choropleth draws - and avoids a second hand-maintained code table, which the geo
 * README warns about.
 */
export function stateNamesFrom(topology: Topology | null): Map<string, string> {
  const out = new Map<string, string>();
  if (!topology) return out;
  const obj = topology.objects.districts as GeometryCollection<DistrictProps>;
  obj.geometries.forEach((g) => {
    // topojson-specification types `properties` as `{} | P`, since a geometry need not
    // carry any - the district file's always do.
    const p = g.properties as DistrictProps | undefined;
    if (p?.state_id && !out.has(p.state_id)) out.set(p.state_id, p.state_name);
  });
  return out;
}

/** `IN-MH-NAGPUR` -> `IN-MH`. The region_id scheme is `IN-<state>-<district>` and the
 * API sends no separate state field, so the prefix is the only state grouping key that
 * exists anywhere in served data - the same derivation RiskTicker uses to roll districts
 * up to states. */
export function stateIdOf(regionId: string): string {
  return regionId.split("-").slice(0, 2).join("-");
}
