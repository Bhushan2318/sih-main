import { geoMercator, geoPath } from "d3-geo";
import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { feature, merge } from "topojson-client";
import type { Feature, FeatureCollection, Geometry, MultiPolygon, Polygon } from "geojson";
import type {
  GeometryCollection,
  MultiPolygon as TopoMultiPolygon,
  Polygon as TopoPolygon,
  Topology,
} from "topojson-specification";
import topoData from "../../assets/geo/india_districts.topojson?url";
import claimedTerritoryUrl from "../../assets/geo/claimed_territory.geojson?url";
import type { RegionSummary, RiskBand } from "../../api/types";
import { bandLabel } from "../../theme";
import {
  inferRiskCuts,
  isScoredRegion,
  riskBandForProbability,
  riskBandForRegion,
  type RiskCuts,
} from "../../lib/riskBands";

const WIDTH = 620;
const HEIGHT = 680;
const MAX_SUGGESTIONS = 8;

interface DistrictProps {
  region_id: string;
  region_name: string;
  state_id: string;
  state_name: string;
}

export type Aggregation = "worst" | "mean";

function districtLabel(p: DistrictProps): string {
  // "Delhi, Delhi" reads as a bug rather than as a location.
  return p.region_name.toLowerCase() === p.state_name.toLowerCase()
    ? p.region_name
    : `${p.region_name}, ${p.state_name}`;
}

function bandFor(p: number | null, cuts?: RiskCuts | null) {
  return riskBandForProbability(p, cuts);
}

export function IndiaChoroplethMap({
  regions,
  selectedRegionId,
  onSelect,
  topology,
  riskCuts,
}: {
  regions: RegionSummary[];
  selectedRegionId: string | null;
  onSelect: (regionId: string) => void;
  topology: Topology | null;
  riskCuts?: RiskCuts;
}) {
  const [hover, setHover] = useState<{ x: number; y: number; title: string; body: string[] } | null>(null);
  const [query, setQuery] = useState("");
  const [activeState, setActiveState] = useState<string | null>(null);
  const [aggregation, setAggregation] = useState<Aggregation>("worst");
  const deferredQuery = useDeferredValue(query);

  // Gilgit-Baltistan, Aksai Chin, the Shaksgam Valley and Siachen: territory India
  // claims but does not administer, so no GADM district - disputed-prefixed or not -
  // exists there and the district file's own northern edge stops at 35.50 deg N,
  // well short of the ~37.05 deg N claim. Loaded once, independent of the district
  // topology, and drawn only as a silhouette - see build_claimed_territory_geo.py.
  const [claimedTerritory, setClaimedTerritory] =
    useState<Feature<Geometry, unknown> | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetch(claimedTerritoryUrl)
      .then((r) => r.json())
      .then((fc: FeatureCollection) => {
        if (!cancelled) setClaimedTerritory(fc.features[0] ?? null);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  const byRegionId = useMemo(() => {
    const m = new Map<string, RegionSummary>();
    regions.forEach((r) => m.set(r.region_id, r));
    return m;
  }, [regions]);

  const cuts = useMemo(
    () => riskCuts ?? inferRiskCuts(regions),
    [riskCuts, regions],
  );

  // Districts, and the state outlines built by merging them. Merging uses the topology's
  // shared arcs, so a state boundary is exactly its districts' outer edge - no second
  // asset to keep in step, and no slivers along the joins.
  const { districts, states } = useMemo(() => {
    if (!topology) return { districts: [], states: [] };
    const obj = topology.objects.districts as GeometryCollection<DistrictProps>;
    const fc = feature(topology, obj) as unknown as FeatureCollection<Geometry, DistrictProps>;

    // Every district is a Polygon or MultiPolygon; merge() takes only those, not the
    // wider GeometryObject union the collection is typed as.
    type Areal = TopoPolygon<DistrictProps> | TopoMultiPolygon<DistrictProps>;
    const grouped = new Map<string, { name: string; geoms: Areal[] }>();
    obj.geometries.forEach((g) => {
      if (g.type !== "Polygon" && g.type !== "MultiPolygon") return;
      const p = g.properties as DistrictProps;
      const entry = grouped.get(p.state_id) ?? { name: p.state_name, geoms: [] };
      entry.geoms.push(g as Areal);
      grouped.set(p.state_id, entry);
    });

    const stateFeatures = [...grouped.entries()].map(([stateId, { name, geoms }]) => ({
      type: "Feature" as const,
      properties: { state_id: stateId, state_name: name },
      geometry: merge(topology, geoms) as MultiPolygon | Polygon,
    }));
    return { districts: fc.features, states: stateFeatures };
  }, [topology]);

  // One district-level summary per state, so a state can be coloured by its districts.
  const stateRollup = useMemo(() => {
    const acc = new Map<string, {
      probs: number[];
      worst: RegionSummary | null;
      bands: RiskBand[];
    }>();
    districts.forEach((f) => {
      const r = byRegionId.get(f.properties.region_id);
      if (!r || !isScoredRegion(r)) return;
      const e = acc.get(f.properties.state_id) ?? { probs: [], worst: null, bands: [] };
      e.probs.push(r.bust_probability as number);
      const band = riskBandForRegion(r, cuts);
      if (band) e.bands.push(band);
      if (!e.worst || (e.worst.bust_probability ?? 0) < (r.bust_probability as number)) e.worst = r;
      acc.set(f.properties.state_id, e);
    });
    const out = new Map<string, {
      value: number;
      worst: RegionSummary | null;
      n: number;
      band: RiskBand | null;
    }>();
    acc.forEach((e, k) => {
      const value = aggregation === "worst"
        ? Math.max(...e.probs)
        : e.probs.reduce((a, b) => a + b, 0) / e.probs.length;
      const sameBand = e.bands.length > 0 && e.bands.every((band) => band === e.bands[0])
        ? e.bands[0]
        : null;
      const band = aggregation === "worst" && e.worst
        ? riskBandForRegion(e.worst, cuts)
        : bandFor(value, cuts) ?? sameBand;
      out.set(k, { value, worst: e.worst, n: e.probs.length, band });
    });
    return out;
  }, [districts, byRegionId, aggregation, cuts]);

  // The API keys regions by whatever the canonical store holds. Until the archive is
  // re-fetched at district resolution it holds states, which match no district id here -
  // and the map would render entirely grey with nothing saying why. Say why.
  const matched = useMemo(
    () => districts.reduce((n, f) => n + (byRegionId.has(f.properties.region_id) ? 1 : 0), 0),
    [districts, byRegionId],
  );
  const staleGrain = regions.length > 0 && matched === 0;

  const shown = useMemo(
    () => (activeState ? districts.filter((f) => f.properties.state_id === activeState) : []),
    [districts, activeState],
  );

  // Refit whenever the level changes, so a drilled state fills the frame. The claimed-
  // territory silhouette joins the fit only at the national level - a drilled state's
  // own frame should still fill with just that state's real districts - so the
  // projection actually zooms out to include it rather than clipping it at the edge.
  const pathFor = useMemo(() => {
    if (!topology) return () => "";
    const fitTo: Feature<Geometry, unknown>[] = activeState
      ? shown
      : claimedTerritory
        ? [...(states as never[]), claimedTerritory]
        : (states as never[]);
    if (!fitTo.length) return () => "";
    const fc = { type: "FeatureCollection", features: fitTo } as FeatureCollection;
    const projection = geoMercator().fitSize([WIDTH, HEIGHT], fc);
    const path = geoPath(projection);
    return (g: Geometry) => path(g) ?? "";
  }, [topology, activeState, shown, states, claimedTerritory]);

  const suggestions = useMemo(() => {
    const q = deferredQuery.trim().toLowerCase();
    if (q.length < 2) return [];
    const hits: DistrictProps[] = [];
    for (const f of districts) {
      const p = f.properties;
      if (p.region_name.toLowerCase().includes(q) || p.state_name.toLowerCase().includes(q)) {
        hits.push(p);
        if (hits.length >= MAX_SUGGESTIONS) break;
      }
    }
    return hits;
  }, [deferredQuery, districts]);

  if (!topology) return <div className="state state--loading">Loading map…</div>;

  const activeStateName = activeState
    ? states.find((s) => s.properties.state_id === activeState)?.properties.state_name
    : null;

  return (
    <div className="map-wrap">
      <div className="map-controls">
        <div className="map-search">
          <input
            type="search"
            value={query}
            placeholder={activeState ? `Find a district…` : "Find any district in India…"}
            aria-label="Find a district"
            onChange={(e) => setQuery(e.target.value)}
          />
          {suggestions.length > 0 ? (
            <ul className="map-search__results" role="listbox">
              {suggestions.map((p) => (
                <li key={p.region_id}>
                  <button
                    type="button"
                    onClick={() => {
                      // Drill to the district's state as well as selecting it, or the
                      // selection would be invisible on a national view.
                      setActiveState(p.state_id);
                      onSelect(p.region_id);
                      setQuery("");
                    }}
                  >
                    {districtLabel(p)}
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </div>

        <div className="map-controls__row">
          {activeState ? (
            <button type="button" className="map-back" onClick={() => setActiveState(null)}>
              ← All India
            </button>
          ) : (
            <span className="map-level">All India · {states.length} states and UTs</span>
          )}
          {!activeState ? (
            <div className="map-agg" role="group" aria-label="How a state is coloured">
              {(["worst", "mean"] as Aggregation[]).map((a) => (
                <button
                  key={a}
                  type="button"
                  className={aggregation === a ? "is-on" : ""}
                  onClick={() => setAggregation(a)}
                >
                  {a === "worst" ? "Worst district" : "Mean of districts"}
                </button>
              ))}
            </div>
          ) : (
            <span className="map-level">{activeStateName} · {shown.length} districts</span>
          )}
        </div>
      </div>

      {staleGrain ? (
        <p className="map-notice">
          This forecast cycle is stored by state, not by district, so the map has nothing
          to colour. It fills in once the archive is re-fetched at district resolution.
        </p>
      ) : null}

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="map"
        role="img"
        aria-label={activeState ? `${activeStateName} by district` : "India forecast bust risk by state"}
      >
        <g>
          {!activeState && claimedTerritory ? (
            <path
              className={`region region--claimed region--${
                stateRollup.get("IN-JK")?.band ?? "nodata"}`}
              d={pathFor(claimedTerritory.geometry)}
              aria-label="Claimed territory, not scored: no district-level forecast data"
              // Hoverable but not clickable: there is nothing to drill into, and the
              // colour is inherited from Jammu and Kashmir rather than measured here.
              // Saying so on hover is the whole point - otherwise this reads as a
              // scored region.
              onMouseMove={(e) => {
                const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                setHover({
                  x: e.clientX - rect.left,
                  y: e.clientY - rect.top,
                  title: "Claimed territory",
                  body: [
                    "Not scored - no forecast or observation data",
                    "Shown so India's outline is complete",
                    "Shaded like Jammu and Kashmir, not measured",
                  ],
                });
              }}
              onMouseLeave={() => setHover(null)}
            />
          ) : null}
          {!activeState
            ? states.map((f) => {
                const roll = stateRollup.get(f.properties.state_id);
                const band = roll?.band ?? null;
                return (
                  <path
                    key={f.properties.state_id}
                    d={pathFor(f.geometry)}
                    className={["region", band ? `region--${band}` : "region--nodata"].join(" ")}
                    tabIndex={0}
                    role="button"
                    aria-label={`${f.properties.state_name}${band ? `, ${bandLabel(band)} risk` : ", no data"}. Open districts.`}
                    onClick={() => setActiveState(f.properties.state_id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setActiveState(f.properties.state_id);
                      }
                    }}
                    onMouseMove={(e) => {
                      const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                      setHover({
                        x: e.clientX - rect.left,
                        y: e.clientY - rect.top,
                        title: f.properties.state_name,
                        body: roll
                          ? [
                              `${aggregation === "worst" ? "Worst district" : "Mean of districts"}: ${(roll.value * 100).toFixed(1)}%`,
                              roll.worst ? `Worst: ${roll.worst.region_name}` : "",
                              `${roll.n} district${roll.n === 1 ? "" : "s"} scored`,
                              "Click to open districts",
                            ].filter(Boolean)
                          : ["No district in this state is scored yet"],
                      });
                    }}
                    onMouseLeave={() => setHover(null)}
                  />
                );
              })
            : shown.map((f) => {
                const p = f.properties;
                const region = byRegionId.get(p.region_id);
                const scored = region ? isScoredRegion(region) : false;
                 const band = scored && region ? riskBandForRegion(region, cuts) : null;
                return (
                  <path
                    key={p.region_id}
                    d={pathFor(f.geometry)}
                    className={[
                      "region",
                      band ? `region--${band}` : "region--nodata",
                      p.region_id === selectedRegionId ? "region--selected" : "",
                    ].join(" ")}
                    tabIndex={scored ? 0 : -1}
                    role={scored ? "button" : undefined}
                    aria-label={`${districtLabel(p)}${band ? `, ${bandLabel(band)} risk` : ", no data"}`}
                    onClick={() => { if (scored) onSelect(p.region_id); }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        if (scored) onSelect(p.region_id);
                      }
                    }}
                    onMouseMove={(e) => {
                      const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                      setHover({
                        x: e.clientX - rect.left,
                        y: e.clientY - rect.top,
                        title: districtLabel(p),
                        body: scored && region
                          ? [
                              `Bust probability: ${((region.bust_probability ?? 0) * 100).toFixed(1)}% (${bandLabel(band)})`,
                              region.dominant_variable ? `Driver: ${region.dominant_variable}` : "",
                              region.confidence != null ? `Mean confidence: ${(region.confidence * 100).toFixed(0)}%` : "",
                            ].filter(Boolean)
                          : ["No forecast data for this district"],
                      });
                    }}
                    onMouseLeave={() => setHover(null)}
                  />
                );
              })}
        </g>
      </svg>

      {hover ? (
        <div className="map-tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          <strong>{hover.title}</strong>
          {hover.body.map((line) => (
            <div key={line}>{line}</div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export async function loadTopology(): Promise<Topology> {
  const res = await fetch(topoData);
  if (!res.ok) throw new Error(`could not load India districts topojson (${res.status})`);
  return (await res.json()) as Topology;
}
