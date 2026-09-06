import { geoMercator, geoPath } from "d3-geo";
import { useDeferredValue, useMemo, useState } from "react";
import { feature } from "topojson-client";
import type { FeatureCollection, Geometry } from "geojson";
import type { Topology } from "topojson-specification";
import topoData from "../../assets/geo/india_districts.topojson?url";
import type { RegionSummary } from "../../api/types";
import { bandLabel } from "../../theme";

const WIDTH = 620;
const HEIGHT = 680;
const MAX_SUGGESTIONS = 8;

// The TopoJSON carries region_id directly, so there is no lookup table between the map
// and the API any more. The states file was keyed by census code and needed one.
interface DistrictProps {
  region_id: string;
  region_name: string;
  state_id: string;
  state_name: string;
}

function label(p: DistrictProps): string {
  // Delhi, Karnataka, Sikkim and Tamil Nadu all have a district called "North".
  return `${p.region_name}, ${p.state_name}`;
}

export function IndiaChoroplethMap({
  regions,
  selectedRegionId,
  onSelect,
  topology,
}: {
  regions: RegionSummary[];
  selectedRegionId: string | null;
  onSelect: (regionId: string) => void;
  topology: Topology | null;
}) {
  const [hover, setHover] = useState<{ x: number; y: number; region: RegionSummary | null; name: string } | null>(null);
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);

  const byRegionId = useMemo(() => {
    const m = new Map<string, RegionSummary>();
    regions.forEach((r) => m.set(r.region_id, r));
    return m;
  }, [regions]);

  const { features, pathFor } = useMemo(() => {
    if (!topology) return { features: [], pathFor: () => "" };
    const fc = feature(topology, topology.objects.districts) as unknown as FeatureCollection<Geometry, DistrictProps>;
    const projection = geoMercator().fitSize([WIDTH, HEIGHT], fc);
    const path = geoPath(projection);
    return { features: fc.features, pathFor: (g: Geometry) => path(g) ?? "" };
  }, [topology]);

  // 666 districts cannot be found by clicking around. Deferred so typing stays responsive
  // while the map itself re-renders.
  const suggestions = useMemo(() => {
    const q = deferredQuery.trim().toLowerCase();
    if (q.length < 2) return [];
    const hits: DistrictProps[] = [];
    for (const f of features) {
      const p = f.properties;
      if (p.region_name.toLowerCase().includes(q) || p.state_name.toLowerCase().includes(q)) {
        hits.push(p);
        if (hits.length >= MAX_SUGGESTIONS) break;
      }
    }
    return hits;
  }, [deferredQuery, features]);

  if (!topology) return <div className="state state--loading">Loading map…</div>;

  return (
    <div className="map-wrap">
      <div className="map-search">
        <input
          type="search"
          value={query}
          placeholder="Find a district…"
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
                    onSelect(p.region_id);
                    setQuery("");
                  }}
                >
                  {label(p)}
                </button>
              </li>
            ))}
          </ul>
        ) : null}
      </div>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="map"
        role="img"
        aria-label="India forecast bust risk by district"
      >
        <g>
          {features.map((f) => {
            const p = f.properties;
            const region = byRegionId.get(p.region_id);
            const band = region?.risk_band ?? null;
            const selected = p.region_id === selectedRegionId;
            return (
              <path
                key={p.region_id}
                d={pathFor(f.geometry)}
                className={[
                  "region",
                  band ? `region--${band}` : "region--nodata",
                  selected ? "region--selected" : "",
                ].join(" ")}
                tabIndex={region ? 0 : -1}
                role={region ? "button" : undefined}
                aria-label={`${label(p)}${band ? `, ${bandLabel(band)} risk` : ", no data"}`}
                onClick={() => onSelect(p.region_id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect(p.region_id);
                  }
                }}
                onMouseMove={(e) => {
                  const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                  setHover({
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top,
                    region: region ?? null,
                    name: label(p),
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
          <strong>{hover.name}</strong>
          {hover.region ? (
            <>
              <div>
                Bust probability:{" "}
                <b>{((hover.region.bust_probability ?? 0) * 100).toFixed(1)}%</b> ({bandLabel(hover.region.risk_band)})
              </div>
              {hover.region.dominant_variable ? <div>Driver: {hover.region.dominant_variable}</div> : null}
              {hover.region.confidence != null ? (
                <div>Mean confidence: {(hover.region.confidence * 100).toFixed(0)}%</div>
              ) : null}
            </>
          ) : (
            <div className="muted">No forecast data for this district</div>
          )}
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
