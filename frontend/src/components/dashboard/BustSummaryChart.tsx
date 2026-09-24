import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { RegionSummary } from "../../api/types";
import { isScoredRegion, riskBandForRegion, type RiskCuts } from "../../lib/riskBands";
import { CHART, bandLabel } from "../../theme";

export function BustSummaryChart({ regions, onSelect, riskCuts }: {
  regions: RegionSummary[];
  onSelect: (regionId: string) => void;
  riskCuts?: RiskCuts;
}) {
  if (!regions.length) return null;
  const data = regions
    .filter(isScoredRegion)
    .flatMap((r) => {
      const band = riskBandForRegion(r, riskCuts);
      return band
        ? [{
            region: r.region_name ?? r.region_id,
            regionId: r.region_id,
            probability: Number((r.bust_probability * 100).toFixed(1)),
            band,
          }]
        : [];
    })
    .slice(0, 10);
  if (!data.length) return null;

  return (
    <section className="card">
      <header className="card__head"><h3>Highest risk regions</h3></header>
      <ResponsiveContainer width="100%" height={Math.max(180, data.length * 26)}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke={CHART.grid} />
          <XAxis type="number" domain={[0, 100]} unit="%" stroke={CHART.axis} />
          <YAxis type="category" dataKey="region" width={132} tick={{ fontSize: 11 }} stroke={CHART.axis} />
          <Tooltip formatter={(v: number, _n, item) => [`${v}%`, bandLabel(item.payload.band)]} />
          <Bar dataKey="probability" radius={[0, 4, 4, 0]} onClick={(d: { regionId?: string }) => d.regionId && onSelect(d.regionId)}>
            {data.map((d) => <Cell key={d.regionId} className={`bar bar--${d.band}`} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}
