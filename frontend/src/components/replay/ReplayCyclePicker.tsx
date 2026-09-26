import type { ReplayCycleSummary } from "../../api/types";
import { formatByMagnitude } from "../../lib/format";

/** The cycle dropdown. Past events, whose outcome is known, come first in their own group;
 * the recent live cycles follow as they always have. With no past events it stays the flat
 * list it was, so an API without them reads exactly as before. */
export function ReplayCyclePicker({
  cycles,
  value,
  onChange,
}: {
  cycles: ReplayCycleSummary[];
  value: string;
  onChange: (initDate: string | undefined) => void;
}) {
  const events = cycles.filter((c) => c.kind === "event");
  const forecasts = cycles.filter((c) => c.kind !== "event");
  return (
    <div className="replay__pick">
      <label htmlFor="replay-cycle" className="muted small">Forecast cycle</label>
      <select id="replay-cycle" value={value} onChange={(e) => onChange(e.target.value || undefined)}>
        {events.length ? (
          <>
            <optgroup label="Past events · outcome known">
              {events.map((c) => (
                <option key={c.init_date} value={c.init_date}>{c.title ?? c.init_date}</option>
              ))}
            </optgroup>
            <optgroup label="Recent forecasts">{forecasts.map(forecastOption)}</optgroup>
          </>
        ) : (
          forecasts.map(forecastOption)
        )}
      </select>
    </div>
  );
}

function forecastOption(c: ReplayCycleSummary) {
  return (
    <option key={c.init_date} value={c.init_date}>
      {c.init_date}
      {c.verified
        ? ` · outcome known for ${c.verified_lead_days} day${c.verified_lead_days === 1 ? "" : "s"}`
        : " · outcome not yet known"}
      {c.peak_bust_probability != null
        ? ` · peak risk ${(c.peak_bust_probability * 100).toFixed(0)}%`
        : ""}
      {c.peak_region_abs_error != null
        ? ` · actual error there: ${formatByMagnitude(c.peak_region_abs_error)}${c.peak_region_unit ? ` ${c.peak_region_unit}` : ""}`
        : ""}
    </option>
  );
}
