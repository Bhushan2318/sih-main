import type { Alert, RiskBand } from "../api/types";

/**
 * A district collapsed from its individual alert-days into one row: its worst bust
 * probability across whichever lead days are on alert, that row's own lead day and
 * cause, and how many lead days it appeared on at all.
 *
 * The Alerts table is one row per district-day today, so a district on alert for three
 * lead days shows three times - correct, but it reads as three separate problems
 * rather than one district worth three days of attention. "One row per district" is a
 * view, not a different dataset: everything here is still exactly what the alerts
 * response sent, just grouped.
 */
export interface GroupedAlert {
  region_id: string;
  region_name: string | null;
  bust_probability: number;
  risk_band: RiskBand;
  lead_time_days: number;
  valid_date: string | null;
  dominant_variable: string | null;
  /** How many of this district's lead days are in the list being grouped. */
  days_on_alert: number;
}

export function groupAlertsByDistrict(alerts: Alert[]): GroupedAlert[] {
  const byRegion = new Map<string, Alert[]>();
  for (const a of alerts) {
    const rows = byRegion.get(a.region_id);
    if (rows) rows.push(a);
    else byRegion.set(a.region_id, [a]);
  }

  const grouped: GroupedAlert[] = [];
  for (const rows of byRegion.values()) {
    let worst = rows[0];
    for (const r of rows) {
      if (r.bust_probability > worst.bust_probability) worst = r;
    }
    grouped.push({
      region_id: worst.region_id,
      region_name: worst.region_name,
      bust_probability: worst.bust_probability,
      risk_band: worst.risk_band,
      lead_time_days: worst.lead_time_days,
      valid_date: worst.valid_date,
      dominant_variable: worst.dominant_variable,
      days_on_alert: rows.length,
    });
  }

  return grouped.sort((a, b) => b.bust_probability - a.bust_probability);
}
