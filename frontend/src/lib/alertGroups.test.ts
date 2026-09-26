import { describe, expect, it } from "vitest";
import type { Alert } from "../api/types";
import { groupAlertsByDistrict } from "./alertGroups";

function alert(over: Partial<Alert> = {}): Alert {
  return {
    alert_id: "a1",
    region_id: "IN-TN-CHENNAI",
    region_name: "Chennai, Tamil Nadu",
    lead_time_days: 3,
    valid_date: "2018-12-31",
    bust_probability: 0.5,
    risk_band: "medium",
    dominant_variable: "temperature_c",
    created_at: "2018-12-28T00:00:00Z",
    training_run_id: "run_1",
    ...over,
  };
}

describe("groupAlertsByDistrict", () => {
  it("collapses a district's repeated lead days into one row, keeping its worst", () => {
    const alerts = [
      alert({ alert_id: "a1", region_id: "IN-UT-CHAMPAWAT", region_name: "Champawat, Uttarakhand",
        lead_time_days: 1, bust_probability: 0.6, risk_band: "medium", valid_date: "2026-09-25" }),
      alert({ alert_id: "a2", region_id: "IN-UT-CHAMPAWAT", region_name: "Champawat, Uttarakhand",
        lead_time_days: 2, bust_probability: 0.91, risk_band: "high", valid_date: "2026-09-26" }),
      alert({ alert_id: "a3", region_id: "IN-TN-CHENNAI", lead_time_days: 1, bust_probability: 0.55 }),
    ];
    const grouped = groupAlertsByDistrict(alerts);
    expect(grouped).toHaveLength(2);

    const champawat = grouped.find((g) => g.region_id === "IN-UT-CHAMPAWAT");
    expect(champawat?.bust_probability).toBe(0.91);
    expect(champawat?.risk_band).toBe("high");
    expect(champawat?.lead_time_days).toBe(2);
    expect(champawat?.valid_date).toBe("2026-09-26");
    expect(champawat?.days_on_alert).toBe(2);
  });

  it("reports a single alert-day as on alert for 1 day", () => {
    const grouped = groupAlertsByDistrict([alert({ region_id: "IN-TN-CHENNAI", lead_time_days: 5 })]);
    expect(grouped).toHaveLength(1);
    expect(grouped[0].days_on_alert).toBe(1);
    expect(grouped[0].lead_time_days).toBe(5);
  });

  it("sorts districts by their worst bust probability, most severe first", () => {
    const grouped = groupAlertsByDistrict([
      alert({ region_id: "A", bust_probability: 0.3 }),
      alert({ region_id: "B", bust_probability: 0.95 }),
      alert({ region_id: "C", bust_probability: 0.6 }),
    ]);
    expect(grouped.map((g) => g.region_id)).toEqual(["B", "C", "A"]);
  });

  it("carries the worst row's dominant cause, not the first one seen", () => {
    const grouped = groupAlertsByDistrict([
      alert({ region_id: "A", bust_probability: 0.4, dominant_variable: "humidity_pct" }),
      alert({ region_id: "A", bust_probability: 0.85, dominant_variable: "rainfall_mm" }),
    ]);
    expect(grouped[0].dominant_variable).toBe("rainfall_mm");
  });

  it("returns an empty array for no alerts", () => {
    expect(groupAlertsByDistrict([])).toEqual([]);
  });
});
