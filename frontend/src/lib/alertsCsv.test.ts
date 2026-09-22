import { describe, expect, it } from "vitest";
import type { Alert } from "../api/types";
import { alertsToCsv, csvFilename } from "./alertsCsv";

function alert(over: Partial<Alert> = {}): Alert {
  return {
    alert_id: "a1",
    region_id: "IN-TN-CHENNAI",
    region_name: "Chennai",
    lead_time_days: 3,
    valid_date: "2018-12-31",
    bust_probability: 0.8712,
    risk_band: "high",
    dominant_variable: "temperature_c",
    created_at: "2018-12-28T00:00:00Z",
    training_run_id: "run_1",
    ...over,
  };
}

describe("alertsToCsv", () => {
  it("writes a header even when there is nothing to export", () => {
    const csv = alertsToCsv([]);
    expect(csv.split("\n")[0]).toContain("region_id");
    expect(csv.trim().split("\n")).toHaveLength(1);
  });

  it("writes one row per alert, in the order given", () => {
    const csv = alertsToCsv([alert({ region_id: "A" }), alert({ region_id: "B" })]);
    const rows = csv.trim().split("\n");
    expect(rows).toHaveLength(3);
    expect(rows[1]).toContain("A");
    expect(rows[2]).toContain("B");
  });

  it("keeps the probability as a number, not a rendered percentage", () => {
    // A spreadsheet should get something it can compute with, not "87%".
    expect(alertsToCsv([alert({ bust_probability: 0.8712 })])).toContain("0.8712");
  });

  it("quotes a field containing a comma, so the columns do not shift", () => {
    const csv = alertsToCsv([alert({ region_name: "Thoothukkudi, Tamil Nadu" })]);
    expect(csv).toContain('"Thoothukkudi, Tamil Nadu"');
  });

  it("escapes embedded quotes by doubling them, as RFC 4180 requires", () => {
    const csv = alertsToCsv([alert({ region_name: 'He said "hi"' })]);
    expect(csv).toContain('"He said ""hi"""');
  });

  it("writes an empty cell for a missing value rather than the word null", () => {
    const csv = alertsToCsv([alert({ region_name: null, dominant_variable: null, valid_date: null })]);
    const cells = csv.trim().split("\n")[1].split(",");
    expect(cells).not.toContain("null");
    expect(cells).not.toContain("undefined");
  });

  it("cannot be used to smuggle a formula into a spreadsheet", () => {
    // A cell starting = + - or @ is executed by Excel and Sheets on open. Region names
    // come from a data file, but this export is a file a person opens, so it is neutered.
    const csv = alertsToCsv([alert({ region_name: "=1+1" })]);
    expect(csv).not.toMatch(/,=1\+1/);
    expect(csv).toContain("'=1+1");
  });

  it("ends every line with CRLF, which is what spreadsheets expect", () => {
    expect(alertsToCsv([alert()])).toContain("\r\n");
  });
});

describe("csvFilename", () => {
  it("names the file after the cycle it came from", () => {
    expect(csvFilename("2018-12-31")).toBe("sanket-alerts-2018-12-31.csv");
  });

  it("still returns a usable name when there is no date", () => {
    expect(csvFilename(null)).toMatch(/^sanket-alerts\.csv$/);
  });

  it("refuses a date that would escape the filename", () => {
    expect(csvFilename("../../etc/passwd")).toBe("sanket-alerts.csv");
  });
});
