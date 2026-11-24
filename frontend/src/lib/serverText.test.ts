import { afterEach, describe, expect, it } from "vitest";
import { useI18n } from "../i18n";
import { serverText, stepNarration } from "./serverText";
import type { TextPart } from "../api/types";

// Parts in the shape app/services/server_text.py sends for one Replay lead day. The
// values are illustrative plumbing, not taken from a scored cycle.
const STEP: TextPart[] = [
  { key: "srv.rp.climbs", params: { region: "Idukki, Kerala", from: "0.41", to: "0.77" } },
  { key: "srv.rp.nowDriven", params: { var: "rainfall", var_id: "rainfall_mm" } },
  { key: "srv.rp.nowHighN", params: { n: "12" } },
];

describe("server-written text", () => {
  afterEach(() => useI18n.setState({ lang: "en", catalog: {} }));

  it("renders in English exactly as the API's own string", () => {
    expect(stepNarration(3, STEP, "fallback")).toBe(
      "Day 3 — bust risk over Idukki, Kerala climbs 0.41→0.77; now driven by rainfall error; 12 regions now high-risk.");
  });

  it("uses the catalogue of the language on screen", () => {
    useI18n.setState({ lang: "hi", catalog: { "srv.rp.nowHighN": "{{n}} क्षेत्र अब उच्च जोखिम में" } });
    expect(stepNarration(3, STEP, "fallback")).toContain("12 क्षेत्र अब उच्च जोखिम में");
  });

  it("falls back to the API's English when there are no parts or an unknown key", () => {
    expect(serverText(undefined, "as sent")).toBe("as sent");
    expect(serverText([], "as sent")).toBe("as sent");
    expect(serverText([{ key: "srv.nope", params: {} }], "as sent")).toBe("as sent");
  });
});
