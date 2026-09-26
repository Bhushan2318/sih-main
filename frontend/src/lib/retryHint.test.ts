import { describe, expect, it } from "vitest";
import { retryingHint, slowHint } from "./retryHint";

/**
 * The point of this helper is that a visitor is told *something* while a query is being
 * retried. Measured against an API returning 500, every tab sat on an unexplained
 * "Loading..." for 28 seconds before the error state appeared, which is indistinguishable
 * from a broken site. So the case that matters is the first failure, not the last.
 */
describe("retryingHint", () => {
  it("says nothing on the first attempt, when nothing has gone wrong yet", () => {
    expect(retryingHint(0)).toBeUndefined();
  });

  it("speaks up as soon as one attempt has failed", () => {
    expect(retryingHint(1)).toBeTruthy();
  });

  it("counts attempts rather than failures, so the number matches what is happening now", () => {
    // One failure means the second attempt is the one in flight.
    expect(retryingHint(1)).toContain("attempt 2");
    expect(retryingHint(4)).toContain("attempt 5");
  });

  it("explains the wait instead of only reporting it", () => {
    // The cold start is the actual reason, and it is the part a visitor cannot guess.
    expect(retryingHint(2)).toMatch(/free tier|sleeps|wak/i);
  });

  it("is not defensive about odd input", () => {
    expect(retryingHint(-1)).toBeUndefined();
  });
});

describe("slowHint", () => {
  it("stays quiet for a normal wait", () => {
    expect(slowHint(0)).toBeUndefined();
    expect(slowHint(4.9)).toBeUndefined();
  });

  it("explains a long first load before anything has failed", () => {
    // A cold free-tier box answers, just slowly: no retry ever fires, so retryingHint
    // alone left a visitor on a blank screen for the whole wait.
    expect(slowHint(5)).toMatch(/free/i);
  });
});
