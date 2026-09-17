/** Fixed-precision metric display. `dp` is required — the same-looking metric (e.g. bss,
 * roc_auc) is shown at different precision in different contexts, so there is no single
 * sane default; callers must say which they mean. */
export function formatMetric(v: unknown, dp: number): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(dp) : "—";
}

/** Precision that scales with magnitude, so a genuinely small value (0.03) still reads as
 * small instead of rounding to "0.0" next to a headline percentage. */
export function formatByMagnitude(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
