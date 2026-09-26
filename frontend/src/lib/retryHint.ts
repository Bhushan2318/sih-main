/**
 * What to say while a request is being retried, or nothing on the first attempt.
 *
 * Queries retry five times with exponential backoff, which is the right policy here -
 * this is a free-tier deployment that sleeps when idle, so the first request after a
 * quiet spell genuinely does fail before it succeeds. What was wrong is that the screen
 * said "Loading..." throughout: measured against an API returning 500, every tab sat on
 * an unexplained spinner for **28 seconds** before admitting anything was wrong. A
 * visitor opening a cold link has no way to tell that from a broken site.
 *
 * Retry counts are not exposed anywhere else, so this has to be read from the query and
 * passed down.
 */
/**
 * What to say while a first request is merely slow. A cold free-tier box does answer, so no
 * retry fires and `retryingHint` stays silent: measured 2026-09-26, the home page sat on one
 * grey line for ~12 s warm and longer after an idle spell.
 */
export function slowHint(elapsedSeconds: number): string | undefined {
  if (elapsedSeconds < 5) return undefined;
  return (
    "Still working: this scores every district for all ten days. It runs on a free server " +
    "that sleeps when idle, so the first visit after a quiet spell can take up to a minute."
  );
}

export function retryingHint(failureCount: number): string | undefined {
  if (failureCount < 1) return undefined;
  return (
    `The server has not answered yet — retrying (attempt ${failureCount + 1}). ` +
    "This runs on a free tier that sleeps when idle, so waking it can take up to a minute."
  );
}
