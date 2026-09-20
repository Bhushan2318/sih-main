import type { ReactNode } from "react";
import { API_BASE } from "../../api/client";
import { bandLabel } from "../../theme";

export function EmptyState({ title, message, action }: {
  title: string;
  message?: string | null;
  action?: ReactNode;
}) {
  return (
    <div className="state state--empty">
      <strong>{title}</strong>
      {message ? <p>{message}</p> : null}
      {action}
    </div>
  );
}

/**
 * `hint` is for waits the user would otherwise read as a broken page. Replay scores a
 * historical cycle on demand, so a cold request can run for seconds with nothing else on
 * screen - on a phone that is an empty viewport under one line of grey text.
 */
export function LoadingState({ label = "Loading…", hint }: { label?: string; hint?: string }) {
  return (
    <div className="state state--loading">
      {label}
      {hint ? <p className="state__hint">{hint}</p> : null}
    </div>
  );
}


export function ErrorState({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="state state--error">
      <strong>Could not reach the API</strong>
      <p>{message}</p>
      <p className="muted">{API_BASE ? `Is the backend running on ${API_BASE}?` : "The API is served from this same origin, so this is a server-side problem rather than a misconfigured address."}</p>
    </div>
  );
}

export function RiskBadge({ band }: { band: string | null }) {
  if (!band) return <span className="badge badge--unknown">unknown</span>;
  return <span className={`badge badge--${band}`}>{bandLabel(band)}</span>;
}
