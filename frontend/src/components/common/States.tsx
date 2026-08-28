import type { ReactNode } from "react";

/**
 * The empty state is load-bearing in this project: when no model or no data exists the
 * UI must say so plainly rather than render a placeholder chart or a fake number.
 */
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

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return <div className="state state--loading">{label}</div>;
}

export function ErrorState({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="state state--error">
      <strong>Could not reach the API</strong>
      <p>{message}</p>
      <p className="muted">Is the backend running on {import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"}?</p>
    </div>
  );
}

export function RiskBadge({ band }: { band: string | null }) {
  if (!band) return <span className="badge badge--unknown">unknown</span>;
  return <span className={`badge badge--${band}`}>{band}</span>;
}

export function StatusBadge({ status, label }: { status: string; label: string }) {
  return <span className={`badge badge--${status}`}>{label}</span>;
}
