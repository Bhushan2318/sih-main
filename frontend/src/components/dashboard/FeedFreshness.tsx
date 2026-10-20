import { translate } from "../../i18n";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { triggerCycleRun } from "../../api/ingest";
import { useIngestStatus } from "../../hooks/useDashboardData";

export function FeedFreshness() {
  const { data, error } = useIngestStatus();
  const qc = useQueryClient();

  const refresh = useMutation({
    mutationFn: triggerCycleRun,
    onSuccess: () => {

      setTimeout(() => qc.invalidateQueries({ queryKey: ["ingestStatus"] }), 2000);
    },
  });

  if (error || !data) return null;

  const last = data.last_forecast;
  const failed = last?.status === "failed";
  const ageH = cycleAgeHours(data.last_cycle_ingested);
  const stale = ageH != null && ageH > STALE_AFTER_HOURS;

  return (
    <div className="feed-strip">
      <div className="feed-strip__main">
        <span className={`feed-dot feed-dot--${feedState(data.scheduler.running, failed)}`} />
        <span className="feed-strip__text">
          {data.last_cycle_ingested ? (
            <>
              {translate("feed.cycle")}{" "}
              <b title={translate("feed.utcTitle")}>
                {data.last_cycle_ingested}Z
              </b>
              {last?.finished_at ? translate("feed.ingested", { when: relative(last.finished_at) }) : null}
            </>
          ) : (
            <>{translate("feed.none")}</>
          )}
        </span>
      </div>

      <div className="feed-strip__meta">
        {data.scheduler.running ? (
          <span className="muted small">
            {translate("feed.auto", { min: Math.round(data.scheduler.tick_seconds / 60) })}
            {data.scheduler.next_tick ? translate("feed.next", { time: clock(data.scheduler.next_tick) }) : ""}
          </span>
        ) : (
          <span className="muted small">{translate("feed.off")}</span>
        )}
        {data.enabled ? (
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={() => refresh.mutate()}
            disabled={refresh.isPending || last?.status === "running"}
          >
            {refresh.isPending || last?.status === "running" ? translate("feed.pulling") : translate("feed.refresh")}
          </button>
        ) : null}
      </div>

      {stale ? (
        <p className="warn small feed-strip__error" role="status">
          {translate("feed.stale", { h: Math.round(ageH as number) })}
        </p>
      ) : null}
      {failed && last?.error ? (
        <p className="warn small feed-strip__error">{translate("feed.failed", { err: last.error })}</p>
      ) : null}
      {refresh.isSuccess ? (
        <p className="muted small feed-strip__error">
          {translate("feed.started")}
        </p>
      ) : null}
    </div>
  );
}

/**
 * A cycle is normally well under a day old: pulled every 6 h, by a cron that runs 3-5 h
 * late, on cycles published ~5.5 h after issue. Past 30 h something has stopped - the
 * 2026-09 outage served days-old data behind a healthy-looking strip (known-issues.md).
 */
export const STALE_AFTER_HOURS = 30;

/** Hours since a cycle labelled "YYYY-MM-DD HH" (UTC) was issued; null if unreadable. */
export function cycleAgeHours(label: string | null | undefined, now: number = Date.now()): number | null {
  const m = label?.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2})/);
  if (!m) return null;
  const issued = Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4]);
  return (now - issued) / 3_600_000;
}

function feedState(running: boolean, failed: boolean): string {
  if (failed) return "bad";
  return running ? "live" : "idle";
}

function relative(iso: string): string {
  const then = new Date(iso.endsWith("Z") ? iso : `${iso}Z`).getTime();
  const mins = Math.round((Date.now() - then) / 60000);
  if (!Number.isFinite(mins)) return translate("time.recently");
  if (mins < 1) return translate("time.now");
  if (mins < 60) return translate("time.min", { n: mins });
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return translate("time.h", { n: hrs });
  return translate("time.d", { n: Math.round(hrs / 24) });
}

function clock(iso: string): string {
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime())
    ? "—"
    : d.toLocaleTimeString("en-GB", {
        timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false,
      });
}
