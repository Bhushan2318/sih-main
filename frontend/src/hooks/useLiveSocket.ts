import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { WS_URL } from "../api/client";
import type { LiveEvent } from "../api/types";
import { useLiveStore } from "../store/liveStore";

/** Polling is deliberately finite: a disconnected browser should recover, not poll forever. */
export const LIVE_FALLBACK_POLL_MS = 15_000;
export const LIVE_FALLBACK_MAX_MS = 5 * 60_000;
const CONNECT_TIMEOUT_MS = 10_000;

export function useLiveSocket() {
  const queryClient = useQueryClient();
  const setStatus = useLiveStore((s) => s.setStatus);
  const pushEvent = useLiveStore((s) => s.pushEvent);
  const retryRef = useRef(0);
  const closedRef = useRef(false);

  useEffect(() => {
    closedRef.current = false;
    retryRef.current = 0;

    let ws: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let connectTimer: number | undefined;
    let fallbackTimer: number | undefined;
    let fallbackDeadline = 0;
    let hasConnected = false;

    const invalidate = (queryKey: readonly string[]) => {
      void queryClient.invalidateQueries({ queryKey });
    };

    const refreshLiveData = () => {
      // A close/open gap can contain training_complete or new_alert frames. Refreshing
      // the two expensive, time-sensitive views on open closes that gap; the other
      // queries are cheap enough to refresh as well so the header cannot get stuck.
      invalidate(["ensemble"]);
      invalidate(["replay"]);
      invalidate(["regions"]);
      invalidate(["alerts"]);
      invalidate(["modelStatus"]);
    };

    const clearFallback = (resetWindow = false) => {
      if (fallbackTimer != null) {
        window.clearTimeout(fallbackTimer);
        fallbackTimer = undefined;
      }
      if (resetWindow) fallbackDeadline = 0;
    };

    const startFallback = () => {
      if (closedRef.current) return;

      // One bounded window per outage. A reconnect resets it; a failed reconnect after
      // the window does not start an accidental infinite polling loop.
      const now = Date.now();
      if (fallbackDeadline && now >= fallbackDeadline) return;
      if (!fallbackDeadline) fallbackDeadline = now + LIVE_FALLBACK_MAX_MS;
      if (fallbackTimer != null) return;

      const poll = () => {
        fallbackTimer = undefined;
        if (closedRef.current) return;
        const remaining = fallbackDeadline - Date.now();
        if (remaining <= 0) return;

        // Only active queries are refetched by React Query. Inactive replay/ensemble
        // queries are marked stale, so opening either view still gets fresh data.
        invalidate(["ensemble"]);
        invalidate(["replay"]);
        fallbackTimer = window.setTimeout(poll, Math.min(LIVE_FALLBACK_POLL_MS, remaining));
      };

      // Poll once immediately on disconnect, then at the bounded interval.
      fallbackTimer = window.setTimeout(poll, 0);
    };

    const scheduleReconnect = () => {
      if (closedRef.current || reconnectTimer != null) return;
      const delay = Math.min(30_000, 1000 * 2 ** retryRef.current++);
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = undefined;
        connect();
      }, delay);
    };

    const armConnectTimeout = () => {
      if (connectTimer != null) window.clearTimeout(connectTimer);
      connectTimer = window.setTimeout(() => {
        connectTimer = undefined;
        if (closedRef.current || ws?.readyState === WebSocket.OPEN) return;
        setStatus("closed");
        startFallback();
        try {
          ws?.close();
        } catch {
          // Some WebSocket shims throw when close is called during CONNECTING.
        }
        scheduleReconnect();
      }, CONNECT_TIMEOUT_MS);
    };

    function connect() {
      if (closedRef.current) return;
      setStatus("connecting");

      if (typeof WebSocket === "undefined") {
        setStatus("closed");
        startFallback();
        return;
      }

      try {
        ws = new WebSocket(WS_URL);
      } catch {
        setStatus("closed");
        startFallback();
        scheduleReconnect();
        return;
      }

      armConnectTimeout();

      ws.onopen = () => {
        if (connectTimer != null) {
          window.clearTimeout(connectTimer);
          connectTimer = undefined;
        }
        if (reconnectTimer != null) {
          window.clearTimeout(reconnectTimer);
          reconnectTimer = undefined;
        }
        const reconnected = hasConnected || fallbackDeadline !== 0;
        hasConnected = true;
        retryRef.current = 0;
        setStatus("open");
        clearFallback(true);
        // A close/open gap can contain training_complete or new_alert frames. Refresh
        // once on every reconnect so the REST views cannot stay stale after a missed event.
        if (reconnected) refreshLiveData();
      };

      ws.onmessage = (raw) => {
        let event: LiveEvent;
        try {
          event = JSON.parse(raw.data as string);
        } catch {
          return;
        }
        pushEvent(event);

        switch (event.event) {
          case "training_complete":
          case "new_alert":
            invalidate(["regions"]);
            invalidate(["alerts"]);
            invalidate(["modelStatus"]);
            invalidate(["ensemble"]);
            invalidate(["replay"]);
            break;
          case "training_started":
          case "training_failed":
          case "upload_received":
          case "mapping_pending":
          case "connected":
            invalidate(["modelStatus"]);
            invalidate(["regions"]);
            invalidate(["alerts"]);
            invalidate(["ensemble"]);
            invalidate(["replay"]);
            break;
        }
      };

      ws.onclose = () => {
        if (connectTimer != null) {
          window.clearTimeout(connectTimer);
          connectTimer = undefined;
        }
        setStatus("closed");
        startFallback();
        if (closedRef.current) return;
        scheduleReconnect();
      };

      ws.onerror = () => {
        try {
          ws?.close();
        } catch {
          // onclose (or the bounded connect timeout) will handle the retry.
        }
      };
    }

    connect();
    return () => {
      closedRef.current = true;
      if (reconnectTimer != null) window.clearTimeout(reconnectTimer);
      if (connectTimer != null) window.clearTimeout(connectTimer);
      clearFallback();
      try {
        ws?.close();
      } catch {
        // Cleanup must not throw while React is unmounting.
      }
    };
  }, [queryClient, setStatus, pushEvent]);
}
