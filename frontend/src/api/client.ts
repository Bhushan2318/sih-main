/**
 * Where the API lives.
 *
 * In development the Vite dev server and the API are two processes on two ports, so the
 * default has to be an absolute localhost URL. In a deployed build FastAPI serves the
 * built SPA itself, so the API is same-origin and the right base is the empty string -
 * hard-coding localhost there would make every request from the hosted page fail.
 *
 * `import.meta.env.DEV` is compiled to a constant by Vite, so this branch is resolved at
 * build time rather than sniffed from window.location at runtime. Either var still wins
 * when set, which is what a split deployment (API on another host) would need.
 */
const BASE = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg?: string }) => d.msg).join("; ");
    return JSON.stringify(body);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

/**
 * Parse a successful response, or explain why it could not be parsed.
 *
 * `res.json()` on its own leaks the browser's raw parser error to the user - Safari words
 * it "The string did not match the expected pattern", which tells nobody anything. A 200
 * carrying HTML is a specific, common situation on a host that sleeps: the platform's own
 * holding page is served while the container wakes, so say that rather than blaming JSON.
 */
async function readJson<T>(res: Response): Promise<T> {
  const text = await res.text();
  try {
    return JSON.parse(text) as T;
  } catch {
    const looksLikeHtml = /^\s*<(?:!doctype|html)/i.test(text);
    throw new ApiError(
      looksLikeHtml
        ? "The server sent a web page instead of data — it is probably still starting up. This usually clears within a minute."
        : `The server sent a response that is not JSON (${text.slice(0, 80)}…)`,
      res.status,
    );
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new ApiError(await parseError(res), res.status);
  return readJson<T>(res);
}

export async function apiPostJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(await parseError(res), res.status);
  return readJson<T>(res);
}

export async function apiPostFile<T>(path: string, file: File): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}${path}`, { method: "POST", body: form });
  if (!res.ok) throw new ApiError(await parseError(res), res.status);
  return readJson<T>(res);
}

export const API_BASE = BASE;

/**
 * The live socket, derived from the page's own origin in a deployed build so it follows
 * http->ws and https->wss without configuration. Render Free terminates TLS and does not
 * proxy websockets, so this will fail there - that is handled, not fatal: the socket
 * retries with backoff, reports "closed", and every query falls back to 60 s polling.
 */
export const WS_URL =
  import.meta.env.VITE_WS_URL ??
  (import.meta.env.DEV
    ? "ws://localhost:8000/ws"
    : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws`);
