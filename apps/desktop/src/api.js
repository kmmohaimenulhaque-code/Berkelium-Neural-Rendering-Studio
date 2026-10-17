// api.js
// One place for talking to the Berkelium backend, so the URL and auth header
// are not copy-pasted across components.

// Where the backend is. The installed app starts its own backend on a free
// port with a private per-install key, and the Electron main process tells us
// both. Running from source (npm run studio, or Vite in a plain browser) uses
// the backend you start yourself on port 8000, which needs no key.
const DEV_BACKEND = { managed: false, url: "http://127.0.0.1:8000", key: "", state: "external", error: null };

// Asked fresh on every call (a local IPC round trip, well under a
// millisecond), because a restarted backend comes back on a new port.
export async function getBackend() {
  const bridge = typeof window !== "undefined" ? window.electronAPI : undefined;
  if (!bridge?.backendInfo) return DEV_BACKEND;
  try {
    return (await bridge.backendInfo()) || DEV_BACKEND;
  } catch {
    return DEV_BACKEND;
  }
}

function unreachable(backend) {
  if (!backend.managed) {
    return new ApiError(0, `Could not reach the backend. Is it running at ${backend.url}?`, "backend_down");
  }
  if (backend.state === "starting") {
    return new ApiError(0, "The AI engine is still starting. Try again in a few seconds.", "backend_starting");
  }
  return new ApiError(0, "The built-in AI engine is not running. Restart Berkelium. If it keeps happening, open the logs from Settings > About.", "backend_down");
}

export class ApiError extends Error {
  constructor(status, detail, code) {
    super(detail || `Backend returned ${status}`);
    this.status = status;
    this.code = code || null;
  }
}

export async function apiFetch(path, { method = "GET", body } = {}) {
  const backend = await getBackend();
  if (!backend.url) throw unreachable(backend); // the built-in one has not picked a port yet
  let res;
  try {
    res = await fetch(`${backend.url}${path}`, {
      method,
      headers: { "Content-Type": "application/json", "X-API-Key": backend.key || "" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    // fetch only throws when the request never reached a server.
    throw unreachable(backend);
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  if (!res.ok) {
    // FastAPI validation errors arrive as a list. Flatten to one readable line.
    let detail = data?.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join("; ");
    throw new ApiError(res.status, detail, data?.code);
  }
  return data;
}

// Other components (the chat model picker) listen for this to refresh
// their provider list after the user saves something in Settings.
export const PROVIDERS_CHANGED = "berkelium:providers-changed";

// Any component can ask the app to open the Settings window, for example
// the chat when the chosen provider is not set up yet.
export const OPEN_SETTINGS = "berkelium:open-settings";
