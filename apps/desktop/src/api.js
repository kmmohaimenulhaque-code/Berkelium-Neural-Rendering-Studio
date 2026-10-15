// api.js
// One place for talking to the Berkelium backend, so the URL and auth header
// are not copy-pasted across components.

// VITE_BERKELIUM_API overrides it at build time (end-to-end tests run the backend on a free port).
export const BACKEND_URL = import.meta.env?.VITE_BERKELIUM_API ?? "http://127.0.0.1:8000";
export const BACKEND_API_KEY = ""; // Backend only enforces this when its config has a non-empty key.

export class ApiError extends Error {
  constructor(status, detail, code) {
    super(detail || `Backend returned ${status}`);
    this.status = status;
    this.code = code || null;
  }
}

export async function apiFetch(path, { method = "GET", body } = {}) {
  let res;
  try {
    res = await fetch(`${BACKEND_URL}${path}`, {
      method,
      headers: { "Content-Type": "application/json", "X-API-Key": BACKEND_API_KEY },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    // fetch only throws when the request never reached a server.
    throw new ApiError(0, `Could not reach the backend. Is it running at ${BACKEND_URL}?`, "backend_down");
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
