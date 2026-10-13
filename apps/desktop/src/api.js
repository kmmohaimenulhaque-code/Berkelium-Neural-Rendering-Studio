// api.js
// One place for talking to the Berkelium backend, so the URL and auth header
// are not copy-pasted across components.

export const BACKEND_URL = "http://127.0.0.1:8000";
const BACKEND_API_KEY = ""; // Backend only enforces this when its config has a non-empty key.

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
    throw new ApiError(0, "Could not reach the backend. Is it running at 127.0.0.1:8000?", "backend_down");
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
