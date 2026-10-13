// Thin client for the Berkelium neural service (/v1/neural on the existing FastAPI backend).
// The renderer only ever sends typed JSON requests and fetches immutable artifacts by id; it never receives
// filesystem paths, device handles or credentials beyond the existing backend API key.

export const API_BASE = import.meta.env.VITE_BERKELIUM_API ?? "http://127.0.0.1:8000";
const BACKEND_API_KEY = ""; // same policy as ChatPanel: enforced only when the backend config sets a key

const HEX64 = /^[0-9a-f]{64}$/;
const HEX32 = /^[0-9a-f]{32}$/;

export class NeuralError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function call(path, init = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", "X-API-Key": BACKEND_API_KEY, ...(init.headers || {}) },
    });
  } catch {
    throw new NeuralError(0, "BACKEND_OFFLINE", `Backend unreachable at ${API_BASE}.`);
  }
  if (!res.ok) {
    let body = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    const d = body?.detail;
    const code = (d && typeof d === "object" && !Array.isArray(d) && d.code) || `HTTP_${res.status}`;
    const msg =
      res.status === 503 ? "No trained model is available on this server for that request."
        : res.status === 429 ? "The neural job queue is full. Wait for a job to finish."
          : res.status === 422 ? "The request was rejected by validation (check the prompt)."
            : res.status === 401 ? "Backend rejected the API key."
              : (d && d.message) || `Backend returned ${res.status}.`;
    throw new NeuralError(res.status, code, msg);
  }
  return res;
}

export async function capabilities() {
  return (await call("/v1/neural/capabilities")).json();
}

export async function submitTextTo3D({ prompt, adapter, seed = 0, resolution = 48 }) {
  const body = { kind: "text_to_3d", prompt, seed, resolution, ...(adapter ? { adapter } : {}) };
  return (await call("/v1/neural/jobs", { method: "POST", body: JSON.stringify(body) })).json();
}

export async function getJob(id) {
  if (!HEX32.test(id)) throw new NeuralError(0, "INVALID_ARGUMENT", "malformed job id");
  return (await call(`/v1/neural/jobs/${id}`)).json();
}

export async function cancelJob(id) {
  if (!HEX32.test(id)) throw new NeuralError(0, "INVALID_ARGUMENT", "malformed job id");
  return (await call(`/v1/neural/jobs/${id}/cancel`, { method: "POST" })).json();
}

export async function getArtifact(id) {
  if (!HEX64.test(id)) throw new NeuralError(0, "INVALID_ARGUMENT", "malformed artifact id");
  return (await call(`/v1/neural/artifacts/${id}`)).json();
}

export function fileUrl(id, name) {
  if (!HEX64.test(id) || !/^[A-Za-z0-9._-]{1,128}$/.test(name)) throw new Error("invalid artifact file reference");
  return `${API_BASE}/v1/neural/artifacts/${id}/files/${name}`;
}

export async function fetchFile(id, name) {
  fileUrl(id, name); // validates id + name before any request
  const res = await call(`/v1/neural/artifacts/${id}/files/${name}`);
  return res.arrayBuffer();
}

// berkelium-gaussian-soa/1 -> { n, means: Float32Array, colors: Float32Array } (centres + sigmoid(colour logits)).
export function parseGaussianScene(payload, manifest) {
  if (manifest.format !== "berkelium-gaussian-soa/1") throw new Error(`unsupported scene format ${manifest.format}`);
  const plane = (name) => manifest.planes.find((p) => p.name === name);
  const m = plane("means");
  const c = plane("color_logits");
  if (!m || !c || m.offset + m.bytes > payload.byteLength || c.offset + c.bytes > payload.byteLength) {
    throw new Error("Gaussian scene payload is truncated");
  }
  const means = new Float32Array(payload.slice(m.offset, m.offset + m.bytes));
  const logits = new Float32Array(payload.slice(c.offset, c.offset + c.bytes));
  const colors = logits.map((x) => 1 / (1 + Math.exp(-x)));
  return { n: manifest.n, means, colors };
}
