// neuralApi.js
// Thin client for the Berkelium neural service (/v1/neural on the same backend as chat).
// The UI only ever sends typed JSON requests and fetches immutable artifacts by id; it never receives
// filesystem paths, device handles or credentials beyond the existing backend API key.

import { BACKEND_URL, BACKEND_API_KEY } from "./api.js";

export const API_BASE = BACKEND_URL;

const HEX64 = /^[0-9a-f]{64}$/;
const HEX32 = /^[0-9a-f]{32}$/;
const FILE_NAME = /^[A-Za-z0-9._-]{1,128}$/;

export const TERMINAL = new Set(["completed", "failed", "cancelled"]);

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
    throw new NeuralError(0, "BACKEND_OFFLINE", `Backend unreachable at ${API_BASE}. Start it with scripts/start_backend.sh.`);
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

const json = async (path, init) => (await call(path, init)).json();

export const capabilities = () => json("/v1/neural/capabilities");

export function submitTextTo3D({ prompt, adapter, seed = 0, resolution = 48 }) {
  const body = { kind: "text_to_3d", prompt, seed, resolution, ...(adapter ? { adapter } : {}) };
  return json("/v1/neural/jobs", { method: "POST", body: JSON.stringify(body) });
}

// Render a stored Gaussian scene with a real renderer from an explicit camera (see scene/neuralCamera.js).
export function submitRender({ artifactId, camera, background = [0, 0, 0], backend = "reference_cpu" }) {
  if (!HEX64.test(artifactId)) throw new NeuralError(0, "INVALID_ARGUMENT", "malformed artifact id");
  const body = { kind: "render", artifact_id: artifactId, camera, backend, config: { background } };
  return json("/v1/neural/jobs", { method: "POST", body: JSON.stringify(body) });
}

export const listJobs = (limit = 12) => json(`/v1/neural/jobs?limit=${limit}`);

export function getJob(id) {
  if (!HEX32.test(id)) throw new NeuralError(0, "INVALID_ARGUMENT", "malformed job id");
  return json(`/v1/neural/jobs/${id}`);
}

export function cancelJob(id) {
  if (!HEX32.test(id)) throw new NeuralError(0, "INVALID_ARGUMENT", "malformed job id");
  return json(`/v1/neural/jobs/${id}/cancel`, { method: "POST" });
}

export function getArtifact(id) {
  if (!HEX64.test(id)) throw new NeuralError(0, "INVALID_ARGUMENT", "malformed artifact id");
  return json(`/v1/neural/artifacts/${id}`);
}

export async function fetchFile(id, name) {
  if (!HEX64.test(id) || !FILE_NAME.test(name)) throw new Error("invalid artifact file reference");
  return (await call(`/v1/neural/artifacts/${id}/files/${name}`)).arrayBuffer();
}

// Object URL for an image artifact file (fetched with the API key header, unlike a plain <img src>).
export async function imageUrl(id, name, type = "image/png") {
  return URL.createObjectURL(new Blob([await fetchFile(id, name)], { type }));
}

// Poll a job until it reaches a terminal state. onUpdate sees every intermediate record.
export async function waitForJob(id, { onUpdate, intervalMs = 400, signal } = {}) {
  for (;;) {
    if (signal?.aborted) throw new NeuralError(0, "ABORTED", "stopped waiting");
    const j = await getJob(id);
    onUpdate?.(j);
    if (TERMINAL.has(j.state)) return j;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
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
