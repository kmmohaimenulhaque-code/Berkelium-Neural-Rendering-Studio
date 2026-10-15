// GeneratePanel.jsx
// Text-to-3D inside the Scene workspace.
//
// A prompt goes to the real model on the backend (/v1/neural). The generated triangle mesh is imported into
// the scene exactly like a model file, the Gaussian scene built from it is splat-rendered by the backend's
// renderer, and "Render this view" renders that Gaussian scene again from the camera on screen. Everything
// shown is the artifact the backend produced, labelled with what it is; nothing is substituted or faked.
// Jobs are durable on the backend, so the Recent list survives app and backend restarts.

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, Camera, History, Loader2, Play, Plus, Sparkles, Square, X } from "lucide-react";
import * as api from "../neuralApi.js";
import { importIntoScene } from "./actions.js";
import { assets } from "./assets.js";
import { sceneStore } from "./store.js";
import { cameraSpecFor } from "./neuralCamera.js";

const MODEL = "berkelium-textshape";
const VIEW_RENDER_MAX_SIDE = 384;
const SCENE_BG = [8 / 255, 9 / 255, 10 / 255]; // the viewport background, so a view render matches it

// mesh artifact id -> scene node id. Module level, so it survives switching workspaces.
const linked = new Map();

const slug = (s) => s.replace(/[^A-Za-z0-9 _-]+/g, "").trim().slice(0, 48) || "generated";
const publish = (patch) => {
  window.__berkeliumNeural = { ...(window.__berkeliumNeural || {}), ...patch };
};

export default function GeneratePanel({ engineRef, onClose }) {
  const [caps, setCaps] = useState(null);
  const [capsError, setCapsError] = useState(null);
  const [prompt, setPrompt] = useState("a red cone on top of a blue box");
  const [adapter, setAdapter] = useState("auto");
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null); // { job, mesh, gaussian, image }
  const [shown, setShown] = useState(null); // { url, label, artifactId }
  const [renderJob, setRenderJob] = useState(null);
  const [error, setError] = useState(null);
  const [recent, setRecent] = useState([]);
  const urlRef = useRef(null);

  const refreshRecent = useCallback(() => {
    api.listJobs(10)
      .then((jobs) => setRecent(jobs.filter((j) => j.request.kind === "text_to_3d")))
      .catch(() => {});
  }, []);

  useEffect(() => {
    api.capabilities().then(setCaps).catch((e) => setCapsError(e.message));
    refreshRecent();
    publish({ ready: true });
    return () => { if (urlRef.current) URL.revokeObjectURL(urlRef.current); };
  }, [refreshRecent]);

  const showImage = (url, label, artifactId) => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = url;
    setShown(url ? { url, label, artifactId } : null);
  };

  const model = caps?.models?.[MODEL];
  const modelReady = Boolean(model?.checkpoint_present);
  const routing = caps?.adapter_routing?.[MODEL] || {};
  const gpuRender = Boolean(caps?.renderers?.gsplat?.available);
  const running = job && !api.TERMINAL.has(job.state);

  // Import the generated mesh into the scene as a normal, selectable, movable model.
  const addToScene = useCallback(async (mesh, label) => {
    const buf = await api.fetchFile(mesh.id, "mesh.glb");
    const file = new File([buf], `${slug(label)}.glb`, { type: "model/gltf-binary" });
    const who = `${mesh.producer.model ?? MODEL}${mesh.producer.adapter ? ` + ${mesh.producer.adapter}` : ""}`;
    const id = await importIntoScene([file], { source: `text-to-3D · ${who} · ${mesh.id.slice(0, 12)}` });
    if (id) {
      linked.set(mesh.id, id);
      requestAnimationFrame(() => engineRef.current?.frameSelection());
    }
    return id;
  }, [engineRef]);

  const openResult = useCallback(async (j, { add }) => {
    const recs = await Promise.all(j.artifacts.map(api.getArtifact));
    const mesh = recs.find((r) => r.kind === "triangle_mesh") || null;
    const gaussian = recs.find((r) => r.kind === "gaussian_scene") || null;
    const image = recs.find((r) => r.kind === "image") || null;
    if (!mesh) throw new Error("The job finished without a mesh artifact.");
    const nodeId = add ? await addToScene(mesh, j.request.prompt) : linked.get(mesh.id) ?? null;
    setResult({ job: j, mesh, gaussian, image });
    if (image) {
      showImage(await api.imageUrl(image.id, "render.png"), `${image.producer.renderer} · 4-view orbit · splat render`,
        image.id);
    } else {
      showImage(null);
    }
    publish({
      jobId: j.id, jobState: j.state, meshArtifact: mesh.id, faces: mesh.validation.n_faces, nodeId,
      gaussianArtifact: gaussian?.id ?? null, gaussians: gaussian?.validation?.n ?? 0,
      renderArtifact: image?.id ?? null, renderer: image?.producer?.renderer ?? null,
    });
  }, [addToScene]);

  const generate = async () => {
    if (!prompt.trim() || running) return;
    setError(null);
    try {
      const body = { prompt: prompt.trim() };
      if (adapter !== "base") body.adapter = adapter;
      const submitted = await api.submitTextTo3D(body);
      setJob(submitted);
      publish({ jobId: submitted.id, jobState: submitted.state });
      const done = await api.waitForJob(submitted.id, {
        onUpdate: (j) => { setJob(j); publish({ jobState: j.state }); },
      });
      if (done.state === "completed") await openResult(done, { add: true });
      else if (done.state === "failed") setError(`Generation failed: ${done.error?.message ?? done.error?.code ?? "error"}`);
    } catch (e) {
      setError(e.message);
    } finally {
      refreshRecent();
    }
  };

  // Splat-render the Gaussian scene from the camera currently on screen.
  const renderThisView = async () => {
    const engine = engineRef.current;
    if (!engine || !result?.gaussian) return;
    setError(null);
    const nodeId = linked.get(result.mesh.id);
    const wrapper = nodeId && sceneStore.get().nodes[nodeId] ? assets.get(nodeId) : null;
    if (!wrapper) {
      setError("Add the generated model to the scene first, so its position is known.");
      return;
    }
    const root = wrapper.children[0] ?? wrapper; // the GLB's own frame = the Gaussian scene's frame
    root.updateWorldMatrix(true, false);
    const el = engine.container;
    const k = VIEW_RENDER_MAX_SIDE / Math.max(el.clientWidth, el.clientHeight, 1);
    const w = Math.max(16, Math.round(el.clientWidth * k));
    const h = Math.max(16, Math.round(el.clientHeight * k));
    try {
      const camera = cameraSpecFor(engine.camera, root.matrixWorld, w, h);
      const submitted = await api.submitRender({
        artifactId: result.gaussian.id, camera, background: SCENE_BG,
        backend: gpuRender ? "berkelium_hip" : "reference_cpu",
      });
      setRenderJob(submitted);
      const done = await api.waitForJob(submitted.id, { onUpdate: setRenderJob });
      if (done.state !== "completed") throw new Error(done.error?.message ?? `render ${done.state}`);
      const img = await api.getArtifact(done.artifacts[0]);
      showImage(await api.imageUrl(img.id, "render.png"), `${img.producer.renderer} · this view · ${w}×${h}`, img.id);
      publish({ viewRender: { artifact: img.id, width: w, height: h, renderer: img.producer.renderer } });
    } catch (e) {
      setError(e.message);
    } finally {
      setRenderJob(null);
    }
  };

  const reopen = async (j, add) => {
    setError(null);
    try {
      await openResult(await api.getJob(j.id), { add });
    } catch (e) {
      setError(e.message);
    }
  };

  const m = result?.mesh?.validation;
  return (
    <div
      className="gen"
      data-testid="generate-panel"
      onKeyDown={(e) => e.stopPropagation()}
      onPointerDown={(e) => e.stopPropagation()}
    >
      <div className="gen__head">
        <Sparkles size={13} />
        <span>Text to 3D</span>
        <button className="gen__close" onClick={onClose} title="Close" aria-label="Close text to 3D"><X size={13} /></button>
      </div>

      {capsError && <div className="gen__msg gen__msg--err"><AlertCircle size={12} /> {capsError}</div>}
      {caps && !modelReady && <div className="gen__msg gen__msg--warn"><AlertCircle size={12} /> No trained text-to-3D checkpoint on this backend.</div>}

      <input
        className="gen__prompt"
        data-testid="prompt-input"
        value={prompt}
        maxLength={300}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") generate(); }}
        placeholder="e.g. a green torus next to a white sphere"
        aria-label="Text-to-3D prompt"
      />
      <div className="gen__row">
        <select className="gen__select" value={adapter} onChange={(e) => setAdapter(e.target.value)}
          aria-label="Adapter" data-testid="adapter-select"
          title="auto applies an adapter only when the prompt names its concept">
          <option value="auto">auto (route by concept)</option>
          <option value="base">base model</option>
          {Object.entries(model?.adapters ?? {}).map(([name, present]) => (
            <option key={name} value={name} disabled={!present}>
              {name}{routing[name] ? ` (${routing[name].join(", ")})` : ""}{present ? "" : " (not trained)"}
            </option>
          ))}
        </select>
        {running ? (
          <button className="code__btn" onClick={() => api.cancelJob(job.id).then(setJob).catch((e) => setError(e.message))}>
            <Square size={12} /> <span>Cancel</span>
          </button>
        ) : (
          <button className="code__btn scene__btn--primary" data-testid="generate-btn" onClick={generate}
            disabled={!modelReady || !prompt.trim()} title={modelReady ? "Generate a 3D model" : "No trained model on the backend"}>
            <Play size={12} /> <span>Generate</span>
          </button>
        )}
      </div>

      {job && (
        <div className="gen__progress" data-testid="job-state">
          <div className="gen__state">
            {running ? <Loader2 size={12} className="settings__spin" /> : null}
            <span>job {job.state}{job.stage ? ` · ${job.stage}` : ""}</span>
            {running && <span className="gen__pct">{Math.round(job.progress * 100)}%</span>}
          </div>
          <div className="gen__bar"><div style={{ width: `${Math.round((job.state === "completed" ? 1 : job.progress) * 100)}%` }} /></div>
        </div>
      )}
      {error && <div className="gen__msg gen__msg--err" data-testid="gen-error"><AlertCircle size={12} /> {error}</div>}

      {result && (
        <div className="gen__result" data-testid="artifact-card">
          <div className="gen__title">Triangle mesh · {result.mesh.producer.model}{result.mesh.producer.adapter ? ` + ${result.mesh.producer.adapter}` : " (base)"}</div>
          <div>{m.n_faces.toLocaleString()} faces · {m.components} component(s) · {m.watertight ? "watertight" : "not watertight"}</div>
          <div>units: {result.mesh.labels.units} · {result.mesh.labels.confidence}</div>
          {result.gaussian && <div>{result.gaussian.validation.n.toLocaleString()} Gaussians{result.gaussian.config?.fit?.steps ? ` · fitted ${result.gaussian.config.fit.steps} steps` : " · surface-initialised"}</div>}
          <div className="gen__id">mesh {result.mesh.id.slice(0, 12)}{result.gaussian ? ` · gaussians ${result.gaussian.id.slice(0, 12)}` : ""}</div>
          {shown && (
            <figure className="gen__figure">
              <img src={shown.url} alt="Splat render of the generated Gaussian scene" data-testid="render-image" />
              <figcaption>{shown.label}</figcaption>
            </figure>
          )}
          <div className="gen__row">
            <button className="code__btn" onClick={renderThisView} disabled={!result.gaussian || Boolean(renderJob)}
              data-testid="render-view-btn" title="Splat-render the Gaussian scene from the current viewport camera">
              {renderJob ? <Loader2 size={12} className="settings__spin" /> : <Camera size={12} />}
              <span>{renderJob ? `Rendering ${renderJob.state}` : "Render this view"}</span>
            </button>
            <button className="code__btn" onClick={() => addToScene(result.mesh, result.job.request.prompt).catch((e) => setError(e.message))}>
              <Plus size={12} /> <span>Add again</span>
            </button>
          </div>
        </div>
      )}

      {recent.length > 0 && (
        <div className="gen__recent" data-testid="recent-jobs">
          <div className="gen__subhead"><History size={11} /> Recent (stored on the backend)</div>
          {recent.slice(0, 6).map((j) => (
            <div key={j.id} className="gen__job">
              <span className={`gen__dot gen__dot--${j.state}`} title={j.state} />
              <span className="gen__job-prompt" title={j.request.prompt}>{j.request.prompt}</span>
              {j.state === "completed" && (
                <>
                  <button className="gen__link" onClick={() => reopen(j, false)}>open</button>
                  <button className="gen__link" onClick={() => reopen(j, true)}>add</button>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
