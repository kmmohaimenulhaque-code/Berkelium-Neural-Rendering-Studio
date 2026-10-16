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
// Generators the backend may offer (text3d/generators.py). "auto" = the native model when every content word is
// in its vocabulary, otherwise the first available open-world generator in E-T3 evidence order (decision D-029).
// Availability and the reason a generator is unavailable come from /capabilities.
const GENERATORS = [
  ["auto", "auto (native if it can read the prompt, else the best available open-world model)"],
  ["berkelium-textshape", "Berkelium native (procedural primitives)"],
  ["shap-e-text300m", "Shap-E text300M (open-world; CPU minutes, GPU ~1 min)"],
  ["zimage-triposr", "Z-Image → TripoSR (open-world, needs a GPU)"],
];
const short = (h) => (h ? String(h).slice(0, 10) : "?");
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
  const [generator, setGenerator] = useState("auto");
  const [compose, setCompose] = useState(false); // two-object planner (A on top of / next to B)
  const [fitSteps, setFitSteps] = useState(0); // differentiable Gaussian refinement on the backend
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
    // zimage-triposr stores the generated anchor image with the mesh: show what the reconstruction was made from.
    const anchor = mesh.files?.some((f) => f.name === "anchor.png") ? await api.imageUrl(mesh.id, "anchor.png") : null;
    setResult({ job: j, mesh, gaussian, image, anchor });
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
      const body = { prompt: prompt.trim(), fitSteps, generator, compose };
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
        <select className="gen__select" value={generator} onChange={(e) => setGenerator(e.target.value)}
          aria-label="Generator" data-testid="generator-select"
          title={caps?.generators?.[generator]?.reason || "Which model turns the prompt into a mesh"}>
          {GENERATORS.map(([name, label]) => {
            const g = caps?.generators?.[name];
            const ok = g ? g.available : name === "auto" || name === MODEL;
            return (
              <option key={name} value={name} disabled={!ok} title={g?.reason || ""}>
                {label}{ok ? "" : " — unavailable"}
              </option>
            );
          })}
        </select>
        <label className="gen__check" title="Split 'A on top of B' / 'A next to B' into two separately generated, validated parts">
          <input type="checkbox" checked={compose} onChange={(e) => setCompose(e.target.checked)} data-testid="compose-check" />
          <span>plan 2 objects</span>
        </label>
      </div>
      {caps?.generators?.[generator] && !caps.generators[generator].available && (
        <div className="gen__msg gen__msg--warn" data-testid="generator-reason"><AlertCircle size={12} /> {caps.generators[generator].reason}</div>
      )}
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
      </div>
      <div className="gen__row">
        <select className="gen__select" value={fitSteps} onChange={(e) => setFitSteps(Number(e.target.value))}
          aria-label="Gaussian refinement" data-testid="fit-select"
          title="Refine the Gaussians against renders of the generated mesh with the differentiable renderer">
          <option value={0}>Gaussians: surface init</option>
          <option value={100}>Gaussians: refine 100 steps</option>
          <option value={300}>Gaussians: refine 300 steps</option>
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
          <div className="gen__title">Triangle mesh · {result.mesh.producer.model}{result.mesh.producer.adapter ? ` + ${result.mesh.producer.adapter}` : result.mesh.producer.generator_identity ? "" : " (base)"}</div>
          <Provenance producer={result.mesh.producer} labels={result.mesh.labels} files={result.mesh.files} />
          {result.anchor && (
            <figure className="gen__figure" data-testid="anchor-image">
              <img src={result.anchor} alt="Generated anchor image the mesh was reconstructed from" />
              <figcaption>anchor image (Z-Image-Turbo, generated; no calibrated camera) → TripoSR</figcaption>
            </figure>
          )}
          <div>{m.n_faces.toLocaleString()} faces · {m.components} component(s) · {m.watertight ? "watertight" : "not watertight"}</div>
          <div>units: {result.mesh.labels.units} · {result.mesh.labels.confidence}</div>
          {result.gaussian && <div>{result.gaussian.validation.n.toLocaleString()} Gaussians{result.gaussian.config?.fit?.steps
            ? ` · refined ${result.gaussian.config.fit.steps} steps (PSNR ${result.gaussian.config.fit.psnr_before?.toFixed(1) ?? "?"} → ${result.gaussian.config.fit.psnr_after?.toFixed(1)} dB vs mesh renders)`
            : " · surface-initialised"}</div>}
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

// What produced this mesh: generator, code commit, weight hashes, routing (including generators skipped because
// they are unavailable here), the hardware it ran on, seed, time and memory. Read from the artifact's producer
// record, i.e. what the backend actually ran; file hashes come from the artifact store.
function Provenance({ producer, labels, files }) {
  const ids = producer.generator_identity ? Object.values(producer.generator_identity) : [];
  const routing = producer.routing || [];
  const hw = producer.execution;
  return (
    <div className="gen__prov" data-testid="provenance-card">
      <div>generator: <b data-testid="provenance-generator">{producer.generator ?? MODEL}</b>
        {producer.requested_generator && producer.requested_generator !== producer.generator ? ` (requested ${producer.requested_generator})` : ""}</div>
      {hw && <div className="gen__dim" data-testid="provenance-hardware">ran on {hw.device}{hw.gpu ? ` (${hw.gpu})` : ""} · {hw.host}</div>}
      {routing.map((r, i) => (
        <div key={i} className="gen__dim">
          route “{r.prompt}” → {r.selected ?? r.generator}: {r.reason}
          {(r.skipped_unavailable || []).map((k) => <div key={k.generator} data-testid="provenance-skipped">skipped {k.generator}: {k.reason}</div>)}
        </div>
      ))}
      {producer.planner && <div data-testid="provenance-planner">planner: {producer.planner.relation} · top/left “{producer.planner.a}” · bottom/right “{producer.planner.b}” (each part generated and validated separately, placed by bounding box)</div>}
      {ids.map((id) => (
        <div key={id.generator} className="gen__dim">
          {id.generator}{id.shap_e_commit ? ` · shap-e ${short(id.shap_e_commit)}` : ""}
          {id.weights_sha256 ? ` · text_cond ${short(id.weights_sha256["text_cond.pt"])} · decoder ${short(id.weights_sha256["vector_decoder.pt"])}` : ""}
          {id.weights?.zimage ? ` · z-image ${short(id.weights.zimage.revision)} (${Object.keys(id.weights.zimage.files || {}).length} safetensors sha256-verified) · triposr ckpt ${short(id.weights.triposr?.model_ckpt_sha256)} · ${id.dtype}` : ""}
          {id.sampler?.karras_steps ? ` · ${id.sampler.karras_steps} steps, cfg ${id.sampler.guidance_scale}` : ""}
          {id.sampler?.steps && !id.sampler.karras_steps ? ` · ${id.sampler.steps} steps, ${id.sampler.height}²` : ""}
          {id.license ? ` · ${id.license}` : ""}
        </div>
      ))}
      {!ids.length && producer.base_checkpoint_sha256 && <div className="gen__dim">checkpoint {short(producer.base_checkpoint_sha256)}</div>}
      <div className="gen__dim">seed {producer.seed ?? 0}{producer.seconds != null ? ` · ${Number(producer.seconds).toFixed(1)} s` : ""}{producer.peak_rss_mib ? ` · peak ${Math.round(producer.peak_rss_mib)} MiB` : ""}</div>
      {files?.length > 0 && <div className="gen__dim" data-testid="provenance-files">files: {files.map((f) => `${f.name} ${short(f.sha256)}`).join(" · ")}</div>}
      {labels?.provenance && <div className="gen__label" data-testid="provenance-label">{labels.provenance}</div>}
    </div>
  );
}
