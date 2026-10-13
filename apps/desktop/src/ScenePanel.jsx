import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { Box, Loader2, Play, Square, AlertCircle, CircleDot } from "lucide-react";

import {
  NeuralError,
  cancelJob,
  capabilities,
  fetchFile,
  getArtifact,
  getJob,
  parseGaussianScene,
  submitTextTo3D,
} from "./neuralApi.js";

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

// Viewport shows exactly what the service produced: a triangle mesh (GLB) or the Gaussian scene's centres.
// Every view is labelled with its artifact kind, units and confidence; nothing is substituted or faked.
export default function ScenePanel() {
  const mountRef = useRef(null);
  const three = useRef(null);
  const [caps, setCaps] = useState(null);
  const [capsError, setCapsError] = useState(null);
  const [prompt, setPrompt] = useState("a red cone on top of a blue box");
  const [adapter, setAdapter] = useState("");
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [view, setView] = useState("mesh");
  const [artifacts, setArtifacts] = useState({ mesh: null, gaussian: null });

  // ------------------------------------------------------------------ three.js scene (created once)
  useEffect(() => {
    const el = mountRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x16181d);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
    camera.position.set(2.2, 1.6, 3.2);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 1.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, 4, 3);
    scene.add(sun);
    const grid = new THREE.GridHelper(4, 16, 0x3a3f4b, 0x2a2e36);
    grid.position.y = -1.0;
    scene.add(grid);
    const content = new THREE.Group();
    scene.add(content);
    const resize = () => {
      const w = el.clientWidth || 1;
      const h = el.clientHeight || 1;
      renderer.setSize(w, h, false);
      renderer.domElement.style.width = "100%";
      renderer.domElement.style.height = "100%";
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();
    let raf = 0;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();
    three.current = { renderer, scene, camera, controls, content };
    window.__berkeliumViewport = { ready: true, artifactId: null, kind: null, faces: 0, points: 0 };
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, []);

  // ------------------------------------------------------------------ capabilities
  useEffect(() => {
    capabilities()
      .then(setCaps)
      .catch((e) => setCapsError(e.message));
  }, []);

  const clearContent = () => {
    const { content } = three.current;
    while (content.children.length) {
      const o = content.children.pop();
      o.traverse?.((n) => {
        n.geometry?.dispose();
        if (n.material) (Array.isArray(n.material) ? n.material : [n.material]).forEach((m) => m.dispose());
      });
    }
  };

  const showMesh = useCallback(async (rec) => {
    const buf = await fetchFile(rec.id, "mesh.glb");
    const gltf = await new GLTFLoader().parseAsync(buf, "");
    clearContent();
    let faces = 0;
    gltf.scene.traverse((n) => {
      if (n.isMesh) {
        n.material = new THREE.MeshStandardMaterial({
          vertexColors: !!n.geometry.attributes.color, roughness: 0.6, metalness: 0.05, side: THREE.DoubleSide,
        });
        faces += n.geometry.index ? n.geometry.index.count / 3 : n.geometry.attributes.position.count / 3;
      }
    });
    three.current.content.add(gltf.scene);
    window.__berkeliumViewport = { ready: true, artifactId: rec.id, kind: rec.kind, faces, points: 0 };
  }, []);

  const showGaussians = useCallback(async (rec) => {
    const [payload, manBuf] = await Promise.all([fetchFile(rec.id, "scene.soa"), fetchFile(rec.id, "scene.json")]);
    const g = parseGaussianScene(payload, JSON.parse(new TextDecoder().decode(manBuf)));
    clearContent();
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(g.means, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(g.colors, 3));
    three.current.content.add(new THREE.Points(geom, new THREE.PointsMaterial({ size: 0.025, vertexColors: true })));
    window.__berkeliumViewport = { ready: true, artifactId: rec.id, kind: rec.kind, faces: 0, points: g.n };
  }, []);

  // reload the viewport when artifacts or the selected view change
  useEffect(() => {
    const rec = view === "mesh" ? artifacts.mesh : artifacts.gaussian;
    if (!rec) return;
    (view === "mesh" ? showMesh(rec) : showGaussians(rec)).catch((e) => setError(`Could not display artifact: ${e.message}`));
  }, [artifacts, view, showMesh, showGaussians]);

  // ------------------------------------------------------------------ job lifecycle (poll real state)
  useEffect(() => {
    if (!job || TERMINAL.has(job.state)) return undefined;
    const id = setTimeout(async () => {
      try {
        const j = await getJob(job.id);
        setJob(j);
        if (j.state === "completed") {
          const recs = await Promise.all(j.artifacts.map(getArtifact));
          setArtifacts({
            mesh: recs.find((r) => r.kind === "triangle_mesh") || null,
            gaussian: recs.find((r) => r.kind === "gaussian_scene") || null,
          });
        } else if (j.state === "failed") {
          setError(`Job failed: ${j.error?.code ?? "error"} — ${j.error?.message ?? ""}`);
        }
      } catch (e) {
        setError(e.message);
      }
    }, 400);
    return () => clearTimeout(id);
  }, [job]);

  const generate = async () => {
    setError(null);
    try {
      const j = await submitTextTo3D({ prompt: prompt.trim(), adapter: adapter || undefined });
      setJob(j);
    } catch (e) {
      setError(e instanceof NeuralError ? e.message : String(e));
    }
  };

  const running = job && !TERMINAL.has(job.state);
  const model = caps?.models?.["berkelium-textshape"];
  const modelReady = !!model?.checkpoint_present;
  const shown = view === "mesh" ? artifacts.mesh : artifacts.gaussian;

  return (
    <div className="scene-panel" data-testid="scene-panel">
      <div className="scene-toolbar">
        <Box size={13} color="var(--text-secondary)" />
        <input
          className="scene-toolbar__prompt"
          data-testid="prompt-input"
          value={prompt}
          maxLength={300}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !running && modelReady && generate()}
          placeholder="Describe a shape, e.g. a green torus next to a white sphere"
          aria-label="Text-to-3D prompt"
        />
        <select
          className="scene-toolbar__select"
          value={adapter}
          onChange={(e) => setAdapter(e.target.value)}
          aria-label="Adapter"
          data-testid="adapter-select"
        >
          <option value="">base model</option>
          {Object.entries(model?.adapters ?? {}).map(([name, present]) => (
            <option key={name} value={name} disabled={!present}>
              {name}{present ? "" : " (not trained)"}
            </option>
          ))}
        </select>
        {running ? (
          <button className="scene-btn" onClick={() => cancelJob(job.id).then(setJob).catch((e) => setError(e.message))}>
            <Square size={12} /> Cancel
          </button>
        ) : (
          <button className="scene-btn scene-btn--primary" onClick={generate} disabled={!modelReady || !prompt.trim()}
            data-testid="generate-btn" title={modelReady ? "Generate a 3D mesh" : "No trained model on the backend"}>
            <Play size={12} /> Generate 3D
          </button>
        )}
      </div>

      <div className="scene-viewport" ref={mountRef} data-testid="viewport" />

      <div className="scene-overlay">
        {capsError && <div className="scene-msg scene-msg--err"><AlertCircle size={12} /> Neural service: {capsError}</div>}
        {caps && !modelReady && (
          <div className="scene-msg scene-msg--warn"><AlertCircle size={12} /> No trained text-to-3D checkpoint on this backend.</div>
        )}
        {job && (
          <div className="scene-msg" data-testid="job-state">
            {running ? <Loader2 size={12} className="spin" /> : <CircleDot size={12} />}
            job {job.state}{job.stage ? ` · ${job.stage}` : ""}{running ? ` · ${Math.round(job.progress * 100)}%` : ""}
          </div>
        )}
        {error && <div className="scene-msg scene-msg--err" data-testid="scene-error"><AlertCircle size={12} /> {error}</div>}
        {shown && (
          <div className="scene-card" data-testid="artifact-card">
            <div className="scene-card__title">
              {shown.kind === "triangle_mesh" ? "Triangle mesh" : "Gaussian scene · centres preview (not splatted)"}
            </div>
            <div>units: {shown.labels.units} · {shown.labels.confidence ?? "model output"}</div>
            {shown.kind === "triangle_mesh" && (
              <div>
                {shown.validation.n_faces} faces · {shown.validation.components} component(s) ·
                {shown.validation.watertight ? " watertight" : " not watertight"}
              </div>
            )}
            {shown.kind === "gaussian_scene" && <div>{shown.validation.n} Gaussians</div>}
            <div className="scene-card__id">artifact {shown.id.slice(0, 12)} · {shown.producer.model ?? ""}{shown.producer.adapter ? ` + ${shown.producer.adapter}` : ""}</div>
            <div className="scene-card__views">
              <button className={view === "mesh" ? "on" : ""} onClick={() => setView("mesh")}>mesh</button>
              <button className={view === "gaussian" ? "on" : ""} onClick={() => setView("gaussian")} disabled={!artifacts.gaussian}>gaussians</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
