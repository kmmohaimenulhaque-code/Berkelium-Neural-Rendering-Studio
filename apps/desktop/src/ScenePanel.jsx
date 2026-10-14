// ScenePanel.jsx
// The Scene workspace: toolbar, stats and the 3D viewport. All three.js work
// lives in scene/SceneEngine.js. This component mounts the engine and wires
// buttons, keys and file drops to it.
//
// What is in the scene and what is selected lives in the scene store, so the
// hierarchy panel and this viewport always agree without talking directly.

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Box, Grid3x3, RotateCcw, AlertTriangle, Focus, Move, Rotate3d, Undo2, Redo2, Upload, Loader2,
} from "lucide-react";
import { usePrefs } from "./prefs.js";
import { SceneEngine } from "./scene/SceneEngine.js";
import { sceneStore, useSceneState } from "./scene/store.js";
import { importIntoScene, handleSceneKey } from "./scene/actions.js";
import { ACCEPT } from "./scene/importer.js";
import PropertiesPanel from "./scene/PropertiesPanel.jsx";

export const IMPORT_EVENT = "berkelium:import";
const isMac = typeof navigator !== "undefined" && /Mac/i.test(navigator.platform || navigator.userAgent);
const MOD = isMac ? "⌘" : "Ctrl+";

export default function ScenePanel() {
  const mountRef = useRef(null);
  const engineRef = useRef(null);
  const labelRef = useRef(null);
  const fileRef = useRef(null);
  const dragDepth = useRef(0);
  const [wireframe, setWireframe] = useState(false);
  const [tool, setToolState] = useState("move");
  const [stats, setStats] = useState({ fps: 0, tris: 0 });
  const [failure, setFailure] = useState(null);
  const [dropping, setDropping] = useState(false);
  const [importing, setImporting] = useState(null); // file name while loading
  const { viewport: prefs } = usePrefs();
  const state = useSceneState();
  const gridOn = state.nodes.grid?.visible ?? true;
  const selected = state.selectedId ? state.nodes[state.selectedId] : null;

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;
    let engine;
    try {
      engine = new SceneEngine({
        container,
        store: sceneStore,
        labelEl: labelRef.current,
        onStats: setStats,
        onContextLost: () =>
          setFailure("The graphics driver reset the 3D view. Switch tabs and back to restart it."),
      });
    } catch {
      setFailure("WebGL is not available on this machine, so the 3D viewport cannot start.");
      return;
    }
    engineRef.current = engine;
    return () => {
      engineRef.current = null;
      engine.dispose();
    };
  }, []);

  useEffect(() => { engineRef.current?.setWireframe(wireframe); }, [wireframe, state.rootIds]);

  const setTool = useCallback((t) => {
    setToolState(t);
    engineRef.current?.setTool(t);
  }, []);

  const runImport = useCallback(async (files) => {
    if (!files?.length || importing) return;
    setImporting(files[0].name);
    try {
      const id = await importIntoScene(files);
      if (id) requestAnimationFrame(() => engineRef.current?.frameSelection());
    } finally {
      setImporting(null);
    }
  }, [importing]);

  // ⌘I / Ctrl+I from anywhere in the app opens the file picker.
  useEffect(() => {
    const open = () => fileRef.current?.click();
    window.addEventListener(IMPORT_EVENT, open);
    return () => window.removeEventListener(IMPORT_EVENT, open);
  }, []);

  // Keys only act while the viewport has focus, so typing in the editor or
  // chat can never move, delete or deselect anything by accident.
  const onKeyDown = (e) => {
    if (handleSceneKey(e, { setTool })) {
      e.preventDefault();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Escape") {
      sceneStore.select(null);
      e.preventDefault();
    } else if (e.key === "f" || e.key === "F") {
      engineRef.current?.frameSelection();
      e.preventDefault();
    }
  };

  // Drag and drop from Finder, File Explorer or a Linux file manager.
  // dragenter/dragleave fire for every child element, so count depth.
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  const dropHandlers = {
    onDragEnter: (e) => { if (!hasFiles(e)) return; e.preventDefault(); dragDepth.current += 1; setDropping(true); },
    onDragOver: (e) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = "copy"; },
    onDragLeave: () => { dragDepth.current = Math.max(0, dragDepth.current - 1); if (!dragDepth.current) setDropping(false); },
    onDrop: (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth.current = 0;
      setDropping(false);
      runImport([...e.dataTransfer.files]);
    },
  };

  const canEdit = Boolean(selected?.transformable);

  return (
    <div className="scene">
      <div className="scene__toolbar">
        <div className="scene__actions">
          <button className="code__btn scene__btn--primary" onClick={() => fileRef.current?.click()} disabled={Boolean(importing)} title={`Import model (${MOD}I)`}>
            <Upload size={14} />
            <span>Import</span>
          </button>
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPT}
            multiple
            hidden
            onChange={(e) => { runImport([...e.target.files]); e.target.value = ""; }}
          />
          <span className="scene__sep" />
          <button
            className={`code__btn ${tool === "move" ? "scene__btn--on" : ""}`}
            onClick={() => setTool("move")}
            aria-pressed={tool === "move"}
            title="Move (W)"
          >
            <Move size={14} />
            <span>Move</span>
          </button>
          <button
            className={`code__btn ${tool === "rotate" ? "scene__btn--on" : ""}`}
            onClick={() => setTool("rotate")}
            aria-pressed={tool === "rotate"}
            title="Rotate (E). Hold Shift to snap 15°"
          >
            <Rotate3d size={14} />
            <span>Rotate</span>
          </button>
          <span className="scene__sep" />
          <button className="code__btn" onClick={() => sceneStore.undo()} disabled={!sceneStore.canUndo()} title={`Undo (${MOD}Z)`}>
            <Undo2 size={14} />
          </button>
          <button className="code__btn" onClick={() => sceneStore.redo()} disabled={!sceneStore.canRedo()} title={isMac ? "Redo (⇧⌘Z)" : "Redo (Ctrl+Y)"}>
            <Redo2 size={14} />
          </button>
          <span className="scene__sep" />
          <button className="code__btn" onClick={() => engineRef.current?.resetView()} title="Reset view">
            <RotateCcw size={14} />
            <span>Reset view</span>
          </button>
          <button
            className={`code__btn ${wireframe ? "scene__btn--on" : ""}`}
            onClick={() => setWireframe((v) => !v)}
            aria-pressed={wireframe}
            title="Toggle wireframe"
          >
            <Box size={14} />
            <span>Wireframe</span>
          </button>
          <button
            className={`code__btn ${gridOn ? "scene__btn--on" : ""}`}
            onClick={() => sceneStore.toggleVisible("grid")}
            aria-pressed={gridOn}
            title="Toggle grid"
          >
            <Grid3x3 size={14} />
            <span>Grid</span>
          </button>
          <button
            className="code__btn"
            onClick={() => engineRef.current?.frameSelection()}
            disabled={!selected}
            title="Frame selection (F)"
          >
            <Focus size={14} />
            <span>Frame</span>
          </button>
        </div>
        {prefs.showStats && (
          <div className="scene__stats" aria-label="Viewport stats">
            <span>{stats.fps} fps</span>
            <span>{stats.tris.toLocaleString()} tris</span>
          </div>
        )}
      </div>

      <div
        className="scene__viewport"
        ref={mountRef}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onPointerDown={() => mountRef.current?.focus({ preventScroll: true })}
        aria-label="3D viewport"
        {...dropHandlers}
      >
        {failure ? (
          <div className="scene__failure" role="alert">
            <AlertTriangle size={20} />
            <span>{failure}</span>
          </div>
        ) : (
          <>
            {selected && (
              <div className="scene__selection" aria-live="polite">
                <span className="scene__selection-dot" />
                {selected.name}
              </div>
            )}
            <PropertiesPanel />
            <div ref={labelRef} className="scene__angle" style={{ display: "none" }} />
            <div className="scene__hint">
              {canEdit && tool === "move" && "Drag an arrow to move · "}
              {canEdit && tool === "rotate" && "Drag the dot to spin · Shift snaps 15° · "}
              Click to select · Drag to orbit · F to frame · Drop a .glb, .gltf or .obj to import
            </div>
            {dropping && (
              <div className="scene__drop">
                <Upload size={22} />
                <b>Drop to import</b>
                <small>.glb · .gltf with its .bin and textures · .obj with .mtl</small>
              </div>
            )}
            {importing && (
              <div className="scene__importing">
                <Loader2 size={18} className="settings__spin" />
                Importing {importing}...
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
