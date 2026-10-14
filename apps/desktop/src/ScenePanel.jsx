// ScenePanel.jsx
// The Scene workspace: toolbar, stats and the 3D viewport. All three.js work
// lives in scene/SceneEngine.js. This component only mounts the engine and
// connects toolbar buttons and keys to it.
//
// What is in the scene and what is selected lives in the scene store, so the
// hierarchy panel and this viewport always agree without talking directly.

import { useEffect, useRef, useState } from "react";
import { Box, Grid3x3, RotateCcw, AlertTriangle, Focus } from "lucide-react";
import { usePrefs } from "./prefs.js";
import { SceneEngine } from "./scene/SceneEngine.js";
import { sceneStore, useSceneState } from "./scene/store.js";

export default function ScenePanel() {
  const mountRef = useRef(null);
  const engineRef = useRef(null);
  const [wireframe, setWireframe] = useState(false);
  const [stats, setStats] = useState({ fps: 0, tris: 0 });
  const [failure, setFailure] = useState(null);
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

  useEffect(() => { engineRef.current?.setWireframe(wireframe); }, [wireframe]);

  // Keys only act while the viewport has focus, so typing in the editor or
  // chat can never deselect or move the camera by accident.
  const onKeyDown = (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Escape") {
      sceneStore.select(null);
      e.preventDefault();
    } else if (e.key === "f" || e.key === "F") {
      engineRef.current?.frameSelection();
      e.preventDefault();
    }
  };

  return (
    <div className="scene">
      <div className="scene__toolbar">
        <div className="scene__actions">
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
            <div className="scene__hint">Click to select · Drag to orbit · Right-drag to pan · Scroll to zoom · F to frame</div>
          </>
        )}
      </div>
    </div>
  );
}
