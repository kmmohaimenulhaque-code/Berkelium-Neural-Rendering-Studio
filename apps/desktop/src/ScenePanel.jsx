// ScenePanel.jsx
// The Scene workspace: a real three.js viewport. Lit cube on a grid floor,
// orbit controls, a toolbar (reset view, wireframe, grid), and live stats.
//
// three.js is driven imperatively from one effect. React state only holds
// toolbar toggles and the stats line, which update twice a second, never per
// frame. Re-rendering React 60 times a second would waste the GPU budget.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Box, Grid3x3, RotateCcw, AlertTriangle } from "lucide-react";

const BG = 0x08090a;
const ACCENT = 0x00d4d4;
const HOME_POSITION = new THREE.Vector3(4.5, 3.5, 6);
const HOME_TARGET = new THREE.Vector3(0, 0.6, 0);

// A floor grid made of 1-unit segments instead of GridHelper's full-length
// lines. Long lines that pass behind the camera get dropped entirely by some
// renderers (software WebGL, some drivers), which erased the grid near the
// cube. Short segments clip cleanly everywhere. ~3,200 segments is trivial.
function buildGrid(half, step, centerColor, lineColor) {
  const positions = [];
  const colors = [];
  const c1 = new THREE.Color(centerColor);
  const c2 = new THREE.Color(lineColor);
  for (let i = -half; i <= half; i += step) {
    const color = i === 0 ? c1 : c2;
    for (let j = -half; j < half; j += step) {
      positions.push(i, 0, j, i, 0, j + step); // line along z
      positions.push(j, 0, i, j + step, 0, i); // line along x
      for (let k = 0; k < 4; k++) colors.push(color.r, color.g, color.b);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true }));
}

function buildScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);
  scene.fog = new THREE.Fog(BG, 18, 40); // grid fades out instead of ending in a hard edge

  // Soft fill from above plus one shadow-casting key light.
  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x0b0c10, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.name = "Directional Light";
  sun.position.set(5, 8, 4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.left = -6;
  sun.shadow.camera.right = 6;
  sun.shadow.camera.top = 6;
  sun.shadow.camera.bottom = -6;
  scene.add(sun);

  const grid = buildGrid(20, 1, 0x3d3f45, 0x24262b);
  grid.name = "Grid";
  // Sit the grid a hair above the shadow floor so the floor never hides it.
  grid.position.y = 0.002;
  scene.add(grid);

  // Invisible floor that only shows the cube's shadow.
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.ShadowMaterial({ opacity: 0.35 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  const cube = new THREE.Mesh(
    new THREE.BoxGeometry(1.2, 1.2, 1.2),
    new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.35, metalness: 0.25 })
  );
  cube.name = "Cube";
  cube.position.y = 0.6;
  cube.castShadow = true;
  scene.add(cube);

  // Thin edge outline so the cube reads clearly at any angle.
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(cube.geometry),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 })
  );
  cube.add(edges);

  return { scene, grid, cube };
}

export default function ScenePanel() {
  const mountRef = useRef(null);
  const apiRef = useRef(null); // handles the toolbar needs into the live scene
  const [wireframe, setWireframe] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [stats, setStats] = useState({ fps: 0, tris: 0 });
  const [failure, setFailure] = useState(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    } catch {
      setFailure("WebGL is not available on this machine, so the 3D viewport cannot start.");
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // retina sharp, not 3x cost
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    mount.appendChild(renderer.domElement);

    const { scene, grid, cube } = buildScene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
    camera.name = "Camera";
    camera.position.copy(HOME_POSITION);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(HOME_TARGET);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 2;
    controls.maxDistance = 30;
    controls.maxPolarAngle = Math.PI * 0.495; // never orbit below the floor
    controls.update();

    // Size follows the panel, not the window, so dragging a resize handle
    // re-fits the canvas instead of stretching it.
    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(mount);
    resize();

    const onLost = (e) => {
      e.preventDefault();
      setFailure("The graphics driver reset the 3D view. Switch tabs and back to restart it.");
    };
    renderer.domElement.addEventListener("webglcontextlost", onLost);

    let frames = 0;
    let last = performance.now();
    const tick = () => {
      if (document.hidden) return; // no GPU work while the app is in the background
      controls.update();
      renderer.render(scene, camera);
      frames += 1;
      const now = performance.now();
      if (now - last >= 500) {
        setStats({ fps: Math.round((frames * 1000) / (now - last)), tris: renderer.info.render.triangles });
        frames = 0;
        last = now;
      }
    };
    renderer.setAnimationLoop(tick);

    apiRef.current = {
      setWireframe: (on) => { cube.material.wireframe = on; },
      setGrid: (on) => { grid.visible = on; },
      resetView: () => {
        camera.position.copy(HOME_POSITION);
        controls.target.copy(HOME_TARGET);
        controls.update();
      },
    };

    return () => {
      apiRef.current = null;
      renderer.setAnimationLoop(null);
      ro.disconnect();
      renderer.domElement.removeEventListener("webglcontextlost", onLost);
      controls.dispose();
      // Free GPU memory. Without this, every hot reload during development
      // leaks a full scene worth of buffers.
      scene.traverse((obj) => {
        obj.geometry?.dispose();
        const mats = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
        mats.forEach((m) => m.dispose());
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  useEffect(() => { apiRef.current?.setWireframe(wireframe); }, [wireframe]);
  useEffect(() => { apiRef.current?.setGrid(showGrid); }, [showGrid]);

  return (
    <div className="scene">
      <div className="scene__toolbar">
        <div className="scene__actions">
          <button className="code__btn" onClick={() => apiRef.current?.resetView()} title="Reset view">
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
            className={`code__btn ${showGrid ? "scene__btn--on" : ""}`}
            onClick={() => setShowGrid((v) => !v)}
            aria-pressed={showGrid}
            title="Toggle grid"
          >
            <Grid3x3 size={14} />
            <span>Grid</span>
          </button>
        </div>
        <div className="scene__stats" aria-label="Viewport stats">
          <span>{stats.fps} fps</span>
          <span>{stats.tris.toLocaleString()} tris</span>
        </div>
      </div>

      <div className="scene__viewport" ref={mountRef}>
        {failure ? (
          <div className="scene__failure" role="alert">
            <AlertTriangle size={20} />
            <span>{failure}</span>
          </div>
        ) : (
          <div className="scene__hint">Drag to orbit · Right-drag to pan · Scroll to zoom</div>
        )}
      </div>
    </div>
  );
}
