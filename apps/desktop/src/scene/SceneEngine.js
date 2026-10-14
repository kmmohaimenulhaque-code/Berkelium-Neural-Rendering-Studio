// SceneEngine.js
// Owns everything three.js: renderer, scene, camera, controls, picking and
// selection visuals. It reads the scene store and writes back only one thing:
// the selection, when the user clicks in the viewport.
//
// Kept out of React on purpose. The render loop runs 60 times a second and
// must never trigger React renders. React only gets stats twice a second.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { isEffectivelyVisible } from "./sceneStore.js";

const BG = 0x08090a;
const ACCENT = 0x00d4d4;
const HOME_POSITION = new THREE.Vector3(4.5, 3.5, 6);
const HOME_TARGET = new THREE.Vector3(0, 0.6, 0);
const CLICK_TOLERANCE_PX = 5; // further than this between press and release is an orbit drag, not a click
const FRAME_MS = 280;

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

export class SceneEngine {
  constructor({ container, store, onStats, onContextLost }) {
    this.container = container;
    this.store = store;
    this.onStats = onStats;
    this.onContextLost = onContextLost;

    // Throws if WebGL is unavailable. The caller shows a friendly message.
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    const r = this.renderer;
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // retina sharp, not 3x cost
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFShadowMap; // PCFSoft was removed in three r18x and falls back to this anyway
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(r.domElement);

    this.#buildScene();
    this.#buildCamera();
    this.#bindEvents();

    this.lastState = null;
    this.unsubscribe = store.subscribe(() => this.#sync());
    this.#sync();

    this.frames = 0;
    this.lastStats = performance.now();
    this.tween = null;
    r.setAnimationLoop(() => this.#tick());
  }

  // ------------------------------------------------------------ build

  #buildScene() {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(BG);
    scene.fog = new THREE.Fog(BG, 18, 40); // grid fades out instead of ending in a hard edge
    this.scene = scene;

    const sky = new THREE.HemisphereLight(0xdfe8ff, 0x0b0c10, 0.9);
    sky.position.set(0, 6, 0);
    scene.add(sky);

    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(5, 8, 4);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    Object.assign(sun.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6 });
    scene.add(sun);

    const grid = buildGrid(20, 1, 0x3d3f45, 0x24262b);
    grid.position.y = 0.002; // a hair above the shadow floor so the floor never hides it
    scene.add(grid);

    // Invisible floor that only shows the cube's shadow.
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.ShadowMaterial({ opacity: 0.35 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    const cube = new THREE.Mesh(
      new THREE.BoxGeometry(1.2, 1.2, 1.2),
      new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.35, metalness: 0.25 })
    );
    cube.position.y = 0.6;
    cube.castShadow = true;
    scene.add(cube);
    // Thin edge outline so the cube reads clearly at any angle.
    cube.add(new THREE.LineSegments(
      new THREE.EdgesGeometry(cube.geometry),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 })
    ));

    // Store id -> three.js object. Everything else finds objects through here.
    this.objects = { cube, grid, sun, sky };
    for (const [id, obj] of Object.entries(this.objects)) obj.userData.nodeId = id;

    // Only real geometry is clickable. Raycasting the 3,200-segment grid or
    // the helpers on every click would be wasted work and wrong UX.
    this.pickables = [cube];

    // Selection visuals. Created once and shown or hidden, never rebuilt.
    this.helpers = {
      cube: this.#outlineFor(cube),
      sun: new THREE.DirectionalLightHelper(sun, 1, ACCENT),
      sky: new THREE.HemisphereLightHelper(sky, 0.6, ACCENT),
    };
    for (const h of Object.values(this.helpers)) {
      h.visible = false;
      h.raycast = () => {};
      if (!h.parent) scene.add(h); // mesh outlines are already parented to their mesh
    }
    this.gridMaterials = {
      normal: grid.material,
      selected: new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.35 }),
    };
  }

  // Selection outline for a mesh: crisp white edges drawn as a child of the
  // mesh, so it follows any move, rotate or scale for free. White, not the
  // cyan accent, because the default cube is itself cyan and an accent
  // outline disappeared against it.
  #outlineFor(mesh) {
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95 })
    );
    outline.scale.setScalar(1.004); // a hair outside the faces so it never z-fights
    mesh.add(outline);
    return outline;
  }

  #buildCamera() {
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
    this.camera.position.copy(HOME_POSITION);
    this.objects.camera = this.camera;

    const c = new OrbitControls(this.camera, this.renderer.domElement);
    c.target.copy(HOME_TARGET);
    c.enableDamping = true;
    c.dampingFactor = 0.08;
    c.minDistance = 2;
    c.maxDistance = 30;
    c.maxPolarAngle = Math.PI * 0.495; // never orbit below the floor
    c.update();
    this.controls = c;
  }

  // ------------------------------------------------------------ events

  #bindEvents() {
    const el = this.renderer.domElement;

    // Size follows the panel, not the window, so dragging a resize handle
    // re-fits the canvas instead of stretching it.
    this.resizeObserver = new ResizeObserver(() => this.#resize());
    this.resizeObserver.observe(this.container);
    this.#resize();

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.down = null;
    this.onPointerDown = (e) => {
      if (e.button !== 0) return;
      this.down = { x: e.clientX, y: e.clientY };
    };
    this.onPointerUp = (e) => {
      if (e.button !== 0 || !this.down) return;
      const moved = Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y);
      this.down = null;
      if (moved > CLICK_TOLERANCE_PX) return; // that was an orbit, not a click
      this.store.select(this.pick(e.clientX, e.clientY));
    };
    el.addEventListener("pointerdown", this.onPointerDown);
    el.addEventListener("pointerup", this.onPointerUp);

    this.onLost = (e) => {
      e.preventDefault();
      this.onContextLost?.();
    };
    el.addEventListener("webglcontextlost", this.onLost);
  }

  #resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // Returns the node id under a screen point, or null for empty space.
  pick(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const visible = this.pickables.filter((o) => o.visible);
    for (const hit of this.raycaster.intersectObjects(visible, true)) {
      for (let o = hit.object; o; o = o.parent) {
        if (o.userData.nodeId) return o.userData.nodeId;
      }
    }
    return null;
  }

  // ------------------------------------------------------------ store sync

  // Apply the store to three.js. Cheap, and only runs when the store changes.
  #sync() {
    const state = this.store.get();
    if (state === this.lastState) return;
    this.lastState = state;

    for (const [id, obj] of Object.entries(this.objects)) {
      if (id === "camera" || !state.nodes[id]) continue;
      obj.visible = isEffectivelyVisible(state, id);
    }

    const sel = state.selectedId;
    for (const [id, helper] of Object.entries(this.helpers)) {
      helper.visible = id === sel && this.objects[id].visible;
      if (helper.visible) helper.update?.();
    }
    this.objects.grid.material = sel === "grid" ? this.gridMaterials.selected : this.gridMaterials.normal;
  }

  // ------------------------------------------------------------ actions

  setWireframe(on) {
    this.objects.cube.material.wireframe = on;
  }

  resetView() {
    this.#animateTo(HOME_POSITION.clone(), HOME_TARGET.clone());
  }

  // Move the camera so the selected object fills the view, keeping the
  // current viewing angle. Like F in Blender or Unity.
  frameSelection() {
    const id = this.store.get().selectedId;
    const obj = id && this.objects[id];
    if (!obj || id === "camera") return this.resetView();

    const sphere = new THREE.Sphere();
    if (id === "grid") sphere.set(new THREE.Vector3(0, 0, 0), 8);
    else if (obj.isLight) sphere.set(obj.getWorldPosition(new THREE.Vector3()), 1.5);
    else new THREE.Box3().setFromObject(obj).getBoundingSphere(sphere);

    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const distance = THREE.MathUtils.clamp(
      (sphere.radius / Math.sin(fov / 2)) * 1.3,
      this.controls.minDistance,
      this.controls.maxDistance
    );
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this.#animateTo(sphere.center.clone().addScaledVector(dir, distance), sphere.center.clone());
  }

  #animateTo(position, target) {
    this.tween = {
      start: performance.now(),
      fromPos: this.camera.position.clone(),
      fromTarget: this.controls.target.clone(),
      toPos: position,
      toTarget: target,
    };
  }

  // ------------------------------------------------------------ loop

  #tick() {
    if (document.hidden) return; // no GPU work while the app is in the background
    if (this.tween) {
      const t = Math.min(1, (performance.now() - this.tween.start) / FRAME_MS);
      const k = 1 - Math.pow(1 - t, 3); // ease out
      this.camera.position.lerpVectors(this.tween.fromPos, this.tween.toPos, k);
      this.controls.target.lerpVectors(this.tween.fromTarget, this.tween.toTarget, k);
      if (t === 1) this.tween = null;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);

    this.frames += 1;
    const now = performance.now();
    if (now - this.lastStats >= 500) {
      this.onStats?.({
        fps: Math.round((this.frames * 1000) / (now - this.lastStats)),
        tris: this.renderer.info.render.triangles,
      });
      this.frames = 0;
      this.lastStats = now;
    }
  }

  // Free everything. Without this, every workspace switch would leak a full
  // scene worth of GPU buffers and eventually a WebGL context.
  dispose() {
    this.renderer.setAnimationLoop(null);
    this.unsubscribe();
    this.resizeObserver.disconnect();
    const el = this.renderer.domElement;
    el.removeEventListener("pointerdown", this.onPointerDown);
    el.removeEventListener("pointerup", this.onPointerUp);
    // Remove before forceContextLoss below, or our own cleanup would show
    // the "graphics driver reset" message.
    el.removeEventListener("webglcontextlost", this.onLost);
    this.controls.dispose();
    this.gridMaterials.selected.dispose();
    this.gridMaterials.normal.dispose();
    this.scene.traverse((obj) => {
      obj.geometry?.dispose();
      const mats = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
      mats.forEach((m) => m.dispose());
    });
    Object.values(this.helpers).forEach((h) => h.dispose?.());
    this.renderer.dispose();
    this.renderer.forceContextLoss(); // release the WebGL context now, not at garbage collection
    el.remove();
  }
}
