// SceneEngine.js
// Owns everything three.js: renderer, scene, camera, controls, picking,
// selection visuals, the move and rotate tools, and placing imported models.
//
// It reads the scene store and writes back only user edits: the selection
// when you click, and transforms when you drag a tool. Everything else flows
// one way, store to viewport.
//
// Kept out of React on purpose. The render loop runs 60 times a second and
// must never trigger React renders. React only gets stats twice a second.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { isEffectivelyVisible, topLevelOf } from "./sceneStore.js";
import { assets } from "./assets.js";
import { Turntable } from "./Turntable.js";
import { freeSpot } from "./shapes.js";

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

const applyTransform = (obj, t) => {
  obj.position.fromArray(t.p);
  obj.rotation.set(t.r[0], t.r[1], t.r[2]);
  obj.scale.fromArray(t.s);
};

const readTransform = (obj) => ({
  p: obj.position.toArray(),
  r: [obj.rotation.x, obj.rotation.y, obj.rotation.z],
  s: obj.scale.toArray(),
});

export class SceneEngine {
  constructor({ container, store, onStats, onContextLost, labelEl }) {
    this.container = container;
    this.store = store;
    this.onStats = onStats;
    this.onContextLost = onContextLost;
    this.labelEl = labelEl; // angle readout while rotating, positioned by the engine
    this.tool = "move";

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
    this.#buildTools();
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

    // The studio environment for imported models is built on demand (see
    // #ensureEnvironment), so opening the viewport stays instant.
    this.envTexture = null;

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

    // Invisible floor that only shows shadows.
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.ShadowMaterial({ opacity: 0.35 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    const cube = new THREE.Mesh(
      new THREE.BoxGeometry(1.2, 1.2, 1.2),
      new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.35, metalness: 0.25, envMapIntensity: 0 })
    );
    cube.castShadow = true;
    cube.userData.pickable = true;
    // Thin edge outline so the cube reads clearly at any angle.
    cube.add(new THREE.LineSegments(
      new THREE.EdgesGeometry(cube.geometry),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 })
    ));

    // Built-in objects. Imported models live in the assets registry instead,
    // so they survive this engine being destroyed on a workspace switch.
    this.builtins = { cube, grid, sun, sky };
    for (const [id, obj] of Object.entries(this.builtins)) obj.userData.nodeId = id;

    // Selection visuals. Created once and shown or hidden, never rebuilt.
    this.cubeOutline = this.#outlineFor(cube);
    this.lightHelpers = {
      sun: new THREE.DirectionalLightHelper(sun, 1, ACCENT),
      sky: new THREE.HemisphereLightHelper(sky, 0.6, ACCENT),
    };
    this.selectionBox = new THREE.Box3Helper(new THREE.Box3(), 0xffffff);
    for (const h of [this.cubeOutline, ...Object.values(this.lightHelpers), this.selectionBox]) {
      h.visible = false;
      h.raycast = () => {};
      if (!h.parent) scene.add(h); // mesh outlines are already parented to their mesh
    }
    this.gridMaterials = {
      normal: grid.material,
      selected: new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.35 }),
    };
  }

  // A soft studio environment so imported PBR models (metal, glass) look
  // right instead of flat black. Built only once a model is in the scene:
  // it costs milliseconds on a GPU but seconds on software rendering
  // (machines without a proper graphics driver), and the cube does not use it.
  #ensureEnvironment() {
    if (this.envTexture) return;
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const room = new RoomEnvironment();
    this.envTexture = pmrem.fromScene(room, 0.04).texture;
    room.dispose();
    pmrem.dispose();
    this.scene.environment = this.envTexture;
    this.scene.environmentIntensity = 0.85;
  }

  // Selection outline for the cube: crisp white edges drawn as a child of
  // the mesh, so it follows any move, rotate or scale for free. White, not
  // the cyan accent, because the cube itself is cyan.
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
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 400);
    this.camera.position.copy(HOME_POSITION);
    const c = new OrbitControls(this.camera, this.renderer.domElement);
    c.target.copy(HOME_TARGET);
    c.enableDamping = true;
    c.dampingFactor = 0.08;
    c.minDistance = 0.5;
    c.maxDistance = 60;
    c.maxPolarAngle = Math.PI * 0.495; // never orbit below the floor
    c.update();
    this.controls = c;
  }

  #buildTools() {
    // Move: the standard three-axis arrows (approved in the preview).
    const tc = new TransformControls(this.camera, this.renderer.domElement);
    tc.setMode("translate");
    tc.setSize(0.9);
    tc.addEventListener("dragging-changed", (e) => {
      this.controls.enabled = !e.value;
      if (e.value) this.store.beginGesture();
      else this.store.endGesture(); // the whole drag becomes one undo step
    });
    tc.addEventListener("objectChange", () => {
      const id = tc.object?.userData.nodeId;
      if (id) this.store.setTransform(id, readTransform(tc.object), { record: false });
    });
    this.transformControls = tc;
    this.scene.add(tc.getHelper());

    // Rotate: the turntable ring (approved over the three-ring gizmo).
    this.turntable = new Turntable({
      scene: this.scene,
      camera: this.camera,
      onStart: () => {
        this.controls.enabled = false;
        this.store.beginGesture();
      },
      onChange: () => {
        const obj = this.turntable.target;
        const id = obj?.userData.nodeId;
        if (id) this.store.setTransform(id, readTransform(obj), { record: false });
      },
      onEnd: () => {
        this.controls.enabled = true;
        this.store.endGesture();
      },
    });
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
    this.down = null;
    this.ndc = (e) => {
      const rect = el.getBoundingClientRect();
      return new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    };

    // Capture phase, so the turntable claims the press before the orbit
    // controls start rotating the camera.
    this.onPointerDown = (e) => {
      if (e.button !== 0) return;
      const onGizmo = this.transformControls.enabled && this.transformControls.axis !== null;
      const onTurntable = this.tool === "rotate" && this.turntable.pointerDown(this.ndc(e));
      if (onTurntable) {
        el.setPointerCapture?.(e.pointerId);
        e.stopImmediatePropagation();
      }
      this.down = { x: e.clientX, y: e.clientY, onTool: onGizmo || onTurntable };
    };
    this.onPointerMove = (e) => {
      if (this.turntable.dragging) this.turntable.pointerMove(this.ndc(e), e.shiftKey);
    };
    this.onPointerUp = (e) => {
      if (this.turntable.pointerUp()) {
        this.down = null;
        return;
      }
      if (e.button !== 0 || !this.down) return;
      const { x, y, onTool } = this.down;
      this.down = null;
      if (onTool) return; // that press was on a move arrow, not a click on the scene
      if (Math.hypot(e.clientX - x, e.clientY - y) > CLICK_TOLERANCE_PX) return; // an orbit, not a click
      this.store.select(this.pick(e.clientX, e.clientY));
    };
    el.addEventListener("pointerdown", this.onPointerDown, { capture: true });
    el.addEventListener("pointermove", this.onPointerMove);
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
  // A click on any part of an imported model selects the whole model.
  pick(clientX, clientY) {
    this.raycaster.setFromCamera(this.ndc({ clientX, clientY }), this.camera);
    const candidates = [];
    this.scene.traverseVisible((o) => { if (o.isMesh && o.userData.pickable) candidates.push(o); });
    const state = this.store.get();
    for (const hit of this.raycaster.intersectObjects(candidates, false)) {
      for (let o = hit.object; o; o = o.parent) {
        const id = o.userData.nodeId;
        if (id && state.nodes[id]) return topLevelOf(state, id);
      }
    }
    return null;
  }

  objectFor(id) {
    return this.builtins[id] || assets.get(id) || null;
  }

  // ------------------------------------------------------------ store sync

  // Apply the store to three.js. Cheap, and only runs when the store changes.
  #sync() {
    const state = this.store.get();
    if (state === this.lastState) return;
    this.lastState = state;

    // Imported objects: put each where the store says it belongs, or take it
    // out of the scene if it was deleted. Nothing is destroyed here; the
    // assets registry frees memory once undo can no longer need it.
    for (const id of assets.ids()) {
      const obj = assets.get(id);
      const node = state.nodes[id];
      if (!node) {
        if (obj.parent) obj.removeFromParent();
        continue;
      }
      if (obj.userData.needsEnvironment) this.#ensureEnvironment(); // models only; shapes and the cube do not use it
      const home = node.parentId === "scene" ? this.scene : obj.userData.homeParent;
      if (home && obj.parent !== home) home.add(obj);
    }

    for (const [id, node] of Object.entries(state.nodes)) {
      const obj = this.objectFor(id);
      if (!obj) continue;
      if (node.transformable) applyTransform(obj, node.transform);
      if (id !== "camera") obj.visible = node.visible;
    }
    // The cube is built in, so add or remove it like an imported object.
    const cube = this.builtins.cube;
    if (state.nodes.cube && !cube.parent) this.scene.add(cube);
    if (!state.nodes.cube && cube.parent) cube.removeFromParent();

    this.#syncSelection(state);
  }

  #syncSelection(state) {
    const sel = state.selectedId;
    const node = sel ? state.nodes[sel] : null;
    const obj = node ? this.objectFor(sel) : null;
    const shown = Boolean(obj && isEffectivelyVisible(state, sel));

    this.cubeOutline.visible = sel === "cube" && shown;
    // Added shapes with corners carry their own outline (see shapes.js).
    const outline = shown ? obj.userData.selectionOutline || null : null;
    if (this.activeOutline && this.activeOutline !== outline) this.activeOutline.visible = false;
    if (outline) outline.visible = true;
    this.activeOutline = outline;
    for (const [id, helper] of Object.entries(this.lightHelpers)) {
      helper.visible = id === sel && shown;
      if (helper.visible) helper.update();
    }
    this.builtins.grid.material = sel === "grid" ? this.gridMaterials.selected : this.gridMaterials.normal;

    const boxed = shown && node.transformable && sel !== "cube" && !outline;
    this.selectionBox.visible = boxed;
    if (boxed) this.selectionBox.box.setFromObject(obj).expandByScalar(0.02);

    // Tools attach only to things you are allowed to move.
    const editable = shown && node.transformable ? obj : null;
    const tc = this.transformControls;
    const wantMove = this.tool === "move" && editable;
    if (wantMove && tc.object !== editable) tc.attach(editable);
    if (!wantMove && tc.object) tc.detach();
    tc.enabled = Boolean(wantMove);

    if (!this.turntable.dragging) this.turntable.attach(this.tool === "rotate" ? editable : null);
    else this.turntable.update();
  }

  // ------------------------------------------------------------ actions

  setTool(tool) {
    if (tool !== "move" && tool !== "rotate") return;
    this.tool = tool;
    this.lastState = null; // force a re-sync so the right tool appears
    this.#sync();
    this.#placeAngleLabel(); // show or hide the readout now, not on the next frame
  }

  setWireframe(on) {
    this.builtins.cube.material.wireframe = on;
    for (const id of assets.ids()) {
      assets.get(id).traverse((o) => {
        if (!o.isMesh) return;
        for (const m of Array.isArray(o.material) ? o.material : [o.material]) if (m && "wireframe" in m) m.wireframe = on;
      });
    }
  }

  // Where a new shape should go: the middle of the view, on the floor,
  // nudged aside if something already stands there (see freeSpot).
  // footprint: the new shape's radius seen from above.
  findFreeSpot(footprint = 0.5) {
    const occupied = [];
    const state = this.store.get();
    for (const id of state.nodes.scene?.children || []) {
      const node = state.nodes[id];
      const obj = this.objectFor(id);
      if (!node?.transformable || !node.visible || !obj?.parent) continue;
      const box = new THREE.Box3().setFromObject(obj);
      if (box.isEmpty()) continue;
      const c = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      occupied.push({ x: c.x, z: c.z, r: Math.hypot(size.x, size.z) / 2 });
    }
    const t = this.controls.target;
    return freeSpot({ center: [t.x, t.z], occupied, footprint });
  }

  resetView() {
    this.#animateTo(HOME_POSITION.clone(), HOME_TARGET.clone());
  }

  // Move the camera so the selected object fills the view, keeping the
  // current viewing angle. Like F in Blender or Unity.
  frameSelection() {
    const state = this.store.get();
    const id = state.selectedId;
    const obj = id && this.objectFor(id);
    if (!obj || id === "camera") return this.resetView();

    const sphere = new THREE.Sphere();
    if (id === "grid") sphere.set(new THREE.Vector3(0, 0, 0), 8);
    else if (obj.isLight) sphere.set(obj.getWorldPosition(new THREE.Vector3()), 1.5);
    else {
      const box = new THREE.Box3().setFromObject(obj);
      if (box.isEmpty()) return;
      box.getBoundingSphere(sphere);
    }

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
    if (this.selectionBox.visible && this.transformControls.dragging) {
      this.selectionBox.box.setFromObject(this.transformControls.object).expandByScalar(0.02);
    }
    this.renderer.render(this.scene, this.camera);
    this.#placeAngleLabel();

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

  // The angle readout next to the turntable dot. Written straight to the DOM:
  // going through React 60 times a second would be wasteful.
  #placeAngleLabel() {
    const el = this.labelEl;
    if (!el) return;
    const tt = this.turntable;
    if (!tt.group.visible) {
      if (el.style.display !== "none") el.style.display = "none";
      return;
    }
    const v = tt.knob.getWorldPosition(new THREE.Vector3()).project(this.camera);
    const x = ((v.x + 1) / 2) * this.container.clientWidth + 14;
    const y = ((1 - v.y) / 2) * this.container.clientHeight - 12;
    el.style.display = "block";
    el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    const text = `${tt.degrees.toFixed(1)}°`;
    if (el.textContent !== text) el.textContent = text;
  }

  // Free everything this engine created. Imported models are detached, not
  // destroyed: they belong to the assets registry and the next engine (after
  // a workspace switch) re-attaches them.
  dispose() {
    this.renderer.setAnimationLoop(null);
    this.unsubscribe();
    this.resizeObserver.disconnect();
    const el = this.renderer.domElement;
    el.removeEventListener("pointerdown", this.onPointerDown, { capture: true });
    el.removeEventListener("pointermove", this.onPointerMove);
    el.removeEventListener("pointerup", this.onPointerUp);
    // Remove before forceContextLoss below, or our own cleanup would show
    // the "graphics driver reset" message.
    el.removeEventListener("webglcontextlost", this.onLost);
    // Shapes outlive this engine (assets registry). Do not leave one looking
    // selected for the next engine.
    if (this.activeOutline) this.activeOutline.visible = false;
    this.transformControls.detach();
    this.transformControls.dispose();
    this.turntable.dispose();
    this.controls.dispose();
    for (const id of assets.ids()) {
      const obj = assets.get(id);
      if (obj.parent === this.scene) this.scene.remove(obj);
    }
    this.gridMaterials.selected.dispose();
    this.gridMaterials.normal.dispose();
    if (!this.builtins.cube.parent) this.scene.add(this.builtins.cube); // a deleted cube must be freed too
    this.scene.traverse((obj) => {
      obj.geometry?.dispose();
      const mats = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
      mats.forEach((m) => m.dispose());
    });
    Object.values(this.lightHelpers).forEach((h) => h.dispose?.());
    this.envTexture?.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss(); // release the WebGL context now, not at garbage collection
    el.remove();
  }
}
