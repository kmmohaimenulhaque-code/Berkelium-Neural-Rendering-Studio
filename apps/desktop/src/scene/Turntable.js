// Turntable.js
// The rotate tool: one ring on the floor around the selected object, with a
// white dot you drag to spin it. Approved over three.js's default three-ring
// rotate gizmo, which looked busy.
//
// Spins around the vertical axis only, which is what you want almost every
// time you turn a model to face the camera. Exact X and Z rotation can be
// typed in the Transform panel.

import * as THREE from "three";

const ACCENT = 0x00d4d4;
const SNAP = THREE.MathUtils.degToRad(15);

// Wraps an angle into (-PI, PI] so the readout never shows 540°.
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export class Turntable {
  constructor({ scene, camera, onStart, onChange, onEnd }) {
    this.camera = camera;
    this.onStart = onStart;
    this.onChange = onChange;
    this.onEnd = onEnd;
    this.target = null;
    this.radius = 1;
    this.dragging = null;

    this.group = new THREE.Group();
    this.group.visible = false;
    this.group.renderOrder = 10;
    scene.add(this.group);

    const flat = (mesh) => { mesh.rotation.x = -Math.PI / 2; return mesh; };
    const mat = (opacity) => new THREE.MeshBasicMaterial({
      color: ACCENT, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false,
    });
    this.ringMat = mat(0.9);
    this.ring = flat(new THREE.Mesh(new THREE.BufferGeometry(), this.ringMat));
    this.sector = flat(new THREE.Mesh(new THREE.BufferGeometry(), mat(0.16)));
    // Wide invisible band so the ring is easy to grab, not a 2px target.
    this.hit = flat(new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide })));
    this.knob = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
    this.knob.renderOrder = 11;
    this.halo = flat(new THREE.Mesh(new THREE.RingGeometry(1.3, 1.9, 32), mat(0.85)));
    this.group.add(this.ring, this.sector, this.hit, this.knob, this.halo);

    this.raycaster = new THREE.Raycaster();
    this.plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  }

  // Attach to an object (or null to hide). Sizes the ring to the object's
  // footprint so a small cube and a large model both get a sensible ring.
  attach(object) {
    this.target = object;
    if (!object || !object.visible) {
      this.group.visible = false;
      return;
    }
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) {
      this.group.visible = false;
      return;
    }
    const size = box.getSize(new THREE.Vector3());
    const radius = THREE.MathUtils.clamp(Math.hypot(size.x, size.z) * 0.62, 0.7, 40);
    const center = box.getCenter(new THREE.Vector3());
    this.group.position.set(center.x, Math.max(box.min.y, 0) + 0.012, center.z);
    if (Math.abs(radius - this.radius) > 1e-4 || !this.ring.geometry.attributes.position) {
      this.radius = radius;
      // Line and dot sizes are capped so a big model gets a bigger ring,
      // not a chunkier one.
      const w = THREE.MathUtils.clamp(radius * 0.012, 0.018, 0.04);
      this.ring.geometry.dispose();
      this.ring.geometry = new THREE.RingGeometry(radius - w, radius + w, 128);
      this.hit.geometry.dispose();
      this.hit.geometry = new THREE.RingGeometry(radius * 0.8, radius * 1.2, 64);
      const k = THREE.MathUtils.clamp(radius * 0.045, 0.06, 0.1);
      this.knob.scale.setScalar(k);
      this.halo.scale.setScalar(k);
    }
    this.group.visible = true;
    this.update();
  }

  // Place the dot and the swept area for the object's current Y rotation.
  update() {
    if (!this.target) return;
    const a = wrap(this.target.rotation.y);
    const R = this.radius;
    // Angle 0 points toward +Z (the front), so the dot starts facing the camera.
    this.knob.position.set(Math.sin(a) * R, 0.02, Math.cos(a) * R);
    this.halo.position.copy(this.knob.position);
    this.sector.geometry.dispose();
    this.sector.geometry = a >= 0
      ? new THREE.CircleGeometry(R * 0.97, 64, -Math.PI / 2, a || 1e-4)
      : new THREE.CircleGeometry(R * 0.97, 64, -Math.PI / 2 + a, -a);
    this.angle = a;
  }

  #pointerAngle(ndc) {
    this.raycaster.setFromCamera(ndc, this.camera);
    this.plane.constant = -this.group.position.y;
    const p = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(this.plane, p)) return null;
    return Math.atan2(p.x - this.group.position.x, p.z - this.group.position.z);
  }

  // Returns true when the press landed on the dot or the ring.
  pointerDown(ndc) {
    if (!this.group.visible || !this.target) return false;
    this.raycaster.setFromCamera(ndc, this.camera);
    const hits = this.raycaster.intersectObjects([this.knob, this.hit], false);
    if (!hits.length) return false;
    const start = this.#pointerAngle(ndc);
    if (start === null) return false;
    this.dragging = { start, rotation: this.target.rotation.y, last: start, turned: 0 };
    this.ringMat.opacity = 1;
    this.onStart?.();
    return true;
  }

  pointerMove(ndc, snap) {
    if (!this.dragging) return;
    const a = this.#pointerAngle(ndc);
    if (a === null) return;
    // Accumulate small steps so crossing the back of the ring does not jump 360°.
    this.dragging.turned += wrap(a - this.dragging.last);
    this.dragging.last = a;
    let next = this.dragging.rotation + this.dragging.turned;
    if (snap) next = Math.round(next / SNAP) * SNAP;
    this.target.rotation.y = wrap(next);
    this.update();
    this.onChange?.(this.target.rotation.y);
  }

  pointerUp() {
    if (!this.dragging) return false;
    this.dragging = null;
    this.ringMat.opacity = 0.9;
    this.onEnd?.();
    return true;
  }

  get degrees() {
    return THREE.MathUtils.radToDeg(this.angle || 0);
  }

  dispose() {
    this.group.removeFromParent();
    this.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
  }
}
