// neuralCamera.test.js
// Run with: npm test

import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { cameraSpecFor, projectPoint } from "./neuralCamera.js";

const camAt = (eye, target, fov = 50, aspect = 1) => {
  const c = new THREE.PerspectiveCamera(fov, aspect, 0.05, 400);
  c.position.set(...eye);
  c.lookAt(new THREE.Vector3(...target));
  c.updateMatrixWorld();
  return c;
};

const rotationIsProper = (T) => {
  const R = T.slice(0, 3).map((r) => r.slice(0, 3));
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const d = R[i][0] * R[j][0] + R[i][1] * R[j][1] + R[i][2] * R[j][2] - (i === j ? 1 : 0);
      assert.ok(Math.abs(d) < 1e-9, `R R^T != I at ${i},${j}`);
    }
  }
  const det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
    + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
  assert.ok(Math.abs(det - 1) < 1e-9, `det ${det}`);
};

test("the look-at target lands in the image centre, in front of the camera (OpenCV axes)", () => {
  const spec = cameraSpecFor(camAt([2, 1.5, 3], [0, 0.4, 0]), null, 320, 240);
  rotationIsProper(spec.T_cw);
  assert.deepEqual(spec.T_cw[3], [0, 0, 0, 1]);
  const p = projectPoint(spec, [0, 0.4, 0]);
  assert.ok(Math.abs(p.u - 160) < 1e-6 && Math.abs(p.v - 120) < 1e-6, JSON.stringify(p));
  assert.ok(p.z > 0, "depth must be positive in front of the camera");
});

test("world up is image up (+Y down in OpenCV) and world right stays right", () => {
  const spec = cameraSpecFor(camAt([0, 0, 4], [0, 0, 0]), null, 200, 200);
  assert.ok(projectPoint(spec, [0, 0.5, 0]).v < 100, "a point above the target must be higher in the image");
  assert.ok(projectPoint(spec, [0.5, 0, 0]).u > 100, "+X must be to the right when looking down -Z");
});

test("matches three.js's own projection for the same camera", () => {
  const cam = camAt([3, 2, -2], [0.2, 0.3, 0.1], 40, 4 / 3);
  const spec = cameraSpecFor(cam, null, 400, 300);
  for (const p of [[0.2, 0.3, 0.1], [0.6, -0.2, 0.4], [-0.5, 0.8, -0.3]]) {
    const ndc = new THREE.Vector3(...p).project(cam);
    const u = (ndc.x + 1) * 200;
    const v = (1 - ndc.y) * 150;
    const q = projectPoint(spec, p);
    assert.ok(Math.abs(q.u - u) < 1e-6 && Math.abs(q.v - v) < 1e-6, `${JSON.stringify(q)} vs ${u},${v}`);
  }
});

test("model placement (move, rotate, uniform scale) is folded into the camera", () => {
  const cam = camAt([3, 2.5, 4], [1, 0.5, 0]);
  const model = new THREE.Group();
  model.position.set(1, 0.3, -0.2);
  model.rotation.set(0, 0.7, 0);
  model.scale.setScalar(1.8);
  model.updateMatrixWorld();
  const spec = cameraSpecFor(cam, model.matrixWorld, 256, 256);
  rotationIsProper(spec.T_cw);
  for (const local of [[0, 0, 0], [0.3, 0.4, -0.2]]) {
    const world = new THREE.Vector3(...local).applyMatrix4(model.matrixWorld);
    const ndc = world.clone().project(cam);
    const q = projectPoint(spec, local);
    assert.ok(Math.abs(q.u - (ndc.x + 1) * 128) < 1e-6 && Math.abs(q.v - (1 - ndc.y) * 128) < 1e-6);
  }
});

test("non-uniform scale is refused instead of producing a sheared camera", () => {
  const m = new THREE.Matrix4().makeScale(1, 2, 1);
  assert.throws(() => cameraSpecFor(camAt([0, 0, 4], [0, 0, 0]), m, 64, 64), /non-uniform/);
});
