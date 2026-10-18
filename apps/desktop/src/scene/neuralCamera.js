// neuralCamera.js
// Converts the viewport's three.js camera into a Berkelium CameraSpec, so the backend renders the Gaussian
// scene from exactly the view on screen.
//
// Berkelium cameras use OpenCV axes (+X right, +Y down, +Z forward) and T_cw maps world -> camera, row-major.
// three.js uses OpenGL axes (+Y up, -Z forward). The generated model sits in the scene under a wrapper
// (placed on the floor, then moved or rotated by the user) while its Gaussian scene lives in the model's own
// frame, so the camera is expressed relative to that wrapper:
//
//   T_cw = F · V · M      V = camera.matrixWorldInverse, M = model.matrixWorld, F = diag(1, -1, -1, 1)
//
// A uniform scale s in M is divided out: it scales camera-space depth but not pixels, and the CameraSpec
// contract requires an orthonormal rotation.

import * as THREE from "three";

export function cameraSpecFor(camera, modelMatrixWorld, width, height) {
  camera.updateMatrixWorld();
  const M = modelMatrixWorld ? modelMatrixWorld.clone() : new THREE.Matrix4();
  const s = new THREE.Vector3().setFromMatrixScale(M);
  if (Math.abs(s.x - s.y) > 1e-6 * s.x || Math.abs(s.x - s.z) > 1e-6 * s.x) {
    throw new Error("The model has a non-uniform scale; reset its scale to render this view.");
  }
  const VM = camera.matrixWorldInverse.clone().multiply(M);
  const e = VM.elements; // column-major
  const flip = [1, -1, -1];
  const T_cw = [0, 1, 2].map((r) => [0, 1, 2, 3].map((c) => (flip[r] * e[c * 4 + r]) / s.x));
  T_cw.push([0, 0, 0, 1]);
  const fy = (0.5 * height) / Math.tan(THREE.MathUtils.degToRad(camera.getEffectiveFOV()) / 2);
  return {
    width, height, fx: fy, fy, cx: width / 2, cy: height / 2, T_cw,
    near: Math.max(1e-3, camera.near / s.x), far: camera.far / s.x,
  };
}

// Pixel coordinates and depth of a model-frame point under a CameraSpec (same maths as the backend).
export function projectPoint(spec, p) {
  const T = spec.T_cw;
  const c = [0, 1, 2].map((r) => T[r][0] * p[0] + T[r][1] * p[1] + T[r][2] * p[2] + T[r][3]);
  return { u: (spec.fx * c[0]) / c[2] + spec.cx, v: (spec.fy * c[1]) / c[2] + spec.cy, z: c[2] };
}
