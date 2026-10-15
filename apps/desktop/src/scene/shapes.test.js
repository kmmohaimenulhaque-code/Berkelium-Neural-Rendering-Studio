// shapes.test.js
// Run with: npm test

import test from "node:test";
import assert from "node:assert/strict";
import { SHAPES, shapeByKind, makeShapeMesh, freeSpot } from "./shapes.js";

test("every shape builds, sits on the floor and has its own geometry and material", () => {
  const seen = new Set();
  for (const s of SHAPES) {
    const mesh = makeShapeMesh(s.kind);
    mesh.geometry.computeBoundingBox();
    const box = mesh.geometry.boundingBox;
    // halfHeight must match the real geometry, or the shape floats or sinks.
    assert.ok(Math.abs(-box.min.y - s.halfHeight) < 0.006, `${s.kind}: bottom at ${box.min.y}, halfHeight ${s.halfHeight}`);
    // footprint must cover the shape seen from above, or overlap checks miss.
    const pos = mesh.geometry.attributes.position;
    let reach = 0;
    for (let i = 0; i < pos.count; i++) reach = Math.max(reach, Math.hypot(pos.getX(i), pos.getZ(i)));
    assert.ok(s.footprint >= reach - 0.01, `${s.kind}: footprint ${s.footprint} smaller than ${reach}`);
    assert.equal(mesh.userData.pickable, true);
    assert.equal(mesh.userData.shape, s.kind);
    assert.equal(Boolean(mesh.userData.selectionOutline), s.edges);
    assert.ok(!seen.has(mesh.geometry) && !seen.has(mesh.material), "geometry and material are never shared");
    seen.add(mesh.geometry).add(mesh.material);
  }
});

test("unknown shapes are rejected", () => {
  assert.equal(shapeByKind("teapot"), null);
  assert.throws(() => makeShapeMesh("teapot"), /unknown shape/);
});

test("free spot: an empty floor uses the view center, snapped", () => {
  assert.deepEqual(freeSpot({ center: [0, 0] }), [0, 0]);
  assert.deepEqual(freeSpot({ center: [1.13, -2.4] }), [1.25, -2.5]);
});

test("free spot: never lands inside something already there", () => {
  const occupied = [{ x: 0, z: 0, r: 0.85 }];
  const [x, z] = freeSpot({ center: [0, 0], occupied, footprint: 0.5 });
  assert.ok(Math.hypot(x, z) >= 0.85 + 0.5 + 0.3, `too close: ${x}, ${z}`);
  assert.ok(Math.hypot(x, z) <= 2, "but stays near the center");
});

test("free spot: ten shapes in a row all get different spots", () => {
  const occupied = [];
  const spots = new Set();
  for (let i = 0; i < 10; i++) {
    const [x, z] = freeSpot({ center: [0, 0], occupied, footprint: 0.5 });
    spots.add(`${x},${z}`);
    occupied.push({ x, z, r: 0.5 });
  }
  assert.equal(spots.size, 10);
});

test("free spot: bad input and far away views stay on the grid", () => {
  assert.deepEqual(freeSpot({ center: [NaN, undefined] }), [0, 0]);
  assert.deepEqual(freeSpot({ center: [500, -500] }), [18, -18]);
});
