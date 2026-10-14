// Run with: npm test   (uses Node's built-in test runner, no extra packages)

import { test } from "node:test";
import assert from "node:assert/strict";
import { createSceneStore, flatten, isEffectivelyVisible } from "./sceneStore.js";
import { DEFAULT_SCENE } from "./defaultScene.js";

const fresh = () => createSceneStore(DEFAULT_SCENE);

test("default scene builds with the expected rows in order", () => {
  const rows = flatten(fresh().get()).map((r) => `${r.depth}:${r.node.id}`);
  assert.deepEqual(rows, ["0:scene", "1:camera", "1:cube", "1:grid", "1:sun", "1:sky"]);
});

test("select, reselect and clear", () => {
  const s = fresh();
  let renders = 0;
  s.subscribe(() => renders++);
  s.select("cube");
  assert.equal(s.get().selectedId, "cube");
  s.select("cube"); // same id: no change, no re-render
  assert.equal(renders, 1);
  s.select(null);
  assert.equal(s.get().selectedId, null);
});

test("unknown or unselectable ids are ignored", () => {
  const s = fresh();
  s.select("cube");
  s.select("does-not-exist");
  assert.equal(s.get().selectedId, "cube");
  s.select("scene"); // root is not selectable
  assert.equal(s.get().selectedId, "cube");
});

test("arrow-key navigation walks selectable rows and stops at the ends", () => {
  const s = fresh();
  s.selectRelative(1);
  assert.equal(s.get().selectedId, "camera");
  s.selectRelative(1);
  assert.equal(s.get().selectedId, "cube");
  s.selectRelative(-5);
  assert.equal(s.get().selectedId, "camera");
  s.selectRelative(50);
  assert.equal(s.get().selectedId, "sky");
});

test("visibility toggles, but the camera and root cannot be hidden", () => {
  const s = fresh();
  s.toggleVisible("cube");
  assert.equal(s.get().nodes.cube.visible, false);
  s.toggleVisible("camera");
  assert.equal(s.get().nodes.camera.visible, true);
  s.setVisible("scene", false);
  assert.equal(s.get().nodes.scene.visible, true);
});

test("state is immutable: old snapshots never change", () => {
  const s = fresh();
  const before = s.get();
  s.toggleVisible("cube");
  assert.equal(before.nodes.cube.visible, true);
  assert.notEqual(before, s.get());
});

test("no-op updates do not notify", () => {
  const s = fresh();
  let renders = 0;
  s.subscribe(() => renders++);
  s.setVisible("cube", true);
  s.rename("cube", "Cube");
  s.rename("cube", "   ");
  assert.equal(renders, 0);
});

test("addNode and removeNode keep the tree and selection consistent", () => {
  const s = fresh();
  s.addNode({ id: "model", name: "Model", type: "group", children: [{ id: "part", type: "mesh" }] }, "scene");
  assert.deepEqual(s.get().nodes.scene.children.slice(-1), ["model"]);
  assert.equal(s.get().nodes.part.parentId, "model");
  s.select("part");
  s.removeNode("model");
  assert.equal(s.get().nodes.part, undefined);
  assert.equal(s.get().selectedId, null);
  assert.throws(() => s.addNode({ id: "cube", type: "mesh" }, "scene"), /duplicate/);
  assert.throws(() => s.addNode({ id: "x", type: "mesh" }, "nope"), /no parent/);
  assert.throws(() => s.addNode({ id: "y", type: "banana" }, "scene"), /unknown node type/);
});

test("a child of a hidden parent is not effectively visible", () => {
  const s = fresh();
  s.addNode({ id: "g", type: "group", children: [{ id: "m", type: "mesh" }] }, "scene");
  assert.equal(isEffectivelyVisible(s.get(), "m"), true);
  s.setVisible("g", false);
  assert.equal(isEffectivelyVisible(s.get(), "m"), false);
});

test("1,000 nodes stay fast", () => {
  const s = fresh();
  const t = performance.now();
  for (let i = 0; i < 1000; i++) s.addNode({ id: `n${i}`, type: "mesh" }, "scene");
  for (let i = 0; i < 1000; i++) s.select(`n${i}`);
  const rows = flatten(s.get());
  assert.equal(rows.length, 1006);
  assert.ok(performance.now() - t < 2000, "1,000 adds and selects should take well under 2s");
});
