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

// ---------------------------------------------------------------- transforms, undo, delete

import { isUntouched, topLevelOf, uniqueName, HISTORY_LIMIT } from "./sceneStore.js";

test("setTransform moves the cube and undo/redo restore it", () => {
  const s = fresh();
  s.setTransform("cube", { p: [2, 0.6, 1] });
  assert.deepEqual(s.get().nodes.cube.transform.p, [2, 0.6, 1]);
  assert.ok(s.canUndo());
  s.undo();
  assert.deepEqual(s.get().nodes.cube.transform.p, [0, 0.6, 0]);
  s.redo();
  assert.deepEqual(s.get().nodes.cube.transform.p, [2, 0.6, 1]);
});

test("lights, grid and camera cannot be moved or deleted", () => {
  const s = fresh();
  for (const id of ["sun", "sky", "grid", "camera", "scene"]) {
    s.setTransform(id, { p: [9, 9, 9] });
    assert.equal(s.removeNode(id), false, id);
    assert.ok(s.get().nodes[id], id);
  }
  assert.equal(s.canUndo(), false);
});

test("bad numbers never reach the scene", () => {
  const s = fresh();
  s.setTransform("cube", { p: [NaN, Infinity, 3], s: [0, -2, 1] });
  const t = s.get().nodes.cube.transform;
  assert.deepEqual(t.p, [0, 0.6, 3]);
  assert.ok(t.s.every((v) => v > 0), "scale stays positive");
});

test("a whole drag is one undo step", () => {
  const s = fresh();
  s.beginGesture();
  for (let i = 1; i <= 30; i++) s.setTransform("cube", { p: [i / 10, 0.6, 0] }, { record: false });
  s.endGesture();
  assert.deepEqual(s.get().nodes.cube.transform.p, [3, 0.6, 0]);
  s.undo();
  assert.deepEqual(s.get().nodes.cube.transform.p, [0, 0.6, 0]);
  assert.equal(s.canUndo(), false);
});

test("a click without moving adds no undo step", () => {
  const s = fresh();
  s.beginGesture();
  s.endGesture();
  assert.equal(s.canUndo(), false);
});

test("delete, undo brings it back, and selection is cleaned up", () => {
  const s = fresh();
  s.select("cube");
  assert.equal(s.removeNode("cube"), true);
  assert.equal(s.get().nodes.cube, undefined);
  assert.equal(s.get().selectedId, null);
  s.undo();
  assert.ok(s.get().nodes.cube);
  assert.ok(s.get().nodes.scene.children.includes("cube"));
});

test("a new edit clears redo", () => {
  const s = fresh();
  s.setTransform("cube", { p: [1, 0.6, 0] });
  s.undo();
  assert.ok(s.canRedo());
  s.toggleVisible("cube");
  assert.equal(s.canRedo(), false);
});

test("selection changes are not undo steps", () => {
  const s = fresh();
  s.select("cube");
  s.select("sun");
  assert.equal(s.canUndo(), false);
});

test("history is capped so memory cannot grow forever", () => {
  const s = fresh();
  for (let i = 0; i < HISTORY_LIMIT + 50; i++) s.setTransform("cube", { p: [i, 0.6, 0] });
  let undos = 0;
  while (s.undo()) undos++;
  assert.equal(undos, HISTORY_LIMIT);
});

test("referencedIds keeps deleted objects alive while undo can still need them", () => {
  const s = fresh();
  s.addNode({ id: "m1", type: "model", transformable: true }, "scene");
  s.removeNode("m1");
  assert.ok(s.referencedIds().has("m1"), "still undoable");
  for (let i = 0; i < HISTORY_LIMIT + 5; i++) s.setTransform("cube", { p: [i, 0.6, 0] });
  assert.equal(s.referencedIds().has("m1"), false, "dropped from history, safe to free");
});

test("starter cube counts as untouched only until the user changes it", () => {
  const s = fresh();
  assert.equal(isUntouched(s.get(), "cube"), true);
  s.select("cube");
  assert.equal(isUntouched(s.get(), "cube"), true, "selecting is not touching");
  s.setTransform("cube", { r: [0, 0.5, 0] });
  assert.equal(isUntouched(s.get(), "cube"), false);
  s.undo();
  assert.equal(isUntouched(s.get(), "cube"), true);
  s.rename("cube", "My box");
  assert.equal(isUntouched(s.get(), "cube"), false);
});

test("clicking a part of a model resolves to the whole model", () => {
  const s = fresh();
  s.addNode({ id: "m", type: "model", children: [{ id: "m/0", type: "group", children: [{ id: "m/1", type: "mesh" }] }] }, "scene");
  assert.equal(topLevelOf(s.get(), "m/1"), "m");
  assert.equal(topLevelOf(s.get(), "cube"), "cube");
});

test("imported names never collide", () => {
  const s = fresh();
  assert.equal(uniqueName(s.get(), "Helmet"), "Helmet");
  s.addNode({ id: "h1", name: "Helmet", type: "model" }, "scene");
  assert.equal(uniqueName(s.get(), "Helmet"), "Helmet (2)");
});

test("startup defaults are not undo steps", () => {
  const s = fresh();
  s.setVisible("grid", false, { record: false });
  assert.equal(s.get().nodes.grid.visible, false);
  assert.equal(s.canUndo(), false);
});
