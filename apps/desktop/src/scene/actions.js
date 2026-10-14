// actions.js
// User-level scene commands: import, delete, undo, redo, and the keyboard
// shortcuts for them. Both the viewport and the hierarchy call these, so a
// key does exactly the same thing wherever the focus is.

import * as THREE from "three";
import { sceneStore } from "./store.js";
import { assets } from "./assets.js";
import { loadModelFiles, placement, ImportError, formatBytes } from "./importer.js";
import { isUntouched, uniqueName } from "./sceneStore.js";
import { STARTER_CUBE_ID } from "./defaultScene.js";
import { toast } from "./toast.js";

const MAX_LISTED_PARTS = 200; // deeper or larger models still render, they just are not all listed
const MAX_DEPTH = 4;

const hasMesh = (o) => {
  let found = false;
  o.traverse((c) => { if (c.isMesh) found = true; });
  return found;
};

const transformOf = (o) => ({
  p: o.position.toArray(),
  r: [o.rotation.x, o.rotation.y, o.rotation.z],
  s: o.scale.toArray(),
});

// Registers the meaningful parts of a model as hierarchy rows. Each listed
// part keeps a reference to its original parent so it can be detached on
// delete and put back exactly where it was on undo.
function listParts(rootId, object) {
  let count = 0;
  const walk = (o, depth) => {
    const rows = [];
    for (const child of o.children) {
      if (count >= MAX_LISTED_PARTS) break;
      if (!child.isMesh && !hasMesh(child)) continue;
      const id = `${rootId}/${count++}`;
      child.userData.homeParent = child.parent;
      assets.set(id, child);
      rows.push({
        id,
        name: child.name || (child.isMesh ? "Mesh" : "Group"),
        type: child.isMesh ? "mesh" : "group",
        transformable: true,
        transform: transformOf(child),
        children: depth < MAX_DEPTH ? walk(child, depth + 1) : [],
      });
    }
    return rows;
  };
  return walk(object, 1);
}

// If the user brings the starter cube back with Undo, they want it.
// Later imports then leave it alone.
let keepStarterCube = false;

// An Undo button on a toast only works while nothing else has changed since,
// otherwise it would undo some other, unrelated edit.
function undoIfStill(nodesAfter, label, onUndo) {
  return {
    label,
    run() {
      if (sceneStore.get().nodes !== nodesAfter) {
        toast({ text: "Too late to undo from here. Use ⌘Z / Ctrl+Z instead.", kind: "info" });
        return;
      }
      sceneStore.undo();
      onUndo?.();
    },
  };
}

export async function importIntoScene(files) {
  let loaded;
  try {
    loaded = await loadModelFiles(files);
  } catch (err) {
    const text = err instanceof ImportError ? err.message : "Import failed for an unexpected reason.";
    if (!(err instanceof ImportError)) console.error("[import]", err);
    toast({ text, kind: "error", timeout: 9000 });
    return null;
  }

  const id = `model-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
  const name = uniqueName(sceneStore.get(), loaded.name);
  let place;
  try {
    place = placement(loaded.object);
  } catch (err) {
    toast({ text: err.message, kind: "error", timeout: 9000 });
    return null;
  }

  // The store controls this wrapper's position, rotation and scale. The
  // file's own internal transforms stay untouched inside it.
  const wrapper = new THREE.Group();
  wrapper.name = name;
  wrapper.add(loaded.object);
  assets.set(id, wrapper);

  sceneStore.addNode(
    {
      id,
      name,
      type: "model",
      transformable: true,
      transform: place.transform,
      meta: { fileName: loaded.fileName, bytes: loaded.bytes, ...loaded.stats },
      children: listParts(id, loaded.object),
    },
    "scene"
  );

  // The starter cube is a placeholder. Once real content arrives it only gets
  // in the way, but only if the user never touched it.
  let cubeRemoved = false;
  if (!keepStarterCube && sceneStore.get().nodes[STARTER_CUBE_ID] && isUntouched(sceneStore.get(), STARTER_CUBE_ID)) {
    cubeRemoved = sceneStore.removeNode(STARTER_CUBE_ID);
  }
  const nodesAfter = sceneStore.get().nodes;
  sceneStore.select(id);

  toast({
    text: `Imported ${loaded.fileName}`,
    detail: cubeRemoved ? "Starter cube removed" : `${loaded.stats.triangles.toLocaleString()} triangles · ${formatBytes(loaded.bytes)}`,
    kind: "ok",
    action: cubeRemoved ? undoIfStill(nodesAfter, "Undo", () => { keepStarterCube = true; }) : null,
  });
  for (const w of [place.note, ...loaded.warnings].filter(Boolean)) toast({ text: w, kind: "warn", timeout: 9000 });
  return id;
}

export function deleteSelected() {
  const state = sceneStore.get();
  const node = state.nodes[state.selectedId];
  if (!node) return;
  if (!node.removable) {
    toast({ text: `${node.name} is part of the studio and cannot be deleted. You can hide it instead.`, kind: "info" });
    return;
  }
  if (sceneStore.removeNode(node.id)) {
    toast({ text: `Deleted ${node.name}`, kind: "info", action: undoIfStill(sceneStore.get().nodes, "Undo") });
  }
}

const isMac = typeof navigator !== "undefined" && /Mac/i.test(navigator.platform || navigator.userAgent);

// Shared shortcuts for the viewport and the hierarchy. Returns true when the
// key was handled, so the caller can stop it reaching anything else.
// tools: optional { setTool } from the viewport.
export function handleSceneKey(e, tools) {
  const mod = isMac ? e.metaKey : e.ctrlKey;
  const key = e.key.toLowerCase();
  if (mod && key === "z") {
    if (e.shiftKey) sceneStore.redo();
    else sceneStore.undo();
    return true;
  }
  if (mod && key === "y" && !isMac) {
    sceneStore.redo();
    return true;
  }
  if (mod || e.altKey) return false;
  // Mac keyboards label Backspace as "delete"; Windows and Linux have both.
  if (key === "delete" || key === "backspace") {
    deleteSelected();
    return true;
  }
  if (tools && key === "w") { tools.setTool("move"); return true; }
  if (tools && key === "e") { tools.setTool("rotate"); return true; }
  return false;
}
