// assets.js
// The three.js objects behind imported models, keyed by scene node id.
//
// This lives outside the viewport engine on purpose. Switching between the
// Scene, Code and Combined workspaces destroys and rebuilds the engine; the
// imported models must survive that, so they are parked here and the new
// engine re-attaches them.
//
// GPU memory is freed only when no state in the undo history still uses a
// node. Deleting a model and pressing Undo therefore always works, and
// models that can never come back do not leak.

const objects = new Map();

export const assets = {
  get: (id) => objects.get(id),
  has: (id) => objects.has(id),
  set(id, obj) {
    obj.userData.nodeId = id;
    objects.set(id, obj);
  },
  ids: () => [...objects.keys()],
  size: () => objects.size,
};

export function disposeObject(root) {
  root.traverse((o) => {
    o.geometry?.dispose();
    const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
    for (const m of mats) {
      for (const v of Object.values(m)) if (v && v.isTexture) v.dispose();
      m.dispose();
    }
  });
}

// Free every registered object that nothing (current state or history) uses.
export function collectGarbage(referencedIds) {
  let freed = 0;
  for (const [id, obj] of objects) {
    if (referencedIds.has(id)) continue;
    obj.removeFromParent();
    disposeObject(obj);
    objects.delete(id);
    freed += 1;
  }
  return freed;
}
