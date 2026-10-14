// sceneStore.js
// The single source of truth for what is in the scene, where each object is,
// and what is selected.
//
// The hierarchy panel (React) and the viewport (three.js) both read from this
// store and never talk to each other directly. A properties panel, the model
// importer and undo all plug into the same store instead of being wired into
// every other component.
//
// State is immutable. Every change produces a new state object, which is what
// React's useSyncExternalStore needs to know when to re-render. It also makes
// undo cheap: a history entry is just an old state, and unchanged nodes are
// shared between entries, not copied.
//
// Shape:
//   nodes: { [id]: {
//     id, name, type, parentId, children: [id],
//     visible, selectable, hideable, removable, transformable,
//     transform: { p: [x,y,z], r: [x,y,z], s: [x,y,z] },   // local, radians
//     origin:    the transform and name it was created with (for "untouched" checks)
//     meta:      optional info about imported files (name, size, triangles...)
//   } }
//   rootIds:    [id]
//   selectedId: id | null

export const NODE_TYPES = ["scene", "camera", "mesh", "light", "helper", "group", "model"];
export const HISTORY_LIMIT = 100;

const IDENTITY = Object.freeze({ p: [0, 0, 0], r: [0, 0, 0], s: [1, 1, 1] });

function cleanVec(v, fallback) {
  if (!Array.isArray(v) || v.length !== 3) return [...fallback];
  return v.map((n, i) => (Number.isFinite(n) ? n : fallback[i]));
}

// Invalid numbers fall back to the matching value in base (the object's
// current transform), so a bad input never teleports anything to 0.
export function cleanTransform(t, base = IDENTITY) {
  return {
    p: cleanVec(t?.p, base.p),
    r: cleanVec(t?.r, base.r),
    // A zero or negative scale makes an object vanish or turn inside out,
    // which a user would read as a crash. Clamp to 0.01, small enough for any
    // real use and still visible as a number in the panel.
    s: cleanVec(t?.s, base.s).map((n) => Math.max(0.01, n)),
  };
}

const sameVec = (a, b) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
export const sameTransform = (a, b) => sameVec(a.p, b.p) && sameVec(a.r, b.r) && sameVec(a.s, b.s);

function makeNode(input, parentId) {
  if (!input || typeof input.id !== "string" || !input.id) throw new Error("node needs a string id");
  if (!NODE_TYPES.includes(input.type)) throw new Error(`unknown node type: ${input.type}`);
  const name = typeof input.name === "string" && input.name.trim() ? input.name.trim() : input.id;
  const transform = cleanTransform(input.transform);
  return {
    id: input.id,
    name,
    type: input.type,
    parentId,
    children: [],
    visible: input.visible !== false,
    selectable: input.selectable !== false,
    // The scene root and the active camera cannot be hidden. Hiding the
    // camera you are looking through would leave a blank viewport.
    hideable: input.hideable !== false,
    removable: input.removable !== false,
    transformable: input.transformable === true,
    transform,
    origin: { name, transform },
    meta: input.meta || null,
  };
}

// Build normalized state from a nested description: [{ id, name, type, children: [...] }]
export function buildState(tree) {
  const nodes = {};
  const rootIds = [];
  const walk = (list, parentId) =>
    list.map((input) => {
      if (nodes[input.id]) throw new Error(`duplicate node id: ${input.id}`);
      const node = makeNode(input, parentId);
      nodes[node.id] = node;
      node.children = walk(input.children || [], node.id);
      return node.id;
    });
  rootIds.push(...walk(tree, null));
  return { nodes, rootIds, selectedId: null };
}

// Depth-first list of rows in display order, with depth for indentation.
export function flatten(state) {
  const rows = [];
  const walk = (ids, depth) => {
    for (const id of ids) {
      const node = state.nodes[id];
      if (!node) continue;
      rows.push({ node, depth });
      walk(node.children, depth + 1);
    }
  };
  walk(state.rootIds, 0);
  return rows;
}

// A node is shown in the viewport only if it and all its parents are visible.
export function isEffectivelyVisible(state, id) {
  for (let n = state.nodes[id]; n; n = state.nodes[n.parentId]) {
    if (!n.visible) return false;
  }
  return Boolean(state.nodes[id]);
}

// True while a node is exactly as it was created: same name, place, and shown.
// Used to decide whether the starter cube can be removed without asking.
export function isUntouched(state, id) {
  const n = state.nodes[id];
  return Boolean(n) && n.visible && n.name === n.origin.name && sameTransform(n.transform, n.origin.transform);
}

// The top-level object a node belongs to (the node itself for top-level ones).
// Clicking any part of an imported model selects the whole model, like in
// Blender or Unity. The hierarchy still lets you pick a single part.
export function topLevelOf(state, id) {
  let n = state.nodes[id];
  while (n && n.parentId && state.nodes[n.parentId]?.type !== "scene") n = state.nodes[n.parentId];
  return n ? n.id : null;
}

// A unique display name: "Helmet", then "Helmet (2)", "Helmet (3)"...
export function uniqueName(state, base) {
  const taken = new Set(Object.values(state.nodes).map((n) => n.name));
  if (!taken.has(base)) return base;
  for (let i = 2; ; i++) if (!taken.has(`${base} (${i})`)) return `${base} (${i})`;
}

export function createSceneStore(tree) {
  let state = buildState(tree);
  let past = []; // older states, newest last
  let future = []; // undone states, next redo last
  let gestureStart = null; // state when a drag began, so a whole drag is one undo step
  const listeners = new Set();

  const notify = () => listeners.forEach((l) => l());

  // record: push the current state to history first, so this change can be undone.
  const commit = (next, record) => {
    if (next === state) return;
    if (record && !gestureStart) {
      past.push(state);
      if (past.length > HISTORY_LIMIT) past.shift();
      future = [];
    }
    state = next;
    notify();
  };

  const patchNode = (id, patch, record = true) => {
    const node = state.nodes[id];
    if (!node) return;
    const changed = Object.keys(patch).some((k) => node[k] !== patch[k]);
    if (!changed) return; // no-op updates do not re-render anything or fill history
    commit({ ...state, nodes: { ...state.nodes, [id]: { ...node, ...patch } } }, record);
  };

  // After undo or redo the selected node may no longer exist.
  const withValidSelection = (s, selectedId) => ({ ...s, selectedId: s.nodes[selectedId] ? selectedId : null });

  const api = {
    get: () => state,

    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    canUndo: () => past.length > 0,
    canRedo: () => future.length > 0,

    // Every node id that the current state or any history entry still uses.
    // The viewport frees GPU memory only for objects outside this set, so an
    // undo can always bring a deleted model back.
    referencedIds() {
      const ids = new Set();
      for (const s of [state, ...past, ...future]) for (const id of Object.keys(s.nodes)) ids.add(id);
      return ids;
    },

    // Selection is not an undo step, same as every major 3D tool.
    // null clears it. Unknown or unselectable ids are ignored, so a stale
    // click (an object removed a moment ago) can never select garbage.
    select(id) {
      if (id === null || id === undefined) {
        if (state.selectedId !== null) commit({ ...state, selectedId: null }, false);
        return;
      }
      const node = state.nodes[id];
      if (!node || !node.selectable || state.selectedId === id) return;
      commit({ ...state, selectedId: id }, false);
    },

    // Move the selection up or down the visible list (arrow keys in the tree).
    selectRelative(delta) {
      const rows = flatten(state).filter((r) => r.node.selectable);
      if (!rows.length) return;
      const i = rows.findIndex((r) => r.node.id === state.selectedId);
      const next = i === -1 ? (delta > 0 ? 0 : rows.length - 1) : Math.min(rows.length - 1, Math.max(0, i + delta));
      api.select(rows[next].node.id);
    },

    // record: false only for app-driven setup (startup defaults), never for
    // anything the user did.
    setVisible(id, visible, { record = true } = {}) {
      const node = state.nodes[id];
      if (!node || !node.hideable) return;
      patchNode(id, { visible: Boolean(visible) }, record);
    },

    toggleVisible(id) {
      const node = state.nodes[id];
      if (node) api.setVisible(id, !node.visible);
    },

    rename(id, name) {
      if (typeof name !== "string" || !name.trim()) return;
      patchNode(id, { name: name.trim() });
    },

    // record: false is for live updates during a drag. Wrap the drag in
    // beginGesture/endGesture so the whole drag becomes one undo step.
    setTransform(id, transform, { record = true } = {}) {
      const node = state.nodes[id];
      if (!node || !node.transformable) return;
      const next = cleanTransform({ ...node.transform, ...transform }, node.transform);
      if (sameTransform(next, node.transform)) return;
      patchNode(id, { transform: next }, record);
    },

    beginGesture() {
      if (!gestureStart) gestureStart = state;
    },

    endGesture() {
      const start = gestureStart;
      gestureStart = null;
      if (!start || start === state) return;
      // Compare without selection: a drag that only changed selection is not an edit.
      if (start.nodes === state.nodes && start.rootIds === state.rootIds) return;
      past.push(start);
      if (past.length > HISTORY_LIMIT) past.shift();
      future = [];
      notify(); // canUndo changed
    },

    undo() {
      if (gestureStart || !past.length) return false;
      future.push(state);
      state = withValidSelection(past.pop(), state.selectedId);
      notify();
      return true;
    },

    redo() {
      if (gestureStart || !future.length) return false;
      past.push(state);
      state = withValidSelection(future.pop(), state.selectedId);
      notify();
      return true;
    },

    // Adds a node (and any nested children) under parentId. One undo step.
    addNode(input, parentId = null) {
      if (parentId !== null && !state.nodes[parentId]) throw new Error(`no parent: ${parentId}`);
      const fresh = buildState([input]);
      for (const id of Object.keys(fresh.nodes)) {
        if (state.nodes[id]) throw new Error(`duplicate node id: ${id}`);
      }
      fresh.nodes[input.id] = { ...fresh.nodes[input.id], parentId };
      const nodes = { ...state.nodes, ...fresh.nodes };
      let rootIds = state.rootIds;
      if (parentId === null) rootIds = [...rootIds, input.id];
      else nodes[parentId] = { ...nodes[parentId], children: [...nodes[parentId].children, input.id] };
      commit({ ...state, nodes, rootIds }, true);
    },

    // Removes a node and everything under it. One undo step. Protected nodes
    // (scene root, camera, grid, lights) cannot be removed.
    removeNode(id) {
      const node = state.nodes[id];
      if (!node || !node.removable) return false;
      const nodes = { ...state.nodes };
      const drop = (nid) => {
        nodes[nid]?.children.forEach(drop);
        delete nodes[nid];
      };
      drop(id);
      let rootIds = state.rootIds;
      if (node.parentId === null) rootIds = rootIds.filter((r) => r !== id);
      else nodes[node.parentId] = { ...nodes[node.parentId], children: nodes[node.parentId].children.filter((c) => c !== id) };
      const selectedId = nodes[state.selectedId] ? state.selectedId : null;
      commit({ ...state, nodes, rootIds, selectedId }, true);
      return true;
    },
  };
  return api;
}
