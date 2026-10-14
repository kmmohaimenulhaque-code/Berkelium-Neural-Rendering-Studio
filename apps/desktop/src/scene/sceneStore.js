// sceneStore.js
// The single source of truth for what is in the scene and what is selected.
//
// The hierarchy panel (React) and the viewport (three.js) both read from this
// store and never talk to each other directly. That is what makes the scene
// scale: a properties panel, a model loader or an undo stack later plugs into
// the same store instead of being wired into every other component.
//
// State is immutable. Every change produces a new state object, which is what
// React's useSyncExternalStore needs to know when to re-render.
//
// Shape:
//   nodes:      { [id]: { id, name, type, parentId, children: [id], visible, selectable, hideable } }
//   rootIds:    [id]
//   selectedId: id | null

export const NODE_TYPES = ["scene", "camera", "mesh", "light", "helper", "group"];

function makeNode(input, parentId) {
  if (!input || typeof input.id !== "string" || !input.id) throw new Error("node needs a string id");
  if (!NODE_TYPES.includes(input.type)) throw new Error(`unknown node type: ${input.type}`);
  return {
    id: input.id,
    name: typeof input.name === "string" && input.name.trim() ? input.name.trim() : input.id,
    type: input.type,
    parentId,
    children: [],
    visible: input.visible !== false,
    selectable: input.selectable !== false,
    // The scene root and the active camera cannot be hidden. Hiding the
    // camera you are looking through would leave a blank viewport.
    hideable: input.hideable !== false,
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

export function createSceneStore(tree) {
  let state = buildState(tree);
  const listeners = new Set();

  const set = (next) => {
    if (next === state) return;
    state = next;
    listeners.forEach((l) => l());
  };

  const patchNode = (id, patch) => {
    const node = state.nodes[id];
    if (!node) return;
    const changed = Object.keys(patch).some((k) => node[k] !== patch[k]);
    if (!changed) return; // no-op updates do not re-render anything
    set({ ...state, nodes: { ...state.nodes, [id]: { ...node, ...patch } } });
  };

  const api = {
    get: () => state,

    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    // null clears the selection. Unknown or unselectable ids are ignored, so a
    // stale click (an object removed a moment ago) can never select garbage.
    select(id) {
      if (id === null || id === undefined) {
        if (state.selectedId !== null) set({ ...state, selectedId: null });
        return;
      }
      const node = state.nodes[id];
      if (!node || !node.selectable || state.selectedId === id) return;
      set({ ...state, selectedId: id });
    },

    // Move the selection up or down the visible list (arrow keys in the tree).
    selectRelative(delta) {
      const rows = flatten(state).filter((r) => r.node.selectable);
      if (!rows.length) return;
      const i = rows.findIndex((r) => r.node.id === state.selectedId);
      const next = i === -1 ? (delta > 0 ? 0 : rows.length - 1) : Math.min(rows.length - 1, Math.max(0, i + delta));
      api.select(rows[next].node.id);
    },

    setVisible(id, visible) {
      const node = state.nodes[id];
      if (!node || !node.hideable) return;
      patchNode(id, { visible: Boolean(visible) });
    },

    toggleVisible(id) {
      const node = state.nodes[id];
      if (node) api.setVisible(id, !node.visible);
    },

    rename(id, name) {
      if (typeof name !== "string" || !name.trim()) return;
      patchNode(id, { name: name.trim() });
    },

    // For loading models later. Adds a node (and any nested children) under parentId.
    addNode(input, parentId = null) {
      if (parentId !== null && !state.nodes[parentId]) throw new Error(`no parent: ${parentId}`);
      const fresh = buildState([input]);
      for (const id of Object.keys(fresh.nodes)) {
        if (state.nodes[id]) throw new Error(`duplicate node id: ${id}`);
      }
      fresh.nodes[input.id].parentId = parentId;
      const nodes = { ...state.nodes, ...fresh.nodes };
      let rootIds = state.rootIds;
      if (parentId === null) rootIds = [...rootIds, input.id];
      else nodes[parentId] = { ...nodes[parentId], children: [...nodes[parentId].children, input.id] };
      set({ ...state, nodes, rootIds });
    },

    // Removes a node and everything under it. Clears the selection if the
    // selected node was inside the removed branch.
    removeNode(id) {
      const node = state.nodes[id];
      if (!node) return;
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
      set({ ...state, nodes, rootIds, selectedId });
    },
  };
  return api;
}
