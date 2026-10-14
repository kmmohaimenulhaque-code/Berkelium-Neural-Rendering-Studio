// store.js
// The app's one scene store, plus the React hook to read it.
// Living at module level means the selection survives switching between
// the Scene, Code and Combined workspaces, which unmount the viewport.

import { useSyncExternalStore } from "react";
import { createSceneStore } from "./sceneStore.js";
import { DEFAULT_SCENE } from "./defaultScene.js";
import { getPrefs, PREFS_CHANGED } from "../prefs.js";
import { collectGarbage } from "./assets.js";

export const sceneStore = createSceneStore(DEFAULT_SCENE);

// The Show grid setting is the default for the Grid row. Apply it once at
// startup, then only when the setting itself changes, so toggling the grid
// from the toolbar or the hierarchy is not undone by an unrelated save.
let lastShowGrid = getPrefs().viewport.showGrid;
sceneStore.setVisible("grid", lastShowGrid, { record: false }); // startup default, not an undo step
if (typeof window !== "undefined") {
  window.addEventListener(PREFS_CHANGED, () => {
    const next = getPrefs().viewport.showGrid;
    if (next !== lastShowGrid) {
      lastShowGrid = next;
      sceneStore.setVisible("grid", next);
    }
  });
}

// Returns the whole immutable state. Components derive what they need with
// useMemo, which keeps the snapshot stable and avoids render loops.
export function useSceneState() {
  return useSyncExternalStore(sceneStore.subscribe, sceneStore.get);
}

// Free GPU memory for imported models that no state in the undo history can
// bring back. Debounced: walking the history on every drag frame would be
// wasted work, and a second of delay costs nothing.
let gcTimer = null;
sceneStore.subscribe(() => {
  clearTimeout(gcTimer);
  gcTimer = setTimeout(() => collectGarbage(sceneStore.referencedIds()), 1500);
});
