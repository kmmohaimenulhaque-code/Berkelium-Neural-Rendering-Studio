// SceneHierarchy.jsx
// The live scene tree in the left sidebar. Rows come from the scene store,
// so clicking a row selects the object in the viewport and clicking the
// object in the viewport highlights its row. Eye buttons hide and show.
//
// Keyboard: Up and Down move the selection, Esc clears it, H hides or shows,
// Delete removes, and undo/redo work the same as in the viewport.

import { useEffect, useMemo, useRef } from "react";
import { Box, Eye, EyeOff, Grid3x3, Layers, Sun, Video, Folder, CloudSun, Package } from "lucide-react";
import { flatten } from "./sceneStore.js";
import { sceneStore, useSceneState } from "./store.js";
import { handleSceneKey } from "./actions.js";

function iconFor(node) {
  if (node.type === "scene") return Layers;
  if (node.type === "camera") return Video;
  if (node.type === "mesh") return Box;
  if (node.type === "helper") return Grid3x3;
  if (node.type === "group") return Folder;
  if (node.type === "model") return Package;
  if (node.type === "light") return node.id === "sky" ? CloudSun : Sun;
  return Box;
}

export default function SceneHierarchy() {
  const state = useSceneState();
  const rows = useMemo(() => flatten(state), [state]);
  const rowRefs = useRef({});

  // When the selection comes from the viewport, bring its row into view.
  useEffect(() => {
    rowRefs.current[state.selectedId]?.scrollIntoView({ block: "nearest" });
  }, [state.selectedId]);

  const onKeyDown = (e) => {
    if (handleSceneKey(e)) {
      e.preventDefault();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      sceneStore.selectRelative(e.key === "ArrowDown" ? 1 : -1);
      e.preventDefault();
    } else if (e.key === "Escape") {
      sceneStore.select(null);
      e.preventDefault();
    } else if ((e.key === "h" || e.key === "H") && state.selectedId) {
      sceneStore.toggleVisible(state.selectedId);
      e.preventDefault();
    }
  };

  return (
    <div
      className="tree-section scene-tree"
      role="tree"
      aria-label="Scene hierarchy"
      tabIndex={0}
      onKeyDown={onKeyDown}
    >
      {rows.map(({ node, depth }) => {
        const Icon = iconFor(node);
        const selected = node.id === state.selectedId;
        return (
          <div
            key={node.id}
            ref={(el) => { rowRefs.current[node.id] = el; }}
            role="treeitem"
            aria-selected={selected}
            aria-level={depth + 1}
            className={[
              "tree-item",
              "scene-tree__row",
              selected && "tree-item--selected",
              !node.visible && "scene-tree__row--hidden",
              !node.selectable && "scene-tree__row--static",
            ].filter(Boolean).join(" ")}
            style={{ paddingLeft: 8 + depth * 16 }}
            onClick={() => node.selectable && sceneStore.select(node.id)}
          >
            <Icon size={13} />
            <span className="scene-tree__name">{node.name}</span>
            {node.hideable && (
              <button
                className="scene-tree__eye"
                onClick={(e) => {
                  e.stopPropagation(); // hiding should not also select
                  sceneStore.toggleVisible(node.id);
                }}
                aria-label={node.visible ? `Hide ${node.name}` : `Show ${node.name}`}
                title={node.visible ? "Hide (H)" : "Show (H)"}
              >
                {node.visible ? <Eye size={12} /> : <EyeOff size={12} />}
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
