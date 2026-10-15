// PropertiesPanel.jsx
// Floating panel in the viewport: exact position, rotation (degrees) and
// scale of the selected object, editable by typing, plus file details for
// imported models. Every committed edit is one undo step.

import { useState } from "react";
import { sceneStore, useSceneState } from "./store.js";
import { formatBytes } from "./importer.js";

const DEG = 180 / Math.PI;
const AXES = ["x", "y", "z"];

// Shows the live value until focused, then edits a draft. Enter or leaving
// the field commits; Esc reverts. Anything that is not a number is ignored.
function NumberField({ value, digits, axis, label, onCommit }) {
  const [draft, setDraft] = useState(null);
  const shown = draft ?? value.toFixed(digits);
  const commit = () => {
    if (draft === null) return;
    const n = Number(draft.replace(",", "."));
    if (Number.isFinite(n) && n.toFixed(digits) !== value.toFixed(digits)) onCommit(n);
    setDraft(null);
  };
  return (
    <input
      className={`props__field props__field--${axis} selectable`}
      aria-label={label}
      value={shown}
      inputMode="decimal"
      spellCheck={false}
      onFocus={(e) => { setDraft(value.toFixed(digits)); requestAnimationFrame(() => e.target.select()); }}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        e.stopPropagation(); // typing here must never trigger viewport shortcuts
        if (e.key === "Enter") e.currentTarget.blur();
        if (e.key === "Escape") { setDraft(null); requestAnimationFrame(() => e.target.blur()); }
      }}
    />
  );
}

export default function PropertiesPanel() {
  const state = useSceneState();
  const node = state.selectedId ? state.nodes[state.selectedId] : null;
  if (!node || !node.transformable) return null;
  const t = node.transform;

  const set = (key, i, n) => {
    const next = [...t[key]];
    next[i] = key === "r" ? n / DEG : n;
    sceneStore.setTransform(node.id, { [key]: next });
  };

  const rows = [
    { key: "p", label: "Position", digits: 2, value: (i) => t.p[i] },
    { key: "r", label: "Rotation", digits: 1, value: (i) => t.r[i] * DEG, unit: "°" },
    { key: "s", label: "Scale", digits: 2, value: (i) => t.s[i] },
  ];
  // Only imported models carry file details. Added shapes have meta too
  // ({ shape }), which must not be read as a file.
  const meta = node.type === "model" ? node.meta : null;

  return (
    <div className="props" onPointerDown={(e) => e.stopPropagation()}>
      <div className="props__title">Transform</div>
      {rows.map((r) => (
        <div className="props__row" key={r.key}>
          <span className="props__label">{r.label}</span>
          {AXES.map((a, i) => (
            <NumberField
              key={`${node.id}-${r.key}-${a}`}
              axis={a}
              label={`${r.label} ${a.toUpperCase()}${r.unit ? " in degrees" : ""}`}
              digits={r.digits}
              value={r.value(i)}
              onCommit={(n) => set(r.key, i, n)}
            />
          ))}
        </div>
      ))}
      {meta && (
        <>
          <div className="props__title props__title--gap">Model</div>
          <dl className="props__meta">
            <dt>File</dt><dd title={meta.fileName}>{meta.fileName}</dd>
            <dt>Triangles</dt><dd>{meta.triangles.toLocaleString()}</dd>
            <dt>Meshes</dt><dd>{meta.meshes}</dd>
            <dt>Materials</dt><dd>{meta.materials}{meta.textures ? ` · ${meta.textures} textures` : ""}</dd>
            <dt>Size</dt><dd>{formatBytes(meta.bytes)}</dd>
            {meta.source && <><dt>Source</dt><dd title={meta.source}>{meta.source}</dd></>}
          </dl>
        </>
      )}
    </div>
  );
}
