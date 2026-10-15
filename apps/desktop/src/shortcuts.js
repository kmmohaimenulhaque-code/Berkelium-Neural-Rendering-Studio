// shortcuts.js
// One list of every keyboard shortcut, shown in Settings > Shortcuts.
// "Mod" means Cmd on Mac and Ctrl on Windows and Linux.
// When you add a shortcut to the app, add it here too.

export const SHORTCUTS = [
  {
    group: "General",
    items: [
      { keys: ["Mod", ","], action: "Open Settings" },
      { keys: ["Mod", "B"], action: "Toggle left panel" },
      { keys: ["Mod", "J"], action: "Toggle bottom panel" },
      { keys: ["Mod", "."], action: "Toggle AI panel" },
      { keys: ["Mod", "I"], action: "Import a 3D model" },
      { keys: ["Esc"], action: "Close Settings" },
    ],
  },
  {
    group: "Code editor",
    items: [
      { keys: ["Mod", "N"], action: "New file" },
      { keys: ["Mod", "O"], action: "Open file" },
      { keys: ["Mod", "S"], action: "Save" },
      { keys: ["Mod", "Shift", "S"], action: "Save as" },
    ],
  },
  {
    group: "AI chat",
    items: [
      { keys: ["Enter"], action: "Send message" },
      { keys: ["Shift", "Enter"], action: "New line" },
    ],
  },
  {
    group: "3D viewport",
    items: [
      { keys: ["Click"], action: "Select an object" },
      { keys: ["Shift", "A"], action: "Add a shape" },
      { keys: ["W"], action: "Move tool" },
      { keys: ["E"], action: "Rotate tool" },
      { keys: ["Shift"], action: "Snap rotation to 15° while dragging" },
      { keys: ["Delete"], action: "Delete the selected object" },
      { keys: ["Mod", "Z"], action: "Undo" },
      { keys: ["Redo"], action: "Redo" },
      { keys: ["Esc"], action: "Clear selection" },
      { keys: ["F"], action: "Frame the selected object" },
      { keys: ["Drag"], action: "Orbit the camera" },
      { keys: ["Right drag"], action: "Pan" },
      { keys: ["Scroll"], action: "Zoom" },
    ],
  },
  {
    group: "Scene hierarchy",
    items: [
      { keys: ["↑"], action: "Select the row above" },
      { keys: ["↓"], action: "Select the row below" },
      { keys: ["H"], action: "Hide or show the selected object" },
      { keys: ["Delete"], action: "Delete the selected object" },
      { keys: ["Esc"], action: "Clear selection" },
    ],
  },
];

export const IS_MAC = typeof navigator !== "undefined" && /Mac/i.test(navigator.platform || navigator.userAgent);

export function keyLabel(k) {
  if (k === "Mod") return IS_MAC ? "⌘" : "Ctrl";
  if (k === "Redo") return IS_MAC ? "⇧⌘Z" : "Ctrl+Y";
  if (k === "Delete") return IS_MAC ? "⌫" : "Delete";
  if (k === "Shift") return IS_MAC ? "⇧" : "Shift";
  return k;
}
