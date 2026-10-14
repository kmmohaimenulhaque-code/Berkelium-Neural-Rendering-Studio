// prefs.js
// App preferences (editor and viewport). Saved in localStorage on this
// computer. Every value is validated on load, so a hand-edited or old
// saved value can never crash the editor: anything invalid falls back to
// its default.

import { useEffect, useState } from "react";

const STORAGE_KEY = "berkelium.prefs.v1";
export const PREFS_CHANGED = "berkelium:prefs-changed";

export const DEFAULT_PREFS = {
  editor: {
    fontSize: 13,
    tabSize: 4,
    wordWrap: true,
    minimap: false,
    lineNumbers: true,
  },
  viewport: {
    showGrid: true,
    showStats: true,
  },
};

// One rule per field. A rule returns the clean value or undefined.
const RULES = {
  editor: {
    fontSize: (v) => (Number.isInteger(v) && v >= 10 && v <= 24 ? v : undefined),
    tabSize: (v) => ([2, 4, 8].includes(v) ? v : undefined),
    wordWrap: (v) => (typeof v === "boolean" ? v : undefined),
    minimap: (v) => (typeof v === "boolean" ? v : undefined),
    lineNumbers: (v) => (typeof v === "boolean" ? v : undefined),
  },
  viewport: {
    showGrid: (v) => (typeof v === "boolean" ? v : undefined),
    showStats: (v) => (typeof v === "boolean" ? v : undefined),
  },
};

export function sanitize(raw) {
  const out = {};
  for (const section of Object.keys(DEFAULT_PREFS)) {
    out[section] = {};
    for (const key of Object.keys(DEFAULT_PREFS[section])) {
      const clean = RULES[section][key](raw?.[section]?.[key]);
      out[section][key] = clean === undefined ? DEFAULT_PREFS[section][key] : clean;
    }
  }
  return out;
}

let current = null;

export function getPrefs() {
  if (current) return current;
  let raw = null;
  try {
    raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
  } catch {
    raw = null; // corrupt or blocked storage: use defaults
  }
  current = sanitize(raw);
  return current;
}

function commit(next) {
  current = sanitize(next);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(current));
  } catch {
    /* storage blocked: the change still applies until the app closes */
  }
  window.dispatchEvent(new Event(PREFS_CHANGED));
}

export function setPref(section, key, value) {
  const prev = getPrefs();
  commit({ ...prev, [section]: { ...prev[section], [key]: value } });
}

export function resetSection(section) {
  const prev = getPrefs();
  commit({ ...prev, [section]: { ...DEFAULT_PREFS[section] } });
}

// React hook: re-renders whenever any preference changes.
export function usePrefs() {
  const [prefs, setPrefs] = useState(getPrefs);
  useEffect(() => {
    const update = () => setPrefs(getPrefs());
    window.addEventListener(PREFS_CHANGED, update);
    return () => window.removeEventListener(PREFS_CHANGED, update);
  }, []);
  return prefs;
}
