// toast.js
// Small notifications at the bottom of the window, with an optional action
// button ("Undo"). Module-level so any part of the app can post one, and they
// survive switching workspaces.
//
// A toast with an action stays longer (15 s) and pauses while the mouse is
// over it, so nobody loses an Undo while reaching for it.

import { useSyncExternalStore } from "react";

let items = [];
let nextId = 1;
const timers = new Map(); // id -> { handle, remaining, startedAt }
const listeners = new Set();
const emit = () => listeners.forEach((l) => l());

export function dismissToast(id) {
  clearTimeout(timers.get(id)?.handle);
  timers.delete(id);
  const before = items.length;
  items = items.filter((t) => t.id !== id);
  if (items.length !== before) emit();
}

function startTimer(id, ms) {
  timers.set(id, { remaining: ms, startedAt: Date.now(), handle: setTimeout(() => dismissToast(id), ms) });
}

export function pauseToast(id) {
  const t = timers.get(id);
  if (!t || t.paused) return;
  clearTimeout(t.handle);
  t.remaining = Math.max(1500, t.remaining - (Date.now() - t.startedAt));
  t.paused = true;
}

export function resumeToast(id) {
  const t = timers.get(id);
  if (t?.paused) startTimer(id, t.remaining);
}

// kind: "ok" | "info" | "warn" | "error". action: { label, run }.
export function toast({ text, detail = null, kind = "info", action = null, timeout }) {
  const id = nextId++;
  const dropped = items.slice(0, Math.max(0, items.length - 3)); // at most 4 on screen
  dropped.forEach((t) => { clearTimeout(timers.get(t.id)?.handle); timers.delete(t.id); });
  items = [...items.slice(-3), { id, text, detail, kind, action }];
  emit();
  const ms = timeout ?? (action ? 15000 : 6000);
  if (ms) startTimer(id, ms);
  return id;
}

export function useToasts() {
  return useSyncExternalStore(
    (l) => { listeners.add(l); return () => listeners.delete(l); },
    () => items
  );
}
