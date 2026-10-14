// SettingsSections.jsx
// The non-provider tabs of the Settings window: Editor, Viewport,
// Shortcuts and About. Editor and Viewport changes apply instantly and are
// saved on this computer (see prefs.js). There is no Save button because
// nothing here can break anything, and Reset brings back the defaults.

import { useCallback, useEffect, useState } from "react";
import { Check, Copy, RotateCcw } from "lucide-react";
import { BACKEND_URL, apiFetch } from "./api.js";
import { usePrefs, setPref, resetSection } from "./prefs.js";
import { SHORTCUTS, keyLabel } from "./shortcuts.js";

// ---------------------------------------------------------------- controls

function Row({ label, hint, children }) {
  return (
    <div className="settings__pref">
      <div className="settings__pref-text">
        <div className="settings__pref-label">{label}</div>
        {hint && <div className="settings__pref-hint">{hint}</div>}
      </div>
      <div className="settings__pref-control">{children}</div>
    </div>
  );
}

function Toggle({ on, onChange, label }) {
  return (
    <button
      className={`settings__toggle ${on ? "settings__toggle--on" : ""}`}
      role="switch"
      aria-checked={on}
      aria-label={label}
      onClick={() => onChange(!on)}
    >
      <span className="settings__toggle-knob" />
    </button>
  );
}

function Segmented({ value, options, onChange, label }) {
  return (
    <div className="settings__seg" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={o.value === value}
          className={`settings__seg-btn ${o.value === value ? "settings__seg-btn--on" : ""}`}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function SectionHead({ title, subtitle, onReset }) {
  return (
    <div className="settings__section-head">
      <div>
        <h2 className="settings__h2">{title}</h2>
        <div className="settings__muted">{subtitle}</div>
      </div>
      {onReset && (
        <button className="settings__btn settings__btn--ghost settings__reset" onClick={onReset}>
          <RotateCcw size={11} /> Reset to defaults
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- editor

export function EditorSection() {
  const { editor } = usePrefs();
  const set = (k) => (v) => setPref("editor", k, v);
  return (
    <div className="settings__detail">
      <SectionHead
        title="Editor"
        subtitle="How the code editor looks and behaves. Changes apply right away."
        onReset={() => resetSection("editor")}
      />
      <Row label="Font size" hint="10 to 24">
        <div className="settings__range-wrap">
          <input
            type="range"
            className="settings__range"
            min={10}
            max={24}
            step={1}
            value={editor.fontSize}
            onChange={(e) => set("fontSize")(Number(e.target.value))}
            aria-label="Font size"
          />
          <span className="settings__range-value">{editor.fontSize}px</span>
        </div>
      </Row>
      <Row label="Tab size" hint="Spaces inserted when you press Tab">
        <Segmented
          label="Tab size"
          value={editor.tabSize}
          onChange={set("tabSize")}
          options={[2, 4, 8].map((n) => ({ value: n, label: String(n) }))}
        />
      </Row>
      <Row label="Word wrap" hint="Wrap long lines instead of scrolling sideways">
        <Toggle label="Word wrap" on={editor.wordWrap} onChange={set("wordWrap")} />
      </Row>
      <Row label="Minimap" hint="Small overview of the file on the right edge">
        <Toggle label="Minimap" on={editor.minimap} onChange={set("minimap")} />
      </Row>
      <Row label="Line numbers">
        <Toggle label="Line numbers" on={editor.lineNumbers} onChange={set("lineNumbers")} />
      </Row>
    </div>
  );
}

// ---------------------------------------------------------------- viewport

export function ViewportSection() {
  const { viewport } = usePrefs();
  const set = (k) => (v) => setPref("viewport", k, v);
  return (
    <div className="settings__detail">
      <SectionHead
        title="Viewport"
        subtitle="The 3D scene view. Changes apply right away."
        onReset={() => resetSection("viewport")}
      />
      <Row label="Show grid" hint="Floor grid under the scene. The Grid button in the viewport still works too.">
        <Toggle label="Show grid" on={viewport.showGrid} onChange={set("showGrid")} />
      </Row>
      <Row label="Show stats" hint="Frames per second and triangle count in the viewport toolbar">
        <Toggle label="Show stats" on={viewport.showStats} onChange={set("showStats")} />
      </Row>
    </div>
  );
}

// ---------------------------------------------------------------- shortcuts

export function ShortcutsSection() {
  return (
    <div className="settings__detail">
      <SectionHead title="Keyboard shortcuts" subtitle="Everything you can do from the keyboard and mouse." />
      {SHORTCUTS.map((g) => (
        <div key={g.group} className="settings__shortcut-group">
          <div className="settings__group settings__group--flush">{g.group}</div>
          {g.items.map((s) => (
            <div key={s.action} className="settings__shortcut">
              <span>{s.action}</span>
              <span className="settings__keys">
                {s.keys.map((k) => (
                  <kbd key={k} className="settings__kbd">{keyLabel(k)}</kbd>
                ))}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- about

// Copy text, with a fallback for contexts where the Clipboard API is blocked.
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  }
}

export function AboutSection() {
  const [app, setApp] = useState(null); // desktop app info, or "browser"
  const [health, setHealth] = useState(null); // null = loading, false = offline
  const [providers, setProviders] = useState([]);
  const [copied, setCopied] = useState(false);

  const load = useCallback(async () => {
    setHealth(null);
    const bridge = window.electronAPI;
    if (bridge?.appInfo) {
      try {
        setApp(await bridge.appInfo());
      } catch {
        setApp("browser");
      }
    } else {
      setApp("browser");
    }
    try {
      setHealth(await apiFetch("/health"));
    } catch {
      setHealth(false);
    }
    try {
      const data = await apiFetch("/providers");
      setProviders(data.providers.filter((p) => p.configured).map((p) => p.label));
    } catch {
      setProviders([]);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const desktop = app && app !== "browser" ? app : null;
  const v = health?.versions || {};
  // Name the checks that are not ok, so "degraded" tells you where to look.
  const failing = health ? Object.entries(health.components || {}).filter(([, c]) => c.status !== "ok").map(([k]) => k) : [];
  const backendLabel = health === null ? "Checking..." : health === false ? "Offline" : failing.length ? `Online, check: ${failing.join(", ")}` : "Online";

  const rows = [
    ["Berkelium", desktop ? desktop.version : "Running in a browser, not the desktop app"],
    ["Electron", desktop?.electron || "n/a"],
    ["Chrome", desktop?.chrome || "n/a"],
    ["Node", desktop?.node || "n/a"],
    ["Platform", desktop ? `${desktop.platform} ${desktop.arch}` : navigator.platform || "unknown"],
    ["Backend", `${backendLabel} at ${BACKEND_URL}`],
    ["Python", v.python || "n/a"],
    ["FastAPI", v.fastapi || "n/a"],
    ["LiteLLM", v.litellm || "n/a"],
    ["AI providers set up", providers.length ? providers.join(", ") : "none"],
  ];

  // Diagnostics never include keys. Only versions, status and provider names.
  const copy = async () => {
    const text = ["Berkelium diagnostics", ...rows.map(([k, val]) => `${k}: ${val}`)].join("\n");
    if (await copyText(text)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="settings__detail">
      <SectionHead title="About" subtitle="Versions and status. Paste the diagnostics into a bug report." />
      <div className="settings__about-brand">
        <div className="logo-lockup logo-lockup--horizontal">
          <div className="logo-icon logo-icon--white" role="img" aria-label="Berkelium" />
          <span className="logo-wordmark logo-wordmark--white">BERKELIUM</span>
        </div>
        <span className="settings__muted">Neural Rendering Studio · Team Berkelium</span>
      </div>
      <dl className="settings__about">
        {rows.map(([k, val]) => (
          <div key={k} className="settings__about-row">
            <dt>{k}</dt>
            <dd className="selectable">
              {k === "Backend" && (
                <span className={`settings__dot ${health === null ? "" : health === false ? "settings__dot--err" : health.status === "ok" ? "settings__dot--ok" : "settings__dot--set"}`} />
              )}
              {val}
            </dd>
          </div>
        ))}
      </dl>
      <div className="settings__actions">
        <button className="settings__btn settings__btn--secondary" onClick={copy}>
          {copied ? <Check size={12} /> : <Copy size={12} />} {copied ? "Copied" : "Copy diagnostics"}
        </button>
        <button className="settings__btn settings__btn--ghost settings__btn--tall" onClick={load}>
          <RotateCcw size={11} /> Refresh
        </button>
      </div>
      <div className="settings__footnote">Diagnostics never include your API keys.</div>
    </div>
  );
}
