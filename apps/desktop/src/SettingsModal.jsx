// SettingsModal.jsx
// AI provider settings. Left: every provider with a status dot. Right: the
// fields for the selected one, plus Save and a free Test connection.
//
// Secrets never come back from the backend. A saved key shows as a masked
// placeholder, and we only send fields the user actually edited. That way
// changing the model can never overwrite a saved key with its mask.

import { useCallback, useEffect, useMemo, useState } from "react";
import { X, Eye, EyeOff, ExternalLink, Check, AlertCircle, Loader2 } from "lucide-react";
import { apiFetch, PROVIDERS_CHANGED } from "./api.js";

const GROUPS = [
  { kind: "local", title: "Local (free)" },
  { kind: "cloud", title: "Cloud (your own key)" },
  { kind: "custom", title: "Custom" },
];

const FIELD_LABELS = {
  api_key: "API key",
  base_url: "Server URL",
  label: "Display name",
  aws_access_key_id: "AWS access key ID",
  aws_secret_access_key: "AWS secret access key",
  aws_region_name: "AWS region",
};

const SECRET_FIELDS = new Set(["api_key", "aws_access_key_id", "aws_secret_access_key"]);
const MAX_CHIPS = 40;

// Turn backend detail codes into sentences a user can act on.
function testMessage(provider, result) {
  const name = provider.label;
  switch (result.detail) {
    case "ok":
      return result.models.length
        ? `Connected. Found ${result.models.length} model${result.models.length === 1 ? "" : "s"}.`
        : "Connected. Your key works.";
    case "not_running":
      return `${name} is not running. Open the ${name} app and try again.`;
    case "key_rejected":
      return `${name} rejected this key. Check that you copied all of it.`;
    case "not_configured":
      return "Fill in the fields above and save first.";
    case "unreachable":
      return `Could not reach ${name}. Check your internet connection or the URL.`;
    case "no_free_check":
      return `${name} has no free key check. Your key will be verified on your first chat message.`;
    default:
      return `Test failed (${result.detail}).`;
  }
}

function dotClass(provider, lastTest) {
  if (lastTest?.ok === true) return "settings__dot--ok";
  if (lastTest?.ok === false) return "settings__dot--err";
  if (provider.configured) return "settings__dot--set";
  return "";
}

export default function SettingsModal({ onClose }) {
  const [providers, setProviders] = useState([]);
  const [loadError, setLoadError] = useState(null);
  const [selectedId, setSelectedId] = useState("ollama");
  const [edits, setEdits] = useState({}); // only fields the user changed
  const [reveal, setReveal] = useState({});
  const [busy, setBusy] = useState(null); // "save" | "test" | null
  const [tests, setTests] = useState({}); // providerId -> last test result
  const [notice, setNotice] = useState(null); // { kind: "ok" | "err", text }

  const load = useCallback(async () => {
    try {
      const data = await apiFetch("/providers");
      setProviders(data.providers);
      setLoadError(null);
    } catch (e) {
      setLoadError(e.message);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const selected = useMemo(
    () => providers.find((p) => p.id === selectedId),
    [providers, selectedId]
  );
  const dirty = Object.keys(edits).length > 0;

  const confirmDiscard = useCallback(
    () => !dirty || window.confirm("You have unsaved changes. Discard them?"),
    [dirty]
  );

  const close = useCallback(() => {
    if (confirmDiscard()) onClose();
  }, [confirmDiscard, onClose]);

  // Esc closes, like every other dialog on macOS.
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") close(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [close]);

  const pick = (id) => {
    if (id === selectedId || !confirmDiscard()) return;
    setSelectedId(id);
    setEdits({});
    setReveal({});
    setNotice(null);
  };

  const setField = (field, value) => setEdits((e) => ({ ...e, [field]: value }));

  const save = async () => {
    if (!selected || !dirty) return true;
    setBusy("save");
    setNotice(null);
    try {
      const updated = await apiFetch(`/providers/${selected.id}`, { method: "PUT", body: edits });
      setProviders((list) => list.map((p) => (p.id === updated.id ? updated : p)));
      // A changed key or URL makes the old test result stale. A changed
      // model alone does not, so the green dot survives picking a model.
      if (Object.keys(edits).some((k) => k !== "model")) {
        setTests((t) => ({ ...t, [selected.id]: undefined }));
      }
      setEdits({});
      setReveal({});
      setNotice({ kind: "ok", text: "Saved." });
      window.dispatchEvent(new Event(PROVIDERS_CHANGED));
      return true;
    } catch (e) {
      setNotice({ kind: "err", text: e.message });
      return false;
    } finally {
      setBusy(null);
    }
  };

  const test = async () => {
    if (!selected) return;
    // Test what the user is looking at. Unsaved edits get saved first.
    if (dirty && !(await save())) return;
    setBusy("test");
    setNotice(null);
    try {
      const result = await apiFetch(`/providers/${selected.id}/test`, { method: "POST" });
      setTests((t) => ({ ...t, [selected.id]: result }));
      setNotice({
        kind: result.ok === false ? "err" : "ok",
        text: testMessage(selected, result),
      });
    } catch (e) {
      setNotice({ kind: "err", text: e.message });
    } finally {
      setBusy(null);
    }
  };

  const clearSecret = async (field) => {
    if (!window.confirm(`Remove the saved ${FIELD_LABELS[field]}?`)) return;
    setBusy("save");
    try {
      const updated = await apiFetch(`/providers/${selected.id}`, { method: "PUT", body: { [field]: "" } });
      setProviders((list) => list.map((p) => (p.id === updated.id ? updated : p)));
      setTests((t) => ({ ...t, [selected.id]: undefined }));
      setNotice({ kind: "ok", text: "Removed." });
      window.dispatchEvent(new Event(PROVIDERS_CHANGED));
    } catch (e) {
      setNotice({ kind: "err", text: e.message });
    } finally {
      setBusy(null);
    }
  };

  const models = tests[selectedId]?.models || [];
  const modelValue = edits.model ?? selected?.model ?? "";

  return (
    <div className="settings__overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) close(); }}>
      <div className="settings" role="dialog" aria-modal="true" aria-label="Settings">
        <div className="settings__head">
          <div className="settings__title">
            Settings <span className="settings__subtitle">AI providers</span>
          </div>
          <button className="panel__btn" onClick={close} aria-label="Close settings" title="Close (Esc)">
            <X size={14} />
          </button>
        </div>

        {loadError ? (
          <div className="settings__load-error">
            <AlertCircle size={16} />
            <div>
              <div>{loadError}</div>
              <button className="settings__btn settings__btn--secondary" onClick={load}>Retry</button>
            </div>
          </div>
        ) : (
          <div className="settings__body">
            <nav className="settings__list" aria-label="Providers">
              {GROUPS.map((g) => (
                <div key={g.kind}>
                  <div className="settings__group">{g.title}</div>
                  {providers.filter((p) => p.kind === g.kind).map((p) => (
                    <button
                      key={p.id}
                      className={`settings__item ${p.id === selectedId ? "settings__item--active" : ""}`}
                      onClick={() => pick(p.id)}
                    >
                      <span className={`settings__dot ${dotClass(p, tests[p.id])}`} />
                      <span className="settings__item-name">{p.label}</span>
                    </button>
                  ))}
                </div>
              ))}
            </nav>

            {selected && (
              <div className="settings__detail">
                <h2 className="settings__h2">
                  {selected.label}
                  <span className="settings__tag">{selected.kind === "local" ? "Local" : selected.kind === "cloud" ? "Cloud" : "Custom"}</span>
                </h2>
                <div className="settings__muted">
                  {selected.kind === "local" ? "Runs models on this computer. No API key needed." : "Uses your own account and key."}
                  {selected.docs_url && (
                    <>
                      {" "}
                      <a href={selected.docs_url} target="_blank" rel="noreferrer" className="settings__link">
                        {selected.kind === "local" ? "Website" : "Get a key"} <ExternalLink size={10} />
                      </a>
                    </>
                  )}
                </div>

                {selected.fields.map((f) => {
                  const secret = SECRET_FIELDS.has(f);
                  const saved = selected.values[f] || "";
                  const value = edits[f] ?? (secret ? "" : saved);
                  return (
                    <div className="settings__field" key={f}>
                      <label className="settings__label" htmlFor={`f-${f}`}>{FIELD_LABELS[f] || f}</label>
                      <div className="settings__row">
                        <input
                          id={`f-${f}`}
                          className="settings__input selectable"
                          type={secret && !reveal[f] ? "password" : "text"}
                          value={value}
                          placeholder={secret ? (saved ? `${saved}  (saved)` : "Paste your key") : ""}
                          onChange={(e) => setField(f, e.target.value)}
                          autoComplete="off"
                          spellCheck={false}
                        />
                        {secret && (
                          <button
                            className="settings__btn settings__btn--ghost"
                            onClick={() => setReveal((r) => ({ ...r, [f]: !r[f] }))}
                            aria-label={reveal[f] ? "Hide" : "Show"}
                            title={reveal[f] ? "Hide" : "Show"}
                          >
                            {reveal[f] ? <EyeOff size={12} /> : <Eye size={12} />}
                          </button>
                        )}
                        {secret && saved && edits[f] === undefined && (
                          <button className="settings__btn settings__btn--ghost" onClick={() => clearSecret(f)} disabled={!!busy}>
                            Remove
                          </button>
                        )}
                      </div>
                      {f === "base_url" && selected.kind === "local" && (
                        <div className="settings__hint">Leave as is unless you changed the app's port.</div>
                      )}
                    </div>
                  );
                })}

                <div className="settings__field">
                  <label className="settings__label" htmlFor="f-model">Default model</label>
                  <div className="settings__row">
                    <input
                      id="f-model"
                      className="settings__input selectable"
                      value={modelValue}
                      placeholder="Type a model name, or click Test to list them"
                      onChange={(e) => setField("model", e.target.value)}
                      spellCheck={false}
                    />
                  </div>
                  {models.length > 0 && (
                    <>
                      <div className="settings__hint">Click a model to use it.</div>
                      <div className="settings__chips">
                        {models.slice(0, MAX_CHIPS).map((m) => (
                          <button
                            key={m}
                            className={`settings__chip ${m === modelValue ? "settings__chip--active" : ""}`}
                            onClick={() => setField("model", m)}
                          >
                            {m}
                          </button>
                        ))}
                        {models.length > MAX_CHIPS && (
                          <span className="settings__hint">+{models.length - MAX_CHIPS} more. Type the name above.</span>
                        )}
                      </div>
                    </>
                  )}
                </div>

                <div className="settings__actions">
                  <button className="settings__btn settings__btn--primary" onClick={save} disabled={!dirty || !!busy}>
                    {busy === "save" ? <Loader2 size={12} className="settings__spin" /> : null} Save
                  </button>
                  <button className="settings__btn settings__btn--secondary" onClick={test} disabled={!!busy}>
                    {busy === "test" ? <Loader2 size={12} className="settings__spin" /> : null} Test connection
                  </button>
                  {dirty && <span className="settings__hint">Unsaved changes</span>}
                </div>

                {notice && (
                  <div className={`settings__notice settings__notice--${notice.kind}`} role="status">
                    {notice.kind === "ok" ? <Check size={14} /> : <AlertCircle size={14} />}
                    <span>{notice.text}</span>
                  </div>
                )}

                <div className="settings__footnote">
                  Keys are stored only in config.json on this computer and are never shown again after saving.
                  Test is free. It only asks the provider for its model list.
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
