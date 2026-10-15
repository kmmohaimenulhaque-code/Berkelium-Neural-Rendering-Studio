import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { Send, Sparkles, Trash2, AlertCircle, ChevronDown, Settings, Check, RotateCcw } from "lucide-react";
import { apiFetch, PROVIDERS_CHANGED, OPEN_SETTINGS } from "./api.js";

// The provider picked in the chat survives restarts. It is a convenience,
// so a blocked or empty localStorage just falls back to the default.
const PICK_KEY = "berkelium.chat.provider";
const HISTORY_TURNS = 20;

function readPick() {
    try { return localStorage.getItem(PICK_KEY); } catch { return null; }
}
function writePick(id) {
    try { localStorage.setItem(PICK_KEY, id); } catch { /* not critical */ }
}

const openSettings = () => window.dispatchEvent(new Event(OPEN_SETTINGS));

// Backend error codes become sentences that tell the user what to do next.
// "action" adds an Open Settings button when the fix lives there.
function describeError(err, provider) {
    const name = provider?.label || "The AI provider";
    switch (err.code) {
        case "backend_down":
            return { text: err.message };
        case "not_running":
            return { text: `${name} is not running. Open the ${name} app and try again.` };
        case "not_configured":
            return { text: `${name} is not set up yet.`, action: "settings" };
        case "key_rejected":
            return { text: `${name} rejected your API key. Update it in Settings.`, action: "settings" };
        case "no_model":
            return { text: `Pick a model for ${name} in Settings.`, action: "settings" };
        case "model_not_found":
            return { text: `${err.message}. Pick another model in Settings.`, action: "settings" };
        case "upstream_rate_limit":
            return { text: `${name} rate limit hit. Wait a minute and try again.` };
        case "unreachable":
            return { text: `Could not reach ${name}. Check your internet connection.` };
        default:
            break;
    }
    if (err.status === 401) return { text: "Backend rejected the API key." };
    if (err.status === 404) return { text: "Chat endpoint not found. The backend may be out of date." };
    if (err.status === 413 || err.status === 422) return { text: "Message is too long or not valid." };
    if (err.status === 429) return { text: "Too many messages. Wait a minute and try again." };
    return { text: err.message || "Something went wrong talking to the backend." };
}

export default function ChatPanel() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null); // { text, action?, retry? }
    const [providers, setProviders] = useState(null); // null = not loaded yet
    const [pickedId, setPickedId] = useState(readPick);
    const [menuOpen, setMenuOpen] = useState(false);
    const scrollRef = useRef(null);
    const textareaRef = useRef(null);
    const pickerRef = useRef(null);

    const loadProviders = useCallback(async () => {
        try {
            const data = await apiFetch("/providers");
            setProviders(data.providers);
        } catch {
            setProviders([]);
        }
    }, []);

    // Load once, then again whenever Settings saves something.
    useEffect(() => {
        loadProviders();
        window.addEventListener(PROVIDERS_CHANGED, loadProviders);
        return () => window.removeEventListener(PROVIDERS_CHANGED, loadProviders);
    }, [loadProviders]);

    const usable = useMemo(() => (providers || []).filter((p) => p.configured), [providers]);

    // Remembered pick if it is still usable, otherwise Fireworks (the original
    // default), otherwise the first usable one. Ollama is always usable because
    // it needs no key, so a fresh install still has something selected.
    const active = useMemo(() => {
        return usable.find((p) => p.id === pickedId)
            || usable.find((p) => p.id === "fireworks")
            || usable[0]
            || null;
    }, [usable, pickedId]);

    const choose = (id) => {
        setPickedId(id);
        writePick(id);
        setMenuOpen(false);
        setError(null);
    };

    // Close the menu on outside click or Esc.
    useEffect(() => {
        if (!menuOpen) return;
        const onDown = (e) => { if (pickerRef.current && !pickerRef.current.contains(e.target)) setMenuOpen(false); };
        const onKey = (e) => { if (e.key === "Escape") setMenuOpen(false); };
        window.addEventListener("mousedown", onDown);
        window.addEventListener("keydown", onKey);
        return () => {
            window.removeEventListener("mousedown", onDown);
            window.removeEventListener("keydown", onKey);
        };
    }, [menuOpen]);

    // Pin the scroll to the latest message. Standard chat UX.
    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [messages, loading, error]);

    // Auto-grow the textarea up to a cap.
    useEffect(() => {
        const ta = textareaRef.current;
        if (!ta) return;
        ta.style.height = "auto";
        ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
    }, [input]);

    // Sends `text` with everything before it as history. Used by both the
    // normal send and Retry, so a retry never duplicates the user message.
    const request = useCallback(async (text, prior) => {
        if (!active) {
            setError({ text: "No AI provider is set up yet.", action: "settings" });
            return;
        }
        setError(null);
        setLoading(true);
        const history = prior
            .filter((m) => m.role === "user" || m.role === "assistant")
            .slice(-HISTORY_TURNS)
            .map(({ role, content }) => ({ role, content }));
        try {
            const data = await apiFetch("/chat", {
                method: "POST",
                body: { message: text, provider: active.id, model: active.model || undefined, history },
            });
            setMessages((p) => [...p, {
                role: "assistant",
                content: data.response,
                via: `${active.label} · ${data.model}`,
                ts: Date.now(),
            }]);
        } catch (err) {
            setError({ ...describeError(err, active), retry: true });
        } finally {
            setLoading(false);
        }
    }, [active]);

    const send = useCallback(() => {
        const text = input.trim();
        if (!text || loading) return;
        const prior = messages;
        setMessages((p) => [...p, { role: "user", content: text, ts: Date.now() }]);
        setInput("");
        request(text, prior);
    }, [input, loading, messages, request]);

    const retry = useCallback(() => {
        const lastUser = [...messages].reverse().find((m) => m.role === "user");
        if (!lastUser || loading) return;
        const idx = messages.lastIndexOf(lastUser);
        request(lastUser.content, messages.slice(0, idx));
    }, [messages, loading, request]);

    const onKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    };

    const groups = [
        { kind: "local", title: "Local" },
        { kind: "cloud", title: "Cloud" },
        { kind: "custom", title: "Custom" },
    ];

    return (
        <div className="panel chat-panel">
            <div className="panel__header">
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <Sparkles size={12} color="var(--accent)" />
                    <span className="panel__title">AI COPILOT</span>
                </div>
                <div className="panel__header-actions">
                    <button
                        className="panel__btn"
                        onClick={() => { setMessages([]); setError(null); }}
                        aria-label="Clear conversation"
                        title="Clear conversation"
                        disabled={messages.length === 0}
                    >
                        <Trash2 size={12} />
                    </button>
                </div>
            </div>

            <div className="chat__messages" ref={scrollRef}>
                {messages.length === 0 && !loading && !error && (
                    <div className="chat__empty">
                        <Sparkles size={20} color="var(--accent)" strokeWidth={1.5} />
                        <div className="chat__empty-title">Ask the Berkelium assistant.</div>
                        <div className="chat__empty-hint">
                            Code help, 3D workflow questions, debugging. Enter to send, Shift+Enter for a new line.
                        </div>
                    </div>
                )}
                {messages.map((m, i) => <Message key={i} {...m} />)}
                {loading && <TypingIndicator />}
                {error && (
                    <div className="chat__error" role="alert">
                        <AlertCircle size={14} />
                        <div className="chat__error-body">
                            <span>{error.text}</span>
                            <div className="chat__error-actions">
                                {error.action === "settings" && (
                                    <button className="chat__error-btn" onClick={openSettings}>
                                        <Settings size={11} /> Open Settings
                                    </button>
                                )}
                                {error.retry && (
                                    <button className="chat__error-btn" onClick={retry}>
                                        <RotateCcw size={11} /> Retry
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <div className="chat__picker" ref={pickerRef}>
                <button
                    className="chat__picker-btn"
                    onClick={() => setMenuOpen((o) => !o)}
                    disabled={providers === null}
                    aria-haspopup="listbox"
                    aria-expanded={menuOpen}
                    title="Choose which AI answers"
                >
                    <span className={`chat__picker-dot ${active ? "chat__picker-dot--on" : ""}`} />
                    <span className="chat__picker-label">
                        {providers === null ? "Loading..." : active ? active.label : "No provider set up"}
                    </span>
                    {active?.model && <span className="chat__picker-model">{active.model}</span>}
                    <ChevronDown size={12} />
                </button>

                {menuOpen && (
                    <div className="chat__menu" role="listbox">
                        {groups.map((g) => {
                            const items = usable.filter((p) => p.kind === g.kind);
                            if (!items.length) return null;
                            return (
                                <div key={g.kind}>
                                    <div className="chat__menu-group">{g.title}</div>
                                    {items.map((p) => (
                                        <button
                                            key={p.id}
                                            className="chat__menu-item"
                                            role="option"
                                            aria-selected={p.id === active?.id}
                                            onClick={() => choose(p.id)}
                                        >
                                            <span className="chat__menu-check">{p.id === active?.id && <Check size={12} />}</span>
                                            <span className="chat__menu-name">{p.label}</span>
                                            <span className="chat__menu-model">{p.model || "no model set"}</span>
                                        </button>
                                    ))}
                                </div>
                            );
                        })}
                        <button className="chat__menu-item chat__menu-manage" onClick={() => { setMenuOpen(false); openSettings(); }}>
                            <span className="chat__menu-check"><Settings size={12} /></span>
                            <span className="chat__menu-name">Manage providers...</span>
                        </button>
                    </div>
                )}
            </div>

            <div className="chat__composer">
                <textarea
                    ref={textareaRef}
                    className="chat__input selectable"
                    placeholder={loading ? "Waiting for response..." : "Ask something..."}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={onKeyDown}
                    disabled={loading}
                    rows={1}
                    aria-label="Message input"
                />
                <button
                    className="chat__send"
                    onClick={send}
                    disabled={!input.trim() || loading}
                    aria-label="Send message"
                    title="Send (Enter)"
                >
                    <Send size={14} />
                </button>
            </div>
        </div>
    );
}

function Message({ role, content, via }) {
    return (
        <div className={`chat__msg chat__msg--${role}`}>
            <div className="chat__msg-label">
                {role === "user" ? "You" : "Assistant"}
                {via && <span className="chat__msg-via"> · {via}</span>}
            </div>
            <div className="chat__msg-body selectable">{renderContent(content)}</div>
        </div>
    );
}

// Fenced ``` code blocks render as monospace boxes. Everything else is a paragraph.
// split() with one capture group returns [segment, lang, segment, lang, ...].
// Segments sit at even indexes and alternate text, code, text, code.
function renderContent(text) {
    const parts = text.split(/```(\w*)\n?/);
    if (parts.length === 1) return <p className="chat__paragraph">{text}</p>;
    const out = [];
    for (let i = 0; i < parts.length; i += 2) {
        const segment = parts[i];
        const isCode = (i / 2) % 2 === 1;
        if (isCode) {
            out.push(
                <pre key={`c-${i}`} className="chat__code">
                    <code>{segment.replace(/\n$/, "")}</code>
                </pre>
            );
        } else if (segment.trim()) {
            out.push(<p key={`p-${i}`} className="chat__paragraph">{segment}</p>);
        }
    }
    return out;
}

function TypingIndicator() {
    return (
        <div className="chat__typing" aria-label="Assistant is typing">
            <span className="chat__typing-dot" />
            <span className="chat__typing-dot" />
            <span className="chat__typing-dot" />
        </div>
    );
}
