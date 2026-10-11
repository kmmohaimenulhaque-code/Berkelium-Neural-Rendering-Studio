import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Sparkles, Trash2, AlertCircle } from "lucide-react";

const CHAT_URL = "http://127.0.0.1:8000/chat";
const BACKEND_API_KEY = ""; // Backend only enforces when its config has a non-empty key.

export default function ChatPanel() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const scrollRef = useRef(null);
    const textareaRef = useRef(null);

    // Pin the scroll to the latest message. Standard chat UX.
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, loading]);

    // Auto-grow the textarea up to a cap.
    useEffect(() => {
        const ta = textareaRef.current;
        if (!ta) return;
        ta.style.height = "auto";
        ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
    }, [input]);

    const send = useCallback(async () => {
        const text = input.trim();
        if (!text || loading) return;

        setMessages((p) => [...p, { role: "user", content: text, ts: Date.now() }]);
        setInput("");
        setError(null);
        setLoading(true);

        try {
            const res = await fetch(CHAT_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json", "X-API-Key": BACKEND_API_KEY },
                body: JSON.stringify({ message: text }),
            });

            if (!res.ok) {
                // Map backend status codes to honest, specific messages. No generic "something broke".
                let detail = `Backend returned ${res.status}.`;
                if (res.status === 401) detail = "Backend rejected the API key.";
                else if (res.status === 404) detail = "Chat endpoint not found. The backend may be out of date or on a different URL.";
                else if (res.status === 413) detail = "Message too long.";
                else if (res.status === 429) detail = "Rate limit hit. Wait a minute and try again.";
                else if (res.status === 502) detail = "AI service unavailable. The Fireworks key may be missing or wrong.";
                setError(detail);
                return;
            }

            const data = await res.json();
            setMessages((p) => [...p, { role: "assistant", content: data.response, ts: Date.now() }]);
        } catch {
            setError("Could not reach the backend. Is it running at 127.0.0.1:8000?");
        } finally {
            setLoading(false);
        }
    }, [input, loading]);

    const onKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    };

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
                {messages.map((m, i) => <Message key={i} role={m.role} content={m.content} />)}
                {loading && <TypingIndicator />}
                {error && (
                    <div className="chat__error" role="alert">
                        <AlertCircle size={14} />
                        <span>{error}</span>
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

function Message({ role, content }) {
    return (
        <div className={`chat__msg chat__msg--${role}`}>
            <div className="chat__msg-label">{role === "user" ? "You" : "Assistant"}</div>
            <div className="chat__msg-body selectable">{renderContent(content)}</div>
        </div>
    );
}

// Fenced ``` code blocks render as monospace boxes. Everything else is a paragraph.
function renderContent(text) {
    const parts = text.split(/```(\w*)\n?/);
    if (parts.length === 1) return <p className="chat__paragraph">{text}</p>;
    const out = [];
    for (let i = 0; i < parts.length; i++) {
        if (i % 3 === 0) {
            if (parts[i]) out.push(<p key={`p-${i}`} className="chat__paragraph">{parts[i]}</p>);
        } else if (i % 3 === 2) {
            out.push(
                <pre key={`c-${i}`} className="chat__code">
                    <code>{parts[i].replace(/```$/, "")}</code>
                </pre>
            );
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