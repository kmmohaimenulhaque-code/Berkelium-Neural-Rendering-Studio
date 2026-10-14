// Toaster.jsx
// Renders the notifications posted with toast() at the bottom of the window.

import { Check, Info, AlertTriangle, AlertCircle, X } from "lucide-react";
import { useToasts, dismissToast, pauseToast, resumeToast } from "./toast.js";

const ICONS = { ok: Check, info: Info, warn: AlertTriangle, error: AlertCircle };

export default function Toaster() {
  const items = useToasts();
  if (!items.length) return null;
  return (
    <div className="toaster" role="status" aria-live="polite">
      {items.map((t) => {
        const Icon = ICONS[t.kind] || Info;
        return (
          <div
            key={t.id}
            className={`toast toast--${t.kind}`}
            onMouseEnter={() => pauseToast(t.id)}
            onMouseLeave={() => resumeToast(t.id)}
          >
            <Icon size={14} className="toast__icon" />
            <span className="toast__text">{t.text}</span>
            {t.detail && <span className="toast__detail">{t.detail}</span>}
            {t.action && (
              <button
                className="toast__action"
                onClick={() => { t.action.run(); dismissToast(t.id); }}
              >
                {t.action.label}
              </button>
            )}
            <button className="toast__close" onClick={() => dismissToast(t.id)} aria-label="Dismiss">
              <X size={12} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
