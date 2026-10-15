// AddShapeMenu.jsx
// The "Add" button in the viewport toolbar and its dropdown of shapes.
// Opens by click or with Shift+A in the viewport (the Blender shortcut).
//
// Fully keyboard driven: arrows move, Enter adds, Esc closes. Focus goes back
// to the viewport afterwards, so W, E and Delete work on the new shape at once.

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Plus, ChevronDown } from "lucide-react";
import { SHAPES } from "./shapes.js";
import { SHAPE_ICONS } from "./shapeIcons.js";

export default function AddShapeMenu({ open, onOpenChange, onPick, onClose }) {
  const rootRef = useRef(null);
  const itemRefs = useRef([]);
  const [active, setActive] = useState(0);

  // Opening always starts at the top and puts focus in the menu, so arrow
  // keys work right away even when it was opened with Shift+A. A layout
  // effect focuses before the next paint: a key pressed right after Shift+A
  // must land in the menu, not in the viewport behind it.
  useLayoutEffect(() => {
    if (!open) return;
    setActive(0);
    itemRefs.current[0]?.focus();
  }, [open]);

  // A click anywhere else closes it, like every native menu.
  useEffect(() => {
    if (!open) return;
    const away = (e) => {
      if (!rootRef.current?.contains(e.target)) onOpenChange(false);
    };
    window.addEventListener("pointerdown", away, true);
    return () => window.removeEventListener("pointerdown", away, true);
  }, [open, onOpenChange]);

  const close = () => {
    onOpenChange(false);
    onClose?.();
  };

  const move = (i) => {
    const n = (i + SHAPES.length) % SHAPES.length;
    setActive(n);
    itemRefs.current[n]?.focus();
  };

  const onKeyDown = (e) => {
    // Keys typed in the menu must not reach the viewport (Delete, W, E...).
    e.stopPropagation();
    if (e.key === "ArrowDown") move(active + 1);
    else if (e.key === "ArrowUp") move(active - 1);
    else if (e.key === "Home") move(0);
    else if (e.key === "End") move(SHAPES.length - 1);
    else if (e.key === "Escape" || e.key === "Tab") close();
    else return;
    e.preventDefault();
  };

  return (
    <span className="addmenu" ref={rootRef}>
      <button
        className={`code__btn ${open ? "scene__btn--on" : ""}`}
        onClick={() => onOpenChange(!open)}
        aria-haspopup="menu"
        aria-expanded={open}
        title="Add a shape (Shift+A)"
      >
        <Plus size={14} />
        <span>Add</span>
        <ChevronDown size={12} className="addmenu__chevron" />
      </button>
      {open && (
        <div className="addmenu__list" role="menu" aria-label="Add a shape" onKeyDown={onKeyDown}>
          <div className="addmenu__head">
            Add shape <kbd>⇧A</kbd>
          </div>
          {SHAPES.map((s, i) => {
            const Icon = SHAPE_ICONS[s.kind];
            return (
              <button
                key={s.kind}
                ref={(el) => (itemRefs.current[i] = el)}
                role="menuitem"
                tabIndex={i === active ? 0 : -1}
                className={`addmenu__item ${i === active ? "is-active" : ""}`}
                // Hover moves focus too, so Enter always adds the highlighted shape.
                onMouseEnter={() => move(i)}
                onClick={() => {
                  onOpenChange(false);
                  onPick(s.kind);
                }}
              >
                <Icon size={14} />
                {s.label}
              </button>
            );
          })}
        </div>
      )}
    </span>
  );
}
