import { useState, useEffect, useCallback, useRef } from "react";
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from "react-resizable-panels";
import {
  Box,
  Code2,
  Columns2,
  FolderTree,
  Layers,
  Terminal,
  Sparkles,
  Command,
  Search,
  Settings,
  PanelLeft,
  PanelRight,
  PanelBottom,
} from "lucide-react";

import "./shell.css";
import "./berkelium-logo.css";
import ChatPanel from "./ChatPanel.jsx";
import CodePanel from "./CodePanel.jsx";
import ScenePanel from "./ScenePanel.jsx";
import SceneHierarchy from "./scene/SceneHierarchy.jsx";
import { useSceneState } from "./scene/store.js";
import Toaster from "./scene/Toaster.jsx";
import { IMPORT_EVENT } from "./ScenePanel.jsx";
import SettingsModal from "./SettingsModal.jsx";
import { OPEN_SETTINGS } from "./api.js";
// Workspace definitions. Switching mode swaps the content of the center region.
// Scene = 3D viewport, Code = Monaco, Combined = both split vertically.
const WORKSPACES = [
  { id: "scene", label: "Scene", icon: Box },
  { id: "code", label: "Code", icon: Code2 },
  { id: "combined", label: "Combined", icon: Columns2 },
];

export default function App() {
  const [workspace, setWorkspace] = useState("combined");
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);
  const [bottomOpen, setBottomOpen] = useState(false);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [settingsOpen, setSettingsOpen] = useState(false);
  // The key handler below is registered once, so it reads the workspace
  // through a ref instead of a stale closure.
  const workspaceRef = useRef(workspace);
  workspaceRef.current = workspace;

  // The chat panel fires this when it needs the user to set up a provider.
  useEffect(() => {
    const open = () => setSettingsOpen(true);
    window.addEventListener(OPEN_SETTINGS, open);
    return () => window.removeEventListener(OPEN_SETTINGS, open);
  }, []);

  // Keyboard shortcuts: Cmd/Ctrl + B/J/. for the three toggles.
  // These match VS Code and Cursor conventions so devs feel at home.
  useEffect(() => {
    const onKey = (e) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      if (e.key === ",") {
        e.preventDefault();
        setSettingsOpen(true);
        return;
      }
      // Import model. Skipped inside the code editor, where Ctrl+I means
      // "trigger suggestions" on Windows and Linux.
      if (e.key.toLowerCase() === "i" && !e.shiftKey && !e.target.closest?.(".monaco-editor")) {
        e.preventDefault();
        if (workspaceRef.current !== "code") window.dispatchEvent(new Event(IMPORT_EVENT));
        return;
      }
      if (e.key === "b") {
        e.preventDefault();
        setLeftOpen((v) => !v);
      } else if (e.key === "j") {
        e.preventDefault();
        setBottomOpen((v) => !v);
      } else if (e.key === ".") {
        e.preventDefault();
        setRightOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Poll the backend /health endpoint. Honest UI: show real state, never fake green.
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const r = await fetch("http://127.0.0.1:8000/health");
        if (cancelled) return;
        setBackendStatus(r.ok ? "ok" : "err");
      } catch {
        if (cancelled) return;
        setBackendStatus("err");
      }
    };
    check();
    const id = setInterval(check, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="app">
      <TopBar
        workspace={workspace}
        onWorkspace={setWorkspace}
        leftOpen={leftOpen}
        rightOpen={rightOpen}
        bottomOpen={bottomOpen}
        onToggleLeft={() => setLeftOpen((v) => !v)}
        onToggleRight={() => setRightOpen((v) => !v)}
        onToggleBottom={() => setBottomOpen((v) => !v)}
        backendStatus={backendStatus}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <PanelGroup direction="vertical" autoSaveId="berkelium-vertical">
        <Panel defaultSize={bottomOpen ? 75 : 100} minSize={30}>
          <PanelGroup direction="horizontal" autoSaveId="berkelium-horizontal">
            {leftOpen && (
              <>
                <Panel defaultSize={18} minSize={12} maxSize={30}>
                  <LeftSidebar workspace={workspace} />
                </Panel>
                <PanelResizeHandle className="resize-handle" />
              </>
            )}

            <Panel defaultSize={60} minSize={30}>
              <Workspace mode={workspace} />
            </Panel>

            {rightOpen && (
              <>
                <PanelResizeHandle className="resize-handle" />
                <Panel defaultSize={22} minSize={16} maxSize={40}>
                  <RightPanel />
                </Panel>
              </>
            )}
          </PanelGroup>
        </Panel>

        {bottomOpen && (
          <>
            <PanelResizeHandle className="resize-handle" />
            <Panel defaultSize={25} minSize={10} maxSize={60}>
              <BottomPanel />
            </Panel>
          </>
        )}
      </PanelGroup>

      <StatusBar backendStatus={backendStatus} workspace={workspace} />

      {settingsOpen && <SettingsModal onClose={() => setSettingsOpen(false)} />}
      <Toaster />
    </div>
  );
}

function TopBar({
  workspace,
  onWorkspace,
  leftOpen,
  rightOpen,
  bottomOpen,
  onToggleLeft,
  onToggleRight,
  onToggleBottom,
  backendStatus,
  onOpenSettings,
}) {
  return (
    <header className="topbar" role="banner">
      <div className="topbar__brand">
        <div className="logo-lockup logo-lockup--horizontal">
          <div className="logo-icon logo-icon--white" role="img" aria-label="Berkelium" />
          <span className="logo-wordmark logo-wordmark--white">BERKELIUM</span>
        </div>
      </div>

      <nav className="topbar__workspaces" role="tablist" aria-label="Workspaces">
        {WORKSPACES.map((w) => {
          const Icon = w.icon;
          const active = workspace === w.id;
          return (
            <button
              key={w.id}
              className={`workspace-tab ${active ? "workspace-tab--active" : ""}`}
              role="tab"
              aria-selected={active}
              onClick={() => onWorkspace(w.id)}
            >
              <Icon size={13} strokeWidth={2} />
              {w.label}
            </button>
          );
        })}
      </nav>

      <div className="topbar__spacer" />

      <BackendChip status={backendStatus} />

      <div className="topbar__actions">
        <IconButton
          label="Command palette (Cmd+K)"
          onClick={() => { }}
          disabled
        >
          <Command size={14} />
        </IconButton>
        <IconButton label="Search" onClick={() => { }} disabled>
          <Search size={14} />
        </IconButton>
        <IconButton label="Settings (Cmd+,)" onClick={onOpenSettings}>
          <Settings size={14} />
        </IconButton>

        <div style={{ width: 1, height: 16, background: "var(--border)", margin: "0 4px" }} />

        <IconButton
          label="Toggle left panel (Cmd+B)"
          onClick={onToggleLeft}
          active={leftOpen}
        >
          <PanelLeft size={14} />
        </IconButton>
        <IconButton
          label="Toggle bottom panel (Cmd+J)"
          onClick={onToggleBottom}
          active={bottomOpen}
        >
          <PanelBottom size={14} />
        </IconButton>
        <IconButton
          label="Toggle AI panel (Cmd+.)"
          onClick={onToggleRight}
          active={rightOpen}
        >
          <PanelRight size={14} />
        </IconButton>
      </div>
    </header>
  );
}

function IconButton({ children, label, onClick, active, disabled }) {
  return (
    <button
      className={`topbar__btn ${active ? "topbar__btn--active" : ""}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      style={disabled ? { opacity: 0.4, cursor: "not-allowed" } : undefined}
    >
      {children}
    </button>
  );
}

function BackendChip({ status }) {
  const label =
    status === "ok" ? "Backend online"
      : status === "err" ? "Backend offline"
        : "Checking...";
  const dotClass =
    status === "ok" ? "status-chip__dot--ok"
      : status === "err" ? "status-chip__dot--err"
        : "";
  return (
    <div className="status-chip" title={label}>
      <span className={`status-chip__dot ${dotClass}`} />
      {label}
    </div>
  );
}

function LeftSidebar({ workspace }) {
  // Scene shows the hierarchy, Code shows the project files, and Combined
  // shows both, because the viewport is visible there too and clicking an
  // object should always have a row to highlight.
  const showProject = workspace !== "scene";
  const showScene = workspace !== "code";
  return (
    <div className="panel">
      {showProject && (
        <SidebarSection icon={FolderTree} title="PROJECT" grow={!showScene}>
          <ProjectTree />
        </SidebarSection>
      )}
      {showScene && (
        <SidebarSection icon={Layers} title="SCENE HIERARCHY" grow>
          <SceneHierarchy />
        </SidebarSection>
      )}
    </div>
  );
}

function SidebarSection({ icon: Icon, title, grow, children }) {
  return (
    <div className={`sidebar-section ${grow ? "sidebar-section--grow" : ""}`}>
      <div className="panel__header">
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Icon size={12} color="var(--text-secondary)" />
          <span className="panel__title">{title}</span>
        </div>
      </div>
      <div className="panel__body">{children}</div>
    </div>
  );
}

function ProjectTree() {
  return (
    <>
      <div className="tree-section">
        <div className="tree-item tree-item--selected">
          <FolderTree size={13} />
          berkelium-studio
        </div>
        <div className="tree-item tree-item--child">src/</div>
        <div className="tree-item tree-item--child">electron/</div>
        <div className="tree-item tree-item--child">backend/</div>
        <div className="tree-item tree-item--child">README.md</div>
      </div>
    </>
  );
}

function Workspace({ mode }) {
  if (mode === "scene") {
    return <ScenePanel />;
  }
  if (mode === "code") {
    return <CodePanel />;
  }
  // Combined mode: code | scene split
  return (
    <div className="workspace">
      <PanelGroup direction="horizontal" autoSaveId="berkelium-workspace">
        <Panel defaultSize={55} minSize={25}>
          <CodePanel />
        </Panel>
        <PanelResizeHandle className="resize-handle" />
        <Panel defaultSize={45} minSize={25}>
          <ScenePanel />
        </Panel>
      </PanelGroup>
    </div>
  );
}

function RightPanel() {
  return <ChatPanel />;
}

function BottomPanel() {
  return (
    <div className="panel">
      <div className="panel__header">
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Terminal size={12} color="var(--text-secondary)" />
          <span className="panel__title">CONSOLE</span>
        </div>
      </div>
      <div className="console selectable">
        <ConsoleLine time="12:00:00" level="info" msg="Berkelium Studio shell initialized." />
        <ConsoleLine time="12:00:00" level="ok" msg="Design tokens loaded." />
        <ConsoleLine time="12:00:01" level="info" msg="Checking backend at http://127.0.0.1:8000/health" />
      </div>
    </div>
  );
}

function ConsoleLine({ time, level, msg }) {
  return (
    <div className="console__line">
      <span className="console__time">{time}</span>
      <span className={`console__level console__level--${level}`}>{level}</span>
      <span className="console__msg">{msg}</span>
    </div>
  );
}

function StatusBar({ backendStatus, workspace }) {
  const scene = useSceneState();
  const selected = scene.selectedId ? scene.nodes[scene.selectedId] : null;
  return (
    <footer className="statusbar" role="contentinfo">
      <div className="statusbar__item">
        <span
          className={`status-chip__dot ${backendStatus === "ok" ? "status-chip__dot--ok" : "status-chip__dot--err"
            }`}
        />
        {backendStatus === "ok" ? "API connected" : "API disconnected"}
      </div>
      <div className="statusbar__item">
        Workspace: {WORKSPACES.find((w) => w.id === workspace)?.label}
      </div>
      {selected && <div className="statusbar__item">Selected: {selected.name}</div>}
      <div className="statusbar__spacer" />
      <div className="statusbar__item">Berkelium Studio v0.1</div>
    </footer>
  );
}