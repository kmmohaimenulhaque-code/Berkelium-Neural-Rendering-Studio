// CodePanel.jsx
// The Code workspace. Monaco editor plus a thin toolbar that drives
// file open and save through the Electron preload bridge.
//
// State we care about:
//   filePath   full path on disk, or null for an unsaved buffer
//   fileName   display name (basename), or "untitled.py" for a new buffer
//   content    current editor text
//   savedContent  the text that matches what's on disk (for dirty detection)
//   language   Monaco language id, derived from the file extension
//
// A buffer is dirty when content !== savedContent. New buffers are
// dirty the moment the user types anything, which triggers Save-As
// on first save.

import { useCallback, useEffect, useRef, useState } from "react";
import "./monacoSetup.js"; // must run before the first <Editor> mounts
import Editor from "@monaco-editor/react";
import { FilePlus, FolderOpen, Save, SaveAll } from "lucide-react";
import { usePrefs } from "./prefs.js";

// Map common extensions to Monaco language ids. Anything missing falls
// back to plain text, which is better than crashing on an unknown lang.
function languageForFile(name) {
    if (!name) return "python";
    const ext = name.split(".").pop().toLowerCase();
    const map = {
        py: "python",
        js: "javascript",
        jsx: "javascript",
        ts: "typescript",
        tsx: "typescript",
        json: "json",
        md: "markdown",
        html: "html",
        css: "css",
        glsl: "cpp", // Monaco has no GLSL out of the box, cpp is the closest
        vert: "cpp",
        frag: "cpp",
        sh: "shell",
        yaml: "yaml",
        yml: "yaml",
    };
    return map[ext] || "plaintext";
}

const STARTER = `# Welcome to Berkelium Studio
# Press Cmd+O to open a file, Cmd+S to save, Cmd+Shift+S to save as.

def hello():
    print("Hello from Berkelium")

if __name__ == "__main__":
    hello()
`;

export default function CodePanel() {
    const [filePath, setFilePath] = useState(null);
    const [fileName, setFileName] = useState("untitled.py");
    const [content, setContent] = useState(STARTER);
    const [savedContent, setSavedContent] = useState(STARTER);
    const [status, setStatus] = useState(""); // transient status line
    const [error, setError] = useState("");
    const editorRef = useRef(null);
    const { editor: prefs } = usePrefs();

    const dirty = content !== savedContent;
    const language = languageForFile(fileName);

    // Clear status messages after a few seconds so the toolbar does not
    // get stuck showing a stale "Saved" from ten minutes ago.
    useEffect(() => {
        if (!status) return;
        const t = setTimeout(() => setStatus(""), 2500);
        return () => clearTimeout(t);
    }, [status]);

    // Tab size belongs to the file model in Monaco, not the editor, so the
    // options prop below cannot change it. Re-apply it whenever the
    // preference changes or a different file is loaded into the model.
    useEffect(() => {
        editorRef.current?.getModel()?.updateOptions({ tabSize: prefs.tabSize, indentSize: prefs.tabSize });
    }, [prefs.tabSize, filePath, fileName]);

    const handleNew = useCallback(() => {
        // Confirm if there are unsaved changes. A real IDE would offer
        // save-first here, but for the demo scope a simple confirm is fine.
        if (dirty && !window.confirm("Discard unsaved changes?")) return;
        setFilePath(null);
        setFileName("untitled.py");
        setContent("");
        setSavedContent("");
        setError("");
        setStatus("New file");
    }, [dirty]);

    const handleOpen = useCallback(async () => {
        if (!window.electronAPI) {
            setError("File operations are only available in the desktop app.");
            return;
        }
        if (dirty && !window.confirm("Discard unsaved changes?")) return;

        const result = await window.electronAPI.openFile();
        if (!result) return; // user cancelled
        if (result.error) {
            setError(`Open failed: ${result.error}`);
            return;
        }
        setFilePath(result.path);
        setFileName(result.name);
        setContent(result.content);
        setSavedContent(result.content);
        setError("");
        setStatus(`Opened ${result.name}`);
    }, [dirty]);

    const handleSave = useCallback(async () => {
        if (!window.electronAPI) {
            setError("File operations are only available in the desktop app.");
            return;
        }
        // No path yet means this is a new buffer. Fall through to Save-As.
        if (!filePath) {
            const result = await window.electronAPI.saveFileAs(content, fileName);
            if (!result) return;
            if (result.error) {
                setError(`Save failed: ${result.error}`);
                return;
            }
            setFilePath(result.path);
            setFileName(result.name);
            setSavedContent(content);
            setError("");
            setStatus(`Saved ${result.name}`);
            return;
        }
        const result = await window.electronAPI.saveFile(filePath, content);
        if (!result.ok) {
            setError(`Save failed: ${result.error}`);
            return;
        }
        setSavedContent(content);
        setError("");
        setStatus(`Saved ${fileName}`);
    }, [content, filePath, fileName]);

    const handleSaveAs = useCallback(async () => {
        if (!window.electronAPI) {
            setError("File operations are only available in the desktop app.");
            return;
        }
        const result = await window.electronAPI.saveFileAs(content, fileName);
        if (!result) return;
        if (result.error) {
            setError(`Save failed: ${result.error}`);
            return;
        }
        setFilePath(result.path);
        setFileName(result.name);
        setSavedContent(content);
        setError("");
        setStatus(`Saved ${result.name}`);
    }, [content, fileName]);
    // Define a custom Monaco theme that matches our near-black app background,
    // so the editor does not look like a lighter grey slab in the middle.
    const handleBeforeMount = (monaco) => {
        monaco.editor.defineTheme("berkelium-dark", {
            base: "vs-dark",
            inherit: true,
            rules: [],
            colors: {
                "editor.background": "#08090A",
                "editor.lineHighlightBackground": "#0E0F11",
                "editorGutter.background": "#08090A",
                "editorLineNumber.foreground": "#4a4a4a",
                "editorLineNumber.activeForeground": "#F7F8F8",
                "editor.selectionBackground": "#1A1B1E",
                "editorCursor.foreground": "#00D4D4",
            },
        });
    };

    // Wire Monaco's keyboard shortcuts after the editor mounts.
    // Doing this through Monaco (not the window) means they only fire
    // when the editor is focused, which is the right behavior.
    const handleMount = (editor, monaco) => {
        editorRef.current = editor;
        editor.getModel()?.updateOptions({ tabSize: prefs.tabSize, indentSize: prefs.tabSize });
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => handleSave());
        editor.addCommand(
            monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.KeyS,
            () => handleSaveAs()
        );
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyO, () => handleOpen());
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyN, () => handleNew());
    };

    return (
        <div className="code">
            <div className="code__toolbar">
                <div className="code__actions">
                    <button className="code__btn" onClick={handleNew} title="New (Cmd+N)">
                        <FilePlus size={14} />
                        <span>New</span>
                    </button>
                    <button className="code__btn" onClick={handleOpen} title="Open (Cmd+O)">
                        <FolderOpen size={14} />
                        <span>Open</span>
                    </button>
                    <button className="code__btn" onClick={handleSave} title="Save (Cmd+S)">
                        <Save size={14} />
                        <span>Save</span>
                    </button>
                    <button className="code__btn" onClick={handleSaveAs} title="Save As (Cmd+Shift+S)">
                        <SaveAll size={14} />
                        <span>Save As</span>
                    </button>
                </div>
                <div className="code__filename">
                    {dirty && <span className="code__dot" title="Unsaved changes" />}
                    <span>{fileName}</span>
                    <span className="code__lang">{language}</span>
                </div>
            </div>

            <div className="code__editor">
                <Editor
                    height="100%"
                    language={language}
                    value={content}
                    onChange={(v) => setContent(v ?? "")}
                    beforeMount={handleBeforeMount}
                    onMount={handleMount}
                    theme="berkelium-dark"
                    options={{
                        fontSize: prefs.fontSize,
                        fontFamily: "'JetBrains Mono', 'SF Mono', Menlo, monospace",
                        minimap: { enabled: prefs.minimap },
                        lineNumbers: prefs.lineNumbers ? "on" : "off",
                        scrollBeyondLastLine: false,
                        smoothScrolling: true,
                        cursorBlinking: "smooth",
                        renderLineHighlight: "line",
                        tabSize: prefs.tabSize,
                        // Keep the user's tab size instead of guessing from the file.
                        detectIndentation: false,
                        wordWrap: prefs.wordWrap ? "on" : "off",
                    }}
                />
            </div>

            {(status || error) && (
                <div className={`code__status ${error ? "code__status--error" : ""}`}>
                    {error || status}
                </div>
            )}
        </div>
    );
}