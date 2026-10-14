// preload.cjs
// Bridge between the renderer (React) and Electron main process.
// We expose only the specific IPC channels we need, nothing more.
// This is the whole point of contextIsolation: the renderer never
// gets raw access to Node or fs. It calls these functions, main
// handles them, and the result comes back.

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
    // Open file dialog, read the chosen file, return { path, content, name }.
    // Returns null if the user cancels.
    openFile: () => ipcRenderer.invoke("file:open"),

    // Save content to a known path (overwrite). Returns { ok, error? }.
    saveFile: (path, content) => ipcRenderer.invoke("file:save", { path, content }),

    // Save-as dialog, write content to chosen path, return { path, name } or null.
    saveFileAs: (content, suggestedName) =>
        ipcRenderer.invoke("file:saveAs", { content, suggestedName }),

    // Version and platform info for Settings > About.
    appInfo: () => ipcRenderer.invoke("app:info"),
});
