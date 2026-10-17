// main.cjs
// Electron main process. Owns the BrowserWindow, the file dialogs,
// and the file system. The renderer talks to us through IPC channels
// exposed by preload.cjs.

const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const fs = require("fs/promises");
const path = require("path");
const { BackendManager, sidecarPath } = require("./backend.cjs");

// Dev mode = running from source (npm run studio), not an installed app.
// app.isPackaged works the same on macOS, Windows and Linux. The old
// NODE_ENV=development prefix in package.json only worked on macOS and Linux.
// --dist runs the built app (dist/) from source, exactly as an installed copy
// loads it. "npm run preview:app" uses it to check a release build without
// packaging. A flag instead of an environment variable works the same on
// macOS, Windows and Linux with no extra tools.
const isDev = !app.isPackaged && !process.argv.includes("--dist");
app.setName("Berkelium");

let mainWindow = null;
let backend = null;

// Running from source keeps the backend you start yourself (uvicorn on 8000).
// The installed app starts the one bundled inside it. BERKELIUM_SIDECAR lets
// a developer try a locally built backend with "npm run preview:app".
const DEV_BACKEND = { managed: false, url: "http://127.0.0.1:8000", key: "", state: "external", error: null };

function backendExe() {
    if (process.env.BERKELIUM_SIDECAR) return process.env.BERKELIUM_SIDECAR;
    if (app.isPackaged) return sidecarPath(process.resourcesPath);
    return null;
}

// One copy of the app at a time. A second copy would start a second backend
// writing to the same config.json; focusing the open window is what people
// expect anyway.
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
    app.quit();
} else {
    app.on("second-instance", () => {
        if (!mainWindow) return;
        if (mainWindow.isMinimized()) mainWindow.restore();
        mainWindow.focus();
    });
}

function createWindow() {
    mainWindow = new BrowserWindow({
        title: "Berkelium",
        width: 1400,
        height: 900,
        minWidth: 900,
        minHeight: 600,
        backgroundColor: "#08090A",
        webPreferences: {
            // Secure defaults. The renderer has no direct Node access.
            // Anything it needs from the OS goes through preload.cjs.
            contextIsolation: true,
            nodeIntegration: false,
            preload: path.join(__dirname, "preload.cjs"),
        },
    });

    // Links like "Get a key" in Settings should open in the user's real
    // browser, not a bare Electron window. Only https links are allowed out,
    // so a crafted link cannot launch file:// or custom URL schemes.
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        if (url.startsWith("https://")) shell.openExternal(url);
        return { action: "deny" };
    });

    if (isDev) {
        mainWindow.loadURL("http://localhost:5173");
        // Devtools are useful while building. Comment out for release builds.
        mainWindow.webContents.openDevTools({ mode: "detach" });
    } else {
        // Production path. Vite build output lives in dist/.
        mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
    }
}

app.whenReady().then(() => {
    if (!gotLock) return; // this copy is quitting; the first one stays in charge
    const exe = backendExe();
    if (exe) {
        backend = new BackendManager({
            exe,
            dataDir: app.getPath("userData"),
            // Push state changes to the window, so the status chip flips to
            // "online" the moment the backend is ready, not on the next poll.
            onChange: (info) => mainWindow?.webContents.send("backend:changed", info),
        });
        // Not awaited: the window opens at once and shows "Starting AI
        // engine..." while the backend boots.
        backend.start();
    }
    createWindow();

    app.on("activate", () => {
        // macOS convention: re-open a window when the dock icon is clicked
        // and no windows are open.
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

// Stop the backend on every way out (Cmd+Q, closing the last window on
// Windows and Linux, logout). If Electron crashes instead, the backend sees
// its stdin close and exits by itself.
app.on("will-quit", () => backend?.stop());

app.on("window-all-closed", () => {
    // Quit on non-mac when all windows are closed. On mac we keep the
    // app running until the user quits explicitly.
    if (process.platform !== "darwin") app.quit();
});

// File open: show dialog, read file, return the path and contents.
// We return null on cancel so the renderer can distinguish cancel
// from an actual error.
ipcMain.handle("file:open", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        title: "Open file",
        properties: ["openFile"],
        filters: [
            { name: "Python", extensions: ["py"] },
            { name: "Shaders", extensions: ["glsl", "vert", "frag"] },
            { name: "JavaScript", extensions: ["js", "jsx", "ts", "tsx"] },
            { name: "All files", extensions: ["*"] },
        ],
    });

    if (result.canceled || result.filePaths.length === 0) return null;

    const filePath = result.filePaths[0];
    try {
        const content = await fs.readFile(filePath, "utf-8");
        return { path: filePath, content, name: path.basename(filePath) };
    } catch (err) {
        // Surface a readable error back to the renderer so it can show
        // a toast rather than a blank screen.
        return { error: err.message };
    }
});

// Save to a known path. Used for Cmd+S when the file already has a path.
ipcMain.handle("file:save", async (_event, { path: filePath, content }) => {
    try {
        await fs.writeFile(filePath, content, "utf-8");
        return { ok: true };
    } catch (err) {
        return { ok: false, error: err.message };
    }
});

// Save-as: show dialog, write to the chosen path, return it.
// Used for new buffers and for Cmd+Shift+S.
ipcMain.handle("file:saveAs", async (_event, { content, suggestedName }) => {
    const result = await dialog.showSaveDialog(mainWindow, {
        title: "Save file as",
        defaultPath: suggestedName || "untitled.py",
        filters: [
            { name: "Python", extensions: ["py"] },
            { name: "Shaders", extensions: ["glsl", "vert", "frag"] },
            { name: "JavaScript", extensions: ["js", "jsx", "ts", "tsx"] },
            { name: "All files", extensions: ["*"] },
        ],
    });

    if (result.canceled || !result.filePath) return null;

    try {
        await fs.writeFile(result.filePath, content, "utf-8");
        return { path: result.filePath, name: path.basename(result.filePath) };
    } catch (err) {
        return { error: err.message };
    }
});
// App info for Settings > About. Read-only facts about this build, so it
// is safe to expose. Nothing here touches the file system or config.json.
ipcMain.handle("app:info", () => ({
    name: app.getName(),
    version: app.getVersion(),
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    node: process.versions.node,
    platform: process.platform,
    arch: process.arch,
}));

// Where the renderer should send API calls, and the private key it needs.
// The key is generated per install and only ever leaves the main process to
// reach our own renderer, never a web page.
ipcMain.handle("backend:info", () => (backend ? backend.info() : DEV_BACKEND));

// Settings > About: open the folder with backend.log so a user can send it
// to us when something goes wrong.
ipcMain.handle("backend:openLogs", async () => {
    const dir = path.join(app.getPath("userData"), "logs");
    await fs.mkdir(dir, { recursive: true });
    const err = await shell.openPath(dir);
    return err ? { ok: false, error: err } : { ok: true };
});
