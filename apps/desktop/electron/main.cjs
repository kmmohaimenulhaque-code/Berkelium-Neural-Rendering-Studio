// main.cjs
// Electron main process. Owns the BrowserWindow, the file dialogs,
// and the file system. The renderer talks to us through IPC channels
// exposed by preload.cjs.

const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const fs = require("fs/promises");
const path = require("path");

const isDev = process.env.NODE_ENV === "development";
app.setName("Berkelium");

let mainWindow = null;

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
    createWindow();

    app.on("activate", () => {
        // macOS convention: re-open a window when the dock icon is clicked
        // and no windows are open.
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

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