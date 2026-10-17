// backend.cjs
// Starts, watches and stops the Python backend that ships inside the
// installed app, so AI chat works with no terminal and no Python install.
//
// Running from source (npm run studio) does not use this: you start uvicorn
// yourself on port 8000, exactly as before.
//
// Life of the backend in the installed app:
//   1. Make sure the user's config.json exists in the app data folder, with
//      a random private key that only this install knows.
//   2. Pick a free port, so a second app or another local server never clashes.
//   3. Start the backend with a pipe on its stdin. If Electron dies for any
//      reason, the OS closes that pipe and the backend exits on its own.
//   4. Poll /health until it answers, then report "ready".
//   5. If it crashes later, restart it a couple of times before giving up.

"use strict";

const crypto = require("crypto");
const fs = require("fs");
const net = require("net");
const path = require("path");
const { spawn } = require("child_process");

// First launch on macOS can be slow: Gatekeeper checks every file in the
// app once, and the backend is a few hundred files. Later launches take
// about a second. A generous limit avoids a false "failed" on that first run.
const STARTUP_TIMEOUT_MS = 60000;
const HEALTH_POLL_MS = 300;
const MAX_RESTARTS = 2;
const STDERR_TAIL_LINES = 40;

// The renderer is loaded from a file, and Chromium sends "Origin: null" for
// requests from file pages. The backend's CORS list must allow it. That is
// safe here because every route that matters also needs the private key.
const FILE_PAGE_ORIGIN = "null";

function sidecarPath(resourcesPath, platform = process.platform) {
    const exe = platform === "win32" ? "berkelium-backend.exe" : "berkelium-backend";
    return path.join(resourcesPath, "backend", exe);
}

// Creates or repairs the per-user config. Never overwrites settings the user
// saved (their provider keys live here). Returns the backend key.
function ensureConfig(configPath) {
    let config = {};
    let changed = false;
    if (fs.existsSync(configPath)) {
        try {
            config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
            if (!config || typeof config !== "object" || Array.isArray(config)) throw new Error("not an object");
        } catch {
            // A broken file must not stop the app from starting. Keep it
            // aside, so a user's saved keys can still be recovered by hand.
            fs.renameSync(configPath, `${configPath}.broken-${Date.now()}`);
            config = {};
        }
    } else {
        changed = true;
    }

    if (typeof config.backend_api_key !== "string" || config.backend_api_key.length < 32) {
        config.backend_api_key = crypto.randomBytes(32).toString("hex");
        changed = true;
    }
    const origins = Array.isArray(config.allowed_origins) ? config.allowed_origins.filter((o) => typeof o === "string") : [];
    if (!origins.includes(FILE_PAGE_ORIGIN)) {
        config.allowed_origins = [...origins, FILE_PAGE_ORIGIN];
        changed = true;
    }

    if (changed) {
        fs.mkdirSync(path.dirname(configPath), { recursive: true });
        // Write then rename, so a crash mid-write never leaves half a file.
        // 0o600: only this user can read it. It holds their provider keys.
        const tmp = `${configPath}.tmp`;
        fs.writeFileSync(tmp, JSON.stringify(config, null, 2), { mode: 0o600 });
        fs.renameSync(tmp, configPath);
    }
    return config.backend_api_key;
}

function freePort() {
    return new Promise((resolve, reject) => {
        const srv = net.createServer();
        srv.unref();
        srv.on("error", reject);
        srv.listen(0, "127.0.0.1", () => {
            const { port } = srv.address();
            srv.close(() => resolve(port));
        });
    });
}

class BackendManager {
    constructor({ exe, dataDir, spawnFn = spawn, fetchFn = globalThis.fetch, onChange = () => {}, timeoutMs = STARTUP_TIMEOUT_MS }) {
        this.exe = exe;
        this.configPath = path.join(dataDir, "config.json");
        this.logDir = path.join(dataDir, "logs");
        this.spawnFn = spawnFn;
        this.fetchFn = fetchFn;
        this.onChange = onChange;
        this.timeoutMs = timeoutMs;
        this.state = "stopped"; // stopped | starting | ready | failed
        this.url = null;
        this.key = null;
        this.error = null;
        this.child = null;
        this.restarts = 0;
        this.stopping = false;
        this.stderrTail = [];
    }

    info() {
        return { managed: true, url: this.url, key: this.key, state: this.state, error: this.error };
    }

    _set(state, error = null) {
        this.state = state;
        this.error = error;
        this.onChange(this.info());
    }

    async start() {
        this.stopping = false;
        this._set("starting");
        try {
            if (!fs.existsSync(this.exe)) throw new Error(`backend program not found at ${this.exe}`);
            this.key = ensureConfig(this.configPath);
            fs.mkdirSync(this.logDir, { recursive: true });
            const port = await freePort();
            this.url = `http://127.0.0.1:${port}`;
            this._spawn(port);
            const child = this.child;
            // Only the start that still owns the live process may say
            // "ready". A crash during startup hands over to a restart.
            // restarts is not reset here on purpose: a backend that keeps
            // crashing after start must end in "failed", not loop forever.
            if (await this._waitHealthy() && this.child === child) {
                this._set("ready");
            }
        } catch (err) {
            this._fail(err.message);
        }
        return this.info();
    }

    _spawn(port) {
        this.stderrTail = [];
        const child = this.spawnFn(this.exe, [], {
            env: {
                ...process.env,
                BERKELIUM_PORT: String(port),
                BERKELIUM_CONFIG: this.configPath,
                BERKELIUM_LOG_DIR: this.logDir,
                BERKELIUM_WATCH_STDIN: "1",
            },
            stdio: ["pipe", "ignore", "pipe"],
            windowsHide: true,
        });
        this.child = child;
        child.stderr?.on("data", (buf) => {
            for (const line of String(buf).split(/\r?\n/)) {
                if (!line) continue;
                this.stderrTail.push(line);
                if (this.stderrTail.length > STDERR_TAIL_LINES) this.stderrTail.shift();
            }
        });
        child.on("error", (err) => this._fail(`could not start the backend: ${err.message}`));
        child.on("exit", (code, signal) => this._onExit(child, code, signal));
    }

    async _waitHealthy() {
        const deadline = Date.now() + this.timeoutMs;
        const child = this.child;
        while (Date.now() < deadline) {
            if (this.child !== child || this.state !== "starting") return false; // exited, failed or stopped meanwhile
            try {
                const r = await this.fetchFn(`${this.url}/health`, { signal: AbortSignal.timeout(2000) });
                if (r.ok) return true;
            } catch {
                // not listening yet
            }
            await new Promise((res) => setTimeout(res, HEALTH_POLL_MS));
        }
        throw new Error(`backend did not answer within ${Math.round(this.timeoutMs / 1000)}s`);
    }

    _onExit(child, code, signal) {
        if (child !== this.child) return; // an old process we already replaced
        this.child = null;
        if (this.stopping) {
            this._set("stopped");
            return;
        }
        this._writeCrashLog(code, signal);
        if (this.restarts < MAX_RESTARTS) {
            this.restarts += 1;
            // A short pause, so a crash loop cannot eat the CPU.
            setTimeout(() => { if (!this.stopping) this.start(); }, 1000 * this.restarts);
            this._set("starting", `restarting after exit code ${code ?? signal}`);
        } else {
            this._fail(`backend stopped (exit code ${code ?? signal}). See logs/backend-crash.log`);
        }
    }

    _writeCrashLog(code, signal) {
        try {
            const lines = [`--- ${new Date().toISOString()} exit=${code ?? signal}`, ...this.stderrTail, ""];
            fs.appendFileSync(path.join(this.logDir, "backend-crash.log"), lines.join("\n"));
        } catch {
            // logging must never take the app down
        }
    }

    _fail(message) {
        if (this.stopping) return;
        if (this.child) {
            const c = this.child;
            this.child = null;
            try { c.kill(); } catch { /* already gone */ }
        }
        this._set("failed", message);
    }

    stop() {
        this.stopping = true;
        const c = this.child;
        if (!c) return;
        // Closing stdin is the clean signal; kill() is the backup for a
        // backend that is stuck and not reading it.
        try { c.stdin?.end(); } catch { /* ignore */ }
        try { c.kill(); } catch { /* ignore */ }
    }
}

module.exports = { BackendManager, ensureConfig, freePort, sidecarPath, FILE_PAGE_ORIGIN };
