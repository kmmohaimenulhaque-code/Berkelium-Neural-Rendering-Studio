// Tests for backend.cjs. No real backend and no Electron: a fake child
// process and a fake fetch stand in for them, so every failure path
// (crash on start, crash later, never answering) can be forced on demand.
// Run: npm test

"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { EventEmitter } = require("events");
const { BackendManager, ensureConfig, sidecarPath, freePort } = require("./backend.cjs");

function tmpDir() {
    return fs.mkdtempSync(path.join(os.tmpdir(), "bk-backend-"));
}

// A stand-in for the real backend process.
function fakeChild() {
    const child = new EventEmitter();
    child.stderr = new EventEmitter();
    child.stdin = { ended: false, end() { this.ended = true; } };
    child.killed = false;
    child.kill = () => { child.killed = true; setImmediate(() => child.emit("exit", null, "SIGTERM")); };
    return child;
}

function setup({ healthy = () => true, timeoutMs = 2000 } = {}) {
    const dir = tmpDir();
    const exe = path.join(dir, "berkelium-backend");
    fs.writeFileSync(exe, "");
    const spawned = [];
    const states = [];
    const mgr = new BackendManager({
        exe,
        dataDir: dir,
        timeoutMs,
        spawnFn: (file, args, opts) => {
            const c = fakeChild();
            c.opts = opts;
            spawned.push(c);
            return c;
        },
        fetchFn: async () => {
            if (!healthy()) throw new Error("connection refused");
            return { ok: true };
        },
        onChange: (info) => states.push(info.state),
    });
    return { dir, mgr, spawned, states };
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

test("sidecar path per OS", () => {
    assert.equal(sidecarPath("/R", "darwin"), path.join("/R", "backend", "berkelium-backend"));
    assert.equal(sidecarPath("/R", "win32"), path.join("/R", "backend", "berkelium-backend.exe"));
});

test("freePort gives a usable port", async () => {
    const port = await freePort();
    assert.ok(port > 0 && port < 65536);
});

test("ensureConfig creates a private config with a strong key", () => {
    const file = path.join(tmpDir(), "nested", "config.json");
    const key = ensureConfig(file);
    const cfg = JSON.parse(fs.readFileSync(file, "utf-8"));
    assert.equal(cfg.backend_api_key, key);
    assert.match(key, /^[0-9a-f]{64}$/);
    assert.deepEqual(cfg.allowed_origins, ["null"]);
    if (process.platform !== "win32") {
        assert.equal(fs.statSync(file).mode & 0o777, 0o600); // only this user can read keys
    }
});

test("ensureConfig keeps the user's saved settings and key", () => {
    const file = path.join(tmpDir(), "config.json");
    const existing = {
        backend_api_key: "k".repeat(64),
        allowed_origins: ["http://localhost:5173"],
        providers: { groq: { api_key: "gsk_users_own" } },
    };
    fs.writeFileSync(file, JSON.stringify(existing));
    const key = ensureConfig(file);
    const cfg = JSON.parse(fs.readFileSync(file, "utf-8"));
    assert.equal(key, existing.backend_api_key);
    assert.deepEqual(cfg.providers, existing.providers);
    assert.deepEqual(cfg.allowed_origins, ["http://localhost:5173", "null"]);
});

test("ensureConfig replaces a weak or empty backend key", () => {
    const file = path.join(tmpDir(), "config.json");
    fs.writeFileSync(file, JSON.stringify({ backend_api_key: "" }));
    assert.match(ensureConfig(file), /^[0-9a-f]{64}$/);
});

test("ensureConfig sets a broken file aside instead of failing", () => {
    const dir = tmpDir();
    const file = path.join(dir, "config.json");
    fs.writeFileSync(file, "{ this is not json");
    const key = ensureConfig(file);
    assert.match(key, /^[0-9a-f]{64}$/);
    const kept = fs.readdirSync(dir).filter((n) => n.startsWith("config.json.broken-"));
    assert.equal(kept.length, 1); // the user's old keys can still be recovered by hand
});

test("start passes port, config, log dir and stdin watch to the backend", async () => {
    const { mgr, spawned, dir } = setup();
    const info = await mgr.start();
    assert.equal(info.state, "ready");
    const env = spawned[0].opts.env;
    assert.match(info.url, /^http:\/\/127\.0\.0\.1:\d+$/);
    assert.equal(env.BERKELIUM_PORT, info.url.split(":").pop());
    assert.equal(env.BERKELIUM_CONFIG, path.join(dir, "config.json"));
    assert.equal(env.BERKELIUM_LOG_DIR, path.join(dir, "logs"));
    assert.equal(env.BERKELIUM_WATCH_STDIN, "1");
    assert.equal(spawned[0].opts.windowsHide, true);
    assert.match(info.key, /^[0-9a-f]{64}$/);
    mgr.stop();
});

test("missing backend program fails clearly without spawning", async () => {
    const { mgr, spawned } = setup();
    mgr.exe = "/definitely/not/here";
    const info = await mgr.start();
    assert.equal(info.state, "failed");
    assert.match(info.error, /not found/);
    assert.equal(spawned.length, 0);
});

test("a backend that never answers ends in failed, and is killed", async () => {
    const { mgr, spawned } = setup({ healthy: () => false, timeoutMs: 600 });
    const info = await mgr.start();
    assert.equal(info.state, "failed");
    assert.match(info.error, /did not answer/);
    assert.equal(spawned[0].killed, true);
});

test("a crash during startup restarts and never reports a stale ready", async () => {
    let up = false;
    const { mgr, spawned, states } = setup({ healthy: () => up });
    const first = mgr.start();
    await wait(50);
    spawned[0].emit("exit", 1, null); // dies before it ever answered
    await first;
    assert.notEqual(mgr.state, "ready"); // the dead process must not count as ready
    up = true;
    await wait(1500); // restart backoff is 1 s
    assert.equal(spawned.length, 2);
    assert.equal(mgr.state, "ready");
    assert.ok(states.includes("starting"));
    mgr.stop();
});

test("crash after ready restarts, then gives up after two restarts", async () => {
    const { mgr, spawned, dir } = setup();
    await mgr.start();
    spawned[0].stderr.emit("data", Buffer.from("Traceback: boom\n"));
    spawned[0].emit("exit", 1, null);
    await wait(1300);
    assert.equal(mgr.state, "ready");
    spawned[1].emit("exit", 1, null);
    await wait(2300);
    assert.equal(mgr.state, "ready");
    spawned[2].emit("exit", 1, null);
    await wait(100);
    assert.equal(mgr.state, "failed");
    assert.equal(spawned.length, 3);
    const crashLog = fs.readFileSync(path.join(dir, "logs", "backend-crash.log"), "utf-8");
    assert.match(crashLog, /Traceback: boom/); // the reason is kept for a bug report
});

test("stop closes stdin, kills, and does not restart", async () => {
    const { mgr, spawned } = setup();
    await mgr.start();
    mgr.stop();
    assert.equal(spawned[0].stdin.ended, true);
    assert.equal(spawned[0].killed, true);
    await wait(1200);
    assert.equal(spawned.length, 1);
    assert.equal(mgr.state, "stopped");
});
