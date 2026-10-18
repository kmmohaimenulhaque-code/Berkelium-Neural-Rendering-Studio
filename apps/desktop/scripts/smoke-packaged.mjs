// smoke-packaged.mjs
// Smoke test for the COMPILED app, the one people actually install.
//
// electron/smoke.cjs tests the build from source. This script starts the
// packaged executable that electron-builder produced (Berkelium.app on macOS,
// Berkelium.exe on Windows, the Linux binary), so it also catches packaging
// mistakes: a file left out of the package, a wrong path inside app.asar, a
// missing icon or a crash on start. It talks to the app over the Chrome
// DevTools protocol, which every Electron app has built in, so it needs no
// extra test library.
//
// Run after "npm run package":   npm run smoke:packaged
// Linux without a screen:        xvfb-run -a npm run smoke:packaged
//
// Backend checks (the app ships its own Python backend):
//   always               the built-in backend starts and answers, with its
//                        private key, and refuses calls without it
//   SMOKE_EXPECT_BUILTIN=1  the free built-in provider is ready and preselected
//   SMOKE_REAL_CHAT=1       send ONE real chat message through the free
//                           provider (Groq free tier, costs nothing) to prove
//                           the shipped key works before anyone downloads it

import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const RELEASE = path.resolve("release");
const TIMEOUT_MS = 90000; // a cold CI runner can take a while to start a fresh app
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function findExecutable() {
  if (!existsSync(RELEASE)) throw new Error("no release/ folder. Run npm run package first.");
  const dirs = readdirSync(RELEASE);
  const mac = dirs.find((d) => d.startsWith("mac"));
  if (mac) return path.join(RELEASE, mac, "Berkelium.app", "Contents", "MacOS", "Berkelium");
  if (dirs.includes("win-unpacked")) return path.join(RELEASE, "win-unpacked", "Berkelium.exe");
  const linux = dirs.find((d) => d.startsWith("linux") && d.endsWith("unpacked"));
  if (linux) {
    // The Linux binary is named after package.json "name". Pick the file that
    // is not a known Chromium helper.
    const helpers = new Set(["chrome-sandbox", "chrome_crashpad_handler"]);
    const bin = readdirSync(path.join(RELEASE, linux)).find((f) => /^[a-z][a-z0-9-]*$/.test(f) && !helpers.has(f));
    if (bin) return path.join(RELEASE, linux, bin);
  }
  throw new Error(`no packaged app found in release/ (saw: ${dirs.join(", ")})`);
}

function fail(message, child) {
  console.error(`[smoke:packaged] FAIL: ${message}`);
  child?.kill();
  process.exit(1);
}

const exe = findExecutable();
if (!existsSync(exe)) fail(`executable missing: ${exe}`);
console.log(`[smoke:packaged] starting ${path.relative(process.cwd(), exe)}`);

const userDataDir = mkdtempSync(path.join(tmpdir(), "berkelium-smoke-"));
const args = [
  `--user-data-dir=${userDataDir}`, // a fresh install: no leftover settings, no developer keys
  "--remote-debugging-port=0", // 0 = any free port. Chromium prints which one.
  "--ignore-gpu-blocklist", // allow software WebGL on runners without a GPU
  "--enable-unsafe-swiftshader",
];
if (process.platform === "linux") args.push("--no-sandbox"); // Ubuntu 24.04+ blocks the sandbox for apps outside apt

const child = spawn(exe, args, { stdio: ["ignore", "pipe", "pipe"] });
const timer = setTimeout(() => fail(`timed out after ${TIMEOUT_MS / 1000}s`, child), TIMEOUT_MS);
child.on("exit", (code) => fail(`the app quit by itself (exit code ${code})`));

// Chromium announces the DevTools endpoint on stderr.
const wsBase = await new Promise((resolve) => {
  let buf = "";
  child.stderr.on("data", (d) => {
    buf += d.toString();
    const m = buf.match(/DevTools listening on (ws:\/\/[^\s]+)/);
    if (m) resolve(m[1]);
  });
});
const port = new URL(wsBase).port;

// Wait for the app window (a "page" target) to exist.
let page;
for (let i = 0; i < 120 && !page; i++) {
  const list = await fetch(`http://127.0.0.1:${port}/json/list`).then((r) => r.json()).catch(() => []);
  page = list.find((t) => t.type === "page" && t.url.includes("index.html"));
  if (!page) await sleep(500);
}
if (!page) fail("the app window never opened", child);

const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
let nextId = 1;
const pending = new Map();
const pageErrors = [];
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
  } else if (msg.method === "Runtime.exceptionThrown") {
    pageErrors.push(msg.params.exceptionDetails?.exception?.description || msg.params.exceptionDetails?.text);
  }
};
const send = (method, params = {}) => new Promise((resolve) => {
  const id = nextId++;
  pending.set(id, resolve);
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) => {
  const res = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  return res.result?.result?.value;
};
async function waitFor(label, expression, ms = 30000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    const value = await evaluate(expression).catch(() => null);
    if (value) return value;
    await sleep(250);
  }
  fail(`${label} never appeared${pageErrors.length ? `. Page error: ${pageErrors[0]}` : ""}`, child);
}

await send("Runtime.enable");

// Same checks as the source smoke test, now against the installed app.
// The window starts as about:blank for a moment before the app page loads,
// so wait for the real page instead of reading the first URL we see.
const loadedFrom = await waitFor("the app page", `location.href.includes("index.html") ? location.href : null`);
if (!loadedFrom.includes("app.asar")) fail(`expected the app to load from inside app.asar, got ${loadedFrom}`, child);
await waitFor("the app shell", `document.querySelectorAll(".workspace-tab").length >= 3`);
await waitFor("the BK logo", `[...document.querySelectorAll(".brand__mark")].every((i) => i.complete && i.naturalWidth > 0) && document.querySelector(".brand__mark") !== null`);
await evaluate(`[...document.querySelectorAll(".workspace-tab")].find((t) => t.textContent.includes("Scene"))?.click(); true`);
const viewport = await waitFor(
  "the 3D viewport",
  `document.querySelector(".scene__viewport canvas") ? "canvas" : document.querySelector(".scene__failure") ? "no-webgl" : null`
);
await waitFor("the starter cube in the hierarchy", `[...document.querySelectorAll(".scene-tree__name")].some((n) => n.textContent === "Cube")`);

// ---- The bundled backend ---------------------------------------------------
// First launch can be slow (macOS checks every file of a fresh app once), so
// allow a full minute before calling the backend broken.
const online = await evaluate(`(async () => {
  const end = Date.now() + 60000;
  while (Date.now() < end) {
    const be = await window.electronAPI.backendInfo();
    if (be.state === "ready") return be;
    if (be.state === "failed") return be;
    await new Promise((r) => setTimeout(r, 500));
  }
  return await window.electronAPI.backendInfo();
})()`);
if (!online?.managed) fail("the app did not start its built-in backend (running from source?)", child);
if (online.state !== "ready") fail(`built-in backend is ${online.state}: ${online.error || "no reason given"}`, child);
console.log(`[smoke:packaged] built-in backend ready at ${online.url}`);

const api = await evaluate(`(async () => {
  const be = await window.electronAPI.backendInfo();
  const H = { "X-API-Key": be.key, "Content-Type": "application/json" };
  const health = await fetch(be.url + "/health").then((r) => r.json());
  const noKey = await fetch(be.url + "/providers").then((r) => r.status);
  const prov = await fetch(be.url + "/providers", { headers: H }).then((r) => r.json());
  const free = prov.providers.find((p) => p.id === prov.default) || null;
  return { versions: health.versions, noKey, default: prov.default, free, chip: document.querySelector(".status-chip")?.textContent };
})()`);
if (api.noKey !== 401) fail(`backend answered without the private key (status ${api.noKey})`, child);
for (const pkg of ["python", "fastapi", "litellm"]) {
  if (!api.versions?.[pkg] || api.versions[pkg] === "not installed") fail(`bundled backend is missing ${pkg}: ${JSON.stringify(api.versions)}`, child);
}
await waitFor("the Backend online chip", `document.querySelector(".status-chip")?.textContent.includes("Backend online")`);
console.log(`[smoke:packaged] backend versions ${JSON.stringify(api.versions)}, refuses calls without key`);

if (process.env.SMOKE_EXPECT_BUILTIN === "1") {
  if (!api.free?.builtin || !api.free.configured) fail(`expected a free built-in provider, got default=${api.default}`, child);
  if (api.free.values?.api_key) fail("Settings would show part of the built-in key", child);
  console.log(`[smoke:packaged] free built-in provider: ${api.free.label} (${api.free.model})`);
}

if (process.env.SMOKE_REAL_CHAT === "1") {
  const chat = await evaluate(`(async () => {
    const be = await window.electronAPI.backendInfo();
    const r = await fetch(be.url + "/chat", {
      method: "POST",
      headers: { "X-API-Key": be.key, "Content-Type": "application/json" },
      body: JSON.stringify({ message: "Reply with the single word OK.", provider: "${api.default}" }),
    });
    return { status: r.status, body: await r.json().catch(() => null) };
  })()`);
  if (chat.status === 200 && chat.body?.response?.trim()) {
    console.log(`[smoke:packaged] real chat through ${chat.body.provider}/${chat.body.model}: ${JSON.stringify(chat.body.response.slice(0, 40))}`);
  } else if (chat.status === 429) {
    // The key was accepted; the shared free quota is just busy right now.
    // Blocking a release on that would be wrong, so warn instead.
    console.log(`::warning::free provider rate limited during the smoke test (${chat.body?.code}). The key itself was accepted.`);
  } else {
    const hint = chat.body?.code === "builtin_unavailable" ? " Check the GROQ_API_KEY secret."
      : chat.body?.code === "model_not_found" ? " The shipped model is not available to this key."
      : "";
    fail(`real chat failed (${chat.status}): ${JSON.stringify(chat.body)}.${hint}`, child);
  }
}

await sleep(1000);
if (pageErrors.length) fail(`page errors: ${pageErrors.slice(0, 3).join(" | ")}`, child);

clearTimeout(timer);
child.removeAllListeners("exit");
child.kill();
ws.close();
console.log(`[smoke:packaged] PASS: installed app rendered from app.asar, logo loaded, viewport: ${viewport}, built-in backend online`);
process.exit(0);
