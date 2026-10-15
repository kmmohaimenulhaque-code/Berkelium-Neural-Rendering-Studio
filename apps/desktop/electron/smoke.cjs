// smoke.cjs
// Release build smoke test. Starts the real app from dist/ (exactly what an
// installed copy loads), checks that the window actually renders and that the
// 3D viewport opens, then quits with exit code 0 (pass) or 1 (fail).
//
// Unit tests cannot catch a blank window. A wrong asset path, a crash on
// startup or a missing bundle only shows up when Electron loads the build,
// so CI runs this on macOS, Windows and Linux after "npm run build".
//
// Run:  npm run build && npm run smoke
// Linux without a screen:  xvfb-run -a npm run smoke -- --no-sandbox

"use strict";

// Force the release path in main.cjs before it loads. Same as --dist.
if (!process.argv.includes("--dist")) process.argv.push("--dist");

const { app } = require("electron");

// Let WebGL run on software rendering when the machine has no GPU (CI
// runners). Without this, Chromium blocklists WebGL there and the viewport
// test would only see the "no WebGL" message, never a real 3D frame.
app.commandLine.appendSwitch("ignore-gpu-blocklist");
app.commandLine.appendSwitch("enable-unsafe-swiftshader");

const TIMEOUT_MS = 60000; // generous: a cold CI runner can be slow to start
const errors = [];
let done = false;

function finish(ok, message) {
  if (done) return;
  done = true;
  const line = `[smoke] ${ok ? "PASS" : "FAIL"}: ${message}`;
  if (ok) console.log(line);
  else console.error(line);
  app.exit(ok ? 0 : 1);
}

setTimeout(() => finish(false, `timed out after ${TIMEOUT_MS / 1000}s`), TIMEOUT_MS).unref();

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Polls the page until check() returns a truthy value or the time runs out.
async function waitFor(wc, label, expression, ms = 20000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    const value = await wc.executeJavaScript(expression).catch(() => null);
    if (value) return value;
    await sleep(250);
  }
  throw new Error(`${label} never appeared`);
}

app.on("browser-window-created", (_event, win) => {
  const wc = win.webContents;

  wc.on("render-process-gone", (_e, details) => finish(false, `renderer crashed (${details.reason})`));
  wc.on("did-fail-load", (_e, code, desc, url) => finish(false, `could not load ${url}: ${desc} (${code})`));

  // Uncaught exceptions in the page. Network errors are ignored on purpose:
  // the backend is not running in CI, so its health check is expected to fail.
  wc.on("console-message", (...args) => {
    const e = args[0];
    const level = typeof e?.level === "string" ? e.level : args[1];
    const message = typeof e?.message === "string" ? e.message : args[2];
    const isError = level === "error" || level === 3;
    if (isError && /Uncaught|is not defined|Cannot read/.test(String(message))) errors.push(String(message));
  });

  wc.once("did-finish-load", async () => {
    try {
      // 1. The app shell rendered: workspace tabs are on screen.
      await waitFor(wc, "the app shell", `document.querySelectorAll(".workspace-tab").length >= 3`);

      // 2. The 3D viewport opens. On a machine without a GPU the app shows a
      //    friendly message instead of a canvas; both mean it did not crash.
      await wc.executeJavaScript(
        `[...document.querySelectorAll(".workspace-tab")].find((t) => t.textContent.includes("Scene"))?.click(); true`
      );
      const viewport = await waitFor(
        wc,
        "the 3D viewport",
        `document.querySelector(".scene__viewport canvas") ? "canvas" : document.querySelector(".scene__failure") ? "no-webgl" : null`
      );

      // 3. The starter scene is listed in the hierarchy.
      await waitFor(wc, "the starter cube in the hierarchy", `[...document.querySelectorAll(".scene-tree__name")].some((n) => n.textContent === "Cube")`);

      // Give late errors a moment to surface before judging.
      await sleep(1000);
      if (errors.length) return finish(false, `page errors: ${errors.slice(0, 3).join(" | ")}`);
      finish(true, `app rendered, viewport: ${viewport}`);
    } catch (err) {
      // Name the page error when there is one: it usually explains the miss.
      finish(false, err.message + (errors.length ? `. Page error: ${errors[0]}` : ""));
    }
  });
});

require("./main.cjs");
