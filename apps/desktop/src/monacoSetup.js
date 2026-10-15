// monacoSetup.js
// Makes the code editor load from the app itself instead of the internet.
//
// @monaco-editor/react downloads Monaco from a CDN (jsdelivr) by default.
// That broke the editor with no Wi-Fi (a blank panel in a demo), ran code
// fetched from a website inside a desktop app, and ignored the Monaco
// version we install and keep patched. Bundling it fixes all three.
//
// Language workers do the heavy lifting (validation, completions) off the
// main thread. Python and GLSL need no worker; they use the core one.

import { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";
import JsonWorker from "monaco-editor/language/json/json.worker?worker";
import CssWorker from "monaco-editor/language/css/css.worker?worker";
import HtmlWorker from "monaco-editor/language/html/html.worker?worker";
import TsWorker from "monaco-editor/language/typescript/ts.worker?worker";

self.MonacoEnvironment = {
  getWorker(_id, label) {
    if (label === "json") return new JsonWorker();
    if (label === "css" || label === "scss" || label === "less") return new CssWorker();
    if (label === "html" || label === "handlebars" || label === "razor") return new HtmlWorker();
    if (label === "typescript" || label === "javascript") return new TsWorker();
    return new EditorWorker();
  },
};

loader.config({ monaco });

// The CDN version created this global as a side effect. Keep it, so DevTools
// debugging and the end-to-end tests can still reach the editor API.
window.monaco = monaco;
