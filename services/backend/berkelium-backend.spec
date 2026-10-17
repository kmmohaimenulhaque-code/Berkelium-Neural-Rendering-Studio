# PyInstaller recipe for the backend that ships inside the desktop app.
# The release workflow and a local build run exactly this file, so what CI
# tests is what users get.
#
#   cd services/backend
#   pip install -r requirements-build.txt
#   pyinstaller --noconfirm berkelium-backend.spec
#
# Output: dist/berkelium-backend/ (a folder, not a single file). One-file
# builds unpack themselves to a temp folder on every launch, which makes the
# app slow to start and trips antivirus tools, so the folder build is used.

import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas, binaries, hiddenimports = [], [], []

# LiteLLM is imported lazily (see providers.py), so PyInstaller cannot see it.
# It also reads JSON price and model tables at runtime. Collect everything.
for pkg in ("litellm", "tiktoken"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += ["tiktoken_ext.openai_public", "tiktoken_ext"]

# Settings > About shows these versions via importlib.metadata.
for pkg in ("fastapi", "litellm", "pydantic"):
    datas += copy_metadata(pkg)

# Release builds add the built-in key file. Missing locally on purpose.
if os.path.exists("builtin.json"):
    datas.append(("builtin.json", "."))

a = Analysis(
    ["server.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="berkelium-backend",
    console=True,  # Electron hides the window; a console app keeps stdin/stdout pipes working on Windows
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="berkelium-backend")
