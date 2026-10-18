# Flatpak build

Wraps `release/linux-unpacked` (electron-builder output, the same files as the AppImage) on
`org.electronjs.Electron2.BaseApp//24.08`. Electron's sandbox runs through zypak.

```bash
# once
flatpak remote-add --user --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
flatpak install --user flathub org.freedesktop.Sdk//24.08 org.electronjs.Electron2.BaseApp//24.08
# from apps/desktop, after the electron-builder linux build
flatpak-builder --user --force-clean --repo=fp-repo fp-build packaging/flatpak/ai.berkelium.studio.yml
flatpak build-bundle --runtime-repo=https://dl.flathub.org/repo/flathub.flatpakrepo fp-repo Berkelium-0.2.0-x86_64.flatpak ai.berkelium.studio
```

Install and run: `flatpak install --user Berkelium-0.2.0-x86_64.flatpak && flatpak run ai.berkelium.studio`.
The app gets network access (model endpoints) and read/write access to your home folder (STEP/GLB exports).
