#!/bin/sh
# Electron's setuid sandbox can't run inside Flatpak; zypak redirects it to Flatpak's own sandbox.
exec zypak-wrapper /app/berkelium/berkelium-studio "$@"
