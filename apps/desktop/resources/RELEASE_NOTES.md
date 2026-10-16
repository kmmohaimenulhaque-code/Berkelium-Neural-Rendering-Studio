## Download

| System | File |
|---|---|
| macOS (Apple Silicon: M1, M2, M3 and newer) | `Berkelium-<version>-mac-arm64.dmg` |
| Windows 10 and 11 (64-bit) | `Berkelium-<version>-win-x64.exe` |
| Linux (64-bit) | `Berkelium-<version>-linux-x86_64.AppImage` |

Intel Macs are not supported.

## First launch

These builds are not signed with a paid Apple or Microsoft certificate yet, so your system asks once before opening them.

**macOS**
1. Open the `.dmg` and drag Berkelium into Applications.
2. Open Berkelium. macOS says it cannot verify the app. Click **Done**.
3. Go to **System Settings > Privacy & Security**, scroll down, and click **Open Anyway** next to Berkelium. Confirm with your password.

If macOS says the app is "damaged", run this once in Terminal, then open it again:
```
xattr -dr com.apple.quarantine /Applications/Berkelium.app
```

**Windows**
1. Run the `.exe`. If SmartScreen shows "Windows protected your PC", click **More info**, then **Run anyway**.
2. Follow the installer.

**Linux**
```
chmod +x Berkelium-*.AppImage
./Berkelium-*.AppImage
```

## Known limits in this build

- AI chat and document reading need the Berkelium backend running on the same computer. It is not bundled yet. The editor, 3D viewport, shapes and model import work without it.

## Verify your download

`SHA256SUMS.txt` lists a checksum for every file. On macOS or Linux: `shasum -a 256 <file>`. On Windows: `certutil -hashfile <file> SHA256`.
