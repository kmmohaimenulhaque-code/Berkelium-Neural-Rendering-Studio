## What's new

- **Team Berkelium's own CAD models work in chat.** Pick BerkeliumGPT 8B (or the 32B CadQuery model) under Ollama in Settings and describe a part. You get one clean design in the exact format the model was trained on.
- **Fixed: CAD answers with stray reasoning text or wrong shapes.** Ollama did not really turn the model's "thinking" off, which made it write reasoning or repeat itself. Studio now sends the trained prompt format directly, so a flange comes out as a round 290 mm disc, not a box.
- **Slow computers no longer time out on big models.** Local models get up to 10 minutes, and answers stream in so a long reply is not cut off.
- Earlier: AI chat works right after install with a free built-in key, and PDF text reading works in the installed app.

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

- The free built-in AI key is one quota shared by everyone using this version, so it can run out for the day. If chat says the free quota is used up, add your own free key in **Settings > AI Providers** (a Groq key takes a minute at console.groq.com/keys).
- Reading text from **images** (OCR) needs the Tesseract engine, which is not included yet. Reading **PDFs** works.
- If something goes wrong, **Settings > About** shows what failed, and **Open logs** opens the backend log to attach to a bug report.

## Verify your download

`SHA256SUMS.txt` lists a checksum for every file. On macOS or Linux: `shasum -a 256 <file>`. On Windows: `certutil -hashfile <file> SHA256`.
