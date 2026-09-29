# VoiceFlow: a free Wispr Flow clone

Hold a hotkey, talk, release. Text appears in whatever app you're typing in.

| Hotkey (hold) | Mode |
|---|---|
| `Ctrl+Shift` | **Raw**: plain speech-to-text |
| `Ctrl+Alt` | **Clean**: removes filler words, fixes grammar and punctuation, handles self-corrections |

## Setup

1. Install Python 3.9+.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Get a free key:
   - **Groq** (recommended): https://console.groq.com/keys
   - **Hugging Face**: https://huggingface.co/settings/tokens
4. Copy `.env.example` to `.env`, set `PROVIDER` and paste your key.
5. Run:
   ```
   python app.py
   ```

## Notes

- **macOS**: grant your terminal Microphone and Accessibility (and Input Monitoring) permissions. The paste shortcut is Cmd+V.
- **Linux**: works on X11. Wayland blocks global hotkeys and synthetic typing. Install `xclip` or `xsel` for the clipboard.
- **Windows**: works out of the box.
- It pastes via the clipboard and then restores your previous clipboard contents.
- Free tiers have rate limits, which is fine for personal dictation. If you get a 429 error, wait a moment.
- Set `LANGUAGE=hi` (etc.) in `.env` to force a language, or leave it empty for auto-detect.

## Ideas to extend

- System tray icon and a start/stop toggle
- Custom dictionary for names and jargon (append to the clean prompt)
- Snippets and voice commands ("new line", "period")
- Package as an exe or app with PyInstaller
