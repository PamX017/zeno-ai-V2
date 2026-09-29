"""
VoiceFlow - a free, Wispr Flow-style dictation app.

Hold a hotkey, speak, release. Your speech is transcribed and typed into
whatever app has focus.

  Hold CTRL+ALT    -> RAW mode    (plain speech-to-text)
  Hold CTRL+SPACE  -> CLEAN mode  (removes filler words, fixes grammar/punctuation)

Providers (both have free tiers):
  groq : Whisper + Llama on Groq   (default, fastest)
  hf   : Whisper + chat models via Hugging Face Inference Providers
"""
import io
import os
import sys
import time
import wave
import threading
import queue
import math
import tkinter as tk

import numpy as np
import pyperclip
import requests
import sounddevice as sd
from dotenv import load_dotenv
from pynput import keyboard

load_dotenv()

# When launched with pythonw (no console) send output to a log file instead.
if sys.stdout is None or sys.stderr is None:
    _log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "voiceflow.log"),
                "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log


def fatal(msg):
    """Exit with a message; show a popup when there is no console window."""
    print(msg)
    if sys.__stdout__ is None:
        try:
            import tkinter.messagebox as mb
            r = tk.Tk()
            r.withdraw()
            mb.showerror("VoiceFlow", msg)
        except Exception:
            pass
    sys.exit(1)

# ----------------------------- configuration -----------------------------
PROVIDER = os.getenv("PROVIDER", "groq").lower()  # groq | hf
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

RAW_HOTKEY = set(os.getenv("RAW_HOTKEY", "ctrl+alt").lower().split("+"))
CLEAN_HOTKEY = set(os.getenv("CLEAN_HOTKEY", "ctrl+space").lower().split("+"))

CLICK_MODE = os.getenv("CLICK_MODE", "clean").lower()  # clean | raw (used by left-click)
LANGUAGE = os.getenv("LANGUAGE", "")  # e.g. "en", "hi"; empty = auto-detect

PROVIDERS = {
    "groq": {
        "stt_url": "https://api.groq.com/openai/v1/audio/transcriptions",
        "stt_model": os.getenv("STT_MODEL", "whisper-large-v3-turbo"),
        "chat_url": "https://api.groq.com/openai/v1/chat/completions",
        "chat_model": os.getenv("CLEAN_MODEL", "llama-3.3-70b-versatile"),
        "key": GROQ_API_KEY,
    },
    "hf": {
        "stt_url": "https://router.huggingface.co/hf-inference/models/"
        + os.getenv("STT_MODEL", "openai/whisper-large-v3"),
        "stt_model": os.getenv("STT_MODEL", "openai/whisper-large-v3"),
        "chat_url": "https://router.huggingface.co/v1/chat/completions",
        "chat_model": os.getenv("CLEAN_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        "key": HF_TOKEN,
    },
}

SAMPLE_RATE = 16000
MIN_SECONDS = 0.4

CLEAN_PROMPT = (
    "You clean up dictated speech. Rewrite the transcript so it reads as polished "
    "written text: remove filler words (um, uh, like, you know), false starts and "
    "repetitions; fix grammar, spelling, capitalization and punctuation; honour "
    "spoken corrections (e.g. 'no wait, Tuesday' means use Tuesday). Keep the "
    "speaker's meaning, tone and language. Do NOT add information, answer "
    "questions in the text, or explain anything. Output ONLY the cleaned text."
)

# ------------------------------- recording -------------------------------
class Recorder:
    def __init__(self):
        self.frames = []
        self.stream = None

    def start(self):
        self.frames = []
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=self._cb
        )
        self.stream.start()

    def _cb(self, indata, frames, time_info, status):
        self.frames.append(indata.copy())

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.frames:
            return None, 0.0
        audio = np.concatenate(self.frames, axis=0)
        seconds = len(audio) / SAMPLE_RATE
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(audio.tobytes())
        return buf.getvalue(), seconds


# ------------------------------- AI calls --------------------------------
RAW_PROMPT = (
    "Umm, uh, so, like, you know, I was, uh, thinking of... Oh, um, well, "
    "hmm, yeah, so basically, ummm, I mean, kind of, sort of."
)


def transcribe(wav_bytes: bytes, keep_fillers: bool = False) -> str:
    cfg = PROVIDERS[PROVIDER]
    headers = {"Authorization": f"Bearer {cfg['key']}"}
    if PROVIDER == "groq":
        data = {"model": cfg["stt_model"], "response_format": "json", "temperature": "0"}
        if LANGUAGE:
            data["language"] = LANGUAGE
        if keep_fillers:
            data["prompt"] = RAW_PROMPT  # nudges Whisper to write "um", "uh" etc.
        r = requests.post(
            cfg["stt_url"],
            headers=headers,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data=data,
            timeout=60,
        )
    else:
        headers["Content-Type"] = "audio/wav"
        r = requests.post(cfg["stt_url"], headers=headers, data=wav_bytes, timeout=120)
    r.raise_for_status()
    return r.json().get("text", "").strip()


FALLBACK_MODELS = {
    "groq": [
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
    ],
    "hf": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ],
}
_working_model = None


def clean_text(text: str) -> str:
    global _working_model
    cfg = PROVIDERS[PROVIDER]
    candidates = [_working_model] if _working_model else []
    for m in [cfg["chat_model"]] + FALLBACK_MODELS[PROVIDER]:
        if m not in candidates:
            candidates.append(m)
    last_err = None
    for model in candidates:
        r = requests.post(
            cfg["chat_url"],
            headers={"Authorization": f"Bearer {cfg['key']}"},
            json={
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": CLEAN_PROMPT},
                    {"role": "user", "content": f"<transcript>{text}</transcript>"},
                ],
            },
            timeout=60,
        )
        if r.status_code in (400, 403, 404):  # model missing/unavailable: try next
            last_err = r
            continue
        r.raise_for_status()
        if _working_model != model:
            _working_model = model
            print(f"  (using cleanup model: {model})")
        out = r.json()["choices"][0]["message"]["content"].strip()
        return out.replace("<transcript>", "").replace("</transcript>", "").strip()
    last_err.raise_for_status()


# ------------------------------- typing ----------------------------------
kb = keyboard.Controller()
PASTE_MOD = keyboard.Key.cmd if sys.platform == "darwin" else keyboard.Key.ctrl


pasting = False


def paste(text: str):
    global pasting
    pasting = True
    try:
        _paste(text)
    finally:
        pasting = False


def _paste(text: str):
    try:
        old = pyperclip.paste()
    except Exception:
        old = None
    pyperclip.copy(text)
    time.sleep(0.05)
    with kb.pressed(PASTE_MOD):
        kb.press("v")
        kb.release("v")
    time.sleep(0.2)
    if old is not None:
        pyperclip.copy(old)  # restore the user's clipboard


# ------------------------------- overlay UI ------------------------------
class Overlay:
    """Small always-on-top glowing circle.
    grey = idle, green = listening, blue = processing, red = finished.
    Left-click = start/stop dictation, right-click = quit, drag = move."""

    KEY = "#010101"  # transparent colour key (Windows)
    COLORS = {
        "idle": "#6b7280",
        "listening": "#22c55e",
        "processing": "#3b82f6",
        "done": "#ef4444",
    }
    SIZE = 60

    def __init__(self):
        self.q = queue.Queue()
        self.state = "idle"
        self.phase = 0.0
        self.root = tk.Tk()
        r = self.root
        r.title("VoiceFlow")
        r.overrideredirect(True)
        r.attributes("-topmost", True)
        r.configure(bg=self.KEY)
        try:
            r.wm_attributes("-transparentcolor", self.KEY)
        except tk.TclError:
            pass  # not Windows: background stays dark
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        r.geometry(f"{self.SIZE}x{self.SIZE}+{sw // 2 - self.SIZE // 2}+{sh - self.SIZE - 220}")
        self.c = tk.Canvas(r, width=self.SIZE, height=self.SIZE, bg=self.KEY, highlightthickness=0)
        self.c.pack()
        self.on_click = None
        self._moved = False
        self.c.bind("<ButtonPress-1>", self._drag_start)
        self.c.bind("<B1-Motion>", self._drag)
        self.c.bind("<ButtonRelease-1>", self._release)
        self.c.bind("<Button-3>", lambda e: self.quit())
        self._tick()

    def set(self, state):  # safe to call from any thread
        self.q.put(state)

    def quit(self):
        self.root.destroy()
        os._exit(0)

    def _drag_start(self, e):
        self._dx, self._dy = e.x, e.y
        self._moved = False

    def _release(self, e):
        if not self._moved and self.on_click:
            self.on_click()

    def _drag(self, e):
        if abs(e.x - self._dx) + abs(e.y - self._dy) > 4:
            self._moved = True
        if not self._moved:
            return
        x = self.root.winfo_x() + e.x - self._dx
        y = self.root.winfo_y() + e.y - self._dy
        self.root.geometry(f"+{x}+{y}")

    @staticmethod
    def _mix(fg, bg, a):
        f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
        return "#%02x%02x%02x" % tuple(int(f[i] * a + b[i] * (1 - a)) for i in range(3))

    def _tick(self):
        try:
            while True:
                self.state = self.q.get_nowait()
                if self.state == "done":
                    self.root.after(1200, lambda: self.set("idle") if self.state == "done" else None)
        except queue.Empty:
            pass
        self.root.attributes("-topmost", True)
        self.phase += 0.18 if self.state == "listening" else 0.10
        pulse = 0.5 + 0.5 * math.sin(self.phase)
        active = self.state != "idle"
        col = self.COLORS[self.state]
        cx = cy = self.SIZE / 2
        core = 9 + (2 * pulse if active else 0)
        self.c.delete("all")
        for i in range(8, 0, -1):  # soft glow layers
            rad = core + i * 2.2
            alpha = (0.55 if active else 0.12) * (1 - i / 9) ** 2 * (0.7 + 0.5 * pulse)
            self.c.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                               fill=self._mix(col, self.KEY, alpha), outline="")
        self.c.create_oval(cx - core, cy - core, cx + core, cy + core, fill=col, outline="")
        self.root.after(33, self._tick)


ui = None


def set_state(state):
    if ui:
        ui.set(state)


# ------------------------------ main logic -------------------------------
recorder = Recorder()
pressed = set()
active_mode = None  # None | "raw" | "clean"
lock = threading.Lock()


def norm(key) -> str:
    if isinstance(key, keyboard.Key):
        name = key.name
        for base in ("ctrl", "alt", "shift", "cmd"):
            if name.startswith(base):
                return base
        if name == "alt_gr":
            return "alt"
        return name
    if isinstance(key, keyboard.KeyCode) and key.char:
        return key.char.lower()
    return str(key)


def process(wav, seconds, mode):
    if not wav or seconds < MIN_SECONDS:
        print("  (too short, ignored)")
        set_state("idle")
        return
    ok = False
    try:
        print(f"  transcribing {seconds:.1f}s of audio...")
        text = transcribe(wav, keep_fillers=(mode == "raw"))
        if not text:
            print("  (nothing heard)")
            return
        if mode == "clean":
            text = clean_text(text)
        print(f"  -> {text}")
        paste(text)
        ok = True
    except requests.HTTPError as e:
        print(f"  API error: {e.response.status_code} {e.response.text[:200]}")
    except Exception as e:
        print(f"  error: {e}")
    finally:
        set_state("done" if ok else "idle")


def stop_and_process():
    global active_mode
    with lock:
        mode, active_mode = active_mode, None
    if mode is None:
        return
    wav, seconds = recorder.stop()
    set_state("processing")
    threading.Thread(target=process, args=(wav, seconds, mode), daemon=True).start()


click_active = False


def begin(mode):
    print(f"[{mode.upper()}] listening...")
    recorder.start()
    set_state("listening")


def on_click():
    """Left-click on the circle: start continuous dictation; click again to stop."""
    global active_mode, click_active
    if active_mode is None:
        active_mode = "clean" if CLICK_MODE == "clean" else "raw"
        click_active = True
        begin(active_mode)
    elif click_active:
        click_active = False
        stop_and_process()


def on_press(key):
    global active_mode
    if pasting:
        return
    pressed.add(norm(key))
    with lock:
        if active_mode is not None:
            return
        if pressed == CLEAN_HOTKEY:
            active_mode = "clean"
        elif pressed == RAW_HOTKEY:
            active_mode = "raw"
        else:
            return
        mode = active_mode
    begin(mode)


def on_release(key):
    if pasting:
        return
    pressed.discard(norm(key))
    if click_active:
        return
    if active_mode is not None and not (
        (active_mode == "clean" and CLEAN_HOTKEY <= pressed)
        or (active_mode == "raw" and RAW_HOTKEY <= pressed)
    ):
        stop_and_process()


_lock_sock = None


def ensure_single_instance():
    """A second copy would paste everything twice, so refuse to start."""
    import socket
    global _lock_sock
    _lock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _lock_sock.bind(("127.0.0.1", 48653))
    except OSError:
        fatal("VoiceFlow is already running. Right-click the circle to close it "
              "first, otherwise text is pasted twice.")


def main():
    ensure_single_instance()
    cfg = PROVIDERS.get(PROVIDER)
    if not cfg:
        fatal(f"Unknown PROVIDER '{PROVIDER}'. Use 'groq' or 'hf'.")
    if not cfg["key"]:
        var = "GROQ_API_KEY" if PROVIDER == "groq" else "HF_TOKEN"
        fatal(f"Missing {var}. Create a .env file next to app.py and add your key.")
    print("VoiceFlow running  (provider: %s)" % PROVIDER)
    print("  Hold CTRL+SPACE        -> cleaned-up dictation")
    print("  Hold CTRL+ALT          -> raw dictation")
    print("  Left-click circle: start/stop dictation (%s mode) | Right-click: quit\n" % CLICK_MODE)
    global ui
    ui = Overlay()
    ui.on_click = on_click
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    try:
        ui.root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()