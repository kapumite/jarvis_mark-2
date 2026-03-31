"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              J.A.R.V.I.S  Mark-2  —  Action-Based Agent System             ║
║         Path : C:\\Users\\P.E.K.K.A\\Desktop\\jarvis\\jarvis.py              ║
║         LLM  : Ollama  qwen2.5:7b  (http://localhost:11434/v1)              ║
║         OS   : Windows 10 / 11                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture Overview
─────────────────────
  Microphone
      │
   STTEngine  →  raw text
      │
  WakeWordFilter
      │
  AgentCore.handle(text)
      │
  LLMPlanner.plan(text)          ← sends text + tool signatures to Ollama
      │                             model returns JSON: { "speech": "...",
      │                                                   "actions": [...] }
  ActionDispatcher.run(plan)
      │
  ToolBox methods  ←─── strictly typed, no eval(), no exec()
      │
  TTSEngine.speak(speech_text)   ← non-blocking, always called

Safety model
────────────
  The LLM *describes* what to do using a closed set of named tools.
  No code is generated or eval'd.  The dispatcher maps tool names to
  Python methods.  Any unknown tool name is silently ignored with a
  warning, not executed.

Setup
─────
  pip install openai pyttsx3 SpeechRecognition pyaudio pygetwindow pyautogui colorama
  (Run: ollama serve  and  ollama pull qwen2.5:7b  before starting)
  python jarvis.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import sys
import json
import time
import queue
import shutil
import signal
import logging
import threading
import subprocess
from datetime import datetime
from typing import Any, Optional

# ── Third-party (hard dependencies) ──────────────────────────────────────────
try:
    import speech_recognition as sr
    import pyttsx3
    from openai import OpenAI
    from colorama import Fore, Style, init as colorama_init
except ImportError as exc:
    print(f"\n[FATAL] Missing package: {exc}")
    print("Run:  pip install openai pyttsx3 SpeechRecognition pyaudio colorama\n")
    sys.exit(1)

# ── Third-party (soft — degrade gracefully if absent) ────────────────────────
try:
    import pygetwindow as gw
    import pyautogui
    pyautogui.FAILSAFE = False
    GUI_OK = True
except ImportError:
    GUI_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  §0  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

colorama_init(autoreset=True)

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL    = "qwen2.5:7b"
LLM_TIMEOUT     = 45          # seconds; Ollama on local GPU can be slow
MAX_HISTORY     = 12          # conversation pairs kept in context

# ── Wake words ────────────────────────────────────────────────────────────────
WAKE_WORDS: tuple[str, ...] = ("джарвис", "jarvis")

# ── Speech recognition ────────────────────────────────────────────────────────
LISTEN_LANG      = "ru-RU"
ENERGY_THRESHOLD = 300
PAUSE_THRESHOLD  = 0.85
PHRASE_LIMIT     = 10.0

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_RATE   = 165
TTS_VOLUME = 1.0

# ── Paths (user-specific) ─────────────────────────────────────────────────────
USER_HOME   = r"C:\Users\P.E.K.K.A"
DESKTOP     = os.path.join(USER_HOME, "Desktop")

APP_PATHS: dict[str, str] = {
    "yandex music":  rf"{USER_HOME}\AppData\Local\Programs\YandexMusicMod\Яндекс Музыка.exe",
    "яндекс музыка": rf"{USER_HOME}\AppData\Local\Programs\YandexMusicMod\Яндекс Музыка.exe",
    "steam":         r"C:\Program Files (x86)\Steam\steam.exe",
    "telegram":      rf"{USER_HOME}\AppData\Roaming\Telegram Desktop\Telegram.exe",
    "firefox":       r"C:\Program Files\Mozilla Firefox\firefox.exe",
    "chrome":        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "vs code":       r"C:\Users\P.E.K.K.A\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    "discord":       rf"{USER_HOME}\AppData\Local\Discord\Update.exe",
    "spotify":       rf"{USER_HOME}\AppData\Roaming\Spotify\Spotify.exe",
    "notepad":       "notepad.exe",
    "explorer":      "explorer.exe",
    "calculator":    "calc.exe",
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("JARVIS-Mk2")


# ══════════════════════════════════════════════════════════════════════════════
#  §1  TERMINAL UI
# ══════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def ui_banner():
    w = 72
    print(Fore.CYAN + Style.BRIGHT + "═" * w)
    print(Fore.CYAN + Style.BRIGHT + "  J.A.R.V.I.S  Mark-2  —  Action-Based Agent")
    print(Fore.CYAN + Style.DIM    + f"  Model: {OLLAMA_MODEL}  |  Endpoint: {OLLAMA_BASE_URL}")
    print(Fore.CYAN + Style.BRIGHT + "═" * w)

def ui_listen():    print(Fore.GREEN  + Style.BRIGHT + f"\n[{_ts()}] 🎤  Слушаю…")
def ui_wake():      print(Fore.YELLOW + Style.BRIGHT + f"[{_ts()}] ✨  Активирован!")
def ui_heard(t):    print(Fore.WHITE  + f"[{_ts()}] 👂  Распознано: " + Style.BRIGHT + f'"{t}"')
def ui_plan(p):     print(Fore.MAGENTA+ f"[{_ts()}] 🧠  Планирование… ответ модели:")
def ui_action(a):   print(Fore.BLUE   + Style.BRIGHT + f"[{_ts()}] ⚡  {a}")
def ui_speak(t):
    d = t if len(t) <= 110 else t[:107] + "…"
    print(Fore.CYAN + Style.BRIGHT + f"[{_ts()}] 🔊  JARVIS: " + Style.NORMAL + d)
def ui_warn(m):     print(Fore.YELLOW + f"[{_ts()}] ⚠   {m}")
def ui_error(m):    print(Fore.RED    + Style.BRIGHT + f"[{_ts()}] ✖   {m}")
def ui_info(m):     print(Fore.WHITE  + Style.DIM    + f"[{_ts()}] ℹ   {m}")
def ui_sep():       print(Fore.CYAN   + Style.DIM    + "─" * 72)
def ui_json(d):     print(Fore.MAGENTA+ Style.DIM    + json.dumps(d, ensure_ascii=False, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
#  §2  TTS ENGINE  (dedicated thread — always non-blocking)
# ══════════════════════════════════════════════════════════════════════════════

class TTSEngine:
    """
    pyttsx3 runs in a dedicated daemon thread with a Queue.
    speak()      → enqueue text, return immediately (non-blocking).
    speak_sync() → enqueue and block until audio finishes.
    """

    def __init__(self):
        self._q: queue.Queue[Optional[str]] = queue.Queue()
        t = threading.Thread(target=self._worker, daemon=True, name="TTS-Worker")
        t.start()

    def _worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate",   TTS_RATE)
            engine.setProperty("volume", TTS_VOLUME)

            # Select Russian voice if available
            for v in engine.getProperty("voices"):
                langs = v.languages or []
                if any("ru" in str(l).lower() for l in langs) \
                        or "elena" in v.name.lower() \
                        or "irina" in v.name.lower() \
                        or "russian" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    ui_info(f"TTS voice: {v.name}")
                    break
            else:
                ui_warn("Russian TTS voice not found — using system default.")

            while True:
                text = self._q.get()
                if text is None:
                    break
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    ui_error(f"TTS playback error: {e}")
                finally:
                    self._q.task_done()
        except Exception as e:
            ui_error(f"TTS init failed: {e}")

    def speak(self, text: str) -> None:
        """Non-blocking. Always call this after every LLM response."""
        ui_speak(text)
        self._q.put(text)

    def speak_sync(self, text: str) -> None:
        """Blocking — used only for the startup greeting."""
        self.speak(text)
        self._q.join()

    def stop(self):
        self._q.put(None)


# ══════════════════════════════════════════════════════════════════════════════
#  §3  STT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class STTEngine:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.rec.energy_threshold        = ENERGY_THRESHOLD
        self.rec.pause_threshold         = PAUSE_THRESHOLD
        self.rec.dynamic_energy_threshold = True
        self._mic: Optional[sr.Microphone] = None
        try:
            self._mic = sr.Microphone()
            ui_info("Microphone initialised.")
        except OSError as e:
            ui_error(f"Microphone not found: {e}")

    def calibrate(self):
        if not self._mic:
            return
        ui_info("Calibrating ambient noise…")
        try:
            with self._mic as src:
                self.rec.adjust_for_ambient_noise(src, duration=1.2)
            ui_info(f"Energy threshold: {self.rec.energy_threshold:.0f}")
        except Exception as e:
            ui_warn(f"Calibration failed: {e}")

    def listen_once(self) -> Optional[str]:
        if not self._mic:
            ui_error("Microphone unavailable.")
            return None
        try:
            with self._mic as src:
                ui_listen()
                audio = self.rec.listen(src, timeout=None,
                                        phrase_time_limit=PHRASE_LIMIT)
            return self.rec.recognize_google(audio, language=LISTEN_LANG).strip()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            ui_warn("Speech not recognised.")
            return None
        except sr.RequestError as e:
            ui_error(f"STT service unavailable: {e}")
            return None
        except OSError as e:
            ui_error(f"Mic OS error: {e}")
            return None

    @property
    def available(self) -> bool:
        return self._mic is not None


# ══════════════════════════════════════════════════════════════════════════════
#  §4  TOOLBOX  —  closed set of safe desktop actions
# ══════════════════════════════════════════════════════════════════════════════

class ToolBox:
    """
    Every public method is a callable tool.  The ActionDispatcher maps
    JSON tool names → ToolBox methods.  Nothing outside this class is
    ever called by the dispatcher, so there is no code-injection surface.

    Tool signatures (what the LLM sees in its system prompt)
    ─────────────────────────────────────────────────────────
    launch_app(name)          – launch app by name from APP_PATHS registry
    focus_or_launch(name)     – focus window if open, else launch
    focus_window(title)       – bring an existing window to the front
    close_app(name)           – terminate a process by image name
    press_key(*keys)          – press one or more keys (hotkey if multiple)
    type_text(text)           – type text into the active window
    open_url(url, browser)    – navigate browser to URL (Ctrl+L → type → Enter)
    create_folder(name)       – create folder on Desktop
    delete_folder(name)       – delete folder/file on Desktop (to Recycle Bin via shell)
    rename_item(old, new)     – rename file/folder on Desktop
    media_play_pause()        – press Play/Pause media key
    media_next()              – press Next Track media key
    media_prev()              – press Previous Track media key
    get_time()                – speak current time (returns string)
    get_date()                – speak current date (returns string)
    """

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_app_path(name: str) -> Optional[str]:
        """Look up an app path from APP_PATHS (case-insensitive)."""
        return APP_PATHS.get(name.lower().strip())

    @staticmethod
    def _find_windows(fragment: str) -> list:
        """Return all open windows whose title contains fragment (case-insensitive)."""
        if not GUI_OK:
            return []
        try:
            frag_l = fragment.lower()
            return [w for w in gw.getAllWindows()
                    if frag_l in w.title.lower() and w.title.strip()]
        except Exception:
            return []

    @staticmethod
    def _popen(path: str, *extra_args: str) -> bool:
        """Launch a process silently.  Returns True on success."""
        cmd = [path] + list(extra_args)
        ui_action(f"Launching: {' '.join(cmd)}")
        try:
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW,
                             close_fds=True)
            return True
        except FileNotFoundError:
            ui_error(f"Executable not found: {path}")
            return False
        except Exception as e:
            ui_error(f"Launch error: {e}")
            return False

    # ── Tool: launch_app ──────────────────────────────────────────────────────

    def launch_app(self, name: str) -> str:
        """Launch an application by its registered name."""
        path = self._resolve_app_path(name)
        if path is None:
            # Last resort: try running the name as a command (e.g. "notepad.exe")
            if self._popen(name):
                return f"Запускаю {name}."
            return f"Приложение «{name}» не найдено в реестре путей."
        if self._popen(path):
            return f"Запускаю {name}."
        return f"Не удалось запустить {name}."

    # ── Tool: focus_or_launch ─────────────────────────────────────────────────

    def focus_or_launch(self, name: str) -> str:
        """
        If a window matching `name` is already open → bring it to front.
        Otherwise → launch the app.
        """
        wins = self._find_windows(name)
        if wins:
            return self.focus_window(name)
        return self.launch_app(name)

    # ── Tool: focus_window ────────────────────────────────────────────────────

    def focus_window(self, title: str) -> str:
        """Bring the first window whose title contains `title` to the foreground."""
        if not GUI_OK:
            return "Модуль управления окнами недоступен (установите pygetwindow)."
        wins = self._find_windows(title)
        if not wins:
            return f"Окно «{title}» не найдено."
        try:
            w = wins[0]
            if w.isMinimized:
                w.restore()
            w.activate()
            time.sleep(0.35)
            ui_action(f"Focused: {w.title[:60]}")
            return f"Переключаюсь на {title}."
        except Exception as e:
            ui_error(f"focus_window error: {e}")
            return f"Не удалось переключиться на {title}: {e}"

    # ── Tool: close_app ───────────────────────────────────────────────────────

    def close_app(self, name: str) -> str:
        """
        Terminate a process by image name (e.g. "firefox.exe", "steam.exe").
        Uses taskkill /F /IM <name>.
        """
        # Normalise: add .exe if not present
        exe = name if name.lower().endswith(".exe") else name + ".exe"
        ui_action(f"taskkill /F /IM {exe}")
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", exe],
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode == 0:
                return f"{name} завершён."
            return f"Не удалось завершить {name}: {result.stderr.strip()}"
        except Exception as e:
            ui_error(f"close_app error: {e}")
            return f"Ошибка при завершении {name}: {e}"

    # ── Tool: press_key ───────────────────────────────────────────────────────

    def press_key(self, *keys: str) -> str:
        """
        Press one key or a hotkey combination.
        Examples: press_key("enter")  /  press_key("ctrl", "l")
        """
        if not GUI_OK:
            return "Модуль ввода клавиш недоступен (установите pyautogui)."
        if not keys:
            return "Не указаны клавиши."
        ui_action(f"Keys: {' + '.join(keys)}")
        try:
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            time.sleep(0.1)
            return "Клавиши нажаты."
        except Exception as e:
            ui_error(f"press_key error: {e}")
            return f"Ошибка нажатия клавиш: {e}"

    # ── Tool: type_text ───────────────────────────────────────────────────────

    def type_text(self, text: str) -> str:
        """Type text into the currently active window."""
        if not GUI_OK:
            return "Модуль ввода текста недоступен (установите pyautogui)."
        ui_action(f"Typing: {text[:60]}")
        try:
            # pyautogui.typewrite doesn't handle non-ASCII well on Windows;
            # pyperclip + Ctrl+V is more reliable for Cyrillic / special chars.
            import pyperclip  # type: ignore
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
        except ImportError:
            # Fallback for plain ASCII
            pyautogui.typewrite(text, interval=0.02)
        time.sleep(0.1)
        return f"Введено: {text[:50]}"

    # ── Tool: open_url ────────────────────────────────────────────────────────

    def open_url(self, url: str, browser: str = "firefox") -> str:
        """
        Navigate a browser to a URL.
        1. Focus or launch the browser.
        2. Ctrl+L → type URL → Enter.
        """
        if not GUI_OK:
            return "Модуль управления браузером недоступен."

        # Ensure URL has a scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Step 1: ensure browser is open and focused
        msg = self.focus_or_launch(browser)
        ui_action(f"Browser: {msg}")

        # Wait for window to appear (up to 8 s)
        for _ in range(16):
            if self._find_windows(browser):
                break
            time.sleep(0.5)
        else:
            return f"Браузер {browser} не открылся вовремя."

        time.sleep(0.5)   # settle

        # Step 2: navigate
        self.press_key("ctrl", "l")
        time.sleep(0.3)
        self.press_key("ctrl", "a")   # clear existing address
        time.sleep(0.1)

        # Type URL (ASCII-safe; URLs are ASCII)
        if GUI_OK:
            pyautogui.typewrite(url, interval=0.025)
        time.sleep(0.15)
        self.press_key("enter")

        ui_action(f"Navigating to: {url}")
        return f"Открываю {url} в {browser}."

    # ── Tool: create_folder ───────────────────────────────────────────────────

    def create_folder(self, name: str) -> str:
        """Create a new folder on the Desktop."""
        # Strip forbidden Windows path chars
        safe = re.sub(r'[<>:"/\\|?*]', "", name).strip()
        if not safe:
            return "Недопустимое имя папки."
        path = os.path.join(DESKTOP, safe)
        if os.path.exists(path):
            return f"Папка «{safe}» уже существует на рабочем столе."
        try:
            os.makedirs(path, exist_ok=True)
            ui_action(f"Created folder: {path}")
            return f"Папка «{safe}» создана на рабочем столе."
        except PermissionError:
            return "Нет прав для создания папки."
        except Exception as e:
            return f"Ошибка создания папки: {e}"

    # ── Tool: delete_folder ───────────────────────────────────────────────────

    def delete_folder(self, name: str) -> str:
        """Delete a file or folder on the Desktop by sending it to the Recycle Bin."""
        safe = re.sub(r'[<>:"/\\|?*]', "", name).strip()
        path = os.path.join(DESKTOP, safe)
        if not os.path.exists(path):
            return f"«{safe}» не найден на рабочем столе."
        try:
            # Use Windows shell to move to Recycle Bin (safer than shutil.rmtree)
            subprocess.run(
                ["powershell", "-Command",
                 f'Add-Type -AssemblyName Microsoft.VisualBasic; '
                 f'[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory('
                 f'"{path}", '
                 f'"OnlyErrorDialogs", "SendToRecycleBin")'],
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=True,
            )
            ui_action(f"Deleted (Recycle Bin): {path}")
            return f"«{safe}» отправлен в корзину."
        except Exception as e:
            # Hard delete as fallback
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                return f"«{safe}» удалён."
            except Exception as e2:
                return f"Ошибка удаления: {e2}"

    # ── Tool: rename_item ─────────────────────────────────────────────────────

    def rename_item(self, old_name: str, new_name: str) -> str:
        """Rename a file or folder on the Desktop."""
        old_path = os.path.join(DESKTOP, old_name.strip())
        new_path = os.path.join(DESKTOP, new_name.strip())
        if not os.path.exists(old_path):
            return f"«{old_name}» не найден на рабочем столе."
        if os.path.exists(new_path):
            return f"«{new_name}» уже существует."
        try:
            os.rename(old_path, new_path)
            ui_action(f"Renamed: {old_name} → {new_name}")
            return f"«{old_name}» переименован в «{new_name}»."
        except PermissionError:
            return "Нет прав для переименования."
        except Exception as e:
            return f"Ошибка переименования: {e}"

    # ── Tool: media controls ──────────────────────────────────────────────────

    def media_play_pause(self) -> str:
        self.press_key("playpause")
        return "Воспроизведение / пауза."

    def media_next(self) -> str:
        self.press_key("nexttrack")
        return "Следующий трек."

    def media_prev(self) -> str:
        self.press_key("prevtrack")
        return "Предыдущий трек."

    def media_volume_up(self) -> str:
        self.press_key("volumeup")
        return "Громкость увеличена."

    def media_volume_down(self) -> str:
        self.press_key("volumedown")
        return "Громкость уменьшена."

    # ── Tool: time / date ─────────────────────────────────────────────────────

    def get_time(self) -> str:
        n = datetime.now()
        return f"Сейчас {n.hour} часов {n.minute} минут."

    def get_date(self) -> str:
        days = ["понедельник","вторник","среда","четверг",
                "пятница","суббота","воскресенье"]
        months = ["января","февраля","марта","апреля","мая","июня",
                  "июля","августа","сентября","октября","ноября","декабря"]
        n = datetime.now()
        return (f"Сегодня {days[n.weekday()]}, "
                f"{n.day} {months[n.month-1]} {n.year} года.")


# ══════════════════════════════════════════════════════════════════════════════
#  §5  ACTION DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

class ActionDispatcher:
    """
    Executes a list of action dicts produced by the LLM planner.

    Each action dict has the shape:
        { "tool": "<tool_name>", "args": { "<param>": "<value>", … } }

    The dispatcher maps tool names → ToolBox methods via a static registry.
    Unknown tool names produce a warning and are skipped — no eval, no exec.
    """

    def __init__(self, toolbox: ToolBox):
        self._tb = toolbox
        # ── Closed tool registry ──────────────────────────────────────────────
        # Maps JSON tool name → (method, positional_arg_keys)
        # positional_arg_keys defines the order args are passed to the method.
        self._registry: dict[str, tuple] = {
            "launch_app":        (self._tb.launch_app,       ["name"]),
            "focus_or_launch":   (self._tb.focus_or_launch,  ["name"]),
            "focus_window":      (self._tb.focus_window,     ["title"]),
            "close_app":         (self._tb.close_app,        ["name"]),
            "press_key":         (self._tb.press_key,        ["keys"]),   # keys = list or str
            "type_text":         (self._tb.type_text,        ["text"]),
            "open_url":          (self._tb.open_url,         ["url", "browser"]),
            "create_folder":     (self._tb.create_folder,    ["name"]),
            "delete_folder":     (self._tb.delete_folder,    ["name"]),
            "rename_item":       (self._tb.rename_item,      ["old_name", "new_name"]),
            "media_play_pause":  (self._tb.media_play_pause, []),
            "media_next":        (self._tb.media_next,       []),
            "media_prev":        (self._tb.media_prev,       []),
            "media_volume_up":   (self._tb.media_volume_up,  []),
            "media_volume_down": (self._tb.media_volume_down,[]),
            "get_time":          (self._tb.get_time,         []),
            "get_date":          (self._tb.get_date,         []),
        }

    def run(self, actions: list[dict]) -> list[str]:
        """
        Execute a sequence of action dicts.
        Returns a list of result strings from each tool call.
        """
        results: list[str] = []
        for action in actions:
            tool_name = action.get("tool", "").strip()
            args_dict  = action.get("args", {})

            if tool_name not in self._registry:
                ui_warn(f"Unknown tool skipped: '{tool_name}'")
                continue

            method, arg_keys = self._registry[tool_name]
            ui_action(f"Tool: {tool_name}  args: {args_dict}")

            try:
                # Build positional args in declared order; missing keys → skip
                pos_args: list[Any] = []
                for key in arg_keys:
                    if key not in args_dict:
                        break
                    val = args_dict[key]
                    # Special case: press_key accepts a list of keys
                    if key == "keys" and isinstance(val, list):
                        pos_args.extend(val)
                    else:
                        pos_args.append(val)

                result = method(*pos_args)
                if result:
                    results.append(str(result))

            except Exception as e:
                msg = f"Ошибка выполнения {tool_name}: {e}"
                ui_error(msg)
                log.exception("ActionDispatcher error in tool '%s'", tool_name)
                results.append(msg)

            # Brief pause between sequential actions so the OS can settle
            time.sleep(0.15)

        return results


# ══════════════════════════════════════════════════════════════════════════════
#  §6  LLM PLANNER
# ══════════════════════════════════════════════════════════════════════════════

# ── System prompt ──────────────────────────────────────────────────────────────
# The prompt is written entirely in English so the model reliably follows the
# JSON schema, while speech output remains Russian.

SYSTEM_PROMPT = f"""You are JARVIS Mark-2, a personal AI assistant running on Windows.
You speak ONLY in Russian, but you must think in English when planning actions.

## IDENTITY
- Witty, elegant, slightly sarcastic — like JARVIS from Iron Man.
- Address the user as "сэр" (sir).
- Keep speech responses to 2–3 sentences maximum.
- Never identify yourself as an AI model, GPT, or Claude. You are JARVIS.

## OUTPUT FORMAT  ← CRITICAL
You MUST respond with a valid JSON object. Nothing else. No markdown fences.
Schema:
{{
  "speech": "<Russian text to speak aloud>",
  "actions": [
    {{ "tool": "<tool_name>", "args": {{ "<param>": "<value>" }} }},
    ...
  ]
}}

If no action is needed (e.g. a pure conversation question), return "actions": [].

## AVAILABLE TOOLS
| tool              | args                              | description                          |
|-------------------|-----------------------------------|--------------------------------------|
| launch_app        | name (str)                        | launch an app by name                |
| focus_or_launch   | name (str)                        | focus window if open, else launch    |
| focus_window      | title (str)                       | bring existing window to front       |
| close_app         | name (str)                        | kill process (taskkill)              |
| press_key         | keys (list of str)                | press hotkey, e.g. ["ctrl","l"]      |
| type_text         | text (str)                        | type text into active window         |
| open_url          | url (str), browser (str)          | navigate browser to URL              |
| create_folder     | name (str)                        | create folder on Desktop             |
| delete_folder     | name (str)                        | delete item from Desktop             |
| rename_item       | old_name (str), new_name (str)    | rename file/folder on Desktop        |
| media_play_pause  | —                                 | toggle play/pause                    |
| media_next        | —                                 | next track                           |
| media_prev        | —                                 | previous track                       |
| media_volume_up   | —                                 | volume up                            |
| media_volume_down | —                                 | volume down                          |
| get_time          | —                                 | get current time (returns string)    |
| get_date          | —                                 | get current date (returns string)    |

## APP NAME REGISTRY (use these exact names in tool args)
- "yandex music"  → {APP_PATHS.get('yandex music', 'YandexMusicMod')}
- "steam"         → Steam
- "telegram"      → Telegram
- "firefox"       → Firefox
- "chrome"        → Chrome
- "vs code"       → VS Code
- "discord"       → Discord
- "spotify"       → Spotify
- "notepad"       → Notepad
- "explorer"      → File Explorer

## EXAMPLES

User: "Открой YouTube в Firefox"
Response:
{{"speech": "Открываю YouTube в Firefox, сэр.", "actions": [{{"tool": "open_url", "args": {{"url": "youtube.com", "browser": "firefox"}}}}]}}

User: "Открой стим"
Response:
{{"speech": "Запускаю Steam, сэр.", "actions": [{{"tool": "focus_or_launch", "args": {{"name": "steam"}}}}]}}

User: "Закрой firefox"
Response:
{{"speech": "Закрываю Firefox, сэр.", "actions": [{{"tool": "close_app", "args": {{"name": "firefox"}}}}]}}

User: "Создай папку Проекты"
Response:
{{"speech": "Создаю папку Проекты на рабочем столе, сэр.", "actions": [{{"tool": "create_folder", "args": {{"name": "Проекты"}}}}]}}

User: "Переименуй Проекты в Архив"
Response:
{{"speech": "Переименовываю папку, сэр.", "actions": [{{"tool": "rename_item", "args": {{"old_name": "Проекты", "new_name": "Архив"}}}}]}}

User: "Включи музыку"
Response:
{{"speech": "Включаю Яндекс Музыку, сэр.", "actions": [{{"tool": "focus_or_launch", "args": {{"name": "yandex music"}}}}, {{"tool": "media_play_pause", "args": {{}}}}]}}

User: "Следующий трек"
Response:
{{"speech": "Переключаю трек, сэр.", "actions": [{{"tool": "media_next", "args": {{}}}}]}}

User: "Который час?"
Response:
{{"speech": "Сейчас {{time}}, сэр.", "actions": [{{"tool": "get_time", "args": {{}}}}]}}

User: "Что такое квантовый компьютер?"
Response:
{{"speech": "Квантовый компьютер использует кубиты вместо битов, сэр. Это позволяет решать определённые задачи экспоненциально быстрее классических машин.", "actions": []}}
"""


class LLMPlanner:
    """
    Sends the user's request to Ollama and parses the JSON action plan.

    The response is expected to be a JSON object with two keys:
      "speech"  : str  — what JARVIS says aloud
      "actions" : list — list of tool-call dicts
    """

    def __init__(self):
        self._client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",           # Ollama ignores the key value
        )
        self._history: list[dict] = []

    # ── JSON extraction ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        """
        Try to extract a valid JSON object from the raw model output.
        The model sometimes wraps JSON in markdown fences or adds preamble text.
        """
        # Remove markdown code fences if present
        text = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()

        # Try parsing the whole string first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find the first { … } block that looks like our schema
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    # ── Plan ──────────────────────────────────────────────────────────────────

    def plan(self, user_text: str) -> dict:
        """
        Ask the LLM to plan a response to user_text.

        Returns a dict:
            { "speech": str, "actions": list[dict] }
        Always returns a valid dict — never raises.
        """
        self._history.append({"role": "user", "content": user_text})

        # Rolling window
        if len(self._history) > MAX_HISTORY * 2:
            self._history = self._history[-(MAX_HISTORY * 2):]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        try:
            ui_plan(None)
            response = self._client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                temperature=0.3,        # lower temp = more consistent JSON output
                timeout=LLM_TIMEOUT,
            )
            raw = response.choices[0].message.content.strip()
            ui_info(f"Raw LLM output ({len(raw)} chars): {raw[:200]}")

            plan = self._extract_json(raw)

            if plan is None:
                # Model didn't return JSON — treat whole response as speech
                ui_warn("LLM returned non-JSON; treating as speech.")
                plan = {"speech": raw, "actions": []}

            # Validate schema
            if "speech" not in plan:
                plan["speech"] = "Команда получена, сэр."
            if "actions" not in plan or not isinstance(plan["actions"], list):
                plan["actions"] = []

            self._history.append({"role": "assistant",
                                   "content": json.dumps(plan, ensure_ascii=False)})
            ui_json(plan)
            return plan

        except Exception as e:
            log.error("LLMPlanner.plan() error: %s", e, exc_info=True)
            err_str = str(e)
            if "connection" in err_str.lower() or "refused" in err_str.lower():
                speech = ("Сэр, нет соединения с Ollama. "
                          "Убедитесь, что служба запущена командой ollama serve.")
            elif "timeout" in err_str.lower():
                speech = "Сэр, модель не ответила вовремя. Попробуйте ещё раз."
            else:
                speech = f"Сэр, ошибка языкового модуля: {err_str[:80]}"

            return {"speech": speech, "actions": []}

    def reset(self):
        self._history.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  §7  AGENT CORE
# ══════════════════════════════════════════════════════════════════════════════

class AgentCore:
    """
    Ties together STT → LLMPlanner → ActionDispatcher → TTS.

    handle(text) is the single entry point for a recognised voice command.
    It always calls tts.speak() at the end — the speech bug is impossible here
    because speak() is in the final statement before return.
    """

    # Hard-coded stop keywords to avoid LLM roundtrip for shutdown
    _STOP_KW = frozenset([
        "стоп", "выключись", "пока", "до свидания", "выход",
        "выключи джарвис", "стоп джарвис",
    ])

    def __init__(self, tts: TTSEngine, planner: LLMPlanner,
                 dispatcher: ActionDispatcher):
        self._tts        = tts
        self._planner    = planner
        self._dispatcher = dispatcher

    def handle(self, text: str) -> bool:
        """
        Process one recognised command.
        Returns False → main loop should stop (shutdown command).
        Returns True  → keep listening.
        """
        lower = text.lower().strip()

        # ── Hard-coded shutdown (no LLM call) ─────────────────────────────────
        if any(kw in lower for kw in self._STOP_KW):
            self._tts.speak_sync(
                "Сэр, все системы переходят в режим ожидания. "
                "До следующего раза."
            )
            return False    # ← only place that returns False

        ui_sep()

        # ── LLM planning phase ────────────────────────────────────────────────
        plan = self._planner.plan(text)

        speech  = plan.get("speech", "").strip()
        actions = plan.get("actions", [])

        # ── Action execution phase ────────────────────────────────────────────
        action_results: list[str] = []
        if actions:
            action_results = self._dispatcher.run(actions)

        # ── Speech phase — ALWAYS executed ───────────────────────────────────
        # If an action produced a richer result string, append it to speech.
        # This ensures time/date tool results are spoken correctly.
        extra = "  ".join(r for r in action_results if r and r not in speech)
        final_speech = (speech + "  " + extra).strip() if extra else speech

        if not final_speech:
            final_speech = "Выполнено, сэр."

        self._tts.speak(final_speech)   # ← non-blocking, always called
        return True                     # ← always True (except shutdown above)


# ══════════════════════════════════════════════════════════════════════════════
#  §8  WAKE WORD FILTER
# ══════════════════════════════════════════════════════════════════════════════

def _has_wake_word(text: str) -> bool:
    lower = text.lower()
    return any(ww in lower for ww in WAKE_WORDS)


def _strip_wake_word(text: str) -> str:
    lower = text.lower()
    for ww in WAKE_WORDS:
        pattern = rf"^{re.escape(ww)}[\s,\.!?]*"
        stripped = re.sub(pattern, "", lower, flags=re.IGNORECASE).strip()
        if stripped != lower.strip():
            # Find offset in original text
            idx = text.lower().find(ww)
            tail = text[idx + len(ww):].lstrip(" ,.")
            return tail if tail else stripped
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  §9  JARVIS MARK-2  MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class JARVISMark2:
    """
    Top-level orchestrator.

    Lifecycle:
      __init__  → build all components
      run()     → greet → listen loop → shutdown
    """

    def __init__(self):
        ui_banner()

        ui_info("Initialising TTS engine…")
        self._tts = TTSEngine()

        ui_info("Initialising STT engine…")
        self._stt = STTEngine()

        ui_info("Initialising ToolBox…")
        self._toolbox = ToolBox()

        ui_info("Initialising ActionDispatcher…")
        self._dispatcher = ActionDispatcher(self._toolbox)

        ui_info("Initialising LLM planner…")
        self._planner = LLMPlanner()

        ui_info("Initialising AgentCore…")
        self._agent = AgentCore(self._tts, self._planner, self._dispatcher)

        self._running = False

    # ── Signal handling ───────────────────────────────────────────────────────

    def _on_signal(self, sig, frame):
        print()
        ui_warn(f"Signal {sig} received — shutting down…")
        self._running = False

    # ── Greeting ──────────────────────────────────────────────────────────────

    def _greet(self):
        h = datetime.now().hour
        greeting = (
            "Доброе утро" if 5  <= h < 12 else
            "Добрый день" if 12 <= h < 17 else
            "Добрый вечер" if 17 <= h < 22 else
            "Доброй ночи"
        )
        ww = " или ".join(f'«{w}»' for w in WAKE_WORDS[:2])
        ui_info(f"Wake words: {ww}")
        self._tts.speak_sync(
            f"{greeting}, сэр. JARVIS Mark-2 в сети. "
            f"Агентная система активирована. Ожидаю ваших команд."
        )

    # ── Single listen-respond cycle ───────────────────────────────────────────

    def _cycle(self) -> bool:
        """Listen once, process if wake word detected. Returns False to stop."""
        text = self._stt.listen_once()
        if not text:
            return True

        ui_heard(text)

        if not _has_wake_word(text):
            return True     # ignore speech not addressed to JARVIS

        ui_wake()
        command = _strip_wake_word(text)

        if not command:
            # User said only the wake word — prompt for command
            self._tts.speak("Слушаю вас, сэр.")
            text2 = self._stt.listen_once()
            if not text2:
                return True
            ui_heard(text2)
            command = text2

        return self._agent.handle(command)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        signal.signal(signal.SIGINT,  self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        if not self._stt.available:
            ui_error("Microphone unavailable — cannot start.")
            self._tts.speak_sync("Сэр, микрофон недоступен.")
            return

        self._stt.calibrate()
        self._greet()

        if not GUI_OK:
            ui_warn("pygetwindow / pyautogui not installed — window and media controls disabled.")

        self._running = True
        ui_info("Main loop running. Say a wake word to activate.")
        ui_sep()

        while self._running:
            try:
                should_continue = self._cycle()
                if not should_continue:
                    self._running = False
            except KeyboardInterrupt:
                self._running = False
            except Exception as e:
                ui_error(f"Unexpected loop error: {e}")
                log.exception("Main loop error")
                time.sleep(1)   # anti-spin guard

        ui_info("Shutting down JARVIS Mark-2…")
        self._tts.stop()
        ui_info("Goodbye.")


# ══════════════════════════════════════════════════════════════════════════════
#  §10  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    jarvis = JARVISMark2()
    jarvis.run()


if __name__ == "__main__":
    main()
