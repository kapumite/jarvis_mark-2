"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              J.A.R.V.I.S  Mark-3.1  —  Action-Based Agent System            ║
║         Local LLM via Ollama · Windows 10/11 · Voice + Hotkey + Text        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Changelog 3.0 → 3.1:
  FIX  ToolBox decorator crash (module-level _TOOL_REGISTRY)
  FIX  _is_shutdown matched "выход" inside arbitrary sentences
  FIX  _confirm_destructive false-positive on "да" inside other words
  FIX  _extract_json non-greedy regex broke nested JSON
  FIX  text-mode required wake word
  FIX  _resolve_app_path static method ignored lowercased apps dict
  FIX  focus_or_launch could not match Russian window titles
  FIX  open_url spoke full URL with scheme; allowed file://
  FIX  volume_up/down unbounded
  FIX  _press_vk did not gate on sys.platform
  FIX  TTSEngine hung if worker died
  FIX  text-mode EOF did not exit
  FIX  close_app did not verify process actually died
  FIX  _has_wake_word triggered on substrings
  NEW  _APP_ALIASES for Russian / alias window titles
  NEW  --reset-config, --list-tools, --no-hotkey, --version
  NEW  tools: get_battery, search_files, show_desktop, remember, recall, sleep_pc
  NEW  TTS-friendly URL display
  NEW  30s timeout on speak_sync
"""

import os
import re
import sys
import json
import time
import copy
import queue
import signal
import ctypes
import logging
import argparse
import threading
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

# ── Hard dependencies ────────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    import pyttsx3
    from openai import OpenAI
    from colorama import Fore, Style, init as colorama_init
except ImportError as exc:
    print(f"\n[FATAL] Missing package: {exc}")
    print("pip install openai pyttsx3 SpeechRecognition pyaudio colorama\n")
    sys.exit(1)

colorama_init(autoreset=True)

# ── Soft dependencies ────────────────────────────────────────────────────────
try:
    import pygetwindow as gw
    import pyautogui
    GUI_OK = True
except ImportError:
    GUI_OK = False

try:
    import pyperclip
    PYPERCLIP_OK = True
except ImportError:
    PYPERCLIP_OK = False

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

try:
    import winsound
    WINSOUND_OK = True
except ImportError:
    WINSOUND_OK = False

try:
    from pynput import keyboard as pynput_kb
    PYNPUT_OK = True
except ImportError:
    PYNPUT_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  §0  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

_HOME  = os.path.expanduser("~")
_LOCAL = os.environ.get("LOCALAPPDATA", os.path.join(_HOME, "AppData", "Local"))
_ROAM  = os.environ.get("APPDATA",      os.path.join(_HOME, "AppData", "Roaming"))
_PF    = os.environ.get("ProgramFiles",     r"C:\Program Files")
_PF86  = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

DEFAULT_APPS: dict[str, str] = {
    "yandex music":  os.path.join(_LOCAL, "Programs", "YandexMusicMod", "Яндекс Музыка.exe"),
    "яндекс музыка": os.path.join(_LOCAL, "Programs", "YandexMusicMod", "Яндекс Музыка.exe"),
    "steam":         os.path.join(_PF86, "Steam", "steam.exe"),
    "telegram":      os.path.join(_ROAM, "Telegram Desktop", "Telegram.exe"),
    "firefox":       os.path.join(_PF, "Mozilla Firefox", "firefox.exe"),
    "chrome":        os.path.join(_PF, "Google", "Chrome", "Application", "chrome.exe"),
    "vs code":       os.path.join(_LOCAL, "Programs", "Microsoft VS Code", "Code.exe"),
    "discord":       "discord:",
    "spotify":       os.path.join(_ROAM, "Spotify", "Spotify.exe"),
    "notepad":       "notepad.exe",
    "explorer":      "explorer.exe",
    "calculator":    "calc.exe",
}

# Window-title aliases — used by focus_or_launch when searching for open windows
_APP_ALIASES: dict[str, list[str]] = {
    "yandex music":  ["Яндекс Музыка", "Yandex Music", "YandexMusic"],
    "яндекс музыка": ["Яндекс Музыка", "Yandex Music", "YandexMusic"],
    "steam":         ["Steam"],
    "telegram":      ["Telegram"],
    "firefox":       ["Firefox"],
    "chrome":        ["Chrome", "Google Chrome"],
    "vs code":       ["Visual Studio Code", "VS Code"],
    "discord":       ["Discord"],
    "spotify":       ["Spotify"],
    "notepad":       ["Notepad", "Блокнот"],
    "calculator":    ["Calculator", "Калькулятор"],
}

DEFAULT_CONFIG: dict = {
    "ollama": {
        "base_url":    "http://localhost:11434/v1",
        "model":       "qwen2.5:7b",
        "timeout":     45,
        "max_history": 12,
    },
    "wake_words": ["джарвис", "jarvis"],
    "stt": {
        "lang":             "ru-RU",
        "energy_threshold": 300,
        "pause_threshold":  0.85,
        "phrase_limit":     10.0,
    },
    "tts": {
        "rate":       165,
        "volume":     1.0,
        "voice_hint": "ru",
        "max_wait":   30.0,
    },
    "hotkey":              "ctrl+alt+j",
    "hotkey_bypass_secs":  8.0,
    "failsafe":            True,
    "confirm_destructive": True,
    "confirm_timeout":     15.0,
    "paths": {
        "desktop": os.path.join(_HOME, "Desktop"),
        "apps":    DEFAULT_APPS,
    },
    "logging": {
        "file":      "jarvis.log",
        "level":     "WARNING",
        "max_bytes": 5_242_880,
        "backups":   3,
    },
}

_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
_MEM_PATH = os.path.join(os.path.dirname(_CFG_PATH), "memory.json")
CFG: dict = {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(explicit_path: Optional[str] = None) -> dict:
    path = explicit_path or _CFG_PATH
    if not os.path.exists(path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
            print(Fore.YELLOW + f"[CFG] Created default config: {path}")
        except Exception as e:
            print(Fore.RED + f"[CFG] Cannot create default config: {e}")
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        with open(path, encoding="utf-8") as f:
            user_cfg = json.load(f)
        return _deep_merge(DEFAULT_CONFIG, user_cfg)
    except Exception as e:
        print(Fore.RED + f"[CFG] Load error: {e}. Using defaults.")
        return copy.deepcopy(DEFAULT_CONFIG)


# ══════════════════════════════════════════════════════════════════════════════
#  §1  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

log = logging.getLogger("JARVIS")


def setup_logging(cfg: dict) -> None:
    lc = cfg.get("logging", {})
    log.setLevel(getattr(logging, str(lc.get("level", "WARNING")).upper(), logging.WARNING))
    for h in list(log.handlers):
        log.removeHandler(h)

    try:
        fh = RotatingFileHandler(
            lc.get("file", "jarvis.log"),
            maxBytes=lc.get("max_bytes", 5_242_880),
            backupCount=lc.get("backups", 3),
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        log.addHandler(fh)
    except Exception as e:
        print(Fore.RED + f"[LOG] Cannot open log file: {e}")

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(ch)


# ══════════════════════════════════════════════════════════════════════════════
#  §2  TERMINAL UI
# ══════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def ui_banner():
    w = 72
    print(Fore.CYAN + Style.BRIGHT + "═" * w)
    print(Fore.CYAN + Style.BRIGHT + "  J.A.R.V.I.S  Mark-3.1  —  Action-Based Agent")
    print(Fore.CYAN + Style.DIM    + f"  Model: {CFG['ollama']['model']}  |  Endpoint: {CFG['ollama']['base_url']}")
    print(Fore.CYAN + Style.BRIGHT + "═" * w)


def ui_listen():  print(Fore.GREEN  + Style.BRIGHT + f"\n[{_ts()}] 🎤  Слушаю…")
def ui_wake():    print(Fore.YELLOW + Style.BRIGHT + f"[{_ts()}] ✨  Активирован!")
def ui_heard(t):  print(Fore.WHITE  + f"[{_ts()}] 👂  Распознано: " + Style.BRIGHT + f'"{t}"')
def ui_plan():    print(Fore.MAGENTA+ f"[{_ts()}] 🧠  Планирование…")
def ui_action(a): print(Fore.BLUE   + Style.BRIGHT + f"[{_ts()}] ⚡  {a}")
def ui_speak(t):
    d = t if len(t) <= 110 else t[:107] + "…"
    print(Fore.CYAN + Style.BRIGHT + f"[{_ts()}] 🔊  JARVIS: " + Style.NORMAL + d)
def ui_warn(m):   print(Fore.YELLOW + f"[{_ts()}] ⚠   {m}")
def ui_error(m):  print(Fore.RED    + Style.BRIGHT + f"[{_ts()}] ✖   {m}")
def ui_info(m):   print(Fore.WHITE  + Style.DIM    + f"[{_ts()}] ℹ   {m}")
def ui_sep():     print(Fore.CYAN   + Style.DIM    + "─" * 72)
def ui_json(d):   print(Fore.MAGENTA+ Style.DIM    + json.dumps(d, ensure_ascii=False, indent=2))


def beep(freq: int = 880, dur: int = 100) -> None:
    if WINSOUND_OK:
        try:
            winsound.Beep(freq, dur)
        except Exception:
            pass


class UserExit(Exception):
    """Raised when user exits via EOF/Ctrl+C in text mode."""


# ══════════════════════════════════════════════════════════════════════════════
#  §3  TOOL REGISTRY  (module-level — safe during class-body execution)
# ══════════════════════════════════════════════════════════════════════════════

_TOOL_REGISTRY: dict[str, dict] = {}


def tool(name: str, args: list[str], desc: str, destructive: bool = False):
    def deco(fn):
        _TOOL_REGISTRY[name] = {
            "args": args, "desc": desc, "destructive": destructive,
        }
        return fn
    return deco


# ══════════════════════════════════════════════════════════════════════════════
#  §4  TTS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class TTSEngine:
    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._dead    = False
        self._q: queue.Queue[Optional[str]] = queue.Queue()
        if enabled:
            threading.Thread(target=self._worker, daemon=True, name="TTS-Worker").start()

    def _worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate",   CFG["tts"]["rate"])
            engine.setProperty("volume", CFG["tts"]["volume"])

            hint = str(CFG["tts"].get("voice_hint", "ru")).lower()
            for v in engine.getProperty("voices") or []:
                langs = v.languages or []
                if any(hint in str(l).lower() for l in langs) or hint in v.name.lower():
                    engine.setProperty("voice", v.id)
                    ui_info(f"TTS voice: {v.name}")
                    break
            else:
                ui_warn("Russian voice not found — using system default.")

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
            self._dead = True

    def speak(self, text: str) -> None:
        ui_speak(text)
        if self._enabled and not self._dead:
            self._q.put(text)

    def speak_sync(self, text: str, timeout: Optional[float] = None) -> None:
        self.speak(text)
        if not self._enabled or self._dead:
            return
        # Bounded wait — never hang forever on a dead/broken worker
        deadline = time.time() + (timeout if timeout is not None else CFG["tts"]["max_wait"])
        while self._q.unfinished_tasks > 0 and time.time() < deadline:
            time.sleep(0.05)

    def stop(self) -> None:
        if self._enabled and not self._dead:
            self._q.put(None)


# ══════════════════════════════════════════════════════════════════════════════
#  §5  STT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class STTEngine:
    def __init__(self, text_mode: bool = False):
        self._text_mode = text_mode
        self._mic: Optional[sr.Microphone] = None
        self.rec = sr.Recognizer()
        stt = CFG["stt"]
        self.rec.energy_threshold         = stt["energy_threshold"]
        self.rec.pause_threshold          = stt["pause_threshold"]
        self.rec.dynamic_energy_threshold = True

        if text_mode:
            ui_info("Text input mode active — no microphone.")
            return
        try:
            self._mic = sr.Microphone()
            ui_info("Microphone initialised.")
        except OSError as e:
            ui_error(f"Microphone not found: {e}")

    @property
    def available(self) -> bool:
        return self._text_mode or self._mic is not None

    @property
    def is_text_mode(self) -> bool:
        return self._text_mode

    def calibrate(self) -> None:
        if self._text_mode or not self._mic:
            return
        ui_info("Calibrating ambient noise…")
        try:
            with self._mic as src:
                self.rec.adjust_for_ambient_noise(src, duration=1.2)
            ui_info(f"Energy threshold: {self.rec.energy_threshold:.0f}")
        except Exception as e:
            ui_warn(f"Calibration failed: {e}")

    def listen_once(self) -> Optional[str]:
        if self._text_mode:
            try:
                val = input(Fore.GREEN + Style.BRIGHT + f"\n[{_ts()}] ⌨  > " + Style.RESET_ALL)
            except (EOFError, KeyboardInterrupt):
                raise UserExit()
            val = val.strip()
            return val or None

        if not self._mic:
            return None
        try:
            with self._mic as src:
                ui_listen()
                audio = self.rec.listen(src, timeout=None,
                                        phrase_time_limit=CFG["stt"]["phrase_limit"])
            return self.rec.recognize_google(audio, language=CFG["stt"]["lang"]).strip()
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


# ══════════════════════════════════════════════════════════════════════════════
#  §6  WIN32 MEDIA KEYS
# ══════════════════════════════════════════════════════════════════════════════

_VK = {
    "playpause":  0xB3,
    "nexttrack":  0xB0,
    "prevtrack":  0xB1,
    "volumeup":   0xAF,
    "volumedown": 0xAE,
    "volumemute": 0xAD,
}


def _press_vk(vk: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.03)
        user32.keybd_event(vk, 0, 2, 0)
        return True
    except Exception as e:
        log.warning("press_vk failed: %s", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  §7  MEMORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_memory() -> dict:
    try:
        with open(_MEM_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_memory(data: dict) -> None:
    try:
        with open(_MEM_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("save_memory failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  §8  TOOLBOX
# ══════════════════════════════════════════════════════════════════════════════

class ToolBox:
    def __init__(self, tts: TTSEngine):
        self._tts     = tts
        self._desktop = CFG["paths"]["desktop"]
        # lowercase app keys for lookup
        self._apps    = {k.lower().strip(): v for k, v in CFG["paths"]["apps"].items()}

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _resolve_app_path(self, name: str) -> Optional[str]:
        return self._apps.get(name.lower().strip())

    @staticmethod
    def _find_windows(fragment: str) -> list:
        if not GUI_OK:
            return []
        try:
            frag = fragment.lower()
            return [w for w in gw.getAllWindows()
                    if frag in w.title.lower() and w.title.strip()]
        except Exception:
            return []

    @staticmethod
    def _popen(path: str, *extra: str) -> bool:
        if path.endswith(":") or "://" in path:
            try:
                os.startfile(path)
                return True
            except Exception as e:
                ui_error(f"startfile error: {e}")
                return False
        cmd = [path, *extra]
        ui_action(f"Launching: {' '.join(cmd)}")
        try:
            subprocess.Popen(cmd,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                             close_fds=True)
            return True
        except FileNotFoundError:
            ui_error(f"Executable not found: {path}")
            return False
        except Exception as e:
            ui_error(f"Launch error: {e}")
            return False

    @staticmethod
    def _normalize_keys(keys: Any) -> list[str]:
        if isinstance(keys, str):
            return [k.strip() for k in re.split(r"[+,]", keys) if k.strip()]
        if isinstance(keys, (list, tuple)):
            return [str(k).strip() for k in keys if str(k).strip()]
        return []

    @staticmethod
    def _process_running(exe: str) -> bool:
        try:
            r = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {exe}"],
                capture_output=True, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return exe.lower() in r.stdout.lower()
        except Exception:
            return False

    # ── Launch / focus / close ───────────────────────────────────────────────

    @tool("launch_app", ["name"], "launch an application by registry name")
    def launch_app(self, name: str) -> str:
        path = self._resolve_app_path(name)
        if path is None:
            if self._popen(name):
                return f"Запускаю {name}."
            return f"Приложение «{name}» не найдено."
        if self._popen(path):
            return f"Запускаю {name}."
        return f"Не удалось запустить {name}."

    @tool("focus_or_launch", ["name"], "focus window if open, else launch app")
    def focus_or_launch(self, name: str) -> str:
        # Try aliases first — Russian titles / alternate names
        for alias in _APP_ALIASES.get(name.lower().strip(), [name]):
            if self._find_windows(alias):
                return self.focus_window(alias)
        return self.launch_app(name)

    @tool("focus_window", ["title"], "bring existing window to front by title fragment")
    def focus_window(self, title: str) -> str:
        if not GUI_OK:
            return "Модуль управления окнами недоступен."
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
            return f"Не удалось переключиться: {e}"

    @tool("close_app", ["name"], "close an app gracefully, force if needed", destructive=True)
    def close_app(self, name: str) -> str:
        exe = name if name.lower().endswith(".exe") else name + ".exe"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # graceful
        subprocess.run(["taskkill", "/IM", exe],
                       capture_output=True, text=True, creationflags=flags)
        time.sleep(0.6)
        if not self._process_running(exe):
            return f"{name} закрыт."
        # force
        subprocess.run(["taskkill", "/F", "/IM", exe],
                       capture_output=True, text=True, creationflags=flags)
        time.sleep(0.3)
        if not self._process_running(exe):
            return f"{name} закрыт принудительно."
        return f"Не удалось закрыть {name}."

    # ── Keyboard / text ──────────────────────────────────────────────────────

    @tool("press_key", ["keys"], "press hotkey, e.g. ['ctrl','l'] or 'ctrl+l'")
    def press_key(self, *keys) -> str:
        if not GUI_OK:
            return "Модуль ввода клавиш недоступен."
        if len(keys) == 1:
            keys = tuple(self._normalize_keys(keys[0]))
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
            return f"Ошибка нажатия: {e}"

    @tool("type_text", ["text"], "type text into the currently active window")
    def type_text(self, text: str) -> str:
        if not GUI_OK:
            return "Модуль ввода недоступен."
        ui_action(f"Typing: {text[:60]}")
        try:
            if PYPERCLIP_OK:
                pyperclip.copy(text)
                pyautogui.hotkey("ctrl", "v")
            else:
                try:
                    pyautogui.typewrite(text, interval=0.02)
                except Exception:
                    return "Нужен pyperclip для кириллицы (pip install pyperclip)."
            time.sleep(0.1)
            return f"Введено: {text[:50]}"
        except Exception as e:
            return f"Ошибка ввода: {e}"

    # ── URL / web ────────────────────────────────────────────────────────────

    @tool("open_url", ["url", "browser"], "open URL; browser='default'|'firefox'|'chrome'")
    def open_url(self, url: str, browser: str = "default") -> str:
        # Reject non-http(s) schemes — no file://, no javascript:
        if url.lower().startswith(("file:", "javascript:", "data:")):
            return f"Схема URL запрещена: {url[:30]}…"
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        browser = (browser or "default").strip().lower()
        if browser not in ("", "default"):
            path = self._resolve_app_path(browser)
            if path and self._popen(path, url):
                speak_url = re.sub(r"^https?://", "", url).rstrip("/")
                return f"Открываю {speak_url} в {browser}."
        try:
            os.startfile(url)
            speak_url = re.sub(r"^https?://", "", url).rstrip("/")
            return f"Открываю {speak_url}."
        except Exception as e:
            return f"Не удалось открыть URL: {e}"

    @tool("play_youtube", ["query"], "search YouTube and open in browser")
    def play_youtube(self, query: str) -> str:
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
        try:
            os.startfile(url)
            return f"Открываю YouTube: {query}."
        except Exception as e:
            return f"Ошибка: {e}"

    @tool("search_web", ["query"], "search the web via DuckDuckGo")
    def search_web(self, query: str) -> str:
        url = "https://duckduckgo.com/?q=" + urllib.parse.quote(query)
        try:
            os.startfile(url)
            return f"Ищу: {query}."
        except Exception as e:
            return f"Ошибка поиска: {e}"

    # ── Files ────────────────────────────────────────────────────────────────

    @tool("create_folder", ["name"], "create folder on Desktop")
    def create_folder(self, name: str) -> str:
        safe = re.sub(r'[<>:"/\\|?*]', "", name).strip()
        if not safe:
            return "Недопустимое имя папки."
        path = os.path.join(self._desktop, safe)
        if os.path.exists(path):
            return f"Папка «{safe}» уже существует."
        try:
            os.makedirs(path, exist_ok=True)
            return f"Папка «{safe}» создана."
        except Exception as e:
            return f"Ошибка: {e}"

    @tool("delete_folder", ["name"], "send file/folder from Desktop to Recycle Bin", destructive=True)
    def delete_folder(self, name: str) -> str:
        safe = re.sub(r'[<>:"/\\|?*]', "", name).strip()
        path = os.path.join(self._desktop, safe)
        if not os.path.exists(path):
            return f"«{safe}» не найден на рабочем столе."
        try:
            is_dir = os.path.isdir(path)
            method = "DeleteDirectory" if is_dir else "DeleteFile"
            ps = (
                f'Add-Type -AssemblyName Microsoft.VisualBasic; '
                f'[Microsoft.VisualBasic.FileIO.FileSystem]::{method}('
                f'"{path}", "OnlyErrorDialogs", "SendToRecycleBin")'
            )
            subprocess.run(["powershell", "-Command", ps],
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                           capture_output=True, check=True)
            return f"«{safe}» отправлен в корзину."
        except Exception as e:
            return f"Не удалось отправить в корзину: {e}. Отмена."

    @tool("rename_item", ["old_name", "new_name"], "rename file/folder on Desktop")
    def rename_item(self, old_name: str, new_name: str) -> str:
        old_path = os.path.join(self._desktop, old_name.strip())
        new_path = os.path.join(self._desktop, new_name.strip())
        if not os.path.exists(old_path):
            return f"«{old_name}» не найден."
        if os.path.exists(new_path):
            return f"«{new_name}» уже существует."
        try:
            os.rename(old_path, new_path)
            return f"«{old_name}» → «{new_name}»."
        except Exception as e:
            return f"Ошибка: {e}"

    @tool("open_path", ["path"], "open a folder or file by absolute path")
    def open_path(self, path: str) -> str:
        if not os.path.exists(path):
            return f"Путь не найден: {path}"
        try:
            os.startfile(path)
            return f"Открываю: {path}"
        except Exception as e:
            return f"Ошибка: {e}"

    @tool("write_note", ["name", "text"], "write a .txt note to Desktop")
    def write_note(self, name: str, text: str) -> str:
        safe = re.sub(r'[<>:"/\\|?*]', "", name).strip() or "note"
        if not safe.endswith(".txt"):
            safe += ".txt"
        path = os.path.join(self._desktop, safe)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return f"Заметка «{safe}» сохранена."
        except Exception as e:
            return f"Ошибка: {e}"

    @tool("search_files", ["query"], "search files on Desktop and Documents")
    def search_files(self, query: str) -> str:
        q = query.lower().strip()
        if not q:
            return "Пустой запрос."
        dirs = [self._desktop, os.path.join(os.path.expanduser("~"), "Documents")]
        results: list[str] = []
        try:
            for base in dirs:
                if not os.path.isdir(base):
                    continue
                for root, subdirs, files in os.walk(base):
                    depth = root[len(base):].count(os.sep)
                    if depth > 3:
                        subdirs.clear()
                        continue
                    for name in files + subdirs:
                        if q in name.lower():
                            results.append(name)
                            if len(results) >= 8:
                                break
                    if len(results) >= 8:
                        break
                if len(results) >= 8:
                    break
        except Exception as e:
            return f"Ошибка поиска: {e}"
        if not results:
            return f"Не найдено файлов по запросу «{query}»."
        return "Найдено: " + "; ".join(results[:5]) + ("…" if len(results) > 5 else ".")

    @tool("screenshot", [], "save a screenshot to Desktop")
    def screenshot(self) -> str:
        if not GUI_OK:
            return "Скриншот недоступен."
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self._desktop, f"shot_{ts}.png")
            pyautogui.screenshot(path)
            return f"Скриншот: {os.path.basename(path)}"
        except Exception as e:
            return f"Ошибка: {e}"

    # ── Media (Win32 VK) ─────────────────────────────────────────────────────

    @tool("media_play_pause", [], "toggle media playback")
    def media_play_pause(self) -> str:
        _press_vk(_VK["playpause"])
        return "Воспроизведение / пауза."

    @tool("media_next", [], "next media track")
    def media_next(self) -> str:
        _press_vk(_VK["nexttrack"])
        return "Следующий трек."

    @tool("media_prev", [], "previous media track")
    def media_prev(self) -> str:
        _press_vk(_VK["prevtrack"])
        return "Предыдущий трек."

    @tool("volume_up", ["steps"], "increase volume (2% per step, default 5, max 50)")
    def volume_up(self, steps=5) -> str:
        try:
            n = max(1, min(50, int(steps)))
        except (TypeError, ValueError):
            n = 5
        for _ in range(n):
            _press_vk(_VK["volumeup"])
        return f"Громкость +{n * 2}%."

    @tool("volume_down", ["steps"], "decrease volume (2% per step, default 5, max 50)")
    def volume_down(self, steps=5) -> str:
        try:
            n = max(1, min(50, int(steps)))
        except (TypeError, ValueError):
            n = 5
        for _ in range(n):
            _press_vk(_VK["volumedown"])
        return f"Громкость −{n * 2}%."

    @tool("mute_toggle", [], "toggle mute")
    def mute_toggle(self) -> str:
        _press_vk(_VK["volumemute"])
        return "Звук переключён."

    # ── Clipboard ────────────────────────────────────────────────────────────

    @tool("clipboard_get", [], "read current clipboard")
    def clipboard_get(self) -> str:
        if not PYPERCLIP_OK:
            return "Нужен pyperclip."
        val = pyperclip.paste() or ""
        return f"В буфере: {val[:200]}" if val else "Буфер пуст."

    @tool("clipboard_set", ["text"], "write text to clipboard")
    def clipboard_set(self, text: str) -> str:
        if not PYPERCLIP_OK:
            return "Нужен pyperclip."
        pyperclip.copy(text)
        return "Скопировано в буфер."

    # ── Windows / system ─────────────────────────────────────────────────────

    @tool("lock_pc", [], "lock the workstation", destructive=True)
    def lock_pc(self) -> str:
        try:
            ctypes.windll.user32.LockWorkStation()
            return "Блокирую компьютер."
        except Exception as e:
            return f"Ошибка: {e}"

    @tool("sleep_pc", [], "put the computer to sleep", destructive=True)
    def sleep_pc(self) -> str:
        try:
            subprocess.run(
                ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return "Перевожу в режим сна."
        except Exception as e:
            return f"Ошибка: {e}"

    @tool("show_desktop", [], "minimize all windows (Win+D)")
    def show_desktop(self) -> str:
        if not GUI_OK:
            return "Недоступно."
        try:
            pyautogui.hotkey("win", "d")
            return "Свернул все окна."
        except Exception as e:
            return f"Ошибка: {e}"

    @tool("system_stats", [], "report CPU/RAM/battery usage")
    def system_stats(self) -> str:
        if not PSUTIL_OK:
            return "Статистика недоступна (pip install psutil)."
        try:
            cpu = psutil.cpu_percent(interval=0.4)
            mem = psutil.virtual_memory().percent
            parts = [f"ЦП {cpu:.0f}%", f"ОЗУ {mem:.0f}%"]
            try:
                bat = psutil.sensors_battery()
                if bat:
                    parts.append(f"Батарея {bat.percent:.0f}%")
            except Exception:
                pass
            return ", ".join(parts) + "."
        except Exception as e:
            return f"Ошибка: {e}"

    @tool("get_battery", [], "report battery level and charging state")
    def get_battery(self) -> str:
        if not PSUTIL_OK:
            return "Недоступно (нужен psutil)."
        try:
            b = psutil.sensors_battery()
            if not b:
                return "Батарея не обнаружена."
            status = "заряжается" if b.power_plugged else "разряжается"
            return f"Батарея {b.percent:.0f}%, {status}."
        except Exception as e:
            return f"Ошибка: {e}"

    @tool("list_windows", [], "list titles of currently open windows")
    def list_windows(self) -> str:
        if not GUI_OK:
            return "Недоступно."
        try:
            titles = [w.title.strip() for w in gw.getAllWindows() if w.title.strip()]
        except Exception:
            return "Ошибка."
        if not titles:
            return "Окон не найдено."
        shown = titles[:10]
        suffix = "…" if len(titles) > 10 else "."
        return "Открыты: " + "; ".join(shown) + suffix

    # ── Time / reminders ─────────────────────────────────────────────────────

    @tool("get_time", [], "get current time as string")
    def get_time(self) -> str:
        n = datetime.now()
        return f"Сейчас {n.hour} часов {n.minute} минут."

    @tool("get_date", [], "get current date as string")
    def get_date(self) -> str:
        days = ["понедельник", "вторник", "среда", "четверг",
                "пятница", "суббота", "воскресенье"]
        months = ["января", "февраля", "марта", "апреля", "мая", "июня",
                  "июля", "августа", "сентября", "октября", "ноября", "декабря"]
        n = datetime.now()
        return (f"Сегодня {days[n.weekday()]}, "
                f"{n.day} {months[n.month - 1]} {n.year} года.")

    @tool("set_timer", ["seconds", "label"], "set a reminder after N seconds")
    def set_timer(self, seconds, label: str = "") -> str:
        try:
            secs = int(float(seconds))
        except (TypeError, ValueError):
            return "Неверная длительность."
        if secs <= 0:
            return "Длительность должна быть положительной."
        msg = f"Сэр, напоминание: {label}." if label else "Сэр, время вышло."
        t = threading.Timer(secs, lambda: self._tts.speak(msg))
        t.daemon = True
        t.start()
        return f"Таймер на {secs} секунд установлен."

    # ── Memory ───────────────────────────────────────────────────────────────

    @tool("remember", ["key", "value"], "remember a key-value pair persistently")
    def remember(self, key: str, value: str) -> str:
        data = _load_memory()
        data[key.lower().strip()] = value
        _save_memory(data)
        return f"Запомнил: {key} — {value}."

    @tool("recall", ["key"], "recall a remembered value")
    def recall(self, key: str) -> str:
        data = _load_memory()
        val = data.get(key.lower().strip())
        if val is None:
            return f"Не помню «{key}»."
        return f"{key}: {val}."

    # ── Weather ──────────────────────────────────────────────────────────────

    @tool("get_weather", ["city"], "get current weather for a city")
    def get_weather(self, city: str = "") -> str:
        try:
            q = urllib.parse.quote(city.strip()) if city else ""
            url = f"https://wttr.in/{q}?format=j1&lang=ru"
            with urllib.request.urlopen(url, timeout=12) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            cur = data["current_condition"][0]
            temp = cur["temp_C"]
            desc = ""
            if "lang_ru" in cur and cur["lang_ru"]:
                desc = cur["lang_ru"][0]["value"]
            elif cur.get("weatherDesc"):
                desc = cur["weatherDesc"][0]["value"]
            where = f" в {city}" if city else ""
            return f"Погода{where}: {desc}, {temp}°C."
        except Exception as e:
            return f"Не удалось получить погоду: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  §9  ACTION DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

class ActionDispatcher:
    def __init__(self, toolbox: ToolBox):
        self._tb = toolbox

    def run(self, actions: list[dict]) -> list[str]:
        results: list[str] = []
        for action in actions:
            name = (action.get("tool") or "").strip()
            args = action.get("args") or {}

            if name not in _TOOL_REGISTRY:
                ui_warn(f"Unknown tool skipped: '{name}'")
                continue

            meta = _TOOL_REGISTRY[name]
            method = getattr(self._tb, name, None)
            if method is None:
                ui_warn(f"Tool '{name}' registered but no method found.")
                continue

            ui_action(f"Tool: {name}  args: {args}")

            try:
                pos_args: list[Any] = []
                for key in meta["args"]:
                    if key not in args:
                        break
                    val = args[key]
                    if key == "keys":
                        pos_args.extend(ToolBox._normalize_keys(val))
                    else:
                        pos_args.append(val)
                result = method(*pos_args)
                if result:
                    results.append(str(result))
            except Exception as e:
                msg = f"Ошибка {name}: {e}"
                ui_error(msg)
                log.exception("Dispatcher error in '%s'", name)
                results.append(msg)

            time.sleep(0.15)

        return results


# ══════════════════════════════════════════════════════════════════════════════
#  §10  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt() -> str:
    rows = ["| tool | args | description |", "|---|---|---|"]
    for name, meta in _TOOL_REGISTRY.items():
        args = ", ".join(meta["args"]) if meta["args"] else "—"
        rows.append(f"| {name} | {args} | {meta['desc']} |")
    tools_table = "\n".join(rows)

    app_names = ", ".join(f'"{n}"' for n in CFG["paths"]["apps"].keys())

    return f"""You are JARVIS, a personal AI assistant on Windows.
You speak ONLY in Russian, but you think in English when planning.

## IDENTITY
- Witty, elegant, slightly sarcastic — like JARVIS from Iron Man.
- Address the user as "сэр".
- Speech responses: 1–3 sentences. No filler.
- Never say you are an AI, GPT, or Claude. You are JARVIS.

## OUTPUT FORMAT  ← CRITICAL
Respond with a valid JSON object and nothing else. No markdown fences.
Schema:
{{
  "speech": "<Russian text to speak>",
  "actions": [ {{ "tool": "<name>", "args": {{ "<param>": "<value>" }} }} ]
}}
If no action is needed (pure conversation), return "actions": [].

## TOOLS
{tools_table}

## APP NAMES (exact strings for launch_app / focus_or_launch / close_app)
{app_names}

## RULES
- "launch_app" for a fresh start; "focus_or_launch" if the user likely already has it open.
- "close_app" takes the bare name (e.g. "firefox").
- "open_url" browser arg: "default", "firefox", or "chrome".
- "get_time" / "get_date" return a string — the tool result will be spoken, so
  DO NOT duplicate the value in "speech"; just say "Сейчас, сэр." and let the tool speak.
- "get_weather" city is optional; empty string = auto-detect.
- For destructive actions (delete_folder, close_app, lock_pc, sleep_pc) — the system
  will ask the user for confirmation, so do not add extra warnings in speech.
- "press_key": pass a list like ["ctrl","l"] OR a string "ctrl+l".
- "set_timer": seconds as integer, label as optional string.
- "remember" / "recall": persist small facts between sessions.
- Never invent tools not in the table.
- Never include code, markdown, or commentary outside the JSON.

## EXAMPLES

User: "Открой YouTube в Firefox"
{{"speech":"Открываю YouTube, сэр.","actions":[{{"tool":"open_url","args":{{"url":"youtube.com","browser":"firefox"}}}}]}}

User: "Открой стим"
{{"speech":"Запускаю Steam, сэр.","actions":[{{"tool":"focus_or_launch","args":{{"name":"steam"}}}}]}}

User: "Закрой firefox"
{{"speech":"Закрываю Firefox, сэр.","actions":[{{"tool":"close_app","args":{{"name":"firefox"}}}}]}}

User: "Создай папку Проекты"
{{"speech":"Создаю папку Проекты, сэр.","actions":[{{"tool":"create_folder","args":{{"name":"Проекты"}}}}]}}

User: "Который час?"
{{"speech":"Сейчас, сэр.","actions":[{{"tool":"get_time","args":{{}}}}]}}

User: "Погода в Москве"
{{"speech":"Сейчас узнаю, сэр.","actions":[{{"tool":"get_weather","args":{{"city":"Москва"}}}}]}}

User: "Поставь таймер на 10 минут на чай"
{{"speech":"Таймер на десять минут установлен, сэр.","actions":[{{"tool":"set_timer","args":{{"seconds":600,"label":"чай"}}}}]}}

User: "Следующий трек"
{{"speech":"Переключаю, сэр.","actions":[{{"tool":"media_next","args":{{}}}}]}}

User: "Найди в интернете рецепт борща"
{{"speech":"Ищу, сэр.","actions":[{{"tool":"search_web","args":{{"query":"рецепт борща"}}}}]}}

User: "Запомни что мой день рождения 12 мая"
{{"speech":"Запомнил, сэр.","actions":[{{"tool":"remember","args":{{"key":"день рождения","value":"12 мая"}}}}]}}

User: "Когда мой день рождения?"
{{"speech":"Проверяю, сэр.","actions":[{{"tool":"recall","args":{{"key":"день рождения"}}}}]}}

User: "Что такое квантовый компьютер?"
{{"speech":"Квантовый компьютер использует кубиты вместо битов, сэр. Это позволяет решать определённые задачи экспоненциально быстрее классических машин.","actions":[]}}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  §11  LLM PLANNER
# ══════════════════════════════════════════════════════════════════════════════

class LLMPlanner:
    def __init__(self):
        self._client = OpenAI(
            base_url=CFG["ollama"]["base_url"],
            api_key="ollama",
        )
        self._system      = build_system_prompt()
        self._history: list[dict] = []
        self._max_history = CFG["ollama"]["max_history"]

    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        text = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()

        # 1) whole-string
        try:
            val = json.loads(text)
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            pass

        # 2) raw_decode from first '{' (handles nested braces correctly)
        idx = text.find("{")
        if idx >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        # 3) greedy brace match (last resort)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                val = json.loads(m.group(0))
                if isinstance(val, dict):
                    return val
            except json.JSONDecodeError:
                pass

        return None

    def _call_model(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=CFG["ollama"]["model"],
            messages=messages,
            temperature=0.3,
            timeout=CFG["ollama"]["timeout"],
        )
        return (resp.choices[0].message.content or "").strip()

    def plan(self, user_text: str) -> dict:
        self._history.append({"role": "user", "content": user_text})
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-(self._max_history * 2):]

        messages = [{"role": "system", "content": self._system}] + self._history

        try:
            ui_plan()
            raw = self._call_model(messages)
            ui_info(f"Raw LLM ({len(raw)} chars): {raw[:200]}")

            plan = self._extract_json(raw)

            if plan is None:
                ui_warn("Non-JSON response — retrying once.")
                retry_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content":
                     "Your reply was not valid JSON. Reply ONLY with the JSON object "
                     "described in the system prompt. No prose, no fences."},
                ]
                raw2 = self._call_model(retry_messages)
                plan = self._extract_json(raw2)
                if plan is None:
                    ui_warn("Retry also failed — using text as speech.")
                    plan = {"speech": raw, "actions": []}

            if not isinstance(plan.get("speech"), str):
                plan["speech"] = "Команда получена, сэр."
            if not isinstance(plan.get("actions"), list):
                plan["actions"] = []

            # Compact assistant turn — no [used: ...] pollution
            self._history.append({"role": "assistant", "content": plan["speech"]})

            ui_json(plan)
            return plan

        except Exception as e:
            log.error("LLMPlanner.plan error: %s", e, exc_info=True)
            err = str(e).lower()
            if "connection" in err or "refused" in err:
                speech = "Сэр, нет соединения с Ollama. Убедитесь, что служба запущена (ollama serve)."
            elif "timeout" in err:
                speech = "Сэр, модель не ответила вовремя."
            else:
                speech = f"Сэр, ошибка языкового модуля: {str(e)[:80]}"
            return {"speech": speech, "actions": []}

    def reset(self) -> None:
        self._history.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  §12  AGENT CORE
# ══════════════════════════════════════════════════════════════════════════════

# Matches ONLY if the whole utterance is a shutdown phrase — no substring bugs
_SHUTDOWN_RE = re.compile(
    r"^\s*(выключись|отбой|выход|пока|до свидания|"
    r"выключи джарвис|стоп джарвис|заверши работу|"
    r"завершение работы|перейди в режим ожидания)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_CANCEL_WORDS = frozenset(["стоп", "отмена", "отставить", "хватит"])

_YES_WORDS = frozenset(["да", "yes", "ок", "ok", "конечно", "ага",
                        "подтверждаю", "верно", "так", "угу", "давай", "конечно да"])
_NO_WORDS  = frozenset(["нет", "no", "отмена", "стоп", "не",
                        "неа", "передумал", "нет не надо", "не надо"])


def _is_shutdown(text: str) -> bool:
    return bool(_SHUTDOWN_RE.match(text))


def _is_cancel(text: str) -> bool:
    return text.lower().strip(" .,!?") in _CANCEL_WORDS


class AgentCore:
    def __init__(self, tts: TTSEngine, stt: STTEngine,
                 planner: LLMPlanner, dispatcher: ActionDispatcher):
        self._tts        = tts
        self._stt        = stt
        self._planner    = planner
        self._dispatcher = dispatcher
        self._confirm    = CFG.get("confirm_destructive", True)

    @staticmethod
    def _destructive_actions(actions: list[dict]) -> list[dict]:
        return [a for a in actions
                if isinstance(a, dict)
                and _TOOL_REGISTRY.get(a.get("tool", ""), {}).get("destructive")]

    def _confirm_destructive(self, actions: list[dict]) -> bool:
        names = ", ".join(sorted({a.get("tool", "?") for a in actions}))
        self._tts.speak_sync(
            f"Сэр, требуется подтверждение для: {names}. Скажите «да» или «нет»."
        )
        try:
            answer = self._stt.listen_once()
        except UserExit:
            return False
        if not answer:
            return False

        al = answer.lower().strip(" .,!?")
        # Exact match first
        if al in _YES_WORDS:
            return True
        if al in _NO_WORDS:
            return False
        # First word only — avoids "нет, давай не надо" → True
        first = al.split()[0] if al.split() else ""
        if first in _YES_WORDS:
            return True
        if first in _NO_WORDS:
            return False
        return False

    def handle(self, text: str) -> bool:
        lower = text.lower().strip()

        if _is_shutdown(lower):
            self._tts.speak_sync("Сэр, перехожу в режим ожидания. До связи.")
            return False

        if _is_cancel(lower):
            self._tts.speak("Отменяю, сэр.")
            return True

        ui_sep()

        filler = threading.Timer(1.5, lambda: self._tts.speak("Секунду, сэр…"))
        filler.daemon = True
        filler.start()
        try:
            plan = self._planner.plan(text)
        finally:
            filler.cancel()

        speech  = (plan.get("speech") or "").strip()
        actions = plan.get("actions") or []

        if actions and self._confirm:
            destructive = self._destructive_actions(actions)
            if destructive and not self._confirm_destructive(destructive):
                self._tts.speak("Отменено, сэр.")
                return True

        results = self._dispatcher.run(actions) if actions else []

        extra = "  ".join(r for r in results if r and r not in speech)
        final = (speech + ("  " + extra if extra else "")).strip() or "Выполнено, сэр."

        self._tts.speak(final)
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  §13  HOTKEY LISTENER
# ══════════════════════════════════════════════════════════════════════════════

def _to_pynput_combo(s: str) -> str:
    parts = re.split(r"[+\s]+", s.lower().strip())
    mapping = {"ctrl": "<ctrl>", "alt": "<alt>", "shift": "<shift>", "win": "<cmd>"}
    return "+".join(mapping.get(p, p) for p in parts)


class HotkeyListener:
    def __init__(self, on_trigger):
        self._listener = None
        if not PYNPUT_OK:
            ui_warn("pynput not installed — global hotkey disabled.")
            return
        combo = _to_pynput_combo(CFG.get("hotkey", "ctrl+alt+j"))
        try:
            self._listener = pynput_kb.GlobalHotKeys({combo: on_trigger})
            self._listener.daemon = True
            self._listener.start()
            ui_info(f"Hotkey armed: {CFG.get('hotkey', 'ctrl+alt+j')} (bypass wake word)")
        except Exception as e:
            ui_warn(f"Hotkey init failed: {e}")

    def stop(self) -> None:
        try:
            if self._listener:
                self._listener.stop()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  §14  WAKE WORD
# ══════════════════════════════════════════════════════════════════════════════

def _has_wake_word(text: str) -> bool:
    tl = text.lower()
    for ww in CFG["wake_words"]:
        if re.search(rf"\b{re.escape(ww)}\b", tl):
            return True
    return False


def _strip_wake_word(text: str) -> str:
    tl = text.lower()
    for ww in CFG["wake_words"]:
        m = re.match(rf"\s*{re.escape(ww)}\b[\s,\.!?]*", tl)
        if m:
            return text[m.end():].strip()
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  §15  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class JARVIS:
    def __init__(self, text_mode: bool = False, no_tts: bool = False, no_hotkey: bool = False):
        ui_banner()

        if GUI_OK and CFG.get("failsafe", True):
            pyautogui.FAILSAFE = True

        ui_info("TTS engine…")
        self._tts = TTSEngine(enabled=not no_tts)

        ui_info("STT engine…")
        self._stt = STTEngine(text_mode=text_mode)

        ui_info("ToolBox…")
        self._toolbox = ToolBox(self._tts)

        ui_info("Dispatcher…")
        self._dispatcher = ActionDispatcher(self._toolbox)

        ui_info("LLM planner…")
        self._planner = LLMPlanner()

        ui_info("Agent core…")
        self._agent = AgentCore(self._tts, self._stt, self._planner, self._dispatcher)

        self._bypass_until = 0.0
        self._hotkey: Optional[HotkeyListener] = None
        if not text_mode and not no_hotkey:
            self._hotkey = HotkeyListener(self._on_hotkey)

        self._running = False

    def _on_hotkey(self) -> None:
        self._bypass_until = time.time() + float(CFG.get("hotkey_bypass_secs", 8.0))
        beep(1000, 120)
        ui_info("Hotkey → bypass wake word")

    def _on_signal(self, sig, frame):
        print()
        ui_warn(f"Signal {sig} — shutting down…")
        self._running = False

    def _greet(self) -> None:
        h = datetime.now().hour
        greeting = (
            "Доброе утро" if 5 <= h < 12 else
            "Добрый день" if 12 <= h < 17 else
            "Добрый вечер" if 17 <= h < 22 else
            "Доброй ночи"
        )
        ww = " или ".join(f"«{w}»" for w in CFG["wake_words"][:2])
        ui_info(f"Wake words: {ww}")
        self._tts.speak_sync(f"{greeting}, сэр. JARVIS в сети. Ожидаю команд.")

    def _cycle(self) -> bool:
        try:
            text = self._stt.listen_once()
        except UserExit:
            ui_info("EOF — exiting.")
            return False

        if not text:
            return True

        ui_heard(text)

        # Text mode: never require wake word
        if self._stt.is_text_mode:
            bypass = True
        else:
            bypass = time.time() < self._bypass_until
            if not bypass and not _has_wake_word(text):
                return True

        if not bypass:
            beep(880, 100)
            ui_wake()
            command = _strip_wake_word(text)
        else:
            command = text.strip()

        if not command:
            self._tts.speak("Слушаю вас, сэр.")
            try:
                text2 = self._stt.listen_once()
            except UserExit:
                return False
            if not text2:
                return True
            ui_heard(text2)
            command = text2

        return self._agent.handle(command)

    def run(self) -> None:
        signal.signal(signal.SIGINT,  self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        if not self._stt.available:
            ui_error("Microphone unavailable — use --text.")
            self._tts.speak_sync("Сэр, микрофон недоступен.")
            return

        self._stt.calibrate()
        self._greet()

        if not GUI_OK:
            ui_warn("pygetwindow/pyautogui not installed — window tools disabled.")

        self._running = True
        ui_info("Ready. Say a wake word (or press hotkey) to activate.")
        ui_sep()

        while self._running:
            try:
                if not self._cycle():
                    self._running = False
            except UserExit:
                self._running = False
            except KeyboardInterrupt:
                self._running = False
            except Exception as e:
                ui_error(f"Loop error: {e}")
                log.exception("Main loop error")
                time.sleep(1)

        ui_info("Shutting down…")
        if self._hotkey:
            self._hotkey.stop()
        self._tts.stop()
        ui_info("Goodbye.")


# ══════════════════════════════════════════════════════════════════════════════
#  §16  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JARVIS Mark-3.1")
    p.add_argument("--text",         action="store_true", help="text input mode (no microphone)")
    p.add_argument("--no-tts",       action="store_true", help="disable speech synthesis")
    p.add_argument("--no-hotkey",    action="store_true", help="disable global hotkey")
    p.add_argument("--config",       default=None,        help="path to config.json")
    p.add_argument("--reset-config", action="store_true", help="reset config.json to defaults")
    p.add_argument("--list-tools",   action="store_true", help="print registered tools and exit")
    p.add_argument("--model",        default=None,        help="override Ollama model")
    p.add_argument("--version",      action="version",    version="JARVIS Mark-3.1")
    return p.parse_args()


def main() -> None:
    global CFG

    args = _parse_cli()

    if args.reset_config:
        try:
            if os.path.exists(_CFG_PATH):
                os.remove(_CFG_PATH)
                print(Fore.YELLOW + f"[CFG] Removed: {_CFG_PATH}")
        except Exception as e:
            print(Fore.RED + f"[CFG] Cannot reset: {e}")

    CFG = load_config(args.config)
    if args.model:
        CFG["ollama"]["model"] = args.model

    setup_logging(CFG)

    if args.list_tools:
        print(Fore.CYAN + f"Registered tools ({len(_TOOL_REGISTRY)}):")
        for name, meta in _TOOL_REGISTRY.items():
            args_str = ", ".join(meta["args"]) if meta["args"] else "—"
            flag = Fore.RED + " [destructive]" + Fore.RESET if meta["destructive"] else ""
            print(f"  {Fore.GREEN}{name}{Fore.RESET}({args_str}) — {meta['desc']}{flag}")
        return

    jarvis = JARVIS(text_mode=args.text, no_tts=args.no_tts, no_hotkey=args.no_hotkey)
    jarvis.run()


if __name__ == "__main__":
    main()