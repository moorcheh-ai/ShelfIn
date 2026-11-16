"""Shelf catalog and spoken retail guidance.

An enrolled object can name its shelf in either place:

- known_objects/<Name>/shelf.txt  (first line is the shelf)
- retail_catalog.json             (overrides the text file)

Voice prefers a standalone Piper binary when PIPER_MODEL is set, then espeak on
Linux, or Windows SAPI. This path does not use Moorcheh voice tooling. Set
RETAIL_VOICE=0 to only print the line.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "retail_catalog.json"
OBJECTS_ROOT = ROOT / "known_objects"


def _clean(value: str) -> str:
    return " ".join(value.replace("_", " ").split())


def shelf_phrase(shelf: str) -> str:
    """Turn a shelf note into something that can be spoken."""
    text = _clean(shelf)
    if not text:
        return ""
    low = text.lower()
    if low.startswith("the ") or "shelf" in low or "aisle" in low:
        return text
    return f"the {text} shelf"


def _load_json_catalog() -> dict[str, dict[str, str]]:
    if not CATALOG_PATH.is_file():
        return {}
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog: dict[str, dict[str, str]] = {}
    if not isinstance(raw, dict):
        return catalog
    for name, info in raw.items():
        if isinstance(info, str):
            info = {"shelf": info}
        if not isinstance(info, dict):
            continue
        catalog[str(name).strip().lower()] = {
            "say": _clean(str(info.get("say") or name)),
            "shelf": _clean(str(info.get("shelf") or "")),
            "description": _clean(str(info.get("description") or "")),
        }
    return catalog


def _load_folder_catalog() -> dict[str, dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}
    if not OBJECTS_ROOT.is_dir():
        return catalog
    for folder in OBJECTS_ROOT.iterdir():
        if not folder.is_dir():
            continue
        note = folder / "shelf.txt"
        shelf = ""
        if note.is_file():
            shelf = _clean(note.read_text(encoding="utf-8").splitlines()[0])
        desc_file = folder / "description.txt"
        description = ""
        if desc_file.is_file():
            description = _clean(desc_file.read_text(encoding="utf-8").splitlines()[0])
        catalog[folder.name.strip().lower()] = {
            "say": _clean(folder.name),
            "shelf": shelf,
            "description": description,
        }
    return catalog


def load_catalog() -> dict[str, dict[str, str]]:
    """Folder notes first, then retail_catalog.json overrides."""
    catalog = _load_folder_catalog()
    catalog.update(_load_json_catalog())
    return catalog


def save_catalog_entry(name: str, shelf: str, description: str = "") -> None:
    """Remember an item the UI enrolled so the live line can name its shelf."""
    import json

    shelf = _clean(shelf)
    say = _clean(name)
    description = _clean(description)
    if not say:
        return
    data: dict = {}
    if CATALOG_PATH.is_file():
        try:
            loaded = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            data = {}
    for key in list(data.keys()):
        if str(key).strip().lower() == say.lower():
            del data[key]
    entry = {"say": say, "shelf": shelf}
    if description:
        entry["description"] = description
    data[say] = entry
    CATALOG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def lookup(catalog: dict[str, dict[str, str]], name: str) -> dict[str, str] | None:
    key = _clean(name).lower()
    if key in catalog:
        return catalog[key]
    # "coffee mug" can still match a folder named Mug.
    for item_key, info in catalog.items():
        if item_key in key or key in item_key:
            return info
    return None


def _item_phrase(name: str) -> str:
    text = _clean(name)
    if text.lower().startswith("the "):
        return text
    return f"the {text}"


def guidance_line(
    objects: list[tuple[str, str]],
    person: str | None = None,
) -> str:
    """Build one spoken line from matched items and an optional known face.

    objects: [(spoken name, shelf phrase), ...]
    """
    parts: list[str] = []
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, shelf in objects:
        key = name.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((name, shelf))

    if len(unique) == 1:
        name, shelf = unique[0]
        sentence = f"This is {_item_phrase(name)}."
        if shelf:
            sentence += f" It belongs on {shelf_phrase(shelf)}."
        parts.append(sentence)
    elif unique:
        sentences = []
        for name, shelf in unique:
            sentence = f"This is {_item_phrase(name)}"
            if shelf:
                sentence += f", and it belongs on {shelf_phrase(shelf)}"
            sentences.append(sentence + ".")
        parts.append(" ".join(sentences))

    if person and person != "Unknown":
        hello = f"Hello {person}."
        return f"{hello} {' '.join(parts)}".strip()
    return " ".join(parts).strip()


class RetailVoice:
    """Speak one line at a time. Repeats of the same line wait out a cooldown."""

    def __init__(self, cooldown_s: float = 12.0):
        self.cooldown_s = cooldown_s
        self.enabled = os.environ.get("RETAIL_VOICE", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._last_spoken: dict[str, float] = {}
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self.enabled:
            print(f"[retail] voice ready ({_voice_backend_label()})")
        else:
            print("[retail] voice disabled (RETAIL_VOICE=0)")

    def say(self, line: str) -> None:
        text = _clean(line)
        if not text:
            return
        now = time.monotonic()
        if now - self._last_spoken.get(text, 0.0) < self.cooldown_s:
            return
        self._last_spoken[text] = now
        print(f"[retail] {text}")
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            return

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _loop(self) -> None:
        while True:
            line = self._queue.get()
            if line is None:
                return
            try:
                _speak(line)
            except Exception as exc:  # noqa: BLE001
                print(f"[retail] voice skipped: {exc}")


def _piper_ready() -> bool:
    model = (os.environ.get("PIPER_MODEL") or "").strip()
    if not model or not Path(model).is_file():
        return False
    binary = (os.environ.get("PIPER_BIN") or "piper").strip() or "piper"
    return shutil.which(binary) is not None or Path(binary).is_file()


def _voice_backend_label() -> str:
    if _piper_ready():
        return "Piper"
    if sys.platform == "win32":
        return "Windows SAPI"
    for cmd in ("espeak-ng", "espeak", "spd-say"):
        if shutil.which(cmd):
            return cmd
    return "print-only (no TTS engine)"


def _speak(text: str) -> None:
    if _piper_ready():
        try:
            _speak_piper(text)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[retail] Piper skipped: {exc}")

    if sys.platform == "win32":
        _speak_windows(text)
        return
    for cmd in ("espeak-ng", "espeak", "spd-say"):
        if shutil.which(cmd):
            subprocess.run([cmd, text], check=False)
            return
    print(
        "[retail] no TTS engine — install Piper (set PIPER_MODEL) or espeak-ng"
    )


def _speak_piper(text: str) -> None:
    binary = (os.environ.get("PIPER_BIN") or "piper").strip() or "piper"
    model = (os.environ.get("PIPER_MODEL") or "").strip()
    speaker = (os.environ.get("PIPER_SPEAKER") or "").strip()
    cmd = [binary, "--model", model, "--output-raw"]
    if speaker:
        cmd.extend(["--speaker", speaker])
    piper = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert piper.stdin is not None
    piper.stdin.write(text.encode("utf-8"))
    piper.stdin.close()
    if not shutil.which("aplay"):
        if piper.stdout is not None:
            piper.stdout.read()
        piper.wait(timeout=30)
        raise RuntimeError("Piper needs aplay for raw PCM playback")
    play = subprocess.Popen(
        ["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
        stdin=piper.stdout,
    )
    play.wait(timeout=60)
    piper.wait(timeout=30)


def _speak_windows(text: str) -> None:
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$synth.Rate = 0; "
        "$line = [Console]::In.ReadToEnd(); "
        "$synth.Speak($line)"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        input=text,
        text=True,
        check=False,
    )
