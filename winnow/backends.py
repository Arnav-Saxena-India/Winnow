"""Everything that touches a model, a device or a voice.

Only ``pipeline`` and ``cli`` import this module (invariant 2). Logic
receives these objects; it never constructs them.

  Speaker   — Windows SAPI via COM: local voices, no network.
  OCR       — EasyOCR / Windows OCR, in ``ocr_backends`` (re-exported here).
  Describer — figure -> words. NullDescriber says nothing (honestly);
              OllamaDescriber talks to a model served on loopback.
"""
from __future__ import annotations

import base64
import json
import sys
import urllib.request

import cv2
import numpy as np

from winnow.ocr_backends import EasyOcr, WindowsOcr, make_ocr  # noqa: F401  (re-exported)
from winnow.types import BackendUnavailable

PROMPT_VERSION = "v5"
# Qwen3-VL-2B: the VLM Qualcomm profiles on the Snapdragon X Elite NPU (docs/measurements.md).
DEFAULT_VLM = "qwen3-vl:2b"
# v2 ("...If you cannot read the figure with confidence, reply with exactly
# UNCLEAR") made Qwen3-VL-2B answer UNCLEAR for every figure, including a clear
# tree. v3 named what counts as unreadable first. v5 asks for two sentences:
# 74 output tokens on average instead of 93, still 9/9 noise images UNCLEAR and
# 4/4 corpus figures described (tools/prompt_check.py). Decode dominates time.
DESCRIBE_PROMPT = (
    "Look at the image. If it is random noise, static, texture, blank, or unreadable, "
    "reply with only the word UNCLEAR. Otherwise describe it for a listener who cannot "
    "see it in two sentences: first the kind of figure, then what it shows (structure, "
    "visible values, trends). Do not invent values that are not shown. "
    "Caption from the page: {context}")
ASK_PROMPT = (
    "Context from the page: {context}\nAnswer the question about this figure in one "
    "or two sentences. If the figure does not show the answer, reply with exactly "
    "UNCLEAR.\nQuestion: {question}")


# --- speech -------------------------------------------------------------

class SapiSpeaker:
    """Asynchronous SAPI voice; poll ``busy()`` from the UI loop."""
    name = "SAPI"

    def __init__(self):
        if sys.platform != "win32":
            raise BackendUnavailable("SAPI speech needs Windows")
        try:
            import comtypes.client
            self._voice = comtypes.client.CreateObject("SAPI.SpVoice")
        except Exception as e:
            raise BackendUnavailable(f"SAPI voice unavailable: {e}") from e

    def say(self, text: str) -> None:
        self._voice.Speak(text, 1 | 2)          # async | purge anything queued

    def busy(self) -> bool:
        return not self._voice.WaitUntilDone(0)   # queued-but-not-started counts as busy

    def wait(self) -> None:
        self._voice.WaitUntilDone(-1)

    def stop(self) -> None:
        self._voice.Speak("", 1 | 2)

    def set_speed(self, multiplier: float) -> None:
        # SAPI rate is -10..10, roughly log-scaled; 1.0x -> 0, 2.0x -> +10.
        self._voice.Rate = int(max(-10, min(10, round(np.log2(multiplier) * 10))))


class PrintSpeaker:
    """Fallback when no voice is available: prints what would be said."""
    name = "print"

    def say(self, text: str) -> None:
        print(f"  » {text}")

    def busy(self) -> bool:
        return False

    def wait(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_speed(self, multiplier: float) -> None:
        pass


def make_speaker():
    try:
        return SapiSpeaker()
    except BackendUnavailable:
        return PrintSpeaker()


# --- figure description --------------------------------------------------

def phash(gray: np.ndarray) -> str:
    """64-bit DCT perceptual hash."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


class DescriptionCache:
    """Keyed on phash + model + prompt version — never the image alone (§10),
    or a prompt edit silently serves stale descriptions."""

    def __init__(self):
        self._d: dict[str, str | None] = {}

    def key(self, crop, describer) -> str:
        return f"{phash(crop)}:{describer.model_id}:{PROMPT_VERSION}"

    def get(self, key):
        return self._d.get(key)

    def __contains__(self, key) -> bool:
        return key in self._d

    def put(self, key, value) -> None:
        self._d[key] = value


class NullDescriber:
    """No VLM installed: every figure falls back to its caption, audibly."""
    name, device, model_id = "none", "-", "none"

    def describe(self, crop: np.ndarray, context: str) -> str | None:
        return None

    def ask(self, crop: np.ndarray, question: str, context: str) -> str | None:
        return None


class OllamaDescriber:
    """A vision model served by Ollama on this machine (loopback only)."""
    name, device = "Ollama", "local"

    def __init__(self, model: str = DEFAULT_VLM, host: str = "http://127.0.0.1:11434",
                 timeout: float = 120.0):   # descriptions are precomputed at open
        self.model_id, self.host, self.timeout = model, host, timeout
        self.last_stats: dict = {}     # Ollama's token counts and ns timings, last call

    def _generate(self, crop: np.ndarray, prompt: str) -> str:
        ok, png = cv2.imencode(".png", crop)
        body = json.dumps({"model": self.model_id, "prompt": prompt, "stream": False,
                           "images": [base64.b64encode(png.tobytes()).decode()],
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except TimeoutError:
            raise
        except Exception as e:
            raise BackendUnavailable(f"Ollama model {self.model_id} unavailable: {e}") from e
        self.last_stats = {k: v for k, v in data.items() if k.endswith(("_count", "_duration"))}
        return data["response"].strip()

    def describe(self, crop: np.ndarray, context: str) -> str | None:
        return self._generate(crop, DESCRIBE_PROMPT.format(context=context or "none"))

    def ask(self, crop: np.ndarray, question: str, context: str) -> str | None:
        return self._generate(crop, ASK_PROMPT.format(context=context or "none",
                                                      question=question))


class LlamaCppDescriber(OllamaDescriber):
    """The same VLM served by llama.cpp's ``llama-server -m MODEL --mmproj PROJ``
    (loopback only). Stats are renamed to Ollama's keys so tools read either."""
    name = "llama.cpp"

    def __init__(self, model: str = DEFAULT_VLM, host: str = "http://127.0.0.1:8080",
                 timeout: float = 120.0):
        super().__init__(model, host, timeout)

    def _generate(self, crop: np.ndarray, prompt: str) -> str:
        ok, png = cv2.imencode(".png", crop)
        image = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
        body = json.dumps({"temperature": 0, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": prompt}]}]}).encode()
        req = urllib.request.Request(f"{self.host}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except TimeoutError:
            raise
        except Exception as e:
            raise BackendUnavailable(f"llama-server at {self.host} unavailable: {e}") from e
        t, u = data.get("timings", {}), data.get("usage", {})
        # usage, not timings.prompt_n: that counts only tokens not already in
        # llama-server's prompt cache (it read 1 on a repeated prompt).
        self.last_stats = {"prompt_eval_count": u.get("prompt_tokens"),
                           "eval_count": u.get("completion_tokens"),
                           "total_duration": 1e6 * (t.get("prompt_ms", 0) + t.get("predicted_ms", 0))}
        return data["choices"][0]["message"]["content"].strip()
