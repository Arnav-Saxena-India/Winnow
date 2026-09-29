"""OCR engines: on-device, no network at read time.

  EasyOcr    — weights fetched once by `winnow fetch-models`; at read time
               downloads are disabled, so the offline guard holds.
  WindowsOcr — Windows.Media.Ocr: ships with Windows, needs no weights.

Imported only through ``backends`` (invariant 2), which re-exports them.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np

from winnow.types import BackendUnavailable


class EasyOcr:
    name, device = "EasyOCR", "CPU"

    def __init__(self, langs=("en",), download: bool = False):
        try:
            import easyocr
        except ImportError as e:
            raise BackendUnavailable("EasyOCR not installed: pip install easyocr") from e
        except OSError as e:      # torch's DLLs fail to load if WinRT OCR was loaded first
            raise BackendUnavailable(f"EasyOCR could not load PyTorch: {e}") from e
        try:
            with _quiet_torch():      # EasyOCR's int8 default was 2x slower on this CPU
                self._reader = easyocr.Reader(list(langs), gpu=False, verbose=False,
                                              download_enabled=download, quantize=False)
        except Exception as e:
            raise BackendUnavailable(
                f"EasyOCR weights missing ({e}); run `python -m winnow fetch-models` once") from e

    def read(self, gray: np.ndarray) -> list[dict]:
        """-> words {x0, top, x1, bottom, text, conf(0-100)}. Nothing filtered
        by confidence: unreadable text is reported, never dropped."""
        with _quiet_torch():
            return _words(self._reader.readtext(gray, detail=1, paragraph=False))

    def read_lines(self, gray: np.ndarray, boxes) -> list[dict]:
        """Recognise given line boxes, skipping EasyOCR's own detector. On a
        long real page: 67/69 words vs 64/69, and 61 s vs 356 s."""
        hl = [[x0, x1, y0, y1] for x0, y0, x1, y1 in boxes]
        with _quiet_torch():
            return _words(self._reader.recognize(gray, horizontal_list=hl, free_list=[], detail=1))


def _words(results) -> list[dict]:
    out = []
    for pts, text, conf in results:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append({"x0": float(min(xs)), "x1": float(max(xs)), "top": float(min(ys)),
                    "bottom": float(max(ys)), "text": text, "conf": float(conf) * 100})
    return out


@contextmanager
def _quiet_torch():
    """PyTorch's own notices, printed on every run on a CPU-only machine; they
    are not about Winnow or its output."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"torch\.quantize_per_tensor", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*pin_memory", category=UserWarning)
        yield


class WindowsOcr:
    """Windows.Media.Ocr: ships with Windows, runs on-device, needs no weights.
    It reports no confidence, so words carry conf -1 (unknown), not a guess."""
    name, device = "Windows OCR", "on-device"

    def __init__(self):
        try:
            from winrt.windows.media.ocr import OcrEngine
        except ImportError as e:
            raise BackendUnavailable("pip install winrt-Windows.Media.Ocr "
                                     "winrt-Windows.Graphics.Imaging winrt-Windows.Storage.Streams") from e
        self._engine = OcrEngine.try_create_from_user_profile_languages()
        if self._engine is None:
            raise BackendUnavailable("no Windows OCR language pack installed")

    def read(self, gray: np.ndarray) -> list[dict]:
        import asyncio
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.storage.streams import DataWriter
        h, w = gray.shape
        writer = DataWriter()
        writer.write_bytes(np.ascontiguousarray(gray, np.uint8).tobytes())
        bmp = SoftwareBitmap.create_copy_from_buffer(writer.detach_buffer(),
                                                     BitmapPixelFormat.GRAY8, w, h)

        async def go():
            return await self._engine.recognize_async(bmp)
        result = asyncio.run(go())
        out = []
        for line in result.lines:
            for word in line.words:
                b = word.bounding_rect
                out.append({"x0": b.x, "x1": b.x + b.width, "top": b.y,
                            "bottom": b.y + b.height, "text": word.text, "conf": -1.0})
        return out


def make_ocr():
    """EasyOCR if its weights are present, else Windows OCR, else None (image
    pages then say 'could not be read' instead of pretending)."""
    for cls in (EasyOcr, WindowsOcr):
        try:
            return cls()
        except BackendUnavailable:
            continue
    return None
