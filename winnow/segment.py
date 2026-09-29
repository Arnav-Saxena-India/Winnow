"""Pixels -> boxes, and the measurements classification reads.

Segmentation writes boxes and ``nature``. Measurement writes
``meta["line_count"]`` (8a: horizontal projection, not block height) and
``meta["ink"]`` (median grey of ink pixels, for the faint rule). Sparse
reclassify flips ``nature`` after text extraction, before ``kind`` exists.

Words -> lines is in ``lines`` (8e).
"""
from __future__ import annotations

import cv2
import numpy as np

from winnow.types import DEFAULT, Config, Region


# --- basics -------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def work_scale(width: float, height: float, cfg: Config = DEFAULT) -> float:
    """Pixels per source unit. A page is ``work_height`` tall, unless that
    would make it narrower than ``work_min_width`` (a long screenshot): then
    it keeps that width and grows taller. Every pixel threshold assumes this."""
    return max(cfg.work_height / float(height), cfg.work_min_width / float(width))


def resize_to_work(gray: np.ndarray, cfg: Config = DEFAULT) -> np.ndarray:
    scale = work_scale(gray.shape[1], gray.shape[0], cfg)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)


def text_zoom(gray: np.ndarray, cfg: Config = DEFAULT) -> float:
    """How much larger to render a page whose letters come out too small to
    OCR (1.0 for any normal page). Thresholds were set at ``body_glyph_h``."""
    g = glyph_height(binarise(gray, cfg), cfg)
    if g >= cfg.min_glyph_h:
        return 1.0
    cap = float(np.sqrt(cfg.max_work_pixels / gray.size))
    return max(1.0, min(cfg.body_glyph_h / g, cap))


def normalise_polarity(gray: np.ndarray, cfg: Config = DEFAULT) -> tuple[np.ndarray, bool]:
    """Dark-mode pages (light text on black) are inverted, so ink is always
    darker than paper. Uninverted, Otsu called the black background ink and
    the whole page became one figure."""
    if float(np.median(gray)) < cfg.dark_paper:
        return 255 - gray, True
    return gray, False


def binarise(gray: np.ndarray, cfg: Config = DEFAULT) -> np.ndarray:
    """Ink = 255. Otsu, but anything clearly darker than the paper is ink
    too: on a page of black text, Otsu alone files grey furniture under
    background and it vanishes without a trace."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    paper = float(np.median(gray))
    t = max(t, paper - cfg.ink_min_contrast)
    return np.where(gray < t, 255, 0).astype(np.uint8)


def is_blank(gray: np.ndarray, cfg: Config = DEFAULT) -> bool:
    if int(gray.max()) - int(gray.min()) < cfg.blank_min_contrast:
        return True       # no contrast at all: Otsu would invent ink from noise
    return (binarise(gray, cfg) > 0).mean() < cfg.blank_ink_frac


def _crop(gray: np.ndarray, r: Region) -> np.ndarray:
    h, w = gray.shape
    return gray[max(0, r.y0):min(h, r.y1), max(0, r.x0):min(w, r.x1)]


# --- measurement (8a) ---------------------------------------------------

def line_count(crop: np.ndarray, cfg: Config = DEFAULT) -> int:
    """Runs of rows carrying ink above ``line_ink_frac`` of the width."""
    if crop.size == 0 or int(crop.max()) - int(crop.min()) < cfg.min_contrast:
        return 1
    rows = (binarise(crop, cfg) > 0).sum(axis=1) > cfg.line_ink_frac * crop.shape[1]
    starts = np.flatnonzero(rows[1:] & ~rows[:-1]).size + int(rows[0])
    return max(1, starts)


def ink_level(crop: np.ndarray, cfg: Config = DEFAULT) -> float | None:
    if crop.size == 0 or int(crop.max()) - int(crop.min()) < cfg.min_contrast:
        return None
    t, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ink = crop[crop <= t]
    return float(np.median(ink)) if ink.size else None


def measure(regions: list[Region], gray: np.ndarray, cfg: Config = DEFAULT) -> None:
    """Fill ``line_count`` (unless the text layer already knows it) and ``ink``."""
    for r in regions:
        if r.nature != "text":
            continue
        crop = _crop(gray, r)
        r.meta.setdefault("line_count", line_count(crop, cfg))
        if not r.text.strip() and crop.size:      # unread: how many letter-sized shapes?
            n, _, st, _ = cv2.connectedComponentsWithStats(binarise(crop, cfg), connectivity=8)
            r.meta["tall_marks"] = int((st[1:n, 3] >= cfg.mark_tall * crop.shape[0]).sum())
        ink = ink_level(crop, cfg)
        if ink is not None:
            r.meta["ink"] = round(ink, 1)


# --- segmentation (image pages) ------------------------------------------

def _boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(x, y, x + w, y + h) for x, y, w, h, _ in stats[1:n]]


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def glyph_height(ink: np.ndarray, cfg: Config) -> float:
    """Median component height on the page: the body glyph size."""
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    hs = st[1:n, 3][st[1:n, 4] >= cfg.noise_area]
    return float(np.median(hs)) if hs.size else float(cfg.fallback_line_h)


def _stroke_share(ink: np.ndarray, box, glyph_h: float, cfg: Config) -> float:
    """Share of a block's ink in components far taller than a body glyph.
    Text: ~0. A drawing (often one connected component): most of it. A row
    of similar tall glyphs is a large-type title, so it scores 0."""
    x0, y0, x1, y1 = box
    n, _, st, _ = cv2.connectedComponentsWithStats(ink[y0:y1, x0:x1], connectivity=8)
    w, h, area = st[1:n, 2], st[1:n, 3], st[1:n, 4]
    tall = h > cfg.stroke_min_ratio * glyph_h
    if not tall.any():
        return 0.0
    glyphlike = (h >= cfg.row_similar * h[tall].min()) & (area < cfg.solid_fill * w * h)
    if glyphlike.sum() >= cfg.text_row_min:
        return 0.0
    return float(area[tall].sum() / max(1, area.sum()))


def segment(gray: np.ndarray, cfg: Config = DEFAULT) -> list[Region]:
    """Boxes and physical nature for a page already at work height."""
    ink = binarise(gray, cfg)
    glyph_h = glyph_height(ink, cfg)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.dilate_h, cfg.dilate_v))
    blocks = _boxes(cv2.dilate(ink, k))
    figs, texts = [], []
    for b in blocks:
        w, h = b[2] - b[0], b[3] - b[1]
        drawing = (w * h >= cfg.fig_min_area and h >= cfg.sparse_min_h
                   and _stroke_share(ink, b, glyph_h, cfg) > cfg.stroke_share_min)
        (figs if drawing else texts).append(b)
    # Join scattered strokes of one drawing into one figure.
    if figs:
        m = np.zeros_like(ink)
        for x0, y0, x1, y1 in figs:
            m[y0:y1, x0:x1] = 255
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.fig_close_kernel,) * 2)
        figs = [b for b in _boxes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc))
                if (b[2] - b[0]) * (b[3] - b[1]) >= cfg.fig_min_area]
    # Text inside a figure is its labels; duplicates of one another merge.
    texts = [t for t in texts if not any(_iou(t, f) > 0 and _inside(t, f) for f in figs)]
    kept: list[tuple] = []
    for t in sorted(texts, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(_iou(t, o) <= cfg.merge_iou for o in kept):
            kept.append(t)
    out = [Region(*map(int, f), nature="figure") for f in figs]
    out += [Region(*map(int, t), nature="text") for t in kept]
    return sorted(out, key=lambda r: (r.y0, r.x0))


def _inside(a, b) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def reclassify_sparse(regions: list[Region], cfg: Config = DEFAULT) -> None:
    """After text extraction: a big block with almost no characters is a
    drawing. Writes ``nature`` and ``meta`` only. The height guard is 8d."""
    for r in regions:
        if r.nature != "text":
            continue
        area = (r.x1 - r.x0) * (r.y1 - r.y0)
        if area < cfg.sparse_min_area or r.y1 - r.y0 < cfg.sparse_min_h:
            continue
        density = len(r.text.strip()) / (area / 1000)
        r.meta["density"] = round(density, 3)
        if density < cfg.sparse_max_density:
            r.nature = "figure"
            r.meta["sparse_reclassified"] = True
            r.meta["figure_text"] = r.text
            r.meta["figure_text_reliable"] = r.conf >= cfg.sparse_conf_floor
            r.text = ""
