"""The OCR stage: pages with no text layer get their text here.

The engine is injected (``backends.make_ocr``); this module never imports a
backend (invariant 2). Nothing is filtered by confidence: unreadable text is
reported, never dropped (§10).
"""
from __future__ import annotations

import cv2
import numpy as np

from winnow.lines import group_lines
from winnow.segment import glyph_height, binarise, measure, reclassify_sparse
from winnow.telemetry import Telemetry
from winnow.types import DEFAULT, Config, Region


def line_boxes(gray: np.ndarray, cfg: Config = DEFAULT) -> list[tuple[int, int, int, int]]:
    """Text lines as (x0, y0, x1, y1): ink joined across gaps under
    ``line_join`` glyph heights. What EasyOCR's detector finds, in 0.5 s
    instead of 5 minutes on a long page (it found 15 of 15 lines)."""
    ink = binarise(gray, cfg)
    g = glyph_height(ink, cfg)
    kw = max(1, int(cfg.line_join * g))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    n, _, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, k), connectivity=8)
    H, W = gray.shape
    out = []
    for x, y, w, h, _ in st[1:n]:
        if h < cfg.line_min_h * g or w < g:
            continue
        x, w = x + kw // 2, w - 2 * (kw // 2)    # undo the dilation's widening: the
        m = int(cfg.line_margin * h)             # blank it left read as '~' or "'"
        out.append((max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)))
    return out


def ocr_page(page, ocr, cfg: Config, tel: Telemetry) -> None:
    """Fill text for a page that had no text layer. Nothing is filtered by
    confidence, and words outside every segmented box get a box of their own."""
    if ocr is None or page.blank:
        for r in page.regions:
            r.meta["unread"] = "no OCR backend installed"
        return
    with tel.timed("ocr", ocr.name, ocr.device, page=page.page_no):
        if hasattr(ocr, "read_lines"):     # our line boxes, its recogniser, at work scale
            words = ocr.read_lines(page.gray, line_boxes(page.gray, cfg))
        else:
            k = cfg.ocr_scale
            big = cv2.resize(page.gray, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            words = [dict(w, x0=w["x0"] / k, x1=w["x1"] / k,
                          top=w["top"] / k, bottom=w["bottom"] / k)
                     for w in read_in_strips(ocr, big, cfg)]
    page.source = "ocr"
    homes: dict[int, list[dict]] = {}
    for w in words:
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        i = next((i for i, r in enumerate(page.regions)
                  if r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1), None)
        if i is None:
            page.regions.append(Region(int(w["x0"]), int(w["top"]), int(w["x1"]) + 1,
                                       int(w["bottom"]) + 1, nature="text"))
            i = len(page.regions) - 1
        homes.setdefault(i, []).append(w)
    for i, ws in homes.items():
        r = page.regions[i]
        text = " ".join(" ".join(w["text"] for w in ln) for ln in group_lines(ws, cfg))
        if r.nature == "figure":
            r.meta["figure_text"] = text
        else:
            known = [w["conf"] for w in ws if w["conf"] >= 0]
            r.text, r.conf = text, (sum(known) / len(known) if known else -1.0)
    # Naive read-aloud gets a fair shot: OCR lines, top to bottom, all of them.
    page.naive = " ".join(" ".join(w["text"] for w in ln) for ln in group_lines(words, cfg))
    escalate(page, ocr, cfg, tel)
    reclassify_sparse(page.regions, cfg)
    measure(page.regions, page.gray, cfg)


def read_in_strips(ocr, img: np.ndarray, cfg: Config = DEFAULT) -> list[dict]:
    """OCR a tall image in overlapping horizontal strips (Windows OCR refuses
    anything over 10000 px). Each strip owns the band between the midpoints of
    its overlaps; a word is kept only by the strip whose band holds its centre,
    so a line in an overlap is read once, never twice or not at all."""
    h, tile, ov = img.shape[0], cfg.ocr_tile, cfg.ocr_tile_overlap
    if h <= tile:
        return ocr.read(img)
    out, step, y = [], tile - ov, 0
    while True:
        last = y + tile >= h
        lo, hi = (y + ov / 2 if y else 0), (h if last else y + tile - ov / 2)
        strip = img[y:y + tile]
        # Skip only a strip with no contrast at all: the page-level blank test
        # (under 0.1% ink) skipped a strip holding one short line.
        if int(strip.max()) - int(strip.min()) >= cfg.blank_min_contrast:
            cols = np.flatnonzero(binarise(strip, cfg).any(axis=0))   # read only inked columns
            x0 = max(0, int(cols[0]) - ov // 2) if cols.size else 0
            x1 = int(cols[-1]) + ov // 2 if cols.size else strip.shape[1]
            for w in ocr.read(strip[:, x0:x1]):
                if lo <= y + (w["top"] + w["bottom"]) / 2 < hi:
                    out.append(dict(w, x0=w["x0"] + x0, x1=w["x1"] + x0,
                                    top=w["top"] + y, bottom=w["bottom"] + y))
        if last:
            return out
        y += step


def escalate(page, ocr, cfg: Config, tel: Telemetry) -> None:
    """Tier 3: spend compute only where unsure. Each text region the page-level
    OCR left empty gets exactly one re-read, cropped and further upscaled."""
    for r in page.regions:
        if r.nature != "text" or r.text.strip():
            continue
        p = cfg.escalate_pad
        y0, x0 = max(0, r.y0 - p), max(0, r.x0 - p)
        crop = page.gray[y0:r.y1 + p, x0:r.x1 + p]
        big = cv2.resize(crop, None, fx=cfg.escalate_scale, fy=cfg.escalate_scale,
                         interpolation=cv2.INTER_CUBIC)
        with tel.timed("ocr-escalate", ocr.name, ocr.device, page=page.page_no):
            words = ocr.read(big)
        if words:
            lines = group_lines(words, cfg)
            r.text = " ".join(" ".join(w["text"] for w in ln) for ln in lines)
            r.meta["escalated"] = f"re-read alone at {cfg.escalate_scale:g}x after page OCR found nothing"
        else:
            r.meta["unread"] = "OCR found no text, even re-read alone at higher resolution"
