"""The synthetic fixture: eleven regions on a 1200x1600 page, plus a drawn
image of that page so the debug render and segmentation have pixels.

Every classification path is exercised, and every §8 bug has its trap:
the title repeats the header's words at a different height (8b) inside the
top band as a single line (8a); a bare ISO date sits at 14% (8c).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from winnow.chrome import norm, y_bucket
from winnow.types import Region

PAGE_W, PAGE_H = 1200, 1600
HEADER = "Balanced Binary Search Trees"
BODY1 = ("A balanced binary search tree maintains O(log n) height by performing "
         "rotations after insertions and deletions. The most common variants are "
         "AVL trees and red-black trees. Both guarantee logarithmic height for every "
         "sequence of operations, which bounds the cost of lookup.")
BODY2 = ("Rotations are constant-time operations, so the cost of rebalancing is "
         "dominated by walking back up the path from the modified leaf to the root. "
         "In practice red-black trees perform fewer rotations than AVL trees.")

# (y0 of each region is unique — tests look regions up by it)
SPEC = [
    # x0,  y0,   x1,   y1,  nature,  text,                          lines
    (100,  30,  1100,   54, "text",   HEADER,                          1),  # 0 header (repeats)
    (100,  90,  1100,  150, "text",   HEADER,                          1),  # 1 title
    (100, 225,   400,  255, "text",   "2026-09-19",                    1),  # 2 date byline at 14%
    (100, 290,   900,  530, "text",   BODY1,                           8),  # 3 body
    (930, 300,  1090,  380, "text",   "ask about AVL",                 2),  # 4 margin note
    (100, 560,   500,  590, "text",   "h = O(log n)",                  1),  # 5 equation
    (100, 630,   600,  930, "figure", "",                              0),  # 6 figure
    (100, 945,   600,  975, "text",   "Fig 3. A balanced binary tree", 1),  # 7 caption
    (100, 1010,  900, 1250, "text",   BODY2,                           8),  # 8 body
    (300, 1480,  900, 1508, "text",   "Scanned by CamScanner",         1),  # 9 watermark
    (560, 1530,  640, 1560, "text",   "14",                            1),  # 10 page number
]


def regions() -> list[Region]:
    out = []
    for x0, y0, x1, y1, nature, text, lines in SPEC:
        meta = {"line_count": lines} if nature == "text" else {}
        out.append(Region(x0, y0, x1, y1, nature=nature, text=text,
                          conf=100.0 if text else 0.0, meta=meta))
    return out


def repeated() -> dict:
    """Cross-page memory as a 4-page document would build it: the header at
    its height, never the title at its own (8b)."""
    return {(norm(HEADER), y_bucket(30, PAGE_H)): 4}


def _wrap(d, text, font, width):
    lines, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if d.textlength(t, font=font) > width and line:
            lines.append(line)
            line = w
        else:
            line = t
    return lines + [line]


def image() -> np.ndarray:
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1, nature, text, lines in SPEC:
        if nature == "figure":
            _tree(d, x0, y0, x1, y1)
            continue
        size = int((y1 - y0) / max(1, lines) * 0.82)   # ~1.2 leading, like real type
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except OSError:
            font = ImageFont.load_default()
        grey = 150 if y0 < 60 or y0 > 1470 else 0
        for n, ln in enumerate(_wrap(d, text, font, x1 - x0)[:max(1, lines)]):
            d.text((x0, y0 + n * (y1 - y0) / max(1, lines)), ln, fill=grey, font=font)
    d.line((100, 58, 1100, 58), fill=150, width=2)     # header rule (8d)
    return np.asarray(img)


def _tree(d, x0, y0, x1, y1):
    w, h = x1 - x0, y1 - y0
    pts = {0: (0.5, 0.15), 1: (0.25, 0.5), 2: (0.75, 0.5),
           3: (0.12, 0.85), 4: (0.38, 0.85), 5: (0.62, 0.85), 6: (0.88, 0.85)}
    xy = {k: (x0 + fx * w, y0 + fy * h) for k, (fx, fy) in pts.items()}
    for a, b in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
        d.line((*xy[a], *xy[b]), fill=0, width=3)
    for cx, cy in xy.values():
        d.ellipse((cx - 22, cy - 22, cx + 22, cy + 22), outline=0, fill=255, width=3)
