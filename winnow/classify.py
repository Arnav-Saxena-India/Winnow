"""Classification — assigns ``kind``, ``meta["why"]`` and ``meta["evidence"]``.

The chrome rules themselves are in ``chrome``; this module measures, runs
them, and decides what everything else is. Fix 8a (line height, not block
height) and 8c (dates are not equations) are here.

Reads ``meta["line_count"]`` and ``meta["ink"]`` written by the measurement
step; never touches pixels itself. Invariant: never mutates boxes.
"""
from __future__ import annotations

import re
from statistics import median

from winnow.chrome import _DATELIKE, band_height, chrome_rule, mark_rule
from winnow.reading_order import column_groups
from winnow.types import DEFAULT, Config, Region

_CAPTION_PREFIX = re.compile(r"^\s*(fig|figure|table|chart)\s*\.?\s*\d+", re.I)
EQUATION_CHARS = set("+-=*/^<>−×÷·≤≥≠≈√∑∫")   # brackets alone are not maths
# "red-black" is a word, and so is a name with a number: "ResNet-152", "top-1"
# were read "ResNet minus 152" from a real paper's table. After one letter,
# "x-1" or "n-1", it is still subtraction.
_WORD_HYPHEN = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])|(?<=[A-Za-z]{2})-(?=\d)")
# A list dash: "- Counting using logic" is a bullet, not "minus Counting". A dash
# before a capitalised word is punctuation; "A - B" (one letter) stays maths.
_BULLET = re.compile(r"(?:^|(?<=\s))[-–•*]\s*(?=[A-Z][a-z]{2,})")   # OCR: "-Counting"
_PROSE_WORD = re.compile(r"[A-Za-z]{3,}")

# Rules that may never suppress the page's largest text (invariant 8).
_TITLE_UNSAFE = ("page number", "date", "handle", "small", "faint")


# --- measurements -------------------------------------------------------

def line_height(r: Region) -> float:
    """Per-line height. Never the block height alone (8a). A text layer knows
    it exactly (``meta["line_h"]``); pixels give block height / line count."""
    if "line_h" in r.meta:
        return r.meta["line_h"]
    return (r.y1 - r.y0) / max(1, r.meta.get("line_count", 1))


def _char_median(pairs: list[tuple[float, int]]) -> float | None:
    """Median weighted by character count: the value half the page's text
    sits at. A one-line footer cannot outvote a paragraph."""
    if not pairs:
        return None
    pairs = sorted(pairs)
    half, run = sum(n for _, n in pairs) / 2, 0
    for value, n in pairs:
        run += n
        if run >= half:
            return value
    return pairs[-1][0]


def _body_line_height(regions: list[Region], cfg: Config) -> float:
    lh = _char_median([(line_height(r), len(r.text.strip())) for r in regions
                       if r.nature == "text" and len(r.text.strip()) > cfg.body_min_chars])
    return lh or cfg.fallback_line_h


def _body_ink(regions: list[Region], cfg: Config) -> float | None:
    return _char_median([(r.meta["ink"], len(r.text.strip())) for r in regions
                         if "ink" in r.meta and len(r.text.strip()) > cfg.body_min_chars])


def body_column_span(regions: list[Region], page_w: int,
                     cfg: Config = DEFAULT) -> tuple[float, float]:
    """Left edge of the first text column to right edge of the last. Wide
    blocks are grouped into columns by left edge, and each edge is a median
    within its column. Never min/max: one page-wide footer stretches that to
    the full page and margin notes vanish. A single median over all
    blocks lands on one column of a two-column page, and every narrow block in
    the other column became a "margin note" (found on a real paper)."""
    wide = [r for r in regions if (r.x1 - r.x0) / max(1, page_w) > cfg.col_span_min]
    if not wide:
        return 0.0, float(page_w)
    cols = column_groups(wide, page_w, cfg)
    return median(r.x0 for r in cols[0]), median(r.x1 for r in cols[-1])


def looks_like_equation(text: str, cfg: Config = DEFAULT) -> bool:
    """Operator glyphs, short, few long words — and not a date (8c)."""
    t = text.strip()
    if not t or _DATELIKE.match(t) or len(t) > cfg.equation_max_chars:
        return False
    if not any(c in EQUATION_CHARS for c in _BULLET.sub("", _WORD_HYPHEN.sub("", t))):
        return False
    words = _PROSE_WORD.findall(t)
    if len(words) > cfg.equation_max_words:
        return False
    long_words = sum(1 for w in t.split() if len(w) > cfg.equation_long_word)
    return long_words <= cfg.equation_max_long_words


# --- main entry point ---------------------------------------------------

def classify(regions: list[Region], page_h: int, page_w: int,
             repeated: dict | set | None = None, n_pages: int | None = None,
             cfg: Config = DEFAULT) -> None:
    """Assign ``kind``, ``meta["why"]`` and ``meta["evidence"]`` in place.

    ``repeated`` maps ``(norm(text), y_bucket)`` -> pages seen on (a bare set
    is accepted for synthetic runs).
    """
    if isinstance(repeated, set):
        repeated = {k: cfg.repeat_min_pages for k in repeated}
    repeated = repeated or {}
    body_lh = _body_line_height(regions, cfg)
    body_ink = _body_ink(regions, cfg)
    col_left, col_right = body_column_span(regions, page_w, cfg)

    ratios = [line_height(r) / body_lh if r.nature == "text" and r.text.strip() else 0.0
              for r in regions]
    title_idx = max(range(len(regions)), key=ratios.__getitem__, default=None)
    if title_idx is not None and ratios[title_idx] <= cfg.heading_ratio:
        title_idx = None

    for i, r in enumerate(regions):
        text = r.text.strip()
        ev = _evidence(r, text, page_h, page_w, body_lh, body_ink, cfg)
        r.meta["evidence"] = ev
        rule = mark_rule(r, text, ev, body_lh, cfg) or chrome_rule(
            r, text, ev, page_h, repeated, n_pages, col_left, col_right, cfg, page_w)
        if rule and i == title_idx and rule[0] in _TITLE_UNSAFE:
            ev["title_guard"] = f"would have been '{rule[1]}'; largest text on page"
            rule = None
        if rule:
            r.kind, r.meta["why"] = "chrome", rule[1]
            ev["rule"] = rule[0]
            continue
        r.kind = _content_kind(r, text, ev, regions, page_w, col_left, col_right, cfg)
        ev["rule"] = r.kind


def _evidence(r, text, page_h, page_w, body_lh, body_ink, cfg) -> dict:
    ev = {"chars": len(text), "line_h": round(line_height(r), 1),
          "size_ratio": round(line_height(r) / body_lh, 2)}
    bh = band_height(page_h, page_w, cfg)
    if r.y0 < cfg.band_top * bh:
        ev["band"] = "top"
    elif r.y1 > page_h - (1 - cfg.band_bottom) * bh:
        ev["band"] = "bottom"
    if body_ink is not None and "ink" in r.meta:
        ev["ink_delta"] = int(round(r.meta["ink"] - body_ink))   # + is lighter
    return ev


def _beside_body(r, regions, page_w, cfg) -> bool:
    """A margin note sits next to body text: some wider text block overlaps it
    vertically on the column side. A short label alone on its line ("Example",
    "Answer:") is not a note, however far left it starts."""
    for o in regions:
        if o is r or o.nature != "text" or not o.text.strip():
            continue
        if (o.x1 - o.x0) <= cfg.margin_max_width * page_w:
            continue
        if min(o.y1, r.y1) - max(o.y0, r.y0) <= 0:
            continue
        if o.x0 >= r.x1 or o.x1 <= r.x0:
            return True
    return False


def _content_kind(r, text, ev, regions, page_w, col_left, col_right, cfg) -> str:
    if r.nature in ("figure", "table"):
        return r.nature
    if _CAPTION_PREFIX.match(text):
        return "caption"
    if looks_like_equation(text, cfg):
        return "equation"
    # Margin note before heading: narrow and outside the body column.
    if (r.x1 - r.x0) / max(1, page_w) < cfg.margin_max_width and (
            r.x1 <= col_left or r.x0 >= col_right):
        ev["column_span"] = [int(col_left), int(col_right)]
        if _beside_body(r, regions, page_w, cfg):
            return "annotation"
        ev["not_beside_body"] = True
    darker = r.meta.get("bold", False) or (
        ev.get("ink_delta") is not None and -ev["ink_delta"] >= cfg.heading_dark_delta)
    short = 0 < len(text) <= cfg.heading_max_chars and not text.endswith(".")
    single_line = r.meta.get("line_count", 1) == 1
    if short and (ev["size_ratio"] > cfg.heading_ratio or (darker and single_line)):
        return "heading"
    return "body"
