"""Chrome detection — the furniture rules, in priority order (§6).

First match wins; order is load-bearing. Fixes 8b (position is part of the
repeat key) and 8c (date chips need no band) are here. Each rule returns
(rule name, user-facing reason) so nothing is suppressed without one.
"""
from __future__ import annotations

import re

from winnow.types import DEFAULT, Config, Region

_DATELIKE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\s*$")
_WATERMARK_TERMS = ("scanned by", "camscanner", "adobe scan",
                    "trial version", "confidential", "©")
_PAGE_NUM = re.compile(
    r"^\s*(page\s*)?[-–]?\s*\d{1,4}\s*[-–]?(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
_HANDLE = re.compile(r"^\s*@\w+\s*$")


# --- document memory (repeated furniture) -------------------------------

def norm(text: str) -> str:
    """Normalise for cross-page repetition: lowercase, digits -> #, strip."""
    return re.sub(r"\d", "#", re.sub(r"\s+", " ", text.strip().lower()))


def y_bucket(y0: int, page_h: int, cfg: Config = DEFAULT) -> int:
    """Quantise vertical position (fix 8b: position is part of the key)."""
    return int((y0 / max(1, page_h)) / cfg.y_bucket)


def collect(regions: list[Region], page_h: int, page_no: int,
            seen: dict, cfg: Config = DEFAULT) -> None:
    """Record every text region's (text, position) key against ``page_no``."""
    for r in regions:
        if r.nature == "text" and r.text.strip():
            seen.setdefault((norm(r.text), y_bucket(r.y0, page_h, cfg)), set()).add(page_no)


def repeated_keys(seen: dict, cfg: Config = DEFAULT) -> dict:
    """Keys seen on ``repeat_min_pages`` or more pages -> number of pages."""
    return {k: len(p) for k, p in seen.items() if len(p) >= cfg.repeat_min_pages}


# --- margins and rules ---------------------------------------------------

def band_height(page_h: int, page_w: int, cfg: Config = DEFAULT) -> float:
    """The height margin bands are fractions of: the page, but never more than
    an A4-proportioned one, so a long screenshot keeps page-sized margins."""
    return min(float(page_h), page_w * cfg.band_ref_aspect)


def mark_rule(r, text, ev, body_lh, cfg):
    """Unread ink that cannot be words: a drawn line, or one lone mark (a code
    block's "}", an ellipsis). On a real page, 11 of these were each announced
    as "Text here could not be read." Two or more letter-sized shapes are
    never a mark: an unreadable word is still reported, not hidden (§10)."""
    if r.nature != "text" or text:
        return None
    w, h = r.x1 - r.x0, r.y1 - r.y0
    if h < cfg.rule_max_lines * body_lh and w >= cfg.rule_min_aspect * h:
        ev["shape"] = [w, h]
        return "rule", f"a drawn line, no text ({w}×{h} px)"
    if (max(w, h) <= cfg.mark_max_lines * body_lh
            and r.meta.get("tall_marks", cfg.mark_max_lines) <= 1):
        ev["shape"] = [w, h]
        return "mark", f"a lone mark, not a word ({w}×{h} px)"
    return None


def chrome_rule(r, text, ev, page_h, repeated, n_pages, col_left, col_right, cfg,
                 page_w=None):
    """Return (rule, user-facing reason) for the first chrome rule that fires."""
    if r.nature != "text" or not text:
        return None
    in_band = "band" in ev
    # 1. Repeats at the same position across pages (8b: position in key).
    key = (norm(text), y_bucket(r.y0, page_h, cfg))
    bh = band_height(page_h, page_w or page_h, cfg)
    near_edge = r.y0 < cfg.repeat_band * bh or r.y1 > page_h - cfg.repeat_band * bh
    if key in repeated and near_edge:
        k = repeated[key]
        ev["repeats_on"] = k
        of = f" of {n_pages}" if n_pages else ""
        return "repeat", f"repeats at this position on {k}{of} pages"
    # 2. Watermark lexicon — anywhere on the page.
    low = text.lower()
    for term in _WATERMARK_TERMS:
        if term in low:
            return "watermark", f"watermark ('{term}')"
    # 2b. Sideways text outside the body column: an arXiv or journal stamp.
    # Sideways axis labels sit inside their figure and never reach here.
    if r.meta.get("sideways") and (r.x1 <= col_left or r.x0 >= col_right):
        ev["column_span"] = [int(col_left), int(col_right)]
        return "sideways", "sideways text in the page margin"
    # 3. Page number — pattern AND band. A bare number mid-paragraph is content.
    if _PAGE_NUM.match(text) and in_band:
        return "page number", f"page number in {ev['band']} margin"
    # 4. Metadata chip — no band requirement (8c).
    if _DATELIKE.match(text):
        return "date", "date stamp"
    if _HANDLE.match(text):
        return "handle", "social-media handle"
    # 5. Faint or small in a band (8a: compared on line height).
    if in_band and len(text) < cfg.chrome_max_chars:
        if ev["size_ratio"] < cfg.small_ratio:
            return "small", (f"small text in {ev['band']} margin "
                             f"({ev['size_ratio']}× body line height)")
        if ev.get("ink_delta", 0) >= cfg.faint_delta:
            return "faint", (f"faint text in {ev['band']} margin "
                             f"({ev['ink_delta']:+d} grey levels vs body)")
    return None
