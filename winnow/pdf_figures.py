"""Figures and tables in a text-layer PDF page (§7 step 3).

A figure is a cluster of vector graphics or embedded images that is sparse in
text, or that holds a diagonal stroke and is not as dense as body text (a
chart's plotted line; a table has only horizontal and vertical rules). White
fills that never stroke are invisible and ignored. Short text blocks beside a
figure are its labels; panels over one caption are one figure; a caption line
inside a figure's box ends the box. Checked on four real papers (46 clusters
labelled by eye: 23/23 figures found, 0/23 tables or algorithms taken).

A table is two or more aligned horizontal rules (booktabs' three, or a grid's)
with a caption just above or below ("Table N", or "Figure N": one paper
captions its tables that way); its cells come from ``tables``, and a block
that is not a grid stays text. The caption keeps framed algorithms out.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from winnow.lines import group_lines
from winnow.types import Config, Region

_CAPTION = re.compile(r"^\s*(fig|figure|table|chart)\s*\.?\s*\d+", re.I)
_CAPTION_LINE = re.compile(r"^\s*(fig|figure|table|chart)\s*\.?\s*\d+\s*[.:]", re.I)
# A figure label caught on the caption's line: "F Figure 2. Residual learning".
# A stray token is never a word (one character, or up to three with a non-letter,
# like "(a)"), and what follows must be a caption ("Figure 2." or "Figure 2:"),
# so prose such as "In Table 3 we" or "as in Fig. 2." is never cut.
_LEADING = re.compile(r"^((?:(?:\S|(?=\S*[^A-Za-z\s])\S{2,3})\s+){1,2})"
                      r"(?=(fig|figure|table|chart)\s*\.?\s*\d+\s*[.:])", re.I)
_CMYK = 4                                  # channels; white is all zero, not all one


def figure_boxes(page, words, s, cfg: Config) -> list[tuple[int, int, int, int]]:
    h, w = int(float(page.height) * s) + 1, int(float(page.width) * s) + 1
    mask = np.zeros((h + 1, w), np.uint8)
    page_area = float(page.width) * float(page.height)
    diag = []                              # centres of diagonal strokes
    for o in page.images + page.rects + page.curves + page.lines:
        ow, oh = o["x1"] - o["x0"], o["bottom"] - o["top"]
        if ow * oh >= cfg.page_bg_frac * page_area or _invisible(o, cfg):
            continue                       # page background, or a white fill on white paper
        p0 = (int(o["x0"] * s), int(o["top"] * s))
        p1 = (int(o["x1"] * s), int(o["bottom"] * s))
        cv2.rectangle(mask, p0, p1, 255, thickness=-1)
        if _diagonal(o, cfg):
            diag.append(((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.fig_close_kernel,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for x, y, bw, bh, _ in stats[1:n]:
        box = (int(x), int(y), int(x + bw), int(y + bh))
        if bw * bh < cfg.fig_min_area or bh < cfg.sparse_min_h:
            continue                       # rules and header underlines (8d)
        chars = sum(len(wd["text"]) for wd in words if centre_in(wd, box))
        density = chars / (bw * bh / 1000)
        plotted = any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in diag)
        if density < cfg.sparse_max_density or (plotted and density < cfg.stroke_fig_max_density):
            out.append(box)                # a callout box or a table full of text is not a figure
    return out


def _near(a: Region, b: Region, gap: int) -> bool:
    return (a.x0 - gap <= b.x1 and b.x0 <= a.x1 + gap
            and a.y0 - gap <= b.y1 and b.y0 <= a.y1 + gap)


def absorb_labels(regions: list[Region], cfg: Config) -> list[Region]:
    """Axis titles, tick labels and panel titles sit just outside a chart's
    drawn area; read as text they came out as "beta sub 2 equals 0.99" six
    times (arXiv 1412.6980, p. 8). A figure takes in short blocks
    beside it, repeatedly, so a row of ticks then its axis title both join.
    A caption is never taken, so it stays a caption and stops the chain.
    Then panels merge: close together, or side by side over one caption (five
    two-chart figures in two real papers had one panel captioned, one not)."""
    figs = [r for r in regions if r.nature == "figure"]
    rest = [r for r in regions if r.nature != "figure"]
    for r in rest:
        m = _LEADING.match(r.text)
        near = [f for f in figs if _near(f, r, cfg.fig_label_gap)]
        if m and near:                     # the stray label goes back to its figure
            near[0].meta["figure_text"] = (near[0].meta["figure_text"] + " " + m.group(1)).strip()
            r.text = r.text[m.end():]
    changed = True
    while changed:
        changed = False
        for f in figs:
            for r in list(rest):
                if (len(r.text) <= cfg.fig_label_max_chars
                        and not _CAPTION.match(r.text) and _near(f, r, cfg.fig_label_gap)):
                    f.x0, f.y0 = min(f.x0, r.x0), min(f.y0, r.y0)
                    f.x1, f.y1 = max(f.x1, r.x1), max(f.y1, r.y1)
                    f.meta["figure_text"] = (f.meta["figure_text"] + " " + r.text).strip()
                    rest.remove(r)
                    changed = True
        for a in figs:
            for b in figs:
                if a is not b and _near(a, b, cfg.fig_panel_gap) and _panels(a, b):
                    a.x0, a.y0, a.x1, a.y1 = (min(a.x0, b.x0), min(a.y0, b.y0),
                                              max(a.x1, b.x1), max(a.y1, b.y1))
                    a.meta["figure_text"] = (a.meta["figure_text"] + " " + b.meta["figure_text"]).strip()
                    figs.remove(b)
                    changed = True
                    break
            if changed:
                break
    for c in (r for r in rest if _CAPTION.match(r.text)):
        span = [f for f in figs if f.x0 >= c.x0 - cfg.fig_panel_gap
                and f.x1 <= c.x1 + cfg.fig_panel_gap and f.y1 <= c.y0]
        near = [f for f in span if c.y0 - f.y1 <= cfg.caption_max_gap]
        if not near:
            continue
        a = min(near, key=lambda f: c.y0 - f.y1)
        for b in [f for f in span if f is not a and (f in near or _level(a, f))]:
            a.x0, a.y0, a.x1, a.y1 = (min(a.x0, b.x0), min(a.y0, b.y0),   # panels over one
                                      max(a.x1, b.x1), max(a.y1, b.y1))   # caption: one figure
            a.meta["figure_text"] = (a.meta["figure_text"] + " " + b.meta["figure_text"]).strip()
            figs.remove(b)
    return figs + rest


def table_boxes(page, words: list[dict], s: float, figs, cfg: Config) -> list[tuple]:
    """-> [(box, rules, rules_x)] for each captioned, ruled table; ``rules``
    are (x0, x1, y), full-width and partial (booktabs' cmidrule over a
    spanning header) alike. Rules group when they overlap by most of the
    shorter one: a table whose inner rules differ in length was cut in two."""
    hs = sorted((e["x0"] * s, e["x1"] * s, e["top"] * s) for e in _edges(page, "h", cfg)
                if (e["x1"] - e["x0"]) * s >= cfg.table_col_gap)
    hs.sort(key=lambda h: (h[2], h[0]))
    rules: list[tuple] = []
    for h in hs:                           # a thin filled rect has two edges: keep one
        if not any(abs(h[2] - r[2]) <= cfg.table_rule_tol and abs(h[0] - r[0]) <= cfg.table_rule_tol
                   and abs(h[1] - r[1]) <= cfg.table_rule_tol for r in rules):
            rules.append(h)
    lines = [(min(w["top"] for w in ln), max(w["bottom"] for w in ln),
              min(w["x0"] for w in ln), max(w["x1"] for w in ln),
              " ".join(w["text"] for w in ln)) for ln in group_lines(words, cfg)]
    caps = []
    for c in (c for c in lines if _CAPTION_LINE.match(c[4])):
        bottom = c[1]                      # a caption above its table runs over several
        for ln in sorted(lines):           # lines: measure from where the paragraph ends
            if (0 <= ln[0] - bottom <= cfg.dilate_v and ln[2] < c[3] and c[2] < ln[3]
                    and not any(bottom - cfg.table_rule_tol <= r[2] <= ln[1]   # never into its
                                for r in rules)):          # table: a rule sits level with its header
                bottom = ln[1]
        caps.append((c[0], bottom, c[2], c[3], c[4]))

    def joins(g, r):
        last = max(g, key=lambda q: q[2])
        overlap = min(r[1], max(q[1] for q in g)) - max(r[0], min(q[0] for q in g))
        return (overlap >= cfg.table_rule_overlap * min(r[1] - r[0], last[1] - last[0])
                and r[2] - last[2] <= cfg.table_max_rule_gap
                and not any(last[2] < c[0] < r[2] and c[2] < r[1] and r[0] < c[3]   # a caption over
                            for c in caps))                            # it, not one in the next column
    groups: list[list[tuple]] = []
    for r in rules:
        g = next((g for g in groups if joins(g, r)), None)
        if g is None:
            groups.append([r])
        else:
            g.append(r)
    out = []
    for g in groups:
        width = max(r[1] for r in g) - min(r[0] for r in g)
        full = [r for r in g if r[1] - r[0] >= max(cfg.table_rule_min_w, cfg.table_full_rule * width)]
        if len(full) < 2:
            continue
        lo, hi = min(r[2] for r in full), max(r[2] for r in full)
        g = [r for r in g if lo <= r[2] <= hi]    # a panel frame above is not part of it
        box = (int(min(r[0] for r in g)), int(lo), int(max(r[1] for r in g)) + 1, int(hi) + 1)
        if any(f[0] < box[2] and box[0] < f[2] and f[1] < box[3] and box[1] < f[3] for f in figs):
            continue
        captioned = any(c[2] < box[2] and box[0] < c[3]
                        and (0 <= box[1] - c[1] <= cfg.caption_max_gap
                             or 0 <= c[0] - box[3] <= cfg.caption_max_gap) for c in caps)
        if not captioned:
            continue
        xs = sorted({round(e["x0"] * s) for e in _edges(page, "v", cfg)
                     if box[0] < e["x0"] * s < box[2] and box[1] <= e["top"] * s <= box[3]})
        out.append((box, g, xs))
    return out


def _edges(page, axis: str, cfg: Config) -> list[dict]:
    """Horizontal ("h") or vertical ("v") rules: drawn edges, and embedded
    images a hairline thin (one paper drew every table rule as a 0.48 pt
    image, so its tables had no rules at all)."""
    thin = [m for m in page.images if min(m["x1"] - m["x0"], m["bottom"] - m["top"])
            <= cfg.rule_image_max_pt]
    if axis == "h":
        return page.horizontal_edges + [m for m in thin if m["x1"] - m["x0"] > m["bottom"] - m["top"]]
    return page.vertical_edges + [m for m in thin if m["bottom"] - m["top"] > m["x1"] - m["x0"]]


def _level(a: Region, b: Region) -> bool:
    """Side by side: overlapping vertically by half the shorter one's height.
    A short panel beside a tall one can end far above their shared caption."""
    return 2 * (min(a.y1, b.y1) - max(a.y0, b.y0)) >= min(a.y1 - a.y0, b.y1 - b.y0)


def _panels(a: Region, b: Region) -> bool:
    """Side by side and mostly level, or stacked and mostly aligned."""
    oy = min(a.y1, b.y1) - max(a.y0, b.y0)
    ox = min(a.x1, b.x1) - max(a.x0, b.x0)
    return (2 * oy >= min(a.y1 - a.y0, b.y1 - b.y0)) or (2 * ox >= min(a.x1 - a.x0, b.x1 - b.x0))


def _invisible(o: dict, cfg: Config) -> bool:
    """Filled white and never stroked: invisible on white paper. A chart's
    white background joined a page header, the chart and its caption into
    one "figure" (arXiv 1412.6980, p. 8)."""
    if o.get("stroke") or not o.get("fill"):
        return False
    c = o.get("non_stroking_color")
    c = c if isinstance(c, (tuple, list)) else (c,)
    if not c or any(v is None or not isinstance(v, (int, float)) for v in c):
        return False
    return max(c) <= 1 - cfg.white_min if len(c) == _CMYK else min(c) >= cfg.white_min


def _diagonal(o: dict, cfg: Config) -> bool:
    """Any segment moving in both x and y: a plotted line, an arrow, a slope."""
    pts = o.get("pts") or []
    return any(abs(x1 - x0) > cfg.diag_min_pt and abs(y1 - y0) > cfg.diag_min_pt
               for (x0, y0), (x1, y1) in zip(pts, pts[1:]))


def above_caption(box, words: list[dict], cfg: Config):
    """End a figure above any caption line inside it. A graphics cluster that
    reached down over its caption took the caption's words as figure labels,
    and the caption was never spoken (arXiv 1706.03762, pp. 14-15)."""
    g = cfg.fig_label_gap                  # whole lines touching the box: "Figure" itself
    inside = [w for w in words if box[1] <= (w["top"] + w["bottom"]) / 2 <= box[3]
              and w["x1"] >= box[0] - g and w["x0"] <= box[2] + g]   # can start just outside
    for ln in group_lines(inside, cfg):
        top = min(w["top"] for w in ln)
        if _CAPTION_LINE.match(" ".join(w["text"] for w in ln)) and top - box[1] >= cfg.sparse_min_h:
            return (box[0], box[1], box[2], int(top) - 1)
    return box


def centre_in(w: dict, box) -> bool:
    cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]
