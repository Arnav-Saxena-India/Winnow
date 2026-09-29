"""Input adapter: PDF text layer -> Regions. No OCR, no models (§7 step 3).

Coordinates are scaled by ``segment.work_scale`` (``work_height`` tall, or
``work_min_width`` wide for a very tall page); every pixel threshold in Config
assumes that scale. Words come from the text
layer; lines are grouped geometrically (8e); lines become blocks when they
are close, overlap horizontally, and share size and weight (so a heading
never merges into the paragraph under it).

Figures are found by ``pdf_figures``; words inside a figure become its
labels, not body text. Lines never join across a column gutter.
"""
from __future__ import annotations

import re
from statistics import median

import numpy as np
import pdfplumber

from winnow.pdf_figures import above_caption, absorb_labels, centre_in, figure_boxes, table_boxes
from winnow.tables import structure
from winnow.lines import group_lines, line_size
from winnow.segment import work_scale
from winnow.types import DEFAULT, Config, EncryptedDocument, Region, UnsupportedDocument

_BOLD = re.compile(r"bold|black|heavy|semibold", re.I)


def open_pdf(path: str):
    try:
        pdf = pdfplumber.open(path)
        _ = pdf.pages   # forces the trailer / encryption check
        return pdf
    except Exception as e:  # pdfminer raises several unrelated types here
        chain = f"{type(e).__name__} {e!r} {e.args!r}".lower()   # pdfplumber wraps the real one
        if "password" in chain or "encrypt" in chain:
            raise EncryptedDocument(f"{path} is encrypted; cannot read it") from e
        raise UnsupportedDocument(f"{path}: {e}") from e


def page_count(path: str) -> int:
    with open_pdf(path) as pdf:
        return len(pdf.pages)


def render(page, cfg: Config = DEFAULT, zoom: float = 1.0) -> np.ndarray:
    """Grayscale page image at work scale (× ``zoom`` for small text)."""
    img = page.to_image(resolution=72 * zoom * work_scale(page.width, page.height, cfg)).original
    return np.asarray(img.convert("L"))


def _words(page, s: float, cfg: Config, attrs=("size", "fontname"), **kw) -> list[dict]:
    return [dict(w, x0=w["x0"] * s, x1=w["x1"] * s, top=w["top"] * s,
                 bottom=w["bottom"] * s, size=w.get("size", w["x1"] - w["x0"]) * s,
                 bold=bool(_BOLD.search(w.get("fontname", ""))))
            for w in page.extract_words(extra_attrs=list(attrs),
                                        x_tolerance_ratio=cfg.word_gap_ratio, **kw)]


def extract_page(page, cfg: Config = DEFAULT) -> list[Region]:
    s = work_scale(page.width, page.height, cfg)
    words = [w for w in _words(page, s, cfg) if w["upright"]]
    side, upward = _sideways_words(page, s, cfg)
    figs = [above_caption(f, words, cfg) for f in figure_boxes(page, words + side, s, cfg)]
    inside = [[w for w in words + side if centre_in(w, f)] for f in figs]
    taken = {id(w) for ws in inside for w in ws}
    regions = [Region(*f, nature="figure", conf=100.0,
                      meta={"figure_text": " ".join(w["text"] for w in ws),
                            "source": "text-layer"})
               for f, ws in zip(figs, inside)]
    for box, rules, rules_x in table_boxes(page, words, s, figs, cfg):
        ws = [w for w in words if id(w) not in taken and centre_in(w, box)]
        grid = structure(ws, rules, rules_x, cfg)
        if grid:                           # else its words stay ordinary text
            taken |= {id(w) for w in ws}
            regions.append(Region(*box, nature="table", conf=100.0,
                                  text=" ".join(w["text"] for w in ws),
                                  meta={"table": grid, "source": "text-layer"}))
    lines = group_lines([w for w in words if id(w) not in taken], cfg,
                        walls=_gutters(words, float(page.width) * s, cfg))
    regions += [_block_region(b, cfg) for b in _group_blocks(lines, cfg)]
    regions += _sideways_regions([w for w in side if id(w) not in taken], upward, cfg)
    return sorted(absorb_labels(regions, cfg), key=lambda r: (r.y0, r.x0))


# --- sideways text --------------------------------------------------------

def _sideways_words(page, s: float, cfg: Config) -> tuple[list[dict], bool]:
    """Rotated words, read in their own direction. An arXiv stamp reads bottom
    to top; read top-down it came out as '5102 ceD 01'."""
    rot = [c for c in page.chars if not c.get("upright", True)]
    if not rot:
        return [], False
    upward = 2 * sum(c["matrix"][1] > 0 for c in rot) >= len(rot)
    # Not split on size: a rotated char's "size" is its advance width, so every
    # letter differed and the stamp came out as 'a r X i v'. Size = column width.
    ws = _words(page, s, cfg, attrs=("fontname",), char_dir_rotated="btt" if upward else "ttb")
    return [w for w in ws if not w["upright"]], upward


def _sideways_regions(words: list[dict], upward: bool, cfg: Config) -> list[Region]:
    """One region per vertical run of sideways words (grouped by x overlap)."""
    runs: list[list[dict]] = []
    for w in sorted(words, key=lambda w: w["x0"]):
        if runs and w["x0"] < max(v["x1"] for v in runs[-1]):
            runs[-1].append(w)
        else:
            runs.append([w])
    out = []
    for run in runs:
        run.sort(key=lambda w: -w["bottom"] if upward else w["top"])
        out.append(Region(
            int(min(w["x0"] for w in run)), int(min(w["top"] for w in run)),
            int(max(w["x1"] for w in run)) + 1, int(max(w["bottom"] for w in run)) + 1,
            nature="text", text=" ".join(w["text"] for w in run), conf=100.0,
            meta={"sideways": True, "line_count": 1, "source": "text-layer",
                  "line_h": round(median(w["x1"] - w["x0"] for w in run), 1),
                  "bold": _is_bold(run, cfg)}))
    return out


# --- columns ------------------------------------------------------------

def _gutters(words: list[dict], page_w: float, cfg: Config) -> list[float]:
    """x positions of column gutters: strips in the middle of the page that
    (almost) no word crosses, with plenty of text on both sides, and text
    lines ending and starting right at its edges (justified columns)."""
    if not words:
        return []
    cover = np.zeros(int(page_w) + 2, np.int32)
    for w in words:
        cover[int(w["x0"]):int(w["x1"]) + 1] += 1
    centres = np.array([(w["x0"] + w["x1"]) / 2 for w in words])
    limit = max(cfg.gutter_min_cover, cfg.gutter_max_cover * len(words))   # titles cross it
    lo, hi = int(page_w * cfg.gutter_side_frac), int(page_w * (1 - cfg.gutter_side_frac))
    walls, x = [], lo
    while x < hi:
        if cover[x] > limit:
            x += 1
            continue
        end = x
        while end < len(cover) and cover[end] <= limit:
            end += 1
        mid = (x + end) / 2
        left = len({int(w["top"] // cfg.y_band) for w in words if abs(w["x1"] - x) <= cfg.gutter_edge})
        right = len({int(w["top"] // cfg.y_band) for w in words if abs(w["x0"] - end) <= cfg.gutter_edge})
        crossing = int(cover[x:end].max())
        if (end - x >= cfg.gutter_min_w and (centres < mid).mean() >= cfg.gutter_side_frac
                and (centres > mid).mean() >= cfg.gutter_side_frac
                and min(left, right) >= max(cfg.gutter_min_edge, cfg.gutter_edge_ratio * crossing)):
            walls.append(mid)
        x = end
    return walls


# --- lines -> blocks ----------------------------------------------------

def _line_text(line: list[dict], cfg: Config) -> str:
    """Join words; mark superscripts/subscripts so x² is not 'x 2'."""
    size = line_size(line)
    normal = [w for w in line if w["size"] >= cfg.sup_size_ratio * size] or line
    mid = median((w["top"] + w["bottom"]) / 2 for w in normal)
    out = ""
    for w in line:
        if w["size"] < cfg.sup_size_ratio * size:
            out += ("^" if (w["top"] + w["bottom"]) / 2 < mid else "_") + w["text"]
        else:
            out += (" " if out else "") + w["text"]
    return out


def _is_bold(line: list[dict], cfg: Config) -> bool:
    return sum(len(w["text"]) for w in line if w["bold"]) >= \
        cfg.bold_frac * sum(len(w["text"]) for w in line)


def _group_blocks(lines: list[list[dict]], cfg: Config) -> list[list[list[dict]]]:
    blocks: list[list[list[dict]]] = []
    for ln in lines:
        x0, x1 = min(w["x0"] for w in ln), max(w["x1"] for w in ln)
        top = min(w["top"] for w in ln)
        for b in blocks:
            last = b[-1]
            bx0 = min(w["x0"] for l in b for w in l)
            bx1 = max(w["x1"] for l in b for w in l)
            gap = top - max(w["bottom"] for w in last)
            same_size = abs(line_size(ln) - line_size(last)) <= cfg.para_size_tol * line_size(last)
            if (0 <= gap <= cfg.dilate_v and min(x1, bx1) > max(x0, bx0) and same_size
                    and _is_bold(ln, cfg) == _is_bold(last, cfg)):
                b.append(ln)
                break
        else:
            blocks.append([ln])
    return blocks


def _block_region(block: list[list[dict]], cfg: Config) -> Region:
    words = [w for ln in block for w in ln]
    text = ""
    for ln in block:
        t = _line_text(ln, cfg)
        if text.endswith("-") and t[:1].islower():
            text = text[:-1] + t           # de-hyphenate a wrapped word
        else:
            text += (" " if text else "") + t
    heights = [max(w["bottom"] for w in ln) - min(w["top"] for w in ln) for ln in block]
    return Region(
        int(min(w["x0"] for w in words)), int(min(w["top"] for w in words)),
        int(max(w["x1"] for w in words)) + 1, int(max(w["bottom"] for w in words)) + 1,
        nature="text", text=text, conf=100.0,
        meta={"line_count": len(block), "line_h": round(median(heights), 1),
              "bold": all(_is_bold(ln, cfg) for ln in block), "source": "text-layer"})
