"""Words -> lines, geometrically (8e): backend line numbers are not trusted.

Pure: word dicts ({x0, top, x1, bottom, text, ...}) in, lists of them out.
"""
from __future__ import annotations

from statistics import median

from winnow.types import DEFAULT, Config


def group_lines(words: list[dict], cfg: Config = DEFAULT, walls=()) -> list[list[dict]]:
    """Words ({x0, top, x1, bottom, text, ...}) -> lines, by vertical overlap
    and horizontal adjacency. Never by a backend's line index."""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        home = next((ln for ln in lines if _same_line(_bbox(ln), _bbox([w]), cfg, walls)), None)
        if home is None:
            lines.append([w])
        else:
            home.append(w)
    # Words that arrived out of x-order can leave one line in two pieces.
    merged = True
    while merged:
        merged = False
        for i in range(len(lines)):
            j = next((j for j in range(i + 1, len(lines))
                      if _same_line(_bbox(lines[i]), _bbox(lines[j]), cfg, walls)), None)
            if j is not None:
                lines[i] += lines.pop(j)
                merged = True
                break
    # Pieces of one row are read left to right. When OCR drops the middle of a
    # line the halves stay apart, and sorting on top alone read the right half
    # first when it sat a few px higher ("often see; In" for "In CP, you will
    # often see:", found on a real page).
    rows: list[list[list[dict]]] = []
    for ln in sorted((sorted(ln, key=lambda w: w["x0"]) for ln in lines),
                     key=lambda ln: min(w["top"] for w in ln)):
        if rows and _v_overlap(_bbox(rows[-1][0]), _bbox(ln), cfg):
            rows[-1].append(ln)
        else:
            rows.append([ln])
    return [ln for row in rows for ln in sorted(row, key=lambda ln: ln[0]["x0"])]


def _bbox(line: list[dict]) -> tuple[float, float, float, float]:
    return (min(w["x0"] for w in line), min(w["top"] for w in line),
            max(w["x1"] for w in line), max(w["bottom"] for w in line))


def _v_overlap(a, b, cfg: Config) -> bool:
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap >= cfg.line_overlap * min(a[3] - a[1], b[3] - b[1])


def _same_line(a, b, cfg: Config, walls=()) -> bool:
    """Level and close; never across a column gutter (``walls``): a paper
    whose gutter was 20 px had its two columns read as one, row by row."""
    gap = max(a[0], b[0]) - min(a[2], b[2])
    if not _v_overlap(a, b, cfg) or gap > cfg.dilate_h:
        return False
    lo, hi = min(a[2], b[2]), max(a[0], b[0])
    return not (gap >= cfg.wall_min_gap and any(lo <= x <= hi for x in walls))


def line_size(line: list[dict]) -> float:
    return median(w.get("size", w["bottom"] - w["top"]) for w in line)
