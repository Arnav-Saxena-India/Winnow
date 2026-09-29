"""Tables: rows, columns and headers from a ruled block's words, and how to
speak them (the taxonomy's ``table``: header-qualified rows).

Pure: words and rule positions in, cells out; no pixels, no backends.

  rows     words grouped by vertical overlap. A row with only one cell filled
           and no label continues the row above (a cell that wrapped); a
           fuller row without a label shares the label above (a group).
  columns  x-strips that no body-row word crosses (at least ``table_col_gap``
           wide), plus any vertical rules. Body rows only: a header cell may
           span columns ("COCO val" over "@.5" and "@[.5,.95]").
  header   rows above the first rule inside the table (booktabs' midrule), or
           the first row when there is none. A header cell is given to every
           column it spans.
"""
from __future__ import annotations

from winnow.types import DEFAULT, Config


def _rows(words: list[dict], cfg: Config) -> list[list[dict]]:
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        for row in rows:
            top, bottom = min(v["top"] for v in row), max(v["bottom"] for v in row)
            if min(bottom, w["bottom"]) - max(top, w["top"]) >= cfg.line_overlap * min(
                    bottom - top, w["bottom"] - w["top"]):
                row.append(w)
                break
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w["x0"]) for r in rows]


def _bounds(rows: list[list[dict]], rules_x: list[float], cfg: Config) -> list[float]:
    """x positions between columns."""
    x0 = int(min(w["x0"] for r in rows for w in r))
    x1 = int(max(w["x1"] for r in rows for w in r)) + 1
    covered = [False] * (x1 - x0 + 1)
    for r in rows:
        for w in r:
            for x in range(int(w["x0"]) - x0, int(w["x1"]) - x0 + 1):
                covered[x] = True
    cuts, x = [], 0
    while x < len(covered):
        if covered[x]:
            x += 1
            continue
        end = x
        while end < len(covered) and not covered[end]:
            end += 1
        if end - x >= cfg.table_col_gap and 0 < x and end < len(covered):
            cuts.append(x0 + (x + end) / 2)
        x = end
    out: list[float] = []
    for c in sorted(cuts + [x for x in rules_x if x0 < x < x1]):
        if not out or c - out[-1] >= cfg.table_col_gap:   # a rule beside a gap: one boundary
            out.append(c)
    return out


def _home(w: dict, edges: list[float]) -> int:
    mid = (w["x0"] + w["x1"]) / 2
    return next(i for i in range(len(edges) - 1) if edges[i] <= mid < edges[i + 1])


def _cells(row: list[dict], edges: list[float]) -> list[str]:
    cols: list[list[str]] = [[] for _ in range(len(edges) - 1)]
    for w in row:
        cols[_home(w, edges)].append(w["text"])
    return [" ".join(c) for c in cols]


def _header_cells(row: list[dict], edges: list[float], extents, spans, cfg: Config) -> list[str]:
    """Header words grouped into cells (closer than a column gap). A cell
    heads its own column, every column whose content it mostly covers, and
    every column under a short rule just below it (booktabs' cmidrule): so
    "BLEU" heads both "EN-DE" and "EN-FR", and "Complexity per Layer" does
    not leak into "Layer Type"."""
    groups: list[list[dict]] = []
    for w in row:
        if groups and w["x0"] - groups[-1][-1]["x1"] < cfg.table_col_gap:
            groups[-1].append(w)
        else:
            groups.append([w])
    cols: list[list[str]] = [[] for _ in range(len(edges) - 1)]
    for g in groups:
        x0, x1 = g[0]["x0"], g[-1]["x1"]
        bottom, height = max(w["bottom"] for w in g), max(w["bottom"] - w["top"] for w in g)
        under = [(a, b) for a, b, y in spans if 0 <= y - bottom <= height and a < x1 and x0 < b]
        text = " ".join(w["text"] for w in g)
        home = _home({"x0": x0, "x1": x1}, edges)
        for i, (a, b) in enumerate(extents):
            covers = b > a and (min(x1, b) - max(x0, a)) >= cfg.table_head_cover * (b - a)
            spanned = b > a and any(ra <= (a + b) / 2 <= rb for ra, rb in under)
            if i == home or covers or spanned:
                cols[i].append(text)
    return [" ".join(c) for c in cols]


def structure(words: list[dict], rules: list[tuple], rules_x: list[float],
              cfg: Config = DEFAULT) -> dict | None:
    """-> {"header": [str per column], "rows": [[str per column]]}, or None
    when the words do not form at least a 2×2 grid. ``rules`` are (x0, x1, y):
    full-width ones end the header, short ones mark spanning header cells."""
    rows = _rows(words, cfg)
    if len(rows) < cfg.table_min_rows or not rules:
        return None
    width = max(r[1] for r in rules) - min(r[0] for r in rules)
    full = [y for a, b, y in rules if b - a >= cfg.table_full_rule * width]
    spans = [r for r in rules if r[1] - r[0] < cfg.table_full_rule * width]
    top, bottom = rows[0][0]["top"], max(w["bottom"] for w in rows[-1])
    inner = [y for y in sorted(full) if top < y < bottom]
    n_head = sum(1 for r in rows if max(w["bottom"] for w in r) <= inner[0]) if inner else 1
    n_head = max(1, min(n_head, len(rows) - 1))
    body = rows[n_head:]
    bounds = _bounds(body, rules_x, cfg)
    if len(bounds) + 1 < cfg.table_min_cols:
        return None
    edges = [float("-inf")] + bounds + [float("inf")]
    extents = []
    for i in range(len(edges) - 1):
        ws = [w for r in body for w in r if _home(w, edges) == i]
        extents.append((min(w["x0"] for w in ws), max(w["x1"] for w in ws)) if ws else (0, 0))
    header = [""] * (len(edges) - 1)
    for r in rows[:n_head]:
        header = [" ".join(p for p in (h, c) if p)
                  for h, c in zip(header, _header_cells(r, edges, extents, spans, cfg))]
    out: list[list[str]] = []
    bands: list[int] = []                             # full rules above each row
    for r in body:
        cells = _cells(r, edges)
        filled = sum(1 for c in cells if c)
        band = sum(1 for y in full if y <= (r[0]["top"] + max(w["bottom"] for w in r)) / 2)
        if out and bands[-1] == band and not cells[0] and filled == 1:
            out[-1] = [" ".join(p for p in (a, b) if p)   # a wrapped cell continues the row
                       for a, b in zip(out[-1], cells)]
        elif filled:
            out.append(cells)
            bands.append(band)
    out = _group_labels(out, bands)
    keep = [i for i in range(len(header)) if header[i] or any(r[i] for r in out)]
    if len(keep) < cfg.table_min_cols or not out:
        return None
    return {"header": [header[i] for i in keep], "rows": [[r[i] for i in keep] for r in out]}


def _group_labels(rows: list[list[str]], bands: list[int]) -> list[list[str]]:
    """Rows without a label belong to a group. When rules separate the
    groups, every row between two rules takes the one label there, which is
    often centred on the group, on a line of its own ("(A)" beside four rows
    was given to the row above it, and read as a row of nothing; 1706.03762,
    Table 3). Otherwise a row takes the label above it."""
    drop: set[int] = set()
    if len(set(bands)) > 1:
        for b in set(bands):
            idx = [i for i, x in enumerate(bands) if x == b]
            labels = {rows[i][0] for i in idx if rows[i][0]}
            if len(labels) == 1 and len(idx) > 1:
                label = labels.pop()
                for i in idx:
                    rows[i][0] = label
                    if not any(rows[i][1:]):
                        drop.add(i)               # the label's own line, now given out
    rows = [r for i, r in enumerate(rows) if i not in drop]
    for i in range(1, len(rows)):
        if not rows[i][0]:
            rows[i][0] = rows[i - 1][0]
    return rows


def speak(label: str, caption: str, table: dict) -> tuple[str, str]:
    """(tier 1 summary, tier 2 full reading). Each row is named by its first
    cell, and every other value by its column header."""
    rows, header = table["rows"], table["header"]
    head = f"{label}: {caption}" if caption else f"{label}."
    head = head if head.endswith((".", ":", "?", "!")) else head + "."
    summary = f"{head} {len(rows)} rows, {len(header)} columns."
    spoken = []
    for r in rows:
        parts = [f"{h} {v}".strip() for h, v in zip(header[1:], r[1:]) if v]
        name = r[0] or (header[0] if header[0] else "Row")
        spoken.append(f"{name}: {', '.join(parts)}." if parts else f"{name}.")
    return summary, summary + " " + " ".join(spoken)
