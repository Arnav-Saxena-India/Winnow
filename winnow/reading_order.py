"""Reading order — the sequence a listener should hear regions in.

Columns are found from the left edges of flowing blocks; a block wider than
``spanning_width`` breaks the page into sections that are read column by
column. Margin notes and furniture are not part of the flow: each is slotted
in after the flow block beside or above it, so a note lands between
paragraphs, never mid-sentence.

Invariant: never mutates ``kind``. Sets ``column`` and returns a new list.
"""
from __future__ import annotations

from statistics import median

from winnow.types import DEFAULT, Config, Region

_SIDE = ("annotation", "chrome")


def column_groups(blocks: list[Region], page_w: int,
                  cfg: Config = DEFAULT) -> list[list[Region]]:
    """Blocks grouped into columns by shared left edge. A column needs
    ``col_min_blocks`` blocks: on a real paper a centred title and an author
    line, one block each at odd x, chained the two columns into one and the
    title was read after the whole left column."""
    groups: list[list[Region]] = []
    for r in sorted(blocks, key=lambda r: r.x0):
        if groups and r.x0 - groups[-1][-1].x0 <= cfg.col_align_tol * page_w:
            groups[-1].append(r)
        else:
            groups.append([r])
    real = [g for g in groups if len(g) >= cfg.col_min_blocks] or groups
    cols = [real[0]] if real else []
    for g in real[1:]:
        if g[0].x0 - cols[-1][0].x0 > cfg.col_separation * page_w:
            cols.append(g)
        else:
            cols[-1] = cols[-1] + g
    return cols


def _column_lefts(flow: list[Region], page_w: int, cfg: Config) -> tuple[list[int], int | None]:
    """Column left edges, and where the column body starts (None if no column
    has enough blocks to count)."""
    cols = column_groups([r for r in flow if cfg.col_min_width * page_w <= r.x1 - r.x0
                          <= cfg.spanning_width * page_w], page_w, cfg)
    lefts = [int(median(r.x0 for r in g)) for g in cols] or [0]
    solid = [r for g in cols if len(g) >= cfg.col_min_blocks for r in g]
    return lefts, min((r.y0 for r in solid), default=None)


def _column_of(r: Region, lefts: list[int], page_w: int, cfg: Config) -> int:
    slack = cfg.col_separation * page_w / 2
    return max((j for j, x in enumerate(lefts) if x <= r.x0 + slack), default=0)


def reading_order(regions: list[Region], page_w: int,
                  cfg: Config = DEFAULT) -> list[Region]:
    flow = [r for r in regions if r.kind not in _SIDE]
    side = [r for r in regions if r.kind in _SIDE]
    lefts, body_top = _column_lefts(flow, page_w, cfg)
    for r in regions:
        r.column = _column_of(r, lefts, page_w, cfg)
    slack = cfg.col_separation * page_w / 2

    def spans(r: Region) -> bool:
        """Wide; or header material above the column body (title, authors);
        or crossing the gutter between two columns."""
        return (r.x1 - r.x0 > cfg.spanning_width * page_w
                or (len(lefts) > 1 and body_top is not None and r.y1 <= body_top)
                or any(r.x0 < x - slack and r.x1 > x + slack for x in lefts[1:]))

    def in_section(r: Region):
        return (r.column, r.y0 // cfg.y_band, r.x0)

    ordered: list[Region] = []
    section: list[Region] = []
    for r in sorted(flow, key=lambda r: (r.y0 // cfg.y_band, r.x0)):
        if spans(r):
            ordered += sorted(section, key=in_section) + [r]
            section = []
        else:
            section.append(r)
    ordered += sorted(section, key=in_section)

    # Slot each side region after the flow block above/beside it.
    anchors: dict[int, list[Region]] = {}
    for s in sorted(side, key=lambda r: (r.y0, r.x0)):
        mid = (s.y0 + s.y1) / 2
        above = [i for i, r in enumerate(ordered) if r.y0 <= mid]
        if above:
            top = max(ordered[i].y0 for i in above)
            same_row = [i for i in above if ordered[i].y0 == top]
            best = min(same_row, key=lambda i: abs(ordered[i].x0 - s.x0))
        else:
            best = -1
        anchors.setdefault(best, []).append(s)

    out = list(anchors.get(-1, []))
    for i, r in enumerate(ordered):
        out.append(r)
        out += anchors.get(i, [])
    return out
