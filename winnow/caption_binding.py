"""Caption binding — link each caption to the figure it describes.

A caption binds to the nearest figure or table that overlaps it
horizontally and lies within ``caption_max_gap`` vertically. One caption per
figure or table: pairs are matched like with like first ("Table" captions to
tables, others to figures), then nearest first; a caption left over is
spoken on its own. (Two captions once bound to one chart, and the
table's caption, bound but never spoken, vanished.) Sets ``context_of`` (an
index into the list it is given, i.e. the reading-ordered list) and ``meta``.
Touches nothing else.
"""
from __future__ import annotations

import re

from winnow.types import DEFAULT, Config, Region

_TABLE = re.compile(r"^\s*table\b", re.I)


def _gap(a: Region, b: Region) -> int:
    """Vertical distance between two boxes; 0 when they overlap."""
    return max(0, max(a.y0, b.y0) - min(a.y1, b.y1))


def _overlaps_x(a: Region, b: Region) -> bool:
    return min(a.x1, b.x1) > max(a.x0, b.x0)


def bind_captions(regions: list[Region], cfg: Config = DEFAULT) -> None:
    figures = [i for i, r in enumerate(regions) if r.kind in ("figure", "table")]
    captions = [i for i, r in enumerate(regions) if r.kind == "caption"]
    pairs = sorted((bool(_TABLE.match(regions[ci].text)) != (regions[fi].kind == "table"),
                    _gap(regions[ci], regions[fi]), ci, fi)
                   for ci in captions for fi in figures
                   if _overlaps_x(regions[ci], regions[fi])
                   and _gap(regions[ci], regions[fi]) <= cfg.caption_max_gap)
    bound: dict[int, int] = {}
    for _, gap, ci, fi in pairs:
        if ci in bound or fi in bound.values():
            continue
        bound[ci] = fi
        r = regions[ci]
        r.context_of = fi
        r.meta["bound_to"] = fi
        r.meta.setdefault("evidence", {})["caption_gap_px"] = gap
        regions[fi].meta["caption"] = r.text.strip()
    for ci in captions:
        if ci not in bound:
            regions[ci].context_of = None
            regions[ci].meta["orphan_caption"] = True
