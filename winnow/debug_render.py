"""Debug render: every region boxed, coloured by kind, labelled with why.

The highest-value file in the repository — thresholds are tuned by looking
at this, not by reading transcripts.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from winnow.types import Region

COLOURS = {
    "chrome": (150, 150, 150), "heading": (30, 90, 220), "body": (40, 40, 40),
    "caption": (230, 130, 0), "figure": (0, 160, 60), "equation": (150, 50, 200),
    "annotation": (0, 150, 160), "table": (140, 90, 30),
}


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(gray: np.ndarray, regions: list[Region], path: str | None = None) -> Image.Image:
    img = Image.fromarray(gray).convert("RGB")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    font = _font(max(12, img.height // 90))
    for i, r in enumerate(regions):
        c = COLOURS.get(r.kind, (220, 0, 0))
        if r.kind == "chrome":
            d.rectangle((r.x0, r.y0, r.x1, r.y1), fill=c + (70,))
        d.rectangle((r.x0, r.y0, r.x1, r.y1), outline=c + (255,), width=3)
        why = r.meta.get("why") or r.meta.get("evidence", {}).get("rule", "")
        label = f"{i}:{r.kind}" + (f" - {why}" if why and why != r.kind else "")
        if r.context_of is not None:
            label += f" -> {r.context_of}"
        tb = d.textbbox((r.x0, r.y0), label, font=font)
        y = max(0, r.y0 - (tb[3] - tb[1]) - 4)
        d.rectangle((r.x0, y, r.x0 + tb[2] - tb[0] + 6, y + tb[3] - tb[1] + 4), fill=c + (230,))
        d.text((r.x0 + 3, y), label, fill=(255, 255, 255, 255), font=font)
    out = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    if path:
        out.save(path)
    return out
