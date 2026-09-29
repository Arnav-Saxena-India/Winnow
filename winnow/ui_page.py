"""The page half of the prototype: rendered pages, greyed furniture, the
region being spoken, and hover-for-evidence (upgrade 2)."""
from __future__ import annotations

import tkinter as tk

from PIL import Image, ImageTk

from winnow.debug_render import COLOURS

VIEW_W = 720


def _hex(rgb):
    return "#%02x%02x%02x" % rgb


def evidence_text(r) -> str:
    """'chrome — repeats at this position on 4 of 4 pages' plus every number."""
    lines = [f"{r.kind} — {r.meta['why']}" if r.meta.get("why") else r.kind]
    ev = r.meta.get("evidence", {})
    lines.append("  ".join(f"{k} {v:+d}" if k == "ink_delta" else f"{k} {v}"
                           for k, v in ev.items() if k != "rule"))
    for k in ("repaired", "escalated", "description_status", "caption", "figure_text", "unread"):
        if r.meta.get(k):
            lines.append(f"{k}: {r.meta[k]}")
    return "\n".join(lines)


class PageView:
    def __init__(self, parent):
        self.canvas = tk.Canvas(parent, bg="#d8d8d8", width=VIEW_W, highlightthickness=0)
        self.doc, self.images, self.offsets, self.boxes, self.scale = None, [], {}, {}, 1.0
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<MouseWheel>",
                         lambda e: self.canvas.yview_scroll(-e.delta // 120, "units"))
        self.canvas.create_text(VIEW_W // 2, 300, text="Drop a PDF or page image here\n(or press Open)",
                                font=("Segoe UI", 16), fill="#555", justify="center")
        self.tip = tk.Label(self.canvas, bg="#222", fg="#fff", font=("Consolas", 9),
                            justify="left", padx=6, pady=4, wraplength=420)

    def show(self, doc):
        self.doc = doc
        c = self.canvas
        c.delete("all")
        self.images, self.offsets, self.boxes = [], {}, {}
        y = 10
        for p in doc.pages:
            h, w = p.gray.shape
            self.scale = VIEW_W / w
            tkimg = ImageTk.PhotoImage(Image.fromarray(p.gray).resize((VIEW_W, int(h * self.scale))))
            self.images.append(tkimg)
            c.create_image(0, y, anchor="nw", image=tkimg)
            self.offsets[p.page_no] = y
            for i, r in enumerate(p.regions):
                self._draw(p.page_no, i, r, y)
            y += int(h * self.scale) + 16
        c.configure(scrollregion=(0, 0, VIEW_W, y))

    def _draw(self, page_no, i, r, y0):
        s = self.scale
        box = (r.x0 * s, y0 + r.y0 * s, r.x1 * s, y0 + r.y1 * s)
        if r.kind == "chrome":        # furniture visibly greyed: the listener sees it passed over
            item = self.canvas.create_rectangle(*box, fill="#9a9a9a", stipple="gray50", outline="")
        else:
            item = self.canvas.create_rectangle(
                *box, outline=_hex(COLOURS.get(r.kind, (200, 0, 0))),
                width=2 if r.kind in ("figure", "table") else 1,
                dash=() if r.kind in ("figure", "table") else (2, 3))
        self.boxes[item] = (page_no, i)

    def highlight(self, page_no, region):
        """Outline the region being spoken and scroll it into view."""
        self.canvas.delete("now")
        y0 = self.offsets.get(page_no, 0)
        if region >= 0 and self.doc:
            r, s = self.doc.pages[page_no - 1].regions[region], self.scale
            self.canvas.create_rectangle(r.x0 * s - 3, y0 + r.y0 * s - 3, r.x1 * s + 3,
                                         y0 + r.y1 * s + 3, outline="#ff8c00", width=3, tags="now")
            y0 += r.y0 * s
        total = float(self.canvas.cget("scrollregion").split()[3] or 1)
        self.canvas.yview_moveto(max(0.0, (y0 - 120) / total))

    def _hover(self, e):
        c = self.canvas
        x, y = c.canvasx(e.x), c.canvasy(e.y)
        hits = [self.boxes[i] for i in c.find_overlapping(x, y, x, y) if i in self.boxes]
        if not hits or not self.doc:
            self.tip.place_forget()
            return
        page_no, i = hits[-1]
        self.tip.config(text=evidence_text(self.doc.pages[page_no - 1].regions[i]))
        self.tip.place(x=min(e.x + 14, VIEW_W - 300), y=e.y + 14)
