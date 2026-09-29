"""Labelling window for the benchmark: one keypress per box.

    python -m winnow label DOC OUT.labels.json --labeller NAME   # blank template
    python -m winnow annotate OUT.labels.json                    # this window

The page is shown with every box; the current one is thick. Keys 1-8 set its
kind and move on to the next box without one. Winnow's own guess is never
shown, so the labeller is not anchored on the classifier being evaluated.
Drag on the page to add a region Winnow missed (content it never produced is
the failure ``content lost`` counts); Delete removes a box that is not a
region at all. Every change is saved at once.
"""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

from winnow import pipeline
from winnow.debug_render import COLOURS
from winnow.evaluate import KINDS

VIEW_W = 900


# --- bookkeeping (pure) -------------------------------------------------

def boxes_in_order(lab: dict) -> list[tuple[str, int]]:
    """Every (page, index) pair, pages in numeric order."""
    return [(p, i) for p in sorted(lab["pages"], key=int) for i in range(len(lab["pages"][p]))]


def next_open(lab: dict, after: tuple[str, int] | None = None) -> tuple[str, int] | None:
    """The next box with no kind, after ``after`` (wrapping round), or None."""
    order = boxes_in_order(lab)
    start = order.index(after) + 1 if after in order else 0
    for p, i in order[start:] + order[:start]:
        if lab["pages"][p][i]["kind"] not in KINDS:
            return p, i
    return None


def progress(lab: dict) -> tuple[int, int]:
    """(boxes labelled, boxes in all)."""
    order = boxes_in_order(lab)
    return sum(1 for p, i in order if lab["pages"][p][i]["kind"] in KINDS), len(order)


def save(lab: dict, path: str | Path) -> None:
    tmp = Path(str(path) + ".tmp")                 # never leave a half-written file
    tmp.write_text(json.dumps(lab, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# --- the window ---------------------------------------------------------

def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(rgb)


class Labeller:
    def __init__(self, root, path: str):
        self.root, self.path = root, Path(path)
        self.lab = json.loads(self.path.read_text(encoding="utf-8"))
        self.doc = str((self.path.parent / self.lab["doc"]).resolve())
        self.cur = next_open(self.lab) or boxes_in_order(self.lab)[0]
        self._page, self._img, self._drag = None, None, None
        root.title(f"Winnow labels: {self.path.name}")
        side = tk.Frame(root, padx=8, pady=8)
        side.pack(side="right", fill="y")
        self.info = tk.Label(side, justify="left", anchor="nw", wraplength=260, font=("Segoe UI", 11))
        self.info.pack(fill="x")
        keys = "\n".join(f"{n}  {k}" for n, k in enumerate(KINDS, 1))
        tk.Label(side, justify="left", font=("Consolas", 11), pady=10,
                 text=f"{keys}\n\n→ ←  next / previous box\nPgDn PgUp  page\n"
                      f"Delete  not a region\ndrag  add a missed region").pack(anchor="w")
        self.canvas = tk.Canvas(root, width=VIEW_W, height=min(900, root.winfo_screenheight() - 120),
                                background="#888")
        sb = tk.Scrollbar(root, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        for n, kind in enumerate(KINDS, 1):
            root.bind(str(n), lambda _e, k=kind: self.set_kind(k))
        root.bind("<Right>", lambda _e: self.step(1))
        root.bind("<Left>", lambda _e: self.step(-1))
        root.bind("<Next>", lambda _e: self.page_step(1))
        root.bind("<Prior>", lambda _e: self.page_step(-1))
        root.bind("<Delete>", lambda _e: self.delete())
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-e.delta // 120, "units"))
        self.show()

    def show(self):
        p, i = self.cur
        if self._page != p:
            gray = pipeline.analyse_page(self.doc, int(p)).gray
            self.scale = VIEW_W / gray.shape[1]
            h = int(gray.shape[0] * self.scale)
            self._img = ImageTk.PhotoImage(Image.fromarray(gray).resize((VIEW_W, h)))
            self._page = p
            self.canvas.configure(scrollregion=(0, 0, VIEW_W, h))
        c, s = self.canvas, self.scale
        c.delete("all")
        c.create_image(0, 0, image=self._img, anchor="nw")
        boxes = self.lab["pages"][p]
        for j, b in enumerate(boxes):
            colour = _hex(COLOURS.get(b["kind"], (220, 0, 0)))
            c.create_rectangle(*(v * s for v in b["box"]), outline=colour,
                               width=4 if j == i else 1, dash=() if b["kind"] in KINDS else (3, 2))
        done, total = progress(self.lab)
        if boxes:
            b = boxes[i]
            y = b["box"][1] * s / max(1, int(c.cget("scrollregion").split()[3]))
            c.yview_moveto(max(0.0, y - 0.2))
            self.info.configure(text=f"page {p}, box {i + 1} of {len(boxes)}\n"
                                     f"labelled {done} of {total}\n\n"
                                     f"kind: {b['kind'] or '—'}\n\n“{b.get('text', '')}”")
        else:
            self.info.configure(text=f"page {p}: no boxes. Drag to add one.")

    def set_kind(self, kind):
        p, i = self.cur
        if not self.lab["pages"][p]:
            return
        self.lab["pages"][p][i]["kind"] = kind
        save(self.lab, self.path)
        self.cur = next_open(self.lab, self.cur) or self.cur
        self.show()

    def step(self, d):
        order = boxes_in_order(self.lab)
        if order:
            self.cur = order[(order.index(self.cur) + d) % len(order)] if self.cur in order else order[0]
            self.show()

    def page_step(self, d):
        pages = sorted(self.lab["pages"], key=int)
        p = pages[(pages.index(self.cur[0]) + d) % len(pages)]
        self.cur = (p, 0)
        self.show()

    def delete(self):
        p, i = self.cur
        if self.lab["pages"][p]:
            del self.lab["pages"][p][i]
            save(self.lab, self.path)
            self.cur = (p, max(0, min(i, len(self.lab["pages"][p]) - 1)))
            self.show()

    def _xy(self, e):
        return self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)

    def _press(self, e):
        self._drag = (*self._xy(e), None)

    def _motion(self, e):
        x0, y0, r = self._drag
        if r:
            self.canvas.delete(r)
        self._drag = (x0, y0, self.canvas.create_rectangle(x0, y0, *self._xy(e), outline="#06c", width=2))

    def _release(self, e):
        x0, y0, _ = self._drag
        x1, y1 = self._xy(e)
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:       # a click, not a drag
            return self.show()
        s = self.scale
        box = [int(min(x0, x1) / s), int(min(y0, y1) / s), int(max(x0, x1) / s), int(max(y0, y1) / s)]
        p = self.cur[0]
        self.lab["pages"][p].append({"box": box, "kind": "", "text": "(added by labeller)"})
        save(self.lab, self.path)
        self.cur = (p, len(self.lab["pages"][p]) - 1)
        self.show()


def main(path: str) -> int:
    root = tk.Tk()
    Labeller(root, path)
    root.mainloop()
    return 0
