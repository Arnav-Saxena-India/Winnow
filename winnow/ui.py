"""The prototype: one window. Drag a document onto it.

Left (ui_page): the page, furniture greyed, the region being spoken outlined
and scrolled into view; hover any region for the rule and its numbers.
Right: the transcript and "Skipped (n)" with every reason.
Bottom: transport, speed, tier slider (filters the existing plan — no
re-analysis), compare naive ⟷ winnow, and the network counter, which reads
telemetry.attempts() under an enforced offline guard. The status line
carries the self-check badge.

Backends arrive as arguments; this module never constructs one.
"""
from __future__ import annotations

import re
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from winnow import pipeline, telemetry
from winnow.speech import at_tier
from winnow.ui_page import PageView

TIERS = ("announce", "summarise", "full")
_SENT = re.compile(r"(?<=[.!?])\s+")


class App:
    def __init__(self, root, speaker, ocr, describer):
        self.root, self.speaker, self.ocr, self.describer = root, speaker, ocr, describer
        self.doc = None
        self.queue: list[tuple] = []        # (page_no, region_idx, text)
        self.pos, self.playing, self._last_tier = 0, False, 2
        self._build()

    # --- layout ---------------------------------------------------------
    def _build(self):
        r = self.root
        r.title("Winnow")
        r.geometry(f"{min(1240, r.winfo_screenwidth() - 40)}x"
                   f"{min(900, r.winfo_screenheight() - 100)}+10+10")
        main = ttk.PanedWindow(r, orient="horizontal")   # packed last: the bars keep their space
        left = ttk.Frame(main)
        self.page = PageView(left)
        sb = ttk.Scrollbar(left, command=self.page.canvas.yview)
        self.page.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.page.canvas.pack(fill="both", expand=True)
        main.add(left, weight=3)

        right = ttk.Frame(main)
        ttk.Label(right, text="TRANSCRIPT", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=6)
        self.text = tk.Text(right, wrap="word", font=("Segoe UI", 11), width=44, relief="flat",
                            padx=8, pady=6, cursor="arrow")
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("now", background="#fff1a8")
        self.skip_btn = ttk.Button(right, text="Skipped (0) ▼", command=self._toggle_skipped)
        self.skip_btn.pack(fill="x", padx=6)
        self.skipped = tk.Text(right, height=10, wrap="word", font=("Segoe UI", 9),
                               foreground="#555", relief="flat", padx=8)
        main.add(right, weight=2)

        bar = ttk.Frame(r, padding=6)
        for label, cmd in (("Open", self._open), ("▶ / ‖", self._toggle_play),
                           ("⏮", lambda: self._jump(-1)), ("⏭", lambda: self._jump(1))):
            ttk.Button(bar, text=label, width=6, command=cmd).pack(side="left", padx=2)
        ttk.Label(bar, text="speed").pack(side="left", padx=(12, 2))
        self.speed = tk.DoubleVar(value=1.0)
        ttk.Spinbox(bar, from_=0.5, to=2.0, increment=0.25, width=5, textvariable=self.speed,
                    command=lambda: self.speaker.set_speed(self.speed.get())).pack(side="left")
        ttk.Label(bar, text="tier: announce").pack(side="left", padx=(16, 2))
        self.tier = tk.IntVar(value=2)
        ttk.Scale(bar, from_=0, to=2, variable=self.tier, length=140,
                  command=self._tier_moved).pack(side="left")
        self.tier_label = ttk.Label(bar, text="full", width=10)
        self.tier_label.pack(side="left")
        self.naive = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="compare: naive ⟷ winnow", variable=self.naive,
                        command=self._rebuild).pack(side="left", padx=12)
        self.net = ttk.Label(bar, text="net 0", font=("Segoe UI", 10, "bold"))
        self.net.pack(side="right", padx=6)
        self.status = ttk.Label(r, text=f"voice: {self.speaker.name} · OCR: "
                                f"{self.ocr.name if self.ocr else 'none'} · describer: "
                                f"{self.describer.model_id}", padding=(8, 2), foreground="#444")
        self.status.pack(side="bottom", fill="x")
        bar.pack(side="bottom", fill="x")
        main.pack(fill="both", expand=True)
        self._tick_net()

    # --- loading --------------------------------------------------------
    def _open(self):
        path = filedialog.askopenfilename(
            filetypes=[("Documents", "*.pdf *.png *.jpg *.jpeg *.tif *.tiff")])
        if path:
            self.load(path)

    def load(self, path: str):
        self._stop()
        self.status.config(text=f"analysing {path} …")

        def work():   # shown at once; figures are described behind the reader
            try:
                doc = pipeline.run(path, ocr=self.ocr, describe=False)
                self.root.after(0, lambda: self._show(doc))
                pipeline.describe_pages(doc, self.describer, on_page=lambda p: self.root.after(
                    0, lambda: self._rebuild() if self.doc is doc else None))
            except Exception as e:  # shown to the user, never silent
                msg = f"could not read {path}: {e}"
                self.root.after(0, lambda: self.status.config(text=msg))
        threading.Thread(target=work, daemon=True).start()

    def _show(self, doc):
        self.doc = doc
        self.page.show(doc)
        bad = f" · {len(doc.errors)} SELF-CHECK ERRORS" if doc.errors else ""
        self.status.config(text=f"{doc.path} · {len(doc.pages)} pages · self-check: {doc.badge()}"
                                f"{bad} · voice {self.speaker.name} · OCR "
                                f"{self.ocr.name if self.ocr else 'none'} · describer "
                                f"{self.describer.model_id} · config {doc.config_hash}")
        self._rebuild()

    # --- transcript -----------------------------------------------------
    def _tier_moved(self, _=None):
        t = round(self.tier.get())
        self.tier_label.config(text=TIERS[t])
        if self._last_tier != t:
            self._last_tier = t
            self._rebuild()

    def _rebuild(self):
        if not self.doc:
            return
        current = self.queue[self.pos][:2] if self.pos < len(self.queue) else None
        self.queue, skipped = [], []
        for p in self.doc.pages:
            if self.naive.get():
                self.queue += [(p.page_no, -1, s)
                               for s in _SENT.split(" ".join(p.naive.split())) if s]
                continue
            for u in at_tier(p.utterances, round(self.tier.get())):
                if u.suppressed:
                    skipped.append((p.page_no, u))
                else:
                    self.queue.append((p.page_no, u.region, u.text))
        self.pos = next((n for n, q in enumerate(self.queue) if q[:2] == current), 0)
        self.text.delete("1.0", "end")
        for n, (_, _, text) in enumerate(self.queue):
            self.text.insert("end", text + "\n\n", (f"u{n}",))
            self.text.tag_bind(f"u{n}", "<Button-1>", lambda e, n=n: self._play_from(n))
        self.skip_btn.config(text=f"Skipped ({len(skipped)}) ▼")
        self.skipped.delete("1.0", "end")
        for page, u in skipped:
            self.skipped.insert("end", f"· p{page} {u.text[:48]}\n    {u.why}\n")
        self._highlight()

    def _toggle_skipped(self):
        if self.skipped.winfo_ismapped():
            self.skipped.pack_forget()
        else:
            self.skipped.pack(fill="x", padx=6, pady=4)

    # --- playback -------------------------------------------------------
    def _toggle_play(self):
        if self.playing:
            self._stop()
        elif self.queue:
            self.playing = True
            self._say()

    def _play_from(self, n):
        self._stop()
        self.pos, self.playing = n, True
        self._say()

    def _jump(self, d):
        was = self.playing
        self._stop()
        self.pos = max(0, min(len(self.queue) - 1, self.pos + d))
        self._highlight()
        if was:
            self.playing = True
            self._say()

    def _stop(self):
        self.playing = False
        self.speaker.stop()

    def _say(self):
        if not self.playing or self.pos >= len(self.queue):
            self.playing = False
            return
        self._highlight()
        if self._pending(*self.queue[self.pos][:2]):   # its description is on the way
            self.root.after(200, self._say)
            return
        self.speaker.set_speed(self.speed.get())
        self.speaker.say(self.queue[self.pos][2])
        self.root.after(250, self._poll)

    def _pending(self, page_no, region) -> bool:
        page = next((p for p in self.doc.pages if p.page_no == page_no), None)
        return bool(page and 0 <= region < len(page.regions)
                    and page.regions[region].meta.get("description_status") == "pending")

    def _poll(self):
        if not self.playing:
            return
        if self.speaker.busy():
            self.root.after(100, self._poll)
        else:
            self.pos += 1
            self._say()

    def _highlight(self):
        self.text.tag_remove("now", "1.0", "end")
        if not self.queue or self.pos >= len(self.queue):
            return
        page_no, region, _ = self.queue[self.pos]
        rng = self.text.tag_ranges(f"u{self.pos}")
        if rng:
            self.text.tag_add("now", *rng)
            self.text.see(rng[0])
        self.page.highlight(page_no, region)

    def _tick_net(self):
        n = telemetry.attempts()
        self.net.config(text=f"net {n}", foreground="#0a7d2c" if n == 0 else "#c00")
        self.root.after(1000, self._tick_net)


def main(path=None, speaker=None, ocr=None, describer=None) -> int:
    try:   # crisp text under Windows display scaling
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        root = TkinterDnD.Tk()
    except Exception:
        root, DND_FILES = tk.Tk(), None
    app = App(root, speaker, ocr, describer)
    if DND_FILES:
        root.drop_target_register(DND_FILES)
        root.dnd_bind("<<Drop>>", lambda e: app.load(root.tk.splitlist(e.data)[0]))
    if path:
        root.after(100, lambda: app.load(path))
    with telemetry.assert_offline():
        root.mainloop()
    return 0
