"""Figures, captions and columns in real papers.

Found running four arXiv papers (1512.03385, 1412.6980, 1502.03167,
1706.03762); they are not redistributable, so each bug is rebuilt here as a
small PDF with the same geometry. Before these fixes, charts went undetected
and their tick labels were read aloud, a table's caption vanished, and one
paper's two columns were read interleaved, row by row.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from winnow import pipeline
from winnow.caption_binding import bind_captions
from winnow.pdf_adapter import extract_page, open_pdf
from winnow.pdf_figures import _LEADING
from winnow.types import Region

BODY = ("Residual networks are easier to optimise and gain accuracy from depth. "
        "We evaluate them on several benchmarks and report the results below.")


def _chart(c, x, y, w, h, legend=True):
    """Axes, a plotted (diagonal) line, ticks and an axis title."""
    c.line(x, y, x, y + h)
    c.line(x, y, x + w, y)
    p = c.beginPath()
    p.moveTo(x, y + h * 0.9)
    for i, f in enumerate((0.6, 0.7, 0.35, 0.3, 0.1), 1):
        p.lineTo(x + w * i / 5, y + h * f)
    c.drawPath(p, stroke=1, fill=0)
    c.setFont("Helvetica", 7)
    for i, t in enumerate(("0", "10", "20")):
        c.drawString(x - 12, y + h * i / 2 - 2, t)
    for i in range(4):
        c.drawString(x + w * i / 3, y - 9, str(i))
    c.drawString(x + w / 2 - 8, y - 19, "iter.")
    if legend:                              # text inside the plot: too dense for 0.25
        for i, t in enumerate(("20-layer plain", "56-layer plain", "20-layer residual",
                               "56-layer residual")):
            c.drawString(x + w * 0.55, y + h * (0.95 - 0.1 * i), t)


def _body(c, y, lines=4, x=72):
    c.setFont("Times-Roman", 10)
    for i in range(lines):
        c.drawString(x, y - 12 * i, BODY[:80])


def _pdf(draw) -> str:
    d = tempfile.mkdtemp()
    path = str(Path(d) / "paper.pdf")
    c = canvas.Canvas(path, pagesize=letter)
    draw(c)
    c.save()
    return path


def _figures(path):
    doc = pipeline.run(path, workers=1)
    page = doc.pages[0]
    return page, [r for r in page.regions if r.kind == "figure"]


def _spoken(page):
    return " ".join(u.text for u in page.utterances if not u.suppressed)


class TestChartsAndTables(unittest.TestCase):

    def _page(self, c, white_background=False):
        c.setFont("Times-Roman", 9)
        c.drawString(72, 770, "Published as a conference paper")
        c.line(72, 765, 540, 765)
        if white_background:               # a figure's own background, invisible on paper
            c.setFillColorRGB(1, 1, 1)
            c.rect(70, 440, 400, 340, fill=1, stroke=0)
            c.setFillColorRGB(0, 0, 0)
        _chart(c, 110, 520, 260, 180)
        c.setFont("Times-Roman", 9)
        c.drawString(72, 470, "Figure 1. Training error falls with depth.")
        _body(c, 420)
        c.drawString(72, 350, "Table 1. Error rates on ImageNet.")
        for y in (340, 322, 250):           # a ruled table: only horizontal and vertical rules
            c.line(72, y, 400, y)
        c.line(200, 250, 200, 340)
        c.setFont("Times-Roman", 9)
        for i, (m, e) in enumerate((("method", "top-1 err."), ("ResNet-34", "25.03"),
                                    ("ResNet-50", "22.85"), ("ResNet-101", "21.75"),
                                    ("ResNet-152", "21.43"))):
            c.drawString(80, 328 - 16 * i, m)
            c.drawString(210, 328 - 16 * i, e)
        _body(c, 220)

    def test_labelled_chart_is_a_figure_and_its_ticks_are_not_read(self):
        page, figs = _figures(_pdf(self._page))
        self.assertEqual(len(figs), 1)
        self.assertTrue(figs[0].meta.get("caption", "").startswith("Figure 1."), figs[0].meta)
        spoken = _spoken(page)
        for label in ("iter.", "56-layer plain"):
            self.assertNotIn(label, spoken)

    def test_ruled_table_is_read_row_by_row_with_its_headers(self):
        """Tables were read as a stream: "method top-1 err. ResNet-34 25.03
        ...", with "ResNet-152" as "ResNet minus 152"."""
        page, _ = _figures(_pdf(self._page))
        tables = [r for r in page.regions if r.kind == "table"]
        self.assertEqual(len(tables), 1)
        spoken = _spoken(page)
        self.assertIn("Table 1: Error rates on ImageNet. 4 rows, 2 columns.", spoken)
        self.assertIn("ResNet-152: top-1 err. 21.43.", spoken)
        self.assertNotIn("Caption: Table 1", spoken)          # bound, not spoken alone

    def test_figure_captioned_table_with_image_rules_is_a_table(self):
        """1502.03167 captions its tables "Figure N" and draws every rule as a
        0.48 pt embedded image, so its tables had no rules and were read as
        a stream of numbers."""
        from PIL import Image
        from reportlab.lib.utils import ImageReader
        ink = ImageReader(Image.new("L", (4, 4), 0))

        def draw(c):
            _body(c, 740)
            for y in (600, 582, 510):
                c.drawImage(ink, 72, y, width=330, height=0.48)
            c.drawImage(ink, 200, 510, width=0.4, height=90)
            c.setFont("Times-Roman", 9)
            for i, (m, e) in enumerate((("Model", "Max accuracy"), ("Inception", "72.2%"),
                                        ("BN-Baseline", "72.7%"), ("BN-x5", "73.0%"),
                                        ("BN-x30", "74.8%"))):
                c.drawString(80, 588 - 16 * i, m)
                c.drawString(210, 588 - 16 * i, e)
            c.drawString(72, 495, "Figure 3: Inception and its batch-normalized variants.")
            _body(c, 460)
        page, figs = _figures(_pdf(draw))
        self.assertEqual(figs, [])
        spoken = _spoken(page)
        self.assertIn("Figure 3: Inception and its batch-normalized variants. 4 rows, 2 columns.",
                      spoken)
        self.assertIn("BN-x30: Max accuracy 74.8%.", spoken)

    def test_framed_prose_over_a_figure_caption_stays_text(self):
        """Relaxing the caption to "Figure N" must not turn a boxed paragraph
        into a table: without aligned columns it is not a grid."""
        def draw(c):
            _body(c, 740)
            c.line(72, 620, 400, 620)
            c.line(72, 540, 400, 540)
            _body(c, 605, lines=5)
            c.setFont("Times-Roman", 9)
            c.drawString(72, 525, "Figure 2: An example prompt.")
            _body(c, 480)
        page, _ = _figures(_pdf(draw))
        self.assertEqual([r for r in page.regions if r.kind == "table"], [])
        self.assertIn("Residual networks are easier", _spoken(page))

    def test_white_background_does_not_swallow_header_and_caption(self):
        """A chart's white fill joined the page header, the chart and its
        caption into one 'figure' (1412.6980, p. 8)."""
        page, figs = _figures(_pdf(lambda c: self._page(c, white_background=True)))
        self.assertEqual(len(figs), 1)
        self.assertTrue(figs[0].meta.get("caption", "").startswith("Figure 1."), figs[0].meta)
        self.assertNotIn("Published", figs[0].meta.get("figure_text", ""))


class TestPanelsAndCaptions(unittest.TestCase):

    def test_panels_over_one_caption_are_one_figure(self):
        """Two charts over one caption: one was captioned, the other said
        "Figure: No caption" (five times in two papers). The right panel is
        short and ends far above the caption (1706.03762, p. 4)."""
        def draw(c):
            _chart(c, 100, 520, 180, 180)
            _chart(c, 340, 600, 150, 100, legend=False)
            c.setFont("Times-Roman", 10)         # a caption as wide as the figure
            c.drawString(72, 480, "Figure 2. (left) Training error of plain and residual "
                                  "networks on CIFAR-10. (right) Test error of the same runs.")
            _body(c, 430)
        page, figs = _figures(_pdf(draw))
        self.assertEqual(len(figs), 1, [(f.x0, f.y0, f.x1, f.y1) for f in figs])
        self.assertTrue(figs[0].meta.get("caption", "").startswith("Figure 2."))

    def test_caption_inside_a_graphics_cluster_is_still_a_caption(self):
        """A drawing reaching down past its caption took the caption's words
        as figure labels; the caption was never spoken (1706.03762, p. 14)."""
        def draw(c):
            _chart(c, 110, 520, 260, 180)
            c.line(95, 440, 95, 700)       # a frame line running past the caption
            c.setFont("Times-Roman", 9)
            c.drawString(90, 470, "Figure 4: Two attention heads in layer 5 of 6.")
            _body(c, 400)
        page, figs = _figures(_pdf(draw))
        self.assertEqual(len(figs), 1)
        self.assertTrue(figs[0].meta.get("caption", "").startswith("Figure 4:"), figs[0].meta)

    def test_one_caption_per_figure_and_the_table_caption_is_spoken(self):
        """A table caption above a chart and the chart's own caption below
        both bound to the chart; the table caption was never spoken."""
        regions = [Region(100, 400, 900, 440, kind="caption", text="Table 1. Architectures."),
                   Region(150, 480, 850, 780, kind="figure", nature="figure"),
                   Region(100, 800, 900, 840, kind="caption", text="Figure 4. Training curves.")]
        bind_captions(regions)
        self.assertEqual(regions[1].meta["caption"], "Figure 4. Training curves.")
        self.assertIsNone(regions[0].context_of)
        self.assertTrue(regions[0].meta.get("orphan_caption"))

    def test_stray_label_is_cut_from_a_caption_but_prose_never_is(self):
        self.assertTrue(_LEADING.match("F Figure 2. Residual learning: a building block."))
        self.assertTrue(_LEADING.match("(a) Figure 3: Two panels."))
        for prose in ("In Table 3 we compare", "See Fig. 4 for details", "as in Fig. 2. The",
                      "In Figure 2. we"):
            self.assertIsNone(_LEADING.match(prose), prose)


class TestColumns(unittest.TestCase):

    def test_narrow_gutter_keeps_two_columns_apart(self):
        """One paper's gutter is 10 pt (20 px at work scale), under line
        joining's 25 px: its two columns were read as one, row by row."""
        words = ("the batch normalized network enjoys higher accuracy and trains faster "
                 "than the baseline while using fewer steps").split()

        def column(c, x0, x1, y):
            c.setFont("Times-Roman", 10)
            for row in range(30):              # justified rows: end exactly at x1
                line = words[row % 5: row % 5 + 7]
                natural = sum(stringWidth(w, "Times-Roman", 10) for w in line)
                gap = (x1 - x0 - natural) / (len(line) - 1)
                x = x0
                for w in line:
                    c.drawString(x, y - 12 * row, w)
                    x += stringWidth(w, "Times-Roman", 10) + gap

        def draw(c):
            column(c, 72, 301, 720)
            column(c, 311, 540, 720)
        with open_pdf(_pdf(draw)) as pdf:
            regions = extract_page(pdf.pages[0])
        mid = 306 * 1600 / 792                   # the gutter, at work scale (Letter)
        crossing = [r.text[:60] for r in regions if r.x0 < mid - 5 and r.x1 > mid + 5]
        self.assertEqual(crossing, [])


if __name__ == "__main__":
    unittest.main()
