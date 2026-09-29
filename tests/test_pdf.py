"""Step 3: a real PDF, text layer only, no OCR, correct utterances."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from winnow import pipeline
from winnow.pdf_adapter import extract_page, open_pdf
from winnow.types import EncryptedDocument

PDF = Path(__file__).parent / "corpus" / "lecture.pdf"


class TestPdf(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.doc = pipeline.run(str(PDF), ocr=None, workers=1)

    def test_tight_latex_spacing_does_not_glue_words(self):
        """Found on a real two-column paper: LaTeX places words with no space
        glyph and ~2pt gaps at 10pt, under pdfplumber's fixed 3pt tolerance, so
        the listener heard "presentaresiduallearningframework"."""
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.pdfgen import canvas
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "tight.pdf")
            c = canvas.Canvas(path)
            for y, line in ((700, "We present a residual learning framework"),
                            (688, "to ease the training of deeper networks.")):
                x = 72
                for word in line.split():          # one draw per word: no space glyphs
                    c.setFont("Times-Roman", 10)
                    c.drawString(x, y, word)
                    x += stringWidth(word, "Times-Roman", 10) + 2.0
            c.save()
            with open_pdf(path) as pdf:
                text = " ".join(r.text for r in extract_page(pdf.pages[0]))
        self.assertIn("a residual learning framework", text)
        self.assertNotIn("residuallearning", text)

    def test_nine_regions_on_page_one_without_ocr(self):
        with open_pdf(str(PDF)) as pdf:
            regions = extract_page(pdf.pages[0])
        self.assertEqual(len(regions), 9)
        self.assertTrue(all(r.meta.get("source") == "text-layer" for r in regions))
        self.assertEqual(self.doc.pages[0].source, "text-layer")

    def test_kinds_on_page_one(self):
        kinds = {r.text or r.kind: r.kind for r in self.doc.pages[0].regions}
        self.assertEqual(kinds["Page 1"], "chrome")
        self.assertEqual(kinds["2026-09-19"], "chrome")
        self.assertEqual(kinds["h = O(log n)"], "equation")
        self.assertEqual(kinds["ask about AVL"], "annotation")
        self.assertEqual(kinds["figure"], "figure")
        headers = [r for r in self.doc.pages[0].regions if r.text == "Balanced Binary Search Trees"]
        self.assertEqual(sorted(r.kind for r in headers), ["chrome", "heading"])

    def test_transcript(self):
        spoken = [u.text for p, u in self.doc.transcript(2) if not u.suppressed]
        self.assertEqual(spoken[0], "Balanced Binary Search Trees")
        self.assertIn("h equals big O of log n", spoken)
        self.assertIn("x squared plus y squared equals r squared", spoken)
        self.assertIn("Margin note: ask about AVL", spoken)
        self.assertTrue(any(s.startswith("Figure 3: A balanced binary search tree") for s in spoken))
        self.assertIn("Red-black trees", spoken)
        joined = " ".join(spoken)
        for junk in ("CamScanner", "Page 2", "@cse_notes", "2026-09-19", "8 4 12"):
            self.assertNotIn(junk, joined)

    def test_figure_labels_are_not_body_text(self):
        fig = next(r for r in self.doc.pages[0].regions if r.kind == "figure")
        self.assertEqual(fig.meta["figure_text"], "8 4 12 2 6 10 14")

    def test_clean_document_passes_self_check(self):
        self.assertTrue(self.doc.ok, self.doc.errors)
        self.assertEqual([f for f in self.doc.findings if f.check == "cross_page_conflict"], [])

    def test_encrypted_pdf_says_so(self):
        from reportlab.pdfgen import canvas
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "locked.pdf")
            c = canvas.Canvas(path, encrypt="secret")
            c.drawString(100, 700, "hidden")
            c.save()
            with self.assertRaises(EncryptedDocument):
                pipeline.run(path, workers=1)

    def test_parallel_matches_serial(self):
        """Workers finish out of order; speech must not."""
        from dataclasses import replace
        from winnow.types import DEFAULT
        cfg = replace(DEFAULT, parallel_min_pages=1, workers=2)
        par = pipeline.run(str(PDF), cfg=cfg)
        self.assertEqual([u.text for _, u in par.transcript(2)],
                         [u.text for _, u in self.doc.transcript(2)])


if __name__ == "__main__":
    unittest.main()
