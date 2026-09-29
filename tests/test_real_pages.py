"""Bugs found on real documents, each rebuilt synthetically.

Found on ``Combinatorics.pdf``: one page 17 × 103 inches, light handwriting-
style text on black, no text layer. Before these fixes Winnow said only
"Figure: No caption. Figure, not described." for the whole page.
"""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from winnow.classify import classify, looks_like_equation
from winnow.ocr import line_boxes, read_in_strips
from winnow.lines import group_lines
from winnow.segment import (normalise_polarity, segment,
                            text_zoom, work_scale)
from winnow.types import DEFAULT, Region

LINES = ["Combinatorics is the study of counting", "But not counting by listing",
         "Instead use logic to count", "Why it is needed in programming"]


def _page(dark: bool, h: int = 1600, w: int = 1132, scale: float = 0.9, gap: int = 60):
    paper, ink = (0, 235) if dark else (255, 20)
    img = np.full((h, w), paper, np.uint8)
    for i, text in enumerate(LINES):
        cv2.putText(img, text, (80, 150 + gap * i), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 2)
    return img


class TestDarkAndTallPages(unittest.TestCase):

    def test_dark_page_is_inverted_and_read_as_text(self):
        gray, inverted = normalise_polarity(_page(dark=True))
        self.assertTrue(inverted)
        regions = segment(gray)
        self.assertFalse(any(r.nature == "figure" for r in regions), regions)
        self.assertGreaterEqual(len(regions), 1)

    def test_light_page_is_left_alone(self):
        img = _page(dark=False)
        gray, inverted = normalise_polarity(img)
        self.assertFalse(inverted)
        self.assertTrue(np.array_equal(gray, img))

    def test_tall_page_keeps_a_readable_width(self):
        s = work_scale(1209.12, 7388.88)                       # the real page, in points
        self.assertGreaterEqual(1209.12 * s, DEFAULT.work_min_width - 1)
        self.assertAlmostEqual(792 * work_scale(612, 792), DEFAULT.work_height)   # letter: as before

    def test_small_text_is_rendered_larger_and_normal_text_is_not(self):
        self.assertEqual(text_zoom(_page(dark=False)), 1.0)
        tiny = _page(dark=False, scale=0.3, gap=20)
        self.assertGreater(text_zoom(tiny), 1.0)


class _FakeOcr:
    """Returns every line box as a word, like an OCR that reads perfectly."""
    name, device = "fake", "-"

    def __init__(self):
        self.shapes = []

    def read(self, img):
        self.shapes.append(img.shape)
        return [{"x0": x0, "x1": x1, "top": y0, "bottom": y1, "text": f"L{y0}", "conf": 100.0}
                for x0, y0, x1, y1 in line_boxes(img)]


class TestStripsAndLines(unittest.TestCase):

    def test_tall_image_is_read_in_strips_each_line_exactly_once(self):
        h = 3 * DEFAULT.ocr_tile
        img = np.full((h, 800), 255, np.uint8)
        ys = list(range(100, h - 100, 173))                   # lines land in every overlap
        for y in ys:
            cv2.putText(img, "counting principles", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        ocr = _FakeOcr()
        words = read_in_strips(ocr, img)
        self.assertTrue(all(sh[0] <= DEFAULT.ocr_tile for sh in ocr.shapes), ocr.shapes)
        centres = sorted((w["top"] + w["bottom"]) / 2 for w in words)
        self.assertEqual(len(centres), len(ys))                  # none lost, none twice
        for c, y in zip(centres, ys):
            self.assertLess(abs(c - (y - 8)), 20)

    def test_blank_strips_are_skipped_but_one_short_line_is_not(self):
        img = np.full((3 * DEFAULT.ocr_tile, 800), 255, np.uint8)
        cv2.putText(img, "only line", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        ocr = _FakeOcr()
        read_in_strips(ocr, img)
        self.assertEqual(len(ocr.shapes), 1)

    def test_line_boxes_are_tight_plus_margin(self):
        """Boxes widened by the dilation read a leading '~' or "'" on every
        line ("'Page 1" then escaped the page-number rule)."""
        img = _page(dark=False)
        boxes = line_boxes(img)
        self.assertEqual(len(boxes), len(LINES))
        ink_x0 = int(np.flatnonzero((img < 128).any(axis=0))[0])
        for x0, y0, x1, y1 in boxes:
            self.assertLessEqual(ink_x0 - x0, DEFAULT.line_margin * (y1 - y0) + 2)

    def test_pieces_of_one_row_read_left_to_right(self):
        """OCR dropped the middle of "In CP, you will often see:"; the right
        half sat 5 px higher and was read first: "often see; In"."""
        words = [{"x0": 10, "x1": 40, "top": 105, "bottom": 134, "text": "In"},
                 {"x0": 300, "x1": 380, "top": 100, "bottom": 136, "text": "often"},
                 {"x0": 390, "x1": 450, "top": 114, "bottom": 136, "text": "see:"}]
        order = [w["text"] for ln in group_lines(words) for w in ln]
        self.assertEqual(order, ["In", "often", "see:"])


class TestClassificationOnRealPages(unittest.TestCase):
    W, H = 2200, 13445                                            # the real page at work scale

    def _body(self, y, x0=120, x1=900):
        return Region(x0, y, x1, y + 60, text="Everything in combinatorics rests on two rules " * 3,
                      meta={"line_h": 30, "line_count": 2})

    def test_top_margin_of_a_long_page_is_page_sized(self):
        """'Top 10%' of a 13445 px page swallowed a screen of content as
        'small text in top margin'."""
        small = Region(151, 1110, 528, 1135, text="up to 10^5, 10^6, or even 10^9",
                       meta={"line_h": 20, "line_count": 1})
        regions = [self._body(y) for y in range(600, 13000, 400)] + [small]
        classify(regions, self.H, self.W)
        self.assertNotEqual(small.kind, "chrome", small.meta)

    def test_label_alone_on_its_line_is_not_a_margin_note(self):
        label = Region(89, 771, 193, 800, text="Instead:", meta={"line_h": 29, "line_count": 1})
        regions = [self._body(y, x0=250, x1=1700) for y in range(600, 13000, 400)] + [label]
        classify(regions, self.H, self.W)
        self.assertNotEqual(label.kind, "annotation", label.meta)

    def test_note_beside_a_paragraph_is_still_a_margin_note(self):
        note = Region(1900, 1010, 2150, 1040, text="ask about AVL", meta={"line_h": 30, "line_count": 1})
        regions = [self._body(y, x0=250, x1=1700) for y in range(600, 13000, 400)] + [note]
        classify(regions, self.H, self.W)
        self.assertEqual(note.kind, "annotation", note.meta)

    def _unread(self, img, box):
        from winnow.segment import measure
        r = Region(*box, text="")
        regions = [self._body(y) for y in range(600, 13000, 400)] + [r]
        measure([r], img)
        classify(regions, self.H, self.W)
        return r

    def test_lone_mark_and_drawn_line_are_skipped_with_a_reason(self):
        """A code block's "}" and a rule were each announced as "Text here
        could not be read." (11 times on one real page)."""
        img = np.full((self.H, self.W), 255, np.uint8)
        cv2.putText(img, "}", (360, 9615), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        cv2.line(img, (431, 12866), (1122, 12866), 0, 3)
        brace = self._unread(img, (355, 9589, 391, 9620))
        rule = self._unread(img, (431, 12861, 1122, 12871))
        for r, word in ((brace, "mark"), (rule, "line")):
            self.assertEqual(r.kind, "chrome")
            self.assertIn(word, r.meta["why"])

    def test_unreadable_word_is_still_announced(self):
        img = np.full((self.H, self.W), 255, np.uint8)
        cv2.putText(img, "Hk", (360, 9615), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        word = self._unread(img, (355, 9589, 400, 9620))
        self.assertNotEqual(word.kind, "chrome", word.meta)

    def test_bullet_dash_is_not_minus(self):
        self.assertFalse(looks_like_equation("- Counting using logic - Counting when numbers are"))
        self.assertFalse(looks_like_equation("-Counting paths -Counting Subsets -Probability"))
        self.assertFalse(looks_like_equation(       # OCR joined two bullets into one region
            "- A ways to do part 1 - B ways to do part 2 and both parts are independent"))
        self.assertTrue(looks_like_equation("Total ways = A + B"))
        self.assertTrue(looks_like_equation("So: 2 x 2 x 2 x ... x 2 = 2^n"))
        self.assertTrue(looks_like_equation("A - B"))
        self.assertTrue(looks_like_equation("3 + 2 = 5"))


if __name__ == "__main__":
    unittest.main()
