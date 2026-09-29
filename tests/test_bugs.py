"""§8 and §16: every bug found is an assertion so it cannot come back."""
from __future__ import annotations

import unittest

from tests import fixture
from winnow.classify import classify, looks_like_equation
from winnow.lines import group_lines
from winnow.segment import line_count, reclassify_sparse, segment
from winnow.types import Region


class TestKnownBugs(unittest.TestCase):

    def test_8a_single_line_title_is_not_small(self):
        """Block height would call the title small; line height does not."""
        regions = fixture.regions()
        classify(regions, fixture.PAGE_H, fixture.PAGE_W)
        title = regions[1]
        self.assertGreater(title.meta["evidence"]["size_ratio"], 1.15)
        self.assertEqual(title.kind, "heading")

    def test_8a_line_count_by_projection(self):
        img = fixture.image()
        body = fixture.regions()[3]
        self.assertGreaterEqual(line_count(img[body.y0:body.y1, body.x0:body.x1]), 3)
        title = fixture.regions()[1]
        self.assertEqual(line_count(img[title.y0:title.y1, title.x0:title.x1]), 1)

    def test_8b_title_echoing_header_survives(self):
        regions = fixture.regions()
        classify(regions, fixture.PAGE_H, fixture.PAGE_W, fixture.repeated(), 4)
        self.assertEqual(regions[0].kind, "chrome")
        self.assertEqual(regions[1].kind, "heading")

    def test_8c_date_is_not_arithmetic_and_needs_no_band(self):
        self.assertFalse(looks_like_equation("2026-09-19"))
        self.assertFalse(looks_like_equation("19/09/2026"))
        self.assertTrue(looks_like_equation("h = O(log n)"))
        regions = fixture.regions()
        classify(regions, fixture.PAGE_H, fixture.PAGE_W)
        date = regions[2]
        self.assertGreater(date.y0 / fixture.PAGE_H, 0.10)      # outside the band
        self.assertEqual((date.kind, date.meta["why"]), ("chrome", "date stamp"))

    def test_8d_header_with_rule_is_never_a_figure(self):
        strip = Region(100, 20, 1100, 60, text="Balanced Binary Search Trees", conf=90)
        reclassify_sparse([strip])
        self.assertEqual(strip.nature, "text")
        regions = segment(fixture.image())
        top = [r for r in regions if r.y1 < 80]
        self.assertTrue(top and all(r.nature == "text" for r in top), top)

    def test_8e_lines_grouped_by_geometry_not_backend_index(self):
        # An Aadhaar-style number reported as three separate blocks, out of order.
        words = [{"x0": 260, "x1": 330, "top": 101, "bottom": 121, "text": "9012", "line": 2},
                 {"x0": 100, "x1": 170, "top": 100, "bottom": 120, "text": "1234", "line": 0},
                 {"x0": 180, "x1": 250, "top": 99, "bottom": 119, "text": "5678", "line": 1},
                 {"x0": 100, "x1": 200, "top": 140, "bottom": 160, "text": "next", "line": 0}]
        lines = group_lines(words)
        self.assertEqual([" ".join(w["text"] for w in ln) for ln in lines],
                         ["1234 5678 9012", "next"])

    def test_hyphenated_heading_is_not_an_equation(self):
        """Found on the test PDF: 'Red-black trees' was read as maths."""
        self.assertFalse(looks_like_equation("Red-black trees"))
        self.assertTrue(looks_like_equation("x^2 + y^2 = r^2"))

    def test_segmentation_finds_the_figure(self):
        figs = [r for r in segment(fixture.image()) if r.nature == "figure"]
        self.assertEqual(len(figs), 1)
        f = figs[0]
        self.assertTrue(abs(f.x0 - 100) < 40 and abs(f.y0 - 630) < 40, f)


if __name__ == "__main__":
    unittest.main()


class TestRealPaperBugs(unittest.TestCase):
    """Found running Winnow on a real two-column paper (arXiv 1512.03385, not
    redistributable, so the page is rebuilt here from its measured boxes)."""

    W, H = 1237, 1600

    def _page(self):
        R = Region
        return [
            R(309, 213, 893, 245, text="Deep Residual Learning for Image Recognition",
              meta={"line_h": 32, "line_count": 1}),
            R(275, 307, 395, 330, text="Kaiming He", meta={"line_h": 22, "line_count": 1}),
            R(653, 307, 791, 330, text="Shaoqing Ren", meta={"line_h": 22, "line_count": 1}),
            R(844, 307, 927, 330, text="Jian Sun", meta={"line_h": 22, "line_count": 1}),
            R(404, 373, 808, 392, text="kahe, v-shren, jiansun @microsoft.com",
              meta={"line_h": 18, "line_count": 1}),
            R(101, 496, 579, 720, text="Deeper neural networks are more difficult to train. " * 4,
              meta={"line_h": 24, "line_count": 9}),
            R(101, 729, 579, 1050, text="deeper than VGG nets but still having lower cost. " * 5,
              meta={"line_h": 24, "line_count": 13}),
            R(101, 1123, 579, 1400, text="Deep convolutional networks have led to breakthroughs. " * 5,
              meta={"line_h": 24, "line_count": 11}),
            R(101, 1405, 575, 1440, text="1 http://image-net.org/challenges/LSVRC/2015/ footnote",
              meta={"line_h": 16, "line_count": 2}),
            R(624, 722, 1100, 1060, text="greatly benefited from very deep models. " * 6,
              meta={"line_h": 24, "line_count": 14}),
            R(624, 1070, 1100, 1440, text="The degradation problem has been exposed. " * 6,
              meta={"line_h": 24, "line_count": 15}),
        ]

    def test_centred_title_is_read_before_the_left_column(self):
        from winnow.reading_order import reading_order
        regions = self._page()
        classify(regions, self.H, self.W)
        order = [r.text for r in reading_order(regions, self.W)]
        self.assertEqual(order[0], "Deep Residual Learning for Image Recognition")
        self.assertLess(order.index("Shaoqing Ren"), order.index(regions[5].text))

    def test_right_column_is_not_a_margin(self):
        regions = self._page()
        classify(regions, self.H, self.W)
        self.assertNotEqual(regions[3].kind, "annotation")     # "Jian Sun"
