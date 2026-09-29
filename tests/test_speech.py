"""Step 5: one plan, three tiers; verbaliser; page-level announcements."""
from __future__ import annotations

import unittest

from tests import fixture
from tests.test_reader import run_pipeline
from winnow.speech import at_tier, figure_label, plan_all, plan_page, verbalise_equation
from winnow.types import Region


class TestSpeech(unittest.TestCase):

    def test_tiers_differ_from_one_plan(self):
        ordered, _ = run_pipeline(fixture.regions())
        utts = plan_all(ordered)
        out = [[u.text for u in at_tier(utts, t) if not u.suppressed] for t in (0, 1, 2)]
        self.assertEqual(out[0][:3], ["Heading.", "Paragraph.", "Margin note."])
        self.assertNotEqual(out[0], out[1])
        self.assertNotEqual(out[1], out[2])
        self.assertEqual(len(out[0]), len(out[1]), "every region present at every tier")
        self.assertTrue(out[1][1].endswith("deletions."), "tier 1 body = first sentence")

    def test_verbaliser(self):
        cases = {"h = O(log n)": "h equals big O of log n",
                 "x^2 + y^2 = r^2": "x squared plus y squared equals r squared",
                 "a <= b": "a less than or equal to b",
                 "n! / k": "n factorial over k",
                 "e^(i π) + 1 = 0": "e to the power i pi plus 1 equals 0"}
        for src, want in cases.items():
            self.assertEqual(verbalise_equation(src), want, src)

    def test_figure_label(self):
        self.assertEqual(figure_label("Fig 3. A tree"), ("Figure 3", "A tree"))
        self.assertEqual(figure_label("Table 2: Costs"), ("Table 2", "Costs"))
        self.assertEqual(figure_label(""), ("Figure", ""))

    def test_blank_and_unreadable_pages_say_so(self):
        self.assertEqual([u.text for u in plan_page([], 4, blank=True)], ["Page 4, blank."])
        unread = [Region(0, 0, 500, 500, text="")]
        self.assertEqual(plan_page(unread, 5, blank=False)[0].text, "Page 5, could not be read.")
        self.assertEqual(plan_page([], 6, blank=False)[0].text, "Page 6, could not be read.")
        chrome = [Region(0, 0, 100, 20, text="Page 7", kind="chrome", meta={"why": "page number"})]
        first = plan_page(chrome, 7, blank=False)[0]
        self.assertEqual(first.text, "Page 7: only page furniture, 1 items skipped.")

    def test_unreadable_region_is_announced_not_dropped(self):
        regs = [Region(0, 0, 500, 100, text="Readable.", kind="body"), Region(0, 200, 500, 300, text="")]
        texts = [u.text for u in at_tier(plan_all(regs), 2)]
        self.assertEqual(texts, ["Readable.", "Text here could not be read."])


if __name__ == "__main__":
    unittest.main()
