"""Step 0 gate: the six assertions on the synthetic fixture. The contract."""
from __future__ import annotations

import unittest

from tests import fixture
from winnow.caption_binding import bind_captions
from winnow.classify import classify
from winnow.reading_order import reading_order
from winnow.speech import plan, plan_all


def run_pipeline(regions):
    classify(regions, fixture.PAGE_H, fixture.PAGE_W, repeated=fixture.repeated(), n_pages=4)
    ordered = reading_order(regions, fixture.PAGE_W)
    bind_captions(ordered)
    return ordered, plan(ordered, tier=2)


class TestReader(unittest.TestCase):

    def setUp(self):
        self.regions = fixture.regions()
        self.ordered, self.utterances = run_pipeline(self.regions)

    def _at(self, y0):
        return next(r for r in self.regions if r.y0 == y0)

    def test_chrome_suppressed(self):
        """Header, page number, watermark — all silent."""
        for y0 in (30, 1480, 1530):
            self.assertEqual(self._at(y0).kind, "chrome", self._at(y0).text)
        silent = {self.ordered[u.region].y0 for u in self.utterances if u.suppressed}
        self.assertTrue({30, 1480, 1530} <= silent)
        spoken = " ".join(u.text for u in self.utterances if not u.suppressed)
        self.assertNotIn("CamScanner", spoken)
        self.assertNotIn("14", spoken.split())

    def test_title_not_suppressed(self):
        """The dangerous case — see 8a and 8b."""
        title = self._at(90)
        self.assertEqual(title.kind, "heading", title.meta)
        spoken = [u.text for u in self.utterances if not u.suppressed]
        self.assertIn(fixture.HEADER, spoken)

    def test_caption_becomes_figure_context(self):
        """No caption emitted alone."""
        for tier in (0, 1, 2):
            utts = plan(self.ordered, tier=tier)
            self.assertFalse([u for u in utts if u.kind == "caption"], f"tier {tier}")
        fig = [u for u in self.utterances if u.kind == "figure"]
        self.assertEqual(len(fig), 1)
        self.assertTrue(fig[0].text.startswith("Figure 3"), fig[0].text)
        self.assertIn("A balanced binary tree", fig[0].text)

    def test_equation_spoken_as_words(self):
        """h = O(log n) -> 'h equals big O of log n'."""
        eq = [u for u in self.utterances if u.kind == "equation"]
        self.assertEqual([u.text for u in eq], ["h equals big O of log n"])

    def test_margin_note_not_mid_sentence(self):
        """Announced separately, never spliced."""
        kinds = [u.kind for u in self.utterances]
        i = kinds.index("annotation")
        self.assertEqual(self.utterances[i].text, "Margin note: ask about AVL")
        self.assertEqual(kinds[i - 1], "body", "note must follow the paragraph beside it")
        for u in self.utterances:
            if u.kind == "body":
                self.assertNotIn("ask about AVL", u.text)

    def test_nothing_silently_dropped(self):
        """Every suppression recoverable, with a reason."""
        for r in self.ordered:
            if r.kind == "chrome":
                self.assertTrue(r.meta.get("why"), f"no reason: {r.text!r}")
        for tier in (0, 1, 2):
            utts = plan(self.ordered, tier=tier)
            for u in utts:
                if u.suppressed:
                    self.assertTrue(u.why)
                    self.assertEqual(u.tier, 0)
            seen = {u.region for u in utts}
            bound = {i for i, r in enumerate(self.ordered)
                     if r.kind == "caption" and r.context_of is not None}
            self.assertEqual(set(range(len(self.ordered))) - seen - bound, set(),
                             f"regions silently dropped at tier {tier}")
        self.assertEqual(len([u for u in plan_all(self.ordered) if u.suppressed]), 4)


if __name__ == "__main__":
    unittest.main()
