"""Step 4b: tier 1 + tier 2. A clean doc reports nothing; injected faults are
caught; a two-page doc refuses to repair; repair is reproducible."""
from __future__ import annotations

import copy
import unittest

from tests import fixture
from tests.test_reader import run_pipeline
from winnow import selfcheck
from winnow.classify import classify
from winnow.speech import plan_all
from winnow.types import Region


def _page(header_kind="chrome", title=True):
    """One classified page with a header line at the top."""
    regs = [Region(100, 30, 1100, 54, text="CSE 2026 | Unit 3", conf=100, kind=header_kind,
                   meta={"why": "repeats"} if header_kind == "chrome" else {}),
            Region(100, 300, 900, 540, text="Body text " * 20, conf=100, kind="body",
                   meta={"line_count": 8})]
    if title:
        regs.insert(1, Region(100, 90, 1100, 150, text="Title", conf=100, kind="heading"))
    return regs, 1600


class TestTier1(unittest.TestCase):

    def setUp(self):
        self.ordered, _ = run_pipeline(fixture.regions())
        self.utts = plan_all(self.ordered)

    def test_clean_page_has_no_findings(self):
        self.assertEqual(selfcheck.check_page(self.ordered, self.utts, 1), [])

    def _errors(self, regions, utts):
        return {f.check for f in selfcheck.check_page(regions, utts, 1) if f.severity == "error"}

    def test_injected_missing_reason_caught(self):
        regs = copy.deepcopy(self.ordered)
        next(r for r in regs if r.kind == "chrome").meta.pop("why")
        self.assertIn("suppressed_without_reason", self._errors(regs, plan_all(regs)))

    def test_injected_caption_spoken_alone_caught(self):
        from winnow.types import Utterance
        i = next(i for i, r in enumerate(self.ordered) if r.kind == "caption")
        utts = self.utts + [Utterance("Fig 3.", 1, "caption", i)]
        self.assertIn("caption_spoken_alone", self._errors(self.ordered, utts))

    def test_injected_over_suppression_caught(self):
        regs = copy.deepcopy(self.ordered)
        for r in regs:
            if r.nature == "text":
                r.kind, r.meta["why"] = "chrome", "injected"
        errs = self._errors(regs, plan_all(regs))
        self.assertTrue({"over_suppression", "title_suppressed"} <= errs, errs)

    def test_injected_silent_page_caught(self):
        regs = [r for r in copy.deepcopy(self.ordered) if r.kind != "figure"]
        for r in regs:
            r.kind, r.meta["why"], r.context_of = "chrome", "injected", None
        self.assertIn("silent_page_with_ink", self._errors(regs, plan_all(regs)))


class TestTier2(unittest.TestCase):

    def test_two_page_doc_refuses_to_repair(self):
        """A 1-1 tie must not un-suppress a correct header (docs/design.md)."""
        pages = [_page("chrome"), _page("body")]
        before = copy.deepcopy(pages)
        findings, repairs = selfcheck.reconcile(pages)
        self.assertEqual(repairs, [])
        self.assertEqual(pages, before)
        self.assertEqual(len([f for f in findings if f.check == "cross_page_conflict"]), 1)
        self.assertIn("insufficient evidence", findings[0].detail)

    def test_clear_majority_repairs_minority_with_votes(self):
        pages = [_page("chrome") for _ in range(4)] + [_page("body")]
        findings, repairs = selfcheck.reconcile(pages)
        self.assertEqual(len(repairs), 1)
        fixed = pages[4][0][0]
        self.assertEqual(fixed.kind, "chrome")
        self.assertEqual(fixed.meta["repaired"]["votes"], "4/5")
        self.assertTrue(fixed.meta["why"])
        self.assertEqual(selfcheck.badge(findings, repairs),
                         "1 cross-page conflict found, 1 repaired (4/5 consensus)")

    def test_no_majority_is_left_alone(self):
        pages = [_page("chrome"), _page("chrome"), _page("body"), _page("body"), _page("heading")]
        findings, repairs = selfcheck.reconcile(pages)
        self.assertEqual(repairs, [])
        self.assertIn("tie", findings[0].detail)

    def test_never_repairs_the_title_toward_chrome(self):
        """Page 1's title sits where later pages carry a running header with the
        same words. 2/3 votes say chrome — enough to repair anything else."""
        title = Region(100, 30, 1100, 90, text="Balanced Trees", conf=100, kind="heading")
        body = Region(100, 300, 900, 540, text="Body text " * 20, conf=100, meta={"line_count": 8})
        pages = [([title, body], 1600)]
        for _ in range(2):
            hdr = Region(100, 30, 1100, 54, text="Balanced Trees", conf=100, kind="chrome",
                         meta={"why": "repeats"})
            pages.append(([hdr, copy.deepcopy(body)], 1600))
        findings, repairs = selfcheck.reconcile(pages)
        self.assertEqual(repairs, [])
        self.assertEqual(title.kind, "heading")
        self.assertIn("would suppress the title", findings[0].detail)

    def test_no_repair_reproduces_raw_exactly(self):
        pages = [_page("chrome") for _ in range(4)] + [_page("body")]
        raw = copy.deepcopy(pages)
        findings, repairs = selfcheck.reconcile(pages, repair=False)
        self.assertEqual(pages, raw)
        self.assertEqual(repairs, [])
        self.assertIn("reported only", findings[0].detail)

    def test_repairs_deterministic(self):
        a = [_page("chrome") for _ in range(4)] + [_page("body")]
        b = copy.deepcopy(a)
        self.assertEqual(selfcheck.reconcile(a)[1], selfcheck.reconcile(b)[1])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
