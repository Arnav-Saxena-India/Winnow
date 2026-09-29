"""Reading starts before figures are described (the window waited for every
figure in the document, ~6 s each on this CPU, before saying anything)."""
from __future__ import annotations

import time
import unittest
from pathlib import Path

from winnow import pipeline

LECTURE = str(Path(__file__).parent / "corpus" / "lecture.pdf")


class SlowDescriber:
    name, device, model_id = "slow", "test", "slow-1"

    def __init__(self, seconds=0.5):
        self.seconds, self.calls = seconds, 0

    def describe(self, crop, context):
        self.calls += 1
        time.sleep(self.seconds)
        return "A binary tree diagram with seven nodes in three levels."

    def ask(self, crop, question, context):
        return None


def _figures(doc):
    return [(p, r) for p in doc.pages for r in p.regions if r.kind == "figure"]


class TestQuickStart(unittest.TestCase):

    def test_document_is_ready_before_any_figure_is_described(self):
        slow = SlowDescriber()
        doc = pipeline.run(LECTURE, describer=slow, workers=1, describe=False)
        self.assertEqual(slow.calls, 0)
        figs = _figures(doc)
        self.assertTrue(figs)
        for p, r in figs:
            self.assertEqual(r.meta["description_status"], "pending")
        spoken = " ".join(u.text for p in doc.pages for u in p.utterances)
        self.assertNotIn("not described", spoken)       # pending is not a failure
        self.assertEqual(doc.errors, [])

    def test_background_pass_fills_in_descriptions_and_replans(self):
        slow = SlowDescriber(0.0)
        doc = pipeline.run(LECTURE, workers=1, describe=False)
        seen = []
        pipeline.describe_pages(doc, slow, on_page=lambda p: seen.append(p.page_no))
        self.assertEqual(slow.calls, len(_figures(doc)))
        self.assertEqual(seen, sorted({p.page_no for p, _ in _figures(doc)}))
        for p, r in _figures(doc):
            self.assertEqual(r.meta["description_status"], "ok")
        full = " ".join(u.text for p in doc.pages for u in p.utterances if u.tier == 2)
        self.assertIn("seven nodes in three levels", full)
        self.assertEqual(doc.errors, [])

    def test_describing_up_front_still_works_the_old_way(self):
        doc = pipeline.run(LECTURE, describer=SlowDescriber(0.0), workers=1)
        self.assertTrue(all(r.meta["description_status"] == "ok" for _, r in _figures(doc)))


if __name__ == "__main__":
    unittest.main()
