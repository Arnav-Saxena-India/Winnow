"""Tier 4 guards: the tuner's traps (docs/design.md) must never be promoted."""
from __future__ import annotations

import unittest
from unittest import mock

from winnow import evaluate
from winnow.evaluate import Score


def S(sup, sup_chrome, chrome=10, content=40, lost=0):
    return Score(suppressed=sup, suppressed_chrome=sup_chrome, chrome=chrome,
                 content=content, content_lost=lost)


class TestMetrics(unittest.TestCase):

    def test_suppress_everything_scores_perfect_recall_and_bad_precision(self):
        s = S(sup=50, sup_chrome=10, lost=40)
        self.assertEqual(s.chrome_recall, 1.0)
        self.assertEqual(s.content_precision, 0.2)
        self.assertFalse(evaluate._better(s, S(sup=8, sup_chrome=8)))

    def test_suppress_nothing_is_not_an_improvement(self):
        self.assertEqual(S(sup=0, sup_chrome=0).content_precision, 1.0)
        self.assertFalse(evaluate._better(S(sup=0, sup_chrome=0), S(sup=8, sup_chrome=8)))

    def test_line_leads_with_precision(self):
        self.assertTrue(S(8, 8).line().startswith("content precision"))

    def test_kappa(self):
        a = {"pages": {"1": [{"box": [0, 0, 10, 10], "kind": "chrome"},
                             {"box": [0, 20, 10, 30], "kind": "body"}]}}
        k, n = evaluate.agreement(a, a)
        self.assertEqual((k, n), (1.0, 2))


class TestTuneGate(unittest.TestCase):

    def _run(self, tuning_scores, holdout_scores):
        calls = {"tuning": iter(tuning_scores), "holdout": iter(holdout_scores)}

        def fake(labels, cfg, ocr=None):
            return next(calls[labels])
        with mock.patch.object(evaluate, "evaluate", fake):
            return evaluate.tune("tuning", "holdout", fields=("small_ratio",), factors=(1.1,))

    def test_recall_bought_with_precision_is_surfaced_not_taken(self):
        cfg, log = self._run([S(8, 8), S(12, 10)], [])
        self.assertEqual(cfg, evaluate.DEFAULT)
        self.assertTrue(any("SURFACED" in line for line in log), log)

    def test_holdout_regression_blocks_promotion(self):
        cfg, log = self._run([S(8, 7), S(9, 9)], [S(8, 8), S(9, 7)])
        self.assertEqual(cfg, evaluate.DEFAULT)
        self.assertIn("REJECTED", log[-1])

    def test_clean_improvement_is_promoted(self):
        cfg, log = self._run([S(8, 7), S(9, 9)], [S(8, 7), S(9, 9)])
        self.assertNotEqual(cfg, evaluate.DEFAULT)
        self.assertIn("PROMOTED", log[-1])


class TestLabelTemplate(unittest.TestCase):

    def test_template_leaves_every_kind_blank_and_load_refuses_it(self):
        import json
        import tempfile
        from pathlib import Path

        from winnow import pipeline
        from winnow.types import WinnowError
        doc = pipeline.run("tests/corpus/lecture.pdf", workers=1)
        lab = evaluate.template(doc, "lecture.pdf", "someone")
        boxes = [b for v in lab["pages"].values() for b in v]
        self.assertEqual(len(boxes), sum(len(p.regions) for p in doc.pages))
        self.assertEqual({b["kind"] for b in boxes}, {""})   # never anchored on Winnow
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.labels.json"
            path.write_text(json.dumps(lab), encoding="utf-8")
            with self.assertRaises(WinnowError):
                evaluate.load(path)


if __name__ == "__main__":
    unittest.main()
