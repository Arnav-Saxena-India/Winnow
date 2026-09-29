"""Every invariant in docs/design.md has a test."""
from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

from tests import fixture
from tests.test_reader import run_pipeline
from winnow.caption_binding import bind_captions
from winnow.classify import classify
from winnow.reading_order import reading_order
from winnow.speech import plan, plan_all

PKG = Path(__file__).resolve().parent.parent / "winnow"


def _classified():
    regions = fixture.regions()
    classify(regions, fixture.PAGE_H, fixture.PAGE_W, fixture.repeated(), 4)
    ordered = reading_order(regions, fixture.PAGE_W)
    bind_captions(ordered)
    return ordered


class TestInvariants(unittest.TestCase):

    def test_1_plan_is_pure(self):
        ordered = _classified()
        before = copy.deepcopy(ordered)
        a, b = plan_all(ordered), plan_all(copy.deepcopy(ordered))
        self.assertEqual(a, b)
        self.assertEqual(ordered, before, "plan() mutated its input")

    def test_2_only_pipeline_and_cli_import_backends(self):
        for f in PKG.glob("*.py"):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            names += [f"{n.module}.{a.name}" for n in ast.walk(tree)
                      if isinstance(n, ast.ImportFrom) for a in n.names]
            uses = any("backends" in n for n in names)
            if f.stem not in ("pipeline", "cli", "backends"):
                self.assertFalse(uses, f"{f.name} imports backends")

    def test_3_classify_keeps_boxes_order_keeps_kind(self):
        regions = fixture.regions()
        boxes = [(r.x0, r.y0, r.x1, r.y1) for r in regions]
        classify(regions, fixture.PAGE_H, fixture.PAGE_W, fixture.repeated(), 4)
        self.assertEqual(boxes, [(r.x0, r.y0, r.x1, r.y1) for r in regions])
        kinds = {id(r): r.kind for r in regions}
        ordered = reading_order(regions, fixture.PAGE_W)
        self.assertEqual(kinds, {id(r): r.kind for r in ordered})
        self.assertEqual(len(ordered), len(regions))

    def test_4_chrome_carries_why(self):
        for r in _classified():
            if r.kind == "chrome":
                self.assertTrue(r.meta.get("why"), r.text)

    def test_5_bound_caption_never_alone(self):
        ordered = _classified()
        cap = next(r for r in ordered if r.kind == "caption")
        self.assertIsNotNone(cap.context_of)
        for tier in (0, 1, 2):
            self.assertFalse([u for u in plan(ordered, tier) if u.kind == "caption"])

    def test_6_suppressed_retained_at_tier_0(self):
        ordered = _classified()
        n_chrome = sum(r.kind == "chrome" for r in ordered)
        for tier in (0, 1, 2):
            sup = [u for u in plan(ordered, tier) if u.suppressed]
            self.assertEqual(len(sup), n_chrome)
            self.assertTrue(all(u.tier == 0 and u.why for u in sup))

    def test_8_title_never_suppressed_even_faint_and_small_band(self):
        """Title in the top band, faint, and echoing the header: the guard holds."""
        regions = fixture.regions()
        for r in regions:
            if r.text.strip():
                r.meta["ink"] = 20.0
        title = regions[1]
        title.meta["ink"] = 200.0                       # very faint
        classify(regions, fixture.PAGE_H, fixture.PAGE_W, fixture.repeated(), 4)
        self.assertNotEqual(title.kind, "chrome", title.meta)
        self.assertIn("title_guard", title.meta["evidence"])
        _, utts = run_pipeline(fixture.regions())
        self.assertIn(fixture.HEADER, [u.text for u in utts if not u.suppressed])


if __name__ == "__main__":
    unittest.main()
