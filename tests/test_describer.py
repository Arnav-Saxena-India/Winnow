"""The VLM branch: never let it invent. UNCLEAR, noise, a parroted caption, a
timeout and a missing model all fall back to the caption, audibly.

Most exercise the contract with stand-in describers. ``TestRealDescriber``
runs the real model when one is served locally, and skips otherwise."""
from __future__ import annotations

import unittest

import numpy as np

from winnow import backends, describe, pipeline
from winnow.speech import at_tier
from winnow.types import BackendUnavailable

PDF = "tests/corpus/lecture.pdf"


class Scripted:
    name, device = "scripted", "CPU"

    def __init__(self, answer, model_id="scripted-1"):
        self.answer, self.model_id, self.calls = answer, model_id, 0

    def describe(self, crop, context):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer(crop, context) if callable(self.answer) else self.answer

    def ask(self, crop, question, context):
        return self.describe(crop, context)


def _figure_lines(doc):
    return [u.text for _, u in doc.transcript(2) if u.kind == "figure"]


class TestDescriber(unittest.TestCase):

    def test_description_replaces_caption(self):
        d = Scripted("A tree of seven nodes in three levels; the root is 8.")
        lines = _figure_lines(pipeline.run(PDF, describer=d, workers=1))
        self.assertEqual(lines[0], "Figure 3: A tree of seven nodes in three levels; the root is 8.")
        self.assertNotIn("balanced binary search tree of height 2", lines[0])

    def test_unclear_falls_back_to_caption(self):
        lines = _figure_lines(pipeline.run(PDF, describer=Scripted("UNCLEAR"), workers=1))
        self.assertEqual(lines[0], "Figure 3, content not recognised. "
                                   "Caption: A balanced binary search tree of height 2.")

    def test_noise_triggers_fallback(self):
        """Feed pure noise: a describer that honours the prompt must say UNCLEAR,
        and the fallback must fire."""
        def honest(crop, context):
            return "UNCLEAR" if crop.std() > 60 else "A chart."
        noise = np.random.default_rng(0).integers(0, 256, (300, 300), dtype=np.uint8)
        self.assertEqual(describe.interpret(honest(noise, ""), "Fig 1. x"), ("unclear", ""))

    def test_unclear_anywhere_or_an_admission_falls_back(self):
        """Real Qwen3-VL-2B replies to noise: UNCLEAR not first, or no UNCLEAR
        at all but a plain admission. Each would have been read aloud."""
        for said in ("The image is UNCLEAR.",
                     "A chaotic, grainy texture with no discernible shapes. UNCLEAR",
                     "Fine lines resembling grainy noise. It is not a clear or readable image."):
            self.assertEqual(describe.interpret(said, "Fig 1. Results"), ("unclear", ""), said)

    def test_a_denial_is_not_an_admission(self):
        said = ("The figure is a bar chart of five rising bars. The chart is not random "
                "noise and is not unreadable.")
        self.assertEqual(describe.interpret(said, "Figure 4. Rotation cost")[0], "ok")

    def test_a_real_description_is_kept(self):
        said = ("The image displays a bar chart with five vertical bars of increasing height. "
                "The chart shows a clear upward trend.")
        self.assertEqual(describe.interpret(said, "Figure 4. Rotation cost by tree size")[0], "ok")

    def test_parroted_caption_is_not_a_description(self):
        status, _ = describe.interpret("A balanced binary search tree of height 2.",
                                       "Fig 3. A balanced binary search tree of height 2")
        self.assertEqual(status, "unclear")

    def test_timeout_and_missing_model_degrade(self):
        for exc, word in ((TimeoutError(), "not described"),
                          (BackendUnavailable("gone"), "not described")):
            doc = pipeline.run(PDF, describer=Scripted(exc), workers=1)
            self.assertIn(word, _figure_lines(doc)[0])
            self.assertTrue(doc.ok)

    def test_cache_key_includes_model_and_prompt_version(self):
        crop = np.zeros((50, 50), np.uint8)
        cache = backends.DescriptionCache()
        k1 = cache.key(crop, Scripted("x", "model-a"))
        k2 = cache.key(crop, Scripted("x", "model-b"))
        self.assertNotEqual(k1, k2)
        self.assertTrue(k1.endswith(backends.PROMPT_VERSION))

    def test_tier_one_speaks_caption_tier_two_description(self):
        doc = pipeline.run(PDF, describer=Scripted("Seven circles joined by lines."), workers=1)
        page = doc.pages[0]
        t1 = [u.text for u in at_tier(page.utterances, 1) if u.kind == "figure"]
        self.assertEqual(t1, ["Figure 3: A balanced binary search tree of height 2."])


def _real_vlm():
    """The default VLM served locally by Ollama or llama-server, or None (tests skip)."""
    for d in (backends.OllamaDescriber(), backends.LlamaCppDescriber()):
        try:
            d.describe(np.full((64, 64), 255, np.uint8), "none")
            return d
        except (BackendUnavailable, TimeoutError):
            continue
    return None


class TestRealDescriber(unittest.TestCase):
    """Step 9 against the real model (§14: a figure produces a description that
    is not its caption). Skipped when no local VLM is served."""

    @classmethod
    def setUpClass(cls):
        cls.vlm = _real_vlm()
        if cls.vlm is None:
            raise unittest.SkipTest(f"no local VLM ({backends.DEFAULT_VLM}) served")

    def test_figure_description_is_not_its_caption(self):
        doc = pipeline.run(PDF, describer=self.vlm, workers=1)
        fig = next(r for r in doc.pages[0].regions if r.kind == "figure")
        self.assertEqual(fig.meta["description_status"], "ok", fig.meta)
        line = _figure_lines(doc)[0]
        self.assertTrue(line.startswith("Figure 3: "))
        self.assertNotIn("balanced binary search tree of height 2", line.lower())

    def test_noise_falls_back_with_the_real_model(self):
        noise = np.random.default_rng(0).integers(0, 256, (300, 300), dtype=np.uint8)
        status, _ = describe.interpret(self.vlm.describe(noise, "Fig 1. Results"),
                                       "Fig 1. Results")
        self.assertEqual(status, "unclear")


if __name__ == "__main__":
    unittest.main()
