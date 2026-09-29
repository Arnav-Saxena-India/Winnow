"""Compare describe prompts against a running llama-server (see README).

For each prompt: are the corpus figures described (not UNCLEAR), how long are
the descriptions, and do 9 noise/blank images still fall back to UNCLEAR?
A prompt is only better if it keeps 9/9 on noise: a fabricated description is
worse than none.

    python tools/prompt_check.py

2026-09-27: "v4 short" cut mean output 92 -> 63 tokens, described 4/4 figures,
but let 1 of 9 noise images through, so v3 stays.
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from winnow import backends, pipeline  # noqa: E402
from winnow.describe import interpret  # noqa: E402

PROMPTS = {
    "v3 (current)": backends.DESCRIBE_PROMPT,
    "v5 two sentences": (
        "Look at the image. If it is random noise, static, texture, blank, or unreadable, "
        "reply with only the word UNCLEAR. Otherwise describe it for a listener who cannot "
        "see it in two sentences: first the kind of figure, then what it shows (structure, "
        "visible values, trends). Do not invent values that are not shown. "
        "Caption from the page: {context}"),
    "v6 one sentence each": (
        "Look at the image. If it is random noise, blank, or unreadable, reply with only "
        "the word UNCLEAR. Otherwise, for a listener who cannot see it, say in one sentence "
        "what kind of figure it is, and in one sentence what it shows: its structure and any "
        "values or trends that are visible. Do not invent values that are not shown. "
        "Caption from the page: {context}"),
    "v7 brief": (
        "Look at the image. If it is random noise, blank, or unreadable, reply with only "
        "the word UNCLEAR. Otherwise describe it for a listener who cannot see it: the kind "
        "of figure, its structure, and any values or trends that are visible. Be brief; do "
        "not repeat yourself. Do not invent values that are not shown. "
        "Caption from the page: {context}"),
}
rng = np.random.default_rng
cases = {f"noise{s}": (rng(s).integers(0, 256, (300, 300), dtype=np.uint8), "Fig 1. Results")
         for s in range(6)}
cases["blank"] = (np.full((300, 300), 255, np.uint8), "Fig 2. Overview")
cases["blurred"] = (cv2.GaussianBlur(rng(4).integers(0, 256, (300, 300), dtype=np.uint8),
                                     (9, 9), 0), "Fig 5. Map")
cases["saltpepper"] = ((rng(7).random((300, 300)) > 0.5).astype(np.uint8) * 255, "Figure 2. Accuracy")
ocr = backends.make_ocr()
for path in ("tests/corpus/lecture.pdf", "tests/corpus/scan_p1.png", "tests/corpus/scan_p3.png"):
    doc = pipeline.run(path, ocr=ocr, workers=1)
    for p in doc.pages:
        for r in p.regions:
            if r.kind == "figure":
                cases[f"REAL {path.split('/')[-1]} p{p.page_no}"] = (
                    p.gray[r.y0:r.y1, r.x0:r.x1], r.meta.get("caption", ""))

d = backends.LlamaCppDescriber()
for name, prompt in PROMPTS.items():
    noise_ok = real_ok = 0
    toks = []
    print(f"== {name}")
    for k, (crop, cap) in cases.items():
        raw = d._generate(crop, prompt.format(context=cap or "none"))
        status = interpret(raw, cap)[0]
        if not k.startswith("REAL") and status != "unclear":
            print(f"   LEAKED on {k}: {raw[:120]}")
        if k.startswith("REAL"):
            real_ok += status == "ok"
            toks.append(d.last_stats["eval_count"])
            print(f"   {k:22s} {d.last_stats['eval_count']:4d} tok  {raw}")
        else:
            noise_ok += status == "unclear"
    print(f"   noise/blank caught {noise_ok}/9   real described {real_ok}/4   "
          f"output tokens mean {sum(toks) / len(toks):.0f} (range {min(toks)}-{max(toks)})")
