"""The VLM branch: figures only. Body text is already text.

``describe_figures`` crops each figure, asks the injected describer (cached on
phash + model + prompt version), and writes ``meta["description"]`` and
``meta["description_status"]`` for the planner. ``interpret`` never lets the
model invent: UNCLEAR, empty, or a parroted caption all fall back to the
caption, audibly; a timeout or a missing model says "not described".
"""
from __future__ import annotations

import re

from winnow import selfcheck
from winnow.types import DEFAULT, BackendUnavailable, Config

_WORD = re.compile(r"[a-z0-9]+")
# The model saying it cannot read the figure. A 2B model does not reliably put
# UNCLEAR first: it wrote "The image is UNCLEAR.", or a paragraph about noise
# ending in UNCLEAR, or "resembling grainy noise. It is not a clear or readable
# image" with no UNCLEAR at all. Every phrase here is one it actually used.
_ADMITS = re.compile(r"\bunclear\b|\bunreadable\b|\bnot (?:a )?(?:clear or )?readable\b"
                     r"|\b(?:random|grainy|static) noise\b|\bstatic or (?:grainy )?noise\b"
                     r"|\bno discernible\b", re.I)
# A denial is not an admission: the model wrote of a clear bar chart "The chart
# is not random noise and is not unreadable", and the check discarded it.
_DENIED = re.compile(r"\b(?:is |are )?not (?:a |an )?(?:random |grainy )?"
                     r"(?:noise|static|texture|blank|unreadable|unclear)\b", re.I)


def interpret(raw: str | None, caption: str, cfg: Config = DEFAULT) -> tuple[str, str]:
    """Raw describer output -> (status, description). Never lets the model
    invent: UNCLEAR, empty, or a parroted caption all fall back to the caption."""
    if raw is None:
        return "unavailable", ""
    t = raw.strip()
    if not t or _ADMITS.search(_DENIED.sub(" ", t)):
        return "unclear", ""
    said, cap = set(_WORD.findall(t.lower())), set(_WORD.findall(caption.lower()))
    if cap and len(said & cap) >= cfg.parrot_overlap * len(said):
        return "unclear", ""
    return "ok", t


def describe_figures(page, describer, cache, cfg: Config, tel) -> list:
    findings = []
    for i, r in enumerate(page.regions):
        if r.kind != "figure":
            continue
        crop = page.gray[r.y0:r.y1, r.x0:r.x1]
        caption = r.meta.get("caption", "")
        context = "; ".join(x for x in (caption, r.meta.get("figure_text", "")) if x)
        key = cache.key(crop, describer)
        try:
            if key not in cache:
                with tel.timed("describe", describer.name, describer.device, page=page.page_no):
                    cache.put(key, describer.describe(crop, context))
            status, text = interpret(cache.get(key), caption, cfg)
        except TimeoutError:
            status, text = "timeout", ""
        except BackendUnavailable as e:
            status, text = "unavailable", ""
            findings.append(selfcheck.Finding(3, "describer_unavailable", "warn",
                                              page.page_no, str(e), i))
        r.meta["description_status"], r.meta["description"] = status, text
        r.meta["describer"] = describer.model_id
    return findings
