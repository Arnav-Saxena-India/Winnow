"""Speech planner — pure: regions in, utterances out. No I/O, no model calls.

``plan_all`` emits, per region, one utterance per tier it has something to say
at: 0 announce ("heading."), 1 summarise, 2 describe/full. ``at_tier`` then
picks, per region, the deepest utterance not deeper than the chosen tier — so
the tier slider filters an existing plan instead of re-analysing the page.

Suppressed chrome is emitted once, at tier 0, with its reason, and is kept at
every tier (invariant 6). A caption bound to a figure is never its own
utterance (invariant 5); it becomes context inside the figure's.

Figure descriptions arrive through ``meta["description"]`` and
``meta["description_status"]``, written by the describer branch before
planning; the planner only reads them.
"""
from __future__ import annotations

import re

from winnow import tables
from winnow.types import DEFAULT, Config, Region, Utterance

_LABEL = re.compile(r"^\s*(fig(?:ure)?|table|chart)\s*\.?\s*(\d+)\s*[.:\-–]?\s*", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_ANNOUNCE = {"heading": "Heading.", "body": "Paragraph.", "equation": "Equation.",
             "annotation": "Margin note.", "caption": "Caption."}

_EQ_PATTERNS = [
    (r"\b([OΘΩ])\s*\(", lambda m: {"O": " big O of (", "Θ": " big theta of (",
                                   "Ω": " big omega of ("}[m.group(1)]),
    (r"\bsqrt\s*\(", " square root of ("),
    (r"\^\s*2\b", " squared "),
    (r"\^\s*3\b", " cubed "),
    (r"\^\s*\(?\s*([\w+\-]+)\s*\)?", r" to the power \1 "),
    (r"(\w)\s*!(?!=)", r"\1 factorial "),
    (r"_\s*\{?(\w+)\}?", r" sub \1 "),
]
_EQ_SYMBOLS = [
    ("<=", " less than or equal to "), (">=", " greater than or equal to "),
    ("!=", " not equal to "), ("≤", " less than or equal to "),
    ("≥", " greater than or equal to "), ("≠", " not equal to "),
    ("≈", " approximately equals "), ("=", " equals "), ("+", " plus "),
    ("−", " minus "), ("-", " minus "), ("×", " times "), ("·", " times "),
    ("*", " times "), ("÷", " divided by "), ("/", " over "),
    ("<", " less than "), (">", " greater than "), ("√", " square root of "),
    ("∑", " sum of "), ("∫", " integral of "), ("∞", " infinity "),
    ("π", " pi "), ("θ", " theta "), ("α", " alpha "), ("β", " beta "),
    ("λ", " lambda "), ("μ", " mu "), ("σ", " sigma "), ("Δ", " delta "),
]


def verbalise_equation(text: str) -> str:
    """'h = O(log n)' -> 'h equals big O of log n'."""
    s = text.strip()
    for pat, rep in _EQ_PATTERNS:
        s = re.sub(pat, rep, s)
    for sym, word in _EQ_SYMBOLS:
        s = s.replace(sym, word)
    s = re.sub(r"[(){}\[\]|]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def figure_label(caption: str) -> tuple[str, str]:
    """'Fig 3. A tree' -> ('Figure 3', 'A tree'). No caption -> ('Figure', '')."""
    m = _LABEL.match(caption or "")
    if not m:
        return "Figure", (caption or "").strip()
    word = m.group(1).lower()
    noun = "Figure" if word.startswith("fig") else word.capitalize()
    return f"{noun} {m.group(2)}", caption[m.end():].strip()


def _first_sentence(text: str, cfg: Config) -> str:
    first = _SENTENCE_END.split(text.strip(), maxsplit=1)[0]
    return first if len(first) <= cfg.summary_max_chars else first[:cfg.summary_max_chars].rsplit(" ", 1)[0] + "…"


def _sentence(s: str) -> str:
    s = s.strip()
    return s if not s or s[-1] in ".!?…:" else s + "."


def _figure(r: Region, i: int) -> list[Utterance]:
    label, rest = figure_label(r.meta.get("caption", ""))
    status = r.meta.get("description_status", "unavailable")
    caption_part = _sentence(rest) if rest else "No caption."
    t2 = {
        "ok": f"{label}: {_sentence(r.meta.get('description', ''))}",
        "unclear": f"{label}, content not recognised. Caption: {caption_part}",
        "timeout": f"{label}: {caption_part} Figure, not described.",
        "pending": f"{label}: {caption_part}",       # being described; the player waits
    }.get(status, f"{label}: {caption_part} Figure, not described.")
    return [Utterance(f"{label}.", 0, "figure", i),
            Utterance(f"{label}: {caption_part}", 1, "figure", i),
            Utterance(t2, 2, "figure", i)]


def _table(r: Region, i: int) -> list[Utterance]:
    """Announce; then caption and size; then every row, header-qualified."""
    label, rest = figure_label(r.meta.get("caption", ""))
    label = label if label != "Figure" else "Table"
    summary, full = tables.speak(label, rest, r.meta["table"])
    return [Utterance(f"{label}.", 0, "table", i),
            Utterance(summary, 1, "table", i),
            Utterance(full, 2, "table", i)]


def plan_all(regions: list[Region], cfg: Config = DEFAULT) -> list[Utterance]:
    out: list[Utterance] = []
    for i, r in enumerate(regions):
        text = r.text.strip()
        if r.kind == "chrome":
            out.append(Utterance(text, 0, "chrome", i, suppressed=True,
                                 why=r.meta.get("why", "")))
        elif r.kind == "caption" and r.context_of is not None:
            continue                                   # spoken inside its figure
        elif r.kind == "figure":
            out += _figure(r, i)
        elif r.kind == "table" and "table" in r.meta:
            out += _table(r, i)
        elif not text:                                 # ink we could not turn into words: say so
            out.append(Utterance("Text here could not be read.", 0, "unread", i))
        else:
            full = {"equation": verbalise_equation(text),
                    "annotation": f"Margin note: {text}",
                    "caption": f"Caption: {text}"}.get(r.kind, text)
            summary = _first_sentence(text, cfg) if r.kind == "body" else full
            out.append(Utterance(_ANNOUNCE.get(r.kind, "Text."), 0, r.kind, i))
            out.append(Utterance(summary, 1, r.kind, i))
            if summary != full:
                out.append(Utterance(full, 2, r.kind, i))
    return out


def at_tier(utts: list[Utterance], tier: int) -> list[Utterance]:
    """Per region, keep the deepest utterance with ``u.tier <= tier``.
    Suppressed and page-level utterances are always kept."""
    chosen: dict[int, Utterance] = {}
    for u in utts:
        if u.suppressed or u.region < 0 or u.tier > tier:
            continue
        if u.region not in chosen or u.tier >= chosen[u.region].tier:
            chosen[u.region] = u
    out: list[Utterance] = []
    placed: set[int] = set()
    for u in utts:
        if u.suppressed or u.region < 0:
            out.append(u)
        elif u.region in chosen and u.region not in placed:
            out.append(chosen[u.region])
            placed.add(u.region)
    return out


def plan(regions: list[Region], tier: int = 2, cfg: Config = DEFAULT) -> list[Utterance]:
    return at_tier(plan_all(regions, cfg), tier)


def plan_page(regions: list[Region], page_no: int, blank: bool,
              cfg: Config = DEFAULT) -> list[Utterance]:
    """``plan_all`` plus the page-level announcements that stop silence from
    being ambiguous: blank, unreadable, or furniture only."""
    if blank:
        return [Utterance(f"Page {page_no}, blank.", 0, "page", -1)]
    utts = plan_all(regions, cfg)
    if utts and all(u.kind == "unread" for u in utts):
        return [Utterance(f"Page {page_no}, could not be read.", 0, "page", -1)] + utts
    if any(not u.suppressed for u in utts):
        return utts
    if not utts:
        return [Utterance(f"Page {page_no}, could not be read.", 0, "page", -1)]
    return [Utterance(f"Page {page_no}: only page furniture, "
                      f"{len(utts)} items skipped.", 0, "page", -1)] + utts
