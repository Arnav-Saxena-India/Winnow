"""Self-correction without labels: contradiction, disagreement, incoherence.

Tier 1 (``check_page``) asserts the §3 invariants on every page; an
``error`` fails the run. Tier 2 (``reconcile``) treats the document as its
own consistency check: the same text at the same position should get the
same kind on every page.

Detection is cheap and always safe; repair must be earned. A tie is not a
signal. Repair needs ``min_pages_for_repair`` observations and a
``min_majority``; everything below that bar is reported and left alone.
Decisions are taken from a snapshot, then applied in sorted-key order, so no
repair reads another's output. Tier 1 runs after repair, so a repair that
over-suppresses fails its own check.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median

from winnow.chrome import norm, y_bucket
from winnow.classify import line_height
from winnow.types import DEFAULT, Config, Region, Utterance

_SIDE = ("chrome", "annotation")
_BOUND_KINDS = ("caption", "figure", "table")    # repairing these would stale caption binding


@dataclass
class Finding:
    tier: int
    check: str
    severity: str          # "error" | "warn" | "info"
    page: int
    detail: str
    region: int | None = None


# --- tier 1 -------------------------------------------------------------

def title_index(regions: list[Region]) -> int | None:
    """The largest text on the page, by per-line height."""
    texts = [i for i, r in enumerate(regions) if r.nature == "text" and r.text.strip()]
    return max(texts, key=lambda i: line_height(regions[i]), default=None)


def check_page(regions: list[Region], utts: list[Utterance], page_no: int,
               cfg: Config = DEFAULT) -> list[Finding]:
    f: list[Finding] = []

    def add(check, sev, detail, region=None):
        f.append(Finding(1, check, sev, page_no, detail, region))

    texts = [r for r in regions if r.nature == "text" and r.text.strip()]
    for i, r in enumerate(regions):
        if r.kind == "chrome" and not r.meta.get("why"):
            add("suppressed_without_reason", "error", f"'{r.text[:40]}'", i)
        if r.kind == "chrome" and len(r.text) > cfg.chrome_max_chars:
            add("chrome_too_long", "warn", f"{len(r.text)} chars suppressed: '{r.text[:40]}…'", i)
        if r.kind == "caption" and r.context_of is None:
            add("orphan_caption", "warn", f"'{r.text[:40]}' is bound to no figure", i)
    for u in utts:
        if (u.kind == "caption" and not u.suppressed and u.region >= 0
                and regions[u.region].context_of is not None):
            add("caption_spoken_alone", "error", f"'{u.text[:40]}'", u.region)
    for u in utts:
        if u.suppressed and not u.why:
            add("suppressed_without_reason", "error", f"utterance '{u.text[:40]}'", u.region)
    unread = [r for r in regions if r.nature == "text" and not r.text.strip()
              and r.kind != "chrome"]              # lone marks and drawn lines are not text
    if unread:
        why = unread[0].meta.get("unread", "no text recovered")
        add("unread_regions", "warn", f"{len(unread)} text regions could not be read ({why})")
    n_sup = sum(r.kind == "chrome" for r in texts)
    if len(texts) >= cfg.over_suppression_min_regions and n_sup > cfg.over_suppression * len(texts):
        add("over_suppression", "error", f"{n_sup} of {len(texts)} text regions suppressed")
    if texts and not any(not u.suppressed and u.region >= 0 for u in utts):
        add("silent_page_with_ink", "error", f"{len(texts)} text regions, nothing spoken")
    t = title_index(regions)
    if page_no == 1 and t is not None and regions[t].kind == "chrome":
        add("title_suppressed", "error",
            f"largest text '{regions[t].text[:40]}' suppressed: {regions[t].meta.get('why')}", t)
    last: dict[int, int] = {}
    for i, r in enumerate(regions):
        if r.kind in _SIDE:
            continue
        if r.y0 < last.get(r.column, -1) - cfg.y_band:
            add("order_inversion", "warn", f"column {r.column} steps back to y={r.y0}", i)
        last[r.column] = r.y0
    return f


# --- tier 2 -------------------------------------------------------------

@dataclass
class Repair:
    page: int
    region: int
    text: str
    before: str
    after: str
    votes: str
    reason: str


def reconcile(pages: list[tuple[list[Region], int]], cfg: Config = DEFAULT,
              repair: bool = True) -> tuple[list[Finding], list[Repair]]:
    """``pages``: (regions, page_h) per page, classified, before ordering."""
    by_key: dict[tuple, list[tuple[int, int, str]]] = defaultdict(list)
    for p, (regions, h) in enumerate(pages, 1):
        for i, r in enumerate(regions):
            if r.nature == "text" and r.text.strip():
                by_key[(norm(r.text), y_bucket(r.y0, h, cfg))].append((p, i, r.kind))
    titles = {(1, title_index(pages[0][0]))} if pages else set()

    findings, planned = [], []
    for key in sorted(by_key):
        hits = by_key[key]
        votes = Counter(k for _, _, k in hits)
        if len(votes) < 2:
            continue
        (winner, n), *rest = votes.most_common()
        n_pages = len({p for p, _, _ in hits})
        text = pages[hits[0][0] - 1][0][hits[0][1]].text.strip()[:50]
        tally = ", ".join(f"{k} ×{c}" for k, c in sorted(votes.items()))
        losers = [(p, i, k) for p, i, k in hits if k != winner]
        if not repair:
            action = "reported only (--no-repair)"
        elif len(pages) < cfg.min_pages_for_repair or n_pages < cfg.min_pages_for_repair:
            action = f"left alone — insufficient evidence ({n_pages} pages)"
        elif rest and rest[0][1] == n:
            action = f"left alone — tie ({tally})"
        elif n / len(hits) < cfg.min_majority:
            action = f"left alone — no clear majority ({n}/{len(hits)})"
        elif winner in _BOUND_KINDS or any(k in _BOUND_KINDS for _, _, k in losers):
            action = "left alone — would invalidate caption binding"
        elif winner == "chrome" and any((p, i) in titles for p, i, _ in losers):
            action = "left alone — would suppress the title"
        else:
            action = f"repaired toward {winner} ({n}/{len(hits)} consensus)"
            planned += [Repair(p, i, text, k, winner, f"{n}/{len(hits)}",
                               f"same text at same position is {winner} on {n} of {len(hits)} pages")
                        for p, i, k in losers]
        findings.append(Finding(2, "cross_page_conflict", "warn", hits[0][0],
                                f"'{text}': {tally} — {action}"))

    for rp in planned:                          # applied after every decision is made
        r = pages[rp.page - 1][0][rp.region]
        r.meta["repaired"] = {"from": rp.before, "to": rp.after,
                              "reason": rp.reason, "votes": rp.votes}
        if r.kind == "chrome":
            r.meta["why_before_repair"] = r.meta.pop("why", "")
        r.kind = rp.after
        if rp.after == "chrome":
            r.meta["why"] = f"page furniture: {rp.reason} (repaired)"
    return findings + _yield_check(pages, cfg), planned


def _yield_check(pages, cfg: Config) -> list[Finding]:
    chars = [sum(len(r.text) for r in regs if r.kind == "body") for regs, _ in pages]
    if len(chars) < cfg.min_pages_for_repair or not median(chars):
        return []
    m = median(chars)
    return [Finding(2, "low_yield", "warn", p, f"{c} body chars vs document median {m:.0f}: "
                    "segmentation may have failed")
            for p, c in enumerate(chars, 1) if c < cfg.low_yield * m]


def badge(findings: list[Finding], repairs: list[Repair]) -> str:
    conflicts = [f for f in findings if f.check == "cross_page_conflict"]
    if not conflicts:
        return "no cross-page conflicts"
    fixed = [f for f in conflicts if "repaired toward" in f.detail]
    left = [f for f in conflicts if "left alone" in f.detail or "reported only" in f.detail]
    s = f"{len(conflicts)} cross-page conflict{'s' * (len(conflicts) != 1)} found"
    if fixed:
        votes = sorted({rp.votes for rp in repairs})
        s += f", {len(fixed)} repaired ({', '.join(votes)} consensus)"
    if left:
        d = left[0].detail
        why = (d.split("left alone — ", 1)[1].split(" (")[0] if "left alone — " in d
               else "repair disabled")
        s += f", {len(left)} left alone — {why}"
    return s
