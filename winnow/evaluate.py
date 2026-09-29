"""Benchmark scoring and offline threshold learning (tier 4).

Labels: JSON ``{"doc": path, "labeller": name, "pages": {"1": [{"box": [x0,y0,x1,y1],
"kind": k}, ...]}}`` in work-height pixels (what `winnow render` draws).

Metrics are always a pair and precision leads:
  content_precision  of everything suppressed, the share that truly was chrome
                     (1.0 when nothing is suppressed — which is why it is never
                     reported alone)
  chrome_recall      of true chrome, the share suppressed
  content_lost       true content regions suppressed or never produced — the
                     failure a listener cannot detect

Tuning (tier 4) is coordinate descent over a few Config fields. A candidate
is kept only if tuning-split precision does not drop and recall rises. The
result is promoted only if the untouched holdout does not regress on either.
A change that buys recall with precision is surfaced, never taken.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from winnow import pipeline
from winnow.types import DEFAULT, Config, WinnowError

MATCH_IOU = 0.5          # evaluation matching, not a runtime threshold
KINDS = ("chrome", "heading", "body", "caption", "figure", "equation", "annotation", "table")
TUNABLE = ("small_ratio", "faint_delta", "heading_ratio", "band_top", "band_bottom",
           "repeat_band", "sparse_max_density", "caption_max_gap")


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _match(label_box, preds):
    """The smallest predicted box containing the label's centre (segmentation
    pads boxes, so IoU alone fails on small lines); IoU as the fallback."""
    cx, cy = (label_box[0] + label_box[2]) / 2, (label_box[1] + label_box[3]) / 2
    holding = [p for p in preds if p[0][0] <= cx <= p[0][2] and p[0][1] <= cy <= p[0][3]]
    if holding:
        return min(holding, key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1]))
    best = max(preds, key=lambda p: _iou(p[0], label_box), default=None)
    return best if best and _iou(best[0], label_box) >= MATCH_IOU else None


def template(doc, doc_ref: str, labeller: str) -> dict:
    """A label file for a person to fill in: Winnow's boxes and text, every kind
    blank. Kinds are left blank on purpose, so neither labeller is anchored on
    the classifier being evaluated. Fix wrong boxes; add missed ones."""
    return {"doc": doc_ref, "labeller": labeller, "kinds": list(KINDS),
            "pages": {str(p.page_no): [{"box": [r.x0, r.y0, r.x1, r.y1], "kind": "",
                                        "text": r.text[:40]} for r in p.regions]
                      for p in doc.pages}}


def load(path: str | Path) -> dict:
    lab = json.loads(Path(path).read_text(encoding="utf-8"))
    todo = [f"p{pg} {b.get('text', '')[:20]!r}" for pg, boxes in lab["pages"].items()
            for b in boxes if b["kind"] not in KINDS]
    if todo:
        raise WinnowError(f"{path}: {len(todo)} boxes without a valid kind, e.g. "
                          f"{', '.join(todo[:3])}; choose from {', '.join(KINDS)}")
    lab["doc"] = str((Path(path).parent / lab["doc"]).resolve())
    return lab


@dataclass
class Score:
    suppressed: int = 0
    suppressed_chrome: int = 0
    chrome: int = 0
    content: int = 0
    content_lost: int = 0
    kind_agree: int = 0
    matched: int = 0

    def __add__(self, o: "Score") -> "Score":
        return Score(*(a + b for a, b in zip(vars(self).values(), vars(o).values())))

    @property
    def content_precision(self) -> float:
        return self.suppressed_chrome / self.suppressed if self.suppressed else 1.0

    @property
    def chrome_recall(self) -> float:
        return self.suppressed_chrome / self.chrome if self.chrome else 1.0

    def line(self) -> str:
        return (f"content precision {self.content_precision:.3f}  chrome recall "
                f"{self.chrome_recall:.3f}  content lost {self.content_lost}/{self.content}  "
                f"kind accuracy {self.kind_agree}/{self.matched}")


def score(labels: dict, cfg: Config = DEFAULT, ocr=None) -> Score:
    doc = pipeline.run(labels["doc"], cfg, ocr=ocr, workers=1)
    s = Score()
    for page in doc.pages:
        preds = [((r.x0, r.y0, r.x1, r.y1), r.kind) for r in page.regions]
        for lab in labels["pages"].get(str(page.page_no), []):
            best = _match(lab["box"], preds)
            kind = best[1] if best else None
            truth_chrome = lab["kind"] == "chrome"
            s.chrome += truth_chrome
            s.content += not truth_chrome
            if kind is not None:
                s.matched += 1
                s.kind_agree += kind == lab["kind"]
            if kind == "chrome":
                s.suppressed += 1
                s.suppressed_chrome += truth_chrome
            if not truth_chrome and kind in (None, "chrome"):
                s.content_lost += 1
    return s


def evaluate(label_sets: list[dict], cfg: Config = DEFAULT, ocr=None) -> Score:
    return sum((score(lab, cfg, ocr) for lab in label_sets), Score())


def agreement(a: dict, b: dict) -> tuple[float, int]:
    """Cohen's kappa between two labellers over boxes they both drew."""
    pairs = []
    for page, boxes in a["pages"].items():
        for la in boxes:
            other = max(b["pages"].get(page, []), key=lambda lb: _iou(la["box"], lb["box"]),
                        default=None)
            if other and _iou(la["box"], other["box"]) >= MATCH_IOU:
                pairs.append((la["kind"], other["kind"]))
    if not pairs:
        return float("nan"), 0
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return (1.0 if pe == 1 else (po - pe) / (1 - pe)), n


def _better(new: Score, old: Score) -> bool:
    return (new.content_precision >= old.content_precision
            and new.chrome_recall > old.chrome_recall)


def tune(tuning: list[dict], holdout: list[dict], cfg: Config = DEFAULT,
         fields=TUNABLE, factors=(0.85, 1.15), ocr=None) -> tuple[Config, list[str]]:
    log = [f"baseline {cfg.hash()} tuning: {(base := evaluate(tuning, cfg, ocr)).line()}"]
    best, best_s = cfg, base
    for name in fields:
        for f in factors:
            v = getattr(best, name)
            cand = replace(best, **{name: type(v)(v * f)})
            s = evaluate(tuning, cand, ocr)
            if _better(s, best_s):
                log.append(f"  kept   {name} {v} -> {getattr(cand, name)}: {s.line()}")
                best, best_s = cand, s
            elif s.chrome_recall > best_s.chrome_recall:
                log.append(f"  SURFACED, not taken: {name} {v} -> {getattr(cand, name)} raises "
                           f"recall but lowers content precision ({s.line()})")
    if best is cfg:
        return cfg, log + ["no change improved the tuning split"]
    h0, h1 = evaluate(holdout, cfg, ocr), evaluate(holdout, best, ocr)
    log.append(f"holdout before: {h0.line()}\nholdout after:  {h1.line()}")
    if h1.content_precision < h0.content_precision or h1.chrome_recall < h0.chrome_recall:
        return cfg, log + ["REJECTED: holdout regressed; config unchanged"]
    return best, log + [f"PROMOTED {best.hash()} (a human still signs off)"]
