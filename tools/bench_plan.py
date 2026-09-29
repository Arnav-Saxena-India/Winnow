"""Plan the benchmark: which documents both labellers do.

    python tools/bench_plan.py bench/docs --overlap-pages 20 --seed 0 > bench/plan.txt

Whole documents are drawn at random (fixed seed) until they hold at least
``--overlap-pages`` pages. Those are labelled by both people: they give the
agreement figure, and they are the holdout, never used for tuning. Everything
else is the tuning split. The draw is made before anyone labels or runs
anything, so the holdout cannot be chosen to flatter a result.
"""
import argparse
import random
import sys
from pathlib import Path

import pdfplumber

DOCS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def pages(path: Path) -> int:
    if path.suffix.lower() != ".pdf":
        return 1
    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--overlap-pages", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    docs = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in DOCS)
    if not docs:
        print(f"no documents in {a.folder}", file=sys.stderr)
        return 1
    counts = {d: pages(d) for d in docs}
    order = docs[:]
    random.Random(a.seed).shuffle(order)
    overlap, n = [], 0
    for d in order:
        if n >= a.overlap_pages:
            break
        overlap.append(d)
        n += counts[d]
    tuning = [d for d in docs if d not in overlap]
    print(f"# {len(docs)} documents, {sum(counts.values())} pages, seed {a.seed}")
    print(f"# overlap + holdout: {len(overlap)} documents, {n} pages (both labellers)")
    print(f"# tuning: {len(tuning)} documents, {sum(counts[d] for d in tuning)} pages (labeller A)")
    lab = lambda d, who: d.with_name(f"{d.stem}.{who}.labels.json")  # noqa: E731
    print("\n# labeller A, every document")
    for d in docs:
        print(f"python -m winnow label {d} {lab(d, 'A')} --labeller A")
    print("\n# labeller B, overlap documents only")
    for d in overlap:
        print(f"python -m winnow label {d} {lab(d, 'B')} --labeller B")
    print("\n# then fill each file in: python -m winnow annotate FILE")
    print("\n# agreement, one line per overlap document")
    for d in overlap:
        print(f"python -m winnow eval {lab(d, 'A')} --agree {lab(d, 'A')} {lab(d, 'B')}")
    print("\n# tune on the tuning split; the holdout only gates the result")
    print("python -m winnow tune --tuning " + " ".join(str(lab(d, "A")) for d in tuning)
          + " --holdout " + " ".join(str(lab(d, "A")) for d in overlap))
    print("\n# the reported numbers: the holdout, scored once")
    print("python -m winnow eval " + " ".join(str(lab(d, "A")) for d in overlap))
    return 0


if __name__ == "__main__":
    sys.exit(main())
