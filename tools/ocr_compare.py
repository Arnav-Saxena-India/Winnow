"""Word recall of each installed OCR engine on a region with known text.

    python tools/ocr_compare.py DOC TRUTH.txt --box X0 Y0 X1 Y1

The box is in Winnow's page pixels (``winnow render`` draws them). TRUTH.txt
holds the region's text, typed by a person. Recall = share of true words the
engine produced (case and punctuation ignored, each word counted once per use).
Each engine is run the way Winnow runs it: EasyOCR on Winnow's own line boxes,
Windows OCR on the page upscaled by ``ocr_scale``, in strips.
"""
import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from winnow import backends, pipeline  # noqa: E402
from winnow.ocr import line_boxes, read_in_strips  # noqa: E402
from winnow.lines import group_lines  # noqa: E402
from winnow.types import DEFAULT, BackendUnavailable  # noqa: E402

_WORD = re.compile(r"[a-z0-9^]+")


def words(text: str) -> Counter:
    return Counter(_WORD.findall(text.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc")
    ap.add_argument("truth")
    ap.add_argument("--box", type=int, nargs=4, required=True)
    ap.add_argument("--page", type=int, default=1)
    a = ap.parse_args()
    truth = words(Path(a.truth).read_text(encoding="utf-8"))
    page = pipeline.analyse_page(a.doc, a.page)
    x0, y0, x1, y1 = a.box
    crop = page.gray[y0:y1, x0:x1]
    for cls in (backends.EasyOcr, backends.WindowsOcr):
        try:
            ocr = cls()
        except BackendUnavailable as e:
            print(f"{cls.__name__}: unavailable ({e})")
            continue
        t = time.time()
        if hasattr(ocr, "read_lines"):
            ws = ocr.read_lines(crop, line_boxes(crop))
        else:
            k = DEFAULT.ocr_scale
            big = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            ws = [dict(w, top=w["top"] / k, bottom=w["bottom"] / k, x0=w["x0"] / k, x1=w["x1"] / k)
                  for w in read_in_strips(ocr, big)]
        ms = 1000 * (time.time() - t)
        got = words(" ".join(w["text"] for w in ws))
        hit = sum((truth & got).values())
        print(f"{ocr.name:12s} word recall {hit}/{sum(truth.values())} "
              f"= {hit / sum(truth.values()):.0%}   extra words {sum((got - truth).values())}"
              f"   {ms:.0f} ms")
        print("   " + " ".join(w["text"] for ln in group_lines(ws) for w in ln)[:300])


if __name__ == "__main__":
    main()
