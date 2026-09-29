"""Orchestration: document -> pages -> regions -> utterances.

  analyse (per page, CPU, parallel)   render, text layer, segmentation, measurement
  OCR (sequential: one device)        only for pages without a text layer
  memory                              repeated furniture across pages
  classify -> reconcile (tier 2) -> reading order -> bind -> describe -> plan -> check (tier 1)

``run`` builds document memory from every page. ``stream`` uses a warm-up
window so the first page can be spoken before the last is read; a string
that only reveals itself as furniture later is suppressed from then on.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from winnow import backends, selfcheck
from winnow.describe import describe_figures, interpret  # noqa: F401  (re-exported)
from winnow.caption_binding import bind_captions
from winnow.chrome import collect, repeated_keys
from winnow.classify import classify
from winnow.pdf_adapter import extract_page, open_pdf, render
from winnow.ocr import ocr_page
from winnow.reading_order import reading_order
from winnow.segment import (is_blank, measure, normalise_polarity, resize_to_work, segment,
                            text_zoom)
from winnow.speech import at_tier, plan_page
from winnow.telemetry import Telemetry, install_offline_guard
from winnow.types import DEFAULT, Config, Region, UnsupportedDocument, Utterance

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


@dataclass
class Page:
    page_no: int
    regions: list[Region]
    gray: np.ndarray
    blank: bool = False
    source: str = "text-layer"          # "text-layer" | "ocr" | "no-ocr"
    inverted: bool = False              # light text on dark: pixels were inverted
    zoom: float = 1.0                   # > 1: rendered larger than work scale (small text)
    naive: str = ""                     # what a naive read-aloud would say
    utterances: list[Utterance] = field(default_factory=list)   # plan_all, every tier
    findings: list = field(default_factory=list)


@dataclass
class Document:
    path: str
    pages: list[Page]
    config_hash: str
    findings: list = field(default_factory=list)
    repairs: list = field(default_factory=list)
    telemetry: Telemetry = field(default_factory=Telemetry)

    @property
    def errors(self):
        return [f for f in self.all_findings() if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def all_findings(self):
        return self.findings + [f for p in self.pages for f in p.findings]

    def transcript(self, tier: int = 2) -> list[tuple[Page, Utterance]]:
        return [(p, u) for p in self.pages for u in at_tier(p.utterances, tier)]

    def badge(self) -> str:
        return selfcheck.badge(self.findings, self.repairs)


# --- per-page analysis (runs in worker processes) ------------------------

def _worker_init():
    install_offline_guard()


def analyse_page(path: str, page_no: int, cfg: Config = DEFAULT) -> Page:
    """Everything up to text: pixels, boxes, text layer, measurements."""
    if path.lower().endswith(IMAGE_EXT):
        src = np.asarray(Image.open(path).convert("L"))
        gray, inv = normalise_polarity(resize_to_work(src, cfg), cfg)
        zoom = text_zoom(gray, cfg)
        if zoom > 1:                    # letters too small to read: resample larger
            gray, inv = normalise_polarity(cv2.resize(
                src, (int(gray.shape[1] * zoom), int(gray.shape[0] * zoom)),
                interpolation=cv2.INTER_CUBIC), cfg)
        return Page(page_no, segment(gray, cfg), gray, is_blank(gray, cfg), "no-ocr", inv, zoom)
    with open_pdf(path) as pdf:
        pg = pdf.pages[page_no - 1]
        gray, inv = normalise_polarity(render(pg, cfg), cfg)
        regions = extract_page(pg, cfg)
        naive = pg.extract_text() or ""
    if not any(r.nature == "text" for r in regions):
        if is_blank(gray, cfg):
            return Page(page_no, [], gray, True, naive=naive, inverted=inv)
        with open_pdf(path) as pdf:     # a scan: re-render larger if letters are too small
            zoom = text_zoom(gray, cfg)
            if zoom > 1:
                gray, inv = normalise_polarity(render(pdf.pages[page_no - 1], cfg, zoom), cfg)
        return Page(page_no, segment(gray, cfg), gray, source="no-ocr", inverted=inv, zoom=zoom)
    measure(regions, gray, cfg)
    return Page(page_no, regions, gray, naive=naive, inverted=inv)


def _count_pages(path: str) -> int:
    if path.lower().endswith(IMAGE_EXT):
        return 1
    if not path.lower().endswith(".pdf"):
        raise UnsupportedDocument(f"{path}: expected a PDF or an image")
    with open_pdf(path) as pdf:
        return len(pdf.pages)


def _analysed(path, cfg, ocr, tel, workers):
    n = _count_pages(path)
    with tel.timed("analyse", "cpu", "CPU", pages=n):
        if n >= cfg.parallel_min_pages and workers != 1:
            with ProcessPoolExecutor(workers or cfg.workers, initializer=_worker_init) as ex:
                # map() yields in submission order: the ordering queue.
                pages = ex.map(analyse_page, [path] * n, range(1, n + 1), [cfg] * n)
                pages = list(pages)
        else:
            pages = [analyse_page(path, i, cfg) for i in range(1, n + 1)]
    for p in pages:                     # OCR is sequential: one device, no contention
        if p.source == "no-ocr":
            ocr_page(p, ocr, cfg, tel)
        yield p


# --- document ------------------------------------------------------------

def _finish(page: Page, describer, cache, cfg: Config, tel: Telemetry) -> None:
    """Reading order onward. Classification (and any repair) already happened.
    With no describer, figures are left "pending" for ``describe_pages``."""
    page.regions = reading_order(page.regions, page.gray.shape[1], cfg)
    bind_captions(page.regions, cfg)
    if describer is None:
        for r in page.regions:
            if r.kind == "figure":
                r.meta["description_status"], r.meta["description"] = "pending", ""
        page.findings = []
    else:
        page.findings = describe_figures(page, describer, cache, cfg, tel)
    page.utterances = plan_page(page.regions, page.page_no, page.blank, cfg)
    page.findings += selfcheck.check_page(page.regions, page.utterances, page.page_no, cfg)


def describe_pages(doc: Document, describer, cfg: Config = DEFAULT, on_page=None) -> None:
    """Describe the figures of a document that is already being read, page by
    page in reading order, replanning each page as its figures come in. One
    worker: the model is one device (§13). A description takes 5-7 s on this
    CPU; the text before a figure usually takes longer to speak, so the
    listener rarely waits, where before nothing was spoken until every figure
    in the document had been described."""
    cache = backends.DescriptionCache()
    for p in doc.pages:
        if not any(r.meta.get("description_status") == "pending" for r in p.regions):
            continue
        found = describe_figures(p, describer, cache, cfg, doc.telemetry)
        p.utterances = plan_page(p.regions, p.page_no, p.blank, cfg)
        p.findings = found + selfcheck.check_page(p.regions, p.utterances, p.page_no, cfg)
        if on_page:
            on_page(p)


def run(path: str, cfg: Config = DEFAULT, describer=None, ocr=None, repair: bool = True,
        tel: Telemetry | None = None, workers: int | None = None,
        describe: bool = True) -> Document:
    """``describe=False`` leaves figures pending, for ``describe_pages``."""
    tel = tel or Telemetry()
    describer = (describer or backends.NullDescriber()) if describe else None
    cache = backends.DescriptionCache()
    pages = list(_analysed(os.fspath(path), cfg, ocr, tel, workers))
    seen: dict = {}
    for p in pages:
        collect(p.regions, p.gray.shape[0], p.page_no, seen, cfg)
    rep = repeated_keys(seen, cfg)
    for p in pages:
        classify(p.regions, p.gray.shape[0], p.gray.shape[1], rep, len(pages), cfg)
    findings, repairs = selfcheck.reconcile(
        [(p.regions, p.gray.shape[0]) for p in pages], cfg, repair=repair)
    for p in pages:
        _finish(p, describer, cache, cfg, tel)
    return Document(os.fspath(path), pages, cfg.hash(), findings, repairs, tel)


def stream(path: str, cfg: Config = DEFAULT, describer=None, ocr=None,
           tel: Telemetry | None = None, workers: int | None = None):
    """Yield finished pages in order after a ``warmup_pages`` window. Tier 2
    runs report-only here: earlier pages have already been spoken."""
    tel = tel or Telemetry()
    describer = describer or backends.NullDescriber()
    cache = backends.DescriptionCache()
    seen: dict = {}
    pending: list[Page] = []
    done = 0

    def flush():
        rep = repeated_keys(seen, cfg)
        for p in pending:
            classify(p.regions, p.gray.shape[0], p.gray.shape[1], rep, done + len(pending), cfg)
            _finish(p, describer, cache, cfg, tel)
            yield p

    for p in _analysed(os.fspath(path), cfg, ocr, tel, workers):
        collect(p.regions, p.gray.shape[0], p.page_no, seen, cfg)
        pending.append(p)
        if done + len(pending) >= cfg.warmup_pages:
            yield from flush()
            done += len(pending)
            pending = []
    yield from flush()
