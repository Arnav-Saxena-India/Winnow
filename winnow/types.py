"""The two data types, the typed errors, and every tunable threshold.

`Config` is the single home of every number a decision is made on (§9).
If you are about to write a literal threshold anywhere else, add a field here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Region:
    x0: int; y0: int; x1: int; y1: int   # page pixels, origin top-left
    nature: str = "text"                   # "text" | "figure" | "table"  — PHYSICAL
    text: str = ""                         # from text layer or OCR
    conf: float = 0.0                      # 0-100; 100 from a text layer; -1 if OCR gives none
    kind: str = "body"                     # SEMANTIC — see taxonomy
    column: int = 0
    context_of: int | None = None          # caption -> index of its figure
    meta: dict = field(default_factory=dict)


@dataclass
class Utterance:
    text: str                              # exactly what TTS receives
    tier: int                              # 0 announce, 1 summarise, 2 describe
    kind: str
    region: int                            # index back into the region list
    suppressed: bool = False
    why: str = ""                          # required when suppressed


@dataclass(frozen=True)
class Config:
    # --- segmentation (image pages) -------------------------------------
    work_height: int = 1600          # px; larger wastes OCR, smaller loses small type
    work_min_width: int = 1000       # px; a tall page (a long screenshot, 6x taller than wide)
                                     # scaled to work_height was 262 px wide, unreadable
    min_glyph_h: float = 8.0         # px; median letter smaller than this at work scale = page
                                     # rendered too small for OCR (a 17x103 in page: 5 px)
    body_glyph_h: float = 11.0       # px; median letter on every corpus page at work scale
    max_work_pixels: int = 60_000_000   # cap on an upscaled page, so memory stays bounded
    dark_paper: int = 110            # median grey below this = light text on dark; page inverted
    dilate_h: int = 25               # joins words into lines
    dilate_v: int = 14               # joins lines into paragraphs; too high merges heading into body
    fig_close_kernel: int = 35       # joins scattered strokes
    fig_min_area: int = 9000         # px²
    merge_iou: float = 0.30
    stroke_min_ratio: float = 3.0    # component taller/wider than this × median glyph = drawing stroke
    line_ink_frac: float = 0.02      # row carries ink if > 2% of region width (8a projection)
    stroke_share_min: float = 0.5    # block is a drawing if most of its ink is oversized strokes
    text_row_min: int = 3            # this many similar tall glyphs is large type, not a drawing
    row_similar: float = 0.5         # "similar" = at least this × the smallest tall component's height
    solid_fill: float = 0.8          # a component filling its box this much is a bar/blob, not a glyph
    ink_min_contrast: int = 40       # grey levels below the paper that always count as ink (faint text)
    noise_area: int = 4              # px²; components smaller than this are specks, not glyphs
    line_overlap: float = 0.5        # words share a line if they overlap this much vertically
    min_contrast: int = 8            # grey levels; below this a crop has no ink to measure
    blank_min_contrast: int = 42     # grey levels; below this a page is blank (Otsu invents ink from noise)
    ocr_tile: int = 4000             # px of upscaled page per OCR call; Windows OCR refuses > 10000
    ocr_tile_overlap: int = 240      # px; a line cut by one strip boundary is whole in the next
    line_join: float = 2.0           # × glyph height: horizontal gap that still joins one text line
    line_min_h: float = 0.5          # × glyph height: shorter ink runs are rules and specks, not lines
    line_margin: float = 0.25        # × line height round each box. Word recall on 4 pages with exact
                                     # truth: 0 -> 84.6%, 0.1 -> 93.4%, 0.25 -> 96.1%, 0.5 -> 88.9%
    ocr_scale: float = 2.0           # OCR on an upscaled page: 3 word errors at 1x, 0 at 2x (scan_p1.png)
    escalate_scale: float = 4.0      # tier 3: one re-OCR of an unread region, cropped, at this scale
    escalate_pad: int = 12           # px of context around a region re-read on escalation

    # --- sparse reclassify ----------------------------------------------
    sparse_min_area: int = 12000     # px²
    sparse_min_h: int = 45           # guards 8d: a short wide strip is a rule, never a drawing
    sparse_max_density: float = 0.25 # chars/kpx²; body scores 3–8, a line drawing 0.03
    sparse_conf_floor: float = 40.0  # OCR text inside a figure below this is marked unreliable

    # --- text-layer adapter ---------------------------------------------
    word_gap_ratio: float = 0.15     # × char size; pdfplumber's fixed 3pt glued 142 words on a
                                     # tight LaTeX paper ("presentaresidual..."), 0.15 glued 2 (URLs)
    gutter_min_w: int = 8            # px; an x-strip crossed by at most gutter_max_cover of the page's
    gutter_max_cover: float = 0.02   # words, at least this wide, is a column gutter (one paper's is
    gutter_min_cover: int = 2        # ...but never fewer than this many words (a sparse page)
    gutter_edge: int = 6             # px; justified columns end and start this close to a gutter...
    gutter_min_edge: int = 8         # ...on at least this many rows each side...
    gutter_edge_ratio: float = 2.0   # ...and this many times the words crossing it. Real gutters:
                                     # <= 6 cross, 15-52 rows aligned; a page of proofs (arXiv
                                     # 1412.6980, p. 14): 18 cross, 15-20 aligned, and it split maths
    gutter_side_frac: float = 0.3    # 20 px, under line joining's 25): this share of words each side
    wall_min_gap: int = 12           # px; only a gap this wide may be cut at a gutter (words: ~5)
    sup_size_ratio: float = 0.8     # char smaller than this × line size, raised = superscript
    para_size_tol: float = 0.15      # lines whose font sizes differ more than this never merge
    bold_frac: float = 0.5           # fraction of bold chars for a block to count as bold
    stroke_fig_max_density: float = 4.0   # chars/kpx²: a cluster with a diagonal stroke under this
                                     # density is a figure. On 4 real papers (46 clusters, labelled by
                                     # eye): charts 0.48-1.70, diagrams 0-3.78, tables 1.07-5.07 but
                                     # never a diagonal stroke; body text 3-8 (§9)
    table_rule_min_w: int = 80       # px; a horizontal rule at least this long may bound a table
    table_rule_tol: int = 12         # px; edges this close are one rule (a thin rect's two edges)
    table_rule_overlap: float = 0.8  # rules overlapping this share of the shorter are one table
    table_max_rule_gap: int = 700    # px; farther apart, two rules are two tables
    table_col_gap: int = 16          # px; a strip this wide no body cell crosses separates columns
                                     # (words in a cell: ~5 px apart; at 10, "Maxout | [10]" split)
    table_head_cover: float = 0.5    # a header cell heads every column whose content it covers this much
    table_full_rule: float = 0.8     # a rule this share of the table's width ends the header
    table_min_rows: int = 2
    table_min_cols: int = 2
    rule_image_max_pt: float = 1.0   # pt; an embedded image this thin is a drawn rule
    fig_label_gap: int = 24          # px; a one-line text block this close to a figure is its label
    fig_label_max_chars: int = 24    # ticks, axis titles, "(a) after 10 epochs"; never a caption
    fig_panel_gap: int = 40          # px; figures this close, side by side or stacked, are panels
    white_min: float = 0.99          # colour component at least this = white (a real fill: 0.9999966)
    diag_min_pt: float = 0.5         # a segment moving this much in both x and y is diagonal
    page_bg_frac: float = 0.9        # a graphic covering this much of the page is background, not a figure

    # --- classification -------------------------------------------------
    band_top: float = 0.10           # page fraction
    band_ref_aspect: float = 1.42    # bands are fractions of at most width × this (A4 portrait):
                                     # on a 6:1 screenshot "top 10%" was a whole screen of content
    band_bottom: float = 0.92
    small_ratio: float = 0.88        # × body LINE height — see 8a
    faint_delta: int = 42            # grey levels lighter than body median ink
    heading_ratio: float = 1.15      # × body line height. Known too strict — fix at step 8 with data
    heading_dark_delta: int = 10
    heading_max_chars: int = 120
    chrome_max_chars: int = 90
    mark_max_lines: float = 1.5      # unread ink no bigger than this × body line height, holding at
    mark_tall: float = 0.5           # most one shape this × its height, is a lone mark ("}", "...")
    rule_max_lines: float = 0.5      # unread ink thinner than this × body line height, and
    rule_min_aspect: float = 4.0     # this many times wider than tall, is a drawn line
    body_min_chars: int = 20         # regions shorter than this don't vote on body line height
    fallback_line_h: float = 30.0    # px; body line height when a page has no body text
    equation_max_chars: int = 80
    equation_long_word: int = 8      # a word longer than this is prose, not maths
    equation_max_long_words: int = 1
    equation_max_words: int = 5      # more words of 3+ letters than this is prose: OCR joined two
                                     # bullets into "- A ways to do part 1 - B ways to do part 2..."
    repeat_band: float = 0.12
    repeat_min_pages: int = 2
    y_bucket: float = 0.02           # page fraction — see 8b

    # --- reading order / binding ----------------------------------------
    y_band: int = 12                 # px, sort quantisation
    col_min_width: float = 0.12
    col_separation: float = 0.18
    col_align_tol: float = 0.03      # × page width; blocks of one column share a left edge
    col_min_blocks: int = 2          # a lone centred title or author line is not a column
    col_span_min: float = 0.25       # blocks wider than this define the text column (§10)
    spanning_width: float = 0.6      # a block wider than this spans all columns
    margin_max_width: float = 0.16
    caption_max_gap: int = 90        # px

    # --- speech ---------------------------------------------------------
    summary_max_chars: int = 200     # tier 1 body: first sentence, capped
    parrot_overlap: float = 0.8      # a "description" sharing this share of words with the caption says nothing new

    # --- self-check -----------------------------------------------------
    over_suppression: float = 0.60
    over_suppression_min_regions: int = 5   # a page of 2 furniture lines is not over-suppressed
    low_yield: float = 0.15          # × document median body chars per page
    min_pages_for_repair: int = 3
    min_majority: float = 0.66

    # --- runtime --------------------------------------------------------
    warmup_pages: int = 5
    workers: int = 3
    parallel_min_pages: int = 6      # below this, process-pool startup costs more than it saves
    blank_ink_frac: float = 0.001    # page with less ink than this is blank

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


DEFAULT = Config()


# --- typed errors: library code raises these; only the CLI catches broadly ---

class WinnowError(Exception):
    """Base for every error Winnow raises on purpose."""


class EncryptedDocument(WinnowError):
    pass


class UnsupportedDocument(WinnowError):
    pass


class BackendUnavailable(WinnowError):
    pass


class NetworkAttempt(WinnowError):
    """Raised by telemetry.assert_offline() on any outbound socket."""
