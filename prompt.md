# Winnow — build prompt

You are building Winnow: a reader for content that has no text layer. Read this
whole file before writing any code. It encodes decisions that took real debugging
to find; re-deriving them will cost you a day each.

---

## 1. Mission

Every text-to-speech tool can read. None knows what is worth reading.

Open a PDF in Edge and press Read Aloud. Before it reaches the title it reads the
running header, the date chip, the byline. At a figure it says "Figure 3." and
moves on — having read the label and skipped the content. At an equation it emits
"x 2 plus y 2 equals r 2". At the bottom it reads "Scanned by CamScanner" and
"Page 14".

Winnow decides **what to say, in what order, and what to say instead** for the
parts that are not text.

Two user-visible behaviours define the product:

- **Substitution.** A caption is never spoken alone. `Fig 3.` carries no
  information. The figure's *description* is what the listener needs, and the
  caption becomes context for producing it.
- **Announce, never silently skip.** Suppressed content stays reachable at tier 0
  with a stated reason. A system that removes content without record is deciding
  what the user is allowed to know.

Getting characters off a page is the easy half. Deciding what each region *is* is
the product.

---

## 2. Stance

Build it the way you would build something you have to debug at 2am.

- **Small files, flat structure.** Six modules, none over ~250 lines. If a file
  is growing past that, the split is usually along a stage boundary.
- **No frameworks.** numpy, opencv, and the adapters' own libraries. No DI
  container, no plugin registry, no config framework, no ORM. You do not need them.
- **No premature abstraction.** One implementation of an interface is a function.
  Write the second implementation before you write the Protocol.
- **Measure, don't guess.** Every threshold in this system was set by looking at
  a number, not by intuition. When you are tempted to tune a value, print the
  distribution first.
- **Pure where it matters.** The speech planner takes regions and returns
  utterances with no I/O and no model calls. That purity is what makes the entire
  product testable without audio, a model, or a device. Do not break it.
- **Readable beats clever.** Someone tuning `sparse_max_density` at 2am should be
  able to read the function and see why 0.25.

---

## 3. Invariants

These are not preferences. Violating any of them is a bug, and each has a test.

1. `speech.plan()` is pure. Same regions in, same utterances out.
2. No module imports `backends` except `pipeline` and `cli`. Backends are
   injected, never constructed inside logic.
3. `classify` never mutates boxes. `reading_order` never mutates `kind`.
4. Every region with `kind == "chrome"` carries a non-empty `meta["why"]`.
5. A caption with a bound figure is never emitted as its own utterance.
6. Suppressed content is retained at tier 0. Nothing is deleted from the output.
7. **A repaired run is reproducible.** `--no-repair` reproduces raw output exactly,
   and every repair is logged with its reason and vote count.
8. **The title is never suppressed.** This has broken twice by two different
   mechanisms. It is the single most dangerous failure in the system, because a
   false suppression is invisible to the listener while a false read is merely
   annoying. Those two failures are not equally bad and the code must reflect that.

---

## 4. Data model

Two types. Everything else is a function.

```python
@dataclass
class Region:
    x0: int; y0: int; x1: int; y1: int   # page pixels, origin top-left
    nature: str = "text"                 # "text" | "figure"  - PHYSICAL
    text: str = ""                       # from text layer or OCR
    conf: float = 0.0                    # 0-100; 100 when from a text layer
    kind: str = "body"                   # SEMANTIC - see taxonomy
    column: int = 0
    context_of: int | None = None        # caption -> index of its figure
    meta: dict = field(default_factory=dict)

@dataclass
class Utterance:
    text: str                            # exactly what TTS receives
    tier: int                            # 0 announce, 1 summarise, 2 describe
    kind: str
    region: int                          # index back into the region list
    suppressed: bool = False
    why: str = ""                        # required when suppressed
```

**`nature` and `kind` are different axes and must not be merged.** `nature` is
physical: text pixels or drawing pixels. `kind` is semantic: heading, caption,
furniture. Sparse reclassification changes `nature` *after* OCR, while `kind` has
not been decided yet. Collapsing them breaks that ordering. This is the first
refactor someone attempts; do not.

**`meta["why"]` is user-facing, not debug output.** When a listener asks what was
skipped, they hear the reason. A suppression the system cannot explain is a
suppression it should not have made.

### Stage contracts

| Stage | Reads | Writes |
| --- | --- | --- |
| Segmentation | page image | boxes, `nature` |
| Text extraction | boxes | `text`, `conf` |
| Sparse reclassify | `text`, `conf`, area | `nature`, `meta` |
| Classification | all above | `kind`, `meta["why"]` |
| Reading order | `kind`, boxes | list order, `column` |
| Caption binding | `kind`, boxes | `context_of`, `meta` |
| Speech planner | everything | `list[Utterance]` |

---

## 5. Pipeline

```
input adapter -> segmentation -> text extraction -> classification
              -> reading order -> speech planner -> TTS
                                       |
                        figures only -> VLM describer -> back to planner

document memory (repeated furniture) feeds classification; built from all pages
```

Two ordering decisions matter:

**Classification runs after text extraction, not before.** A region's class
depends on what it says as much as where it sits. "Page 14" in a footer is
furniture; the same string mid-paragraph is content. Splitting on geometry alone
produces exactly the Edge behaviour.

**The VLM is a branch, not a stage.** Only figure regions reach it. Routing every
region through a 2B model costs two orders of magnitude more compute for nothing,
because body text is already text.

---

## 6. The taxonomy

This is the contribution. Published work on image accessibility addresses *how* to
describe an image; deciding *whether*, and what to say instead, is an open gap.

| Class | Signals | Handling |
| --- | --- | --- |
| `chrome` | repeats at same position across pages; watermark lexicon; page-number pattern in band; faint/small in band | suppressed, tier 0 with reason |
| `heading` | line height > body median; darker; short; no terminal period | spoken, tier 1 |
| `body` | default | verbatim |
| `caption` | `Fig/Figure/Table/Chart + number` prefix | **never spoken alone** — context for its figure |
| `figure` | `nature == "figure"` | VLM description, prefixed with label |
| `equation` | operator glyphs, few long words, short, not date-like | verbalised |
| `annotation` | narrow, outside the body column | "Margin note: …", between sentences |
| `table` | ruled or aligned grid | header-qualified rows *(not built)* |

### Chrome detection, in priority order

First match wins. Order is load-bearing.

1. **Repeats at the same position across pages.** Normalise digits to `#`, key on
   `(normalised_text, y_bucket)`, flag anything appearing in two or more pages.
   Strongest signal in the system and no shipping read-aloud tool uses it.
2. **Watermark lexicon.** `Scanned by`, `CamScanner`, `Adobe Scan`,
   `Trial version`, `Confidential`, `©`. Anywhere on the page.
3. **Page number.** Pattern match *and* inside a band. The band is essential — a
   bare number mid-paragraph is content.
4. **Metadata chip.** ISO dates, `@handle`. **No band requirement** — see §8c.
5. **Faint or small in a band.** Line height < 0.88× body median, or ink 42 grey
   levels lighter, within top 10% / bottom 8%, under 90 characters.

---

## 7. Build order

Each step has a gate. **Do not proceed past a failing gate.**

| # | Build | Gate |
| --- | --- | --- |
| 0 | Engine runs on the synthetic fixture | six assertions pass |
| 1 | Debug render | eleven boxes, correctly coloured and labelled |
| 2 | AI Hub latency probe *(async, start now)* | ms/figure on Snapdragon X Elite |
| 3 | PDF text-layer adapter | nine regions on the test PDF, no OCR |
| 4 | Wire adapter into classifier | §8 fixes applied, §0 gate still green |
| 4b | Tier 1 + 2 self-checks | clean doc reports zero findings; injected fault is caught |
| 5 | TTS + tier slider | tier 0/1/2 produce different output from one plan |
| 5b | Prototype UI | hover shows evidence; offline guard green; compare toggle works |
| 6 | Config extraction | no function reads a literal threshold |
| 7 | Benchmark, 100 pages labelled | inter-labeller agreement reported |
| 8 | Tune and evaluate | chrome recall + content precision, on holdout |
| 8b | Tier 4 offline loop | a promoted change improves tuning split without regressing holdout |
| 9 | VLM describer | figure description ≠ its caption; `UNCLEAR` tested |
| 10 | Package | README opens with the A/B transcript |

**Step 1 before step 8.** You cannot tune twenty-five thresholds by reading
transcripts. The debug render is ~40 lines and it is the highest-value file in the
repository. It is also the demo slide.

**Step 2 before step 9.** Under 1.5 s/figure means description is on-demand. Over
3 s means it must be precomputed at document open. These are different
architectures. Get the number before you build the wrong one.

**Step 3 before any OCR work.** The motivating failure happens on documents that
*have* text layers. `pdfplumber.extract_words()` gives boxes; the classifier
consumes them unchanged. No OCR, no models, real demo.

---

## 8. Known failure modes

Four bugs, already found and fixed. Implement the fixes; do not rediscover them.

### 8a — Title suppressed, mechanism 1: block height vs line height

`small` was measured against the median **block** height. A three-line paragraph
is ~90 px; a single-line 26 pt title is ~36 px. Every single-line region therefore
tested as "small", and the title sat inside the top band. The system deleted the
most important text on the page.

**Fix:** compare **line heights**. Count lines by horizontal projection of the
binarised crop — runs of rows carrying ink above 2% of region width — then divide
block height by line count.

### 8b — Title suppressed, mechanism 2: text match without position

A document whose title repeats the running header's words (extremely common) has
both matched by cross-page repetition, and both suppressed.

**Fix:** position is part of the key.

```python
Y_BUCKET = 0.02

def y_bucket(y0, page_h):
    return int((y0 / max(1, page_h)) / Y_BUCKET)

# collection
seen[(_norm(t), y_bucket(r.y0, h))].add(page_no)

# classification
if (_norm(t), y_bucket(r.y0, page_h)) in repeated and (top or bottom):
    r.kind, r.meta["why"] = "chrome", "repeats across pages"
```

Furniture repeats at the *same place* on every page. A title that echoes the
header appears at that place only once.

### 8c — A date read as arithmetic

`2026-09-19` → *"2026 minus 09 minus 19"*. The hyphen is in `EQUATION_CHARS`, the
string is short, no long words, so it passes the equation test. Separately, the
date-chip rule required the top 10% of the page; on A4 at 150 dpi a byline sits at
14%, so it never fired.

**Fix:** reject date-like strings in `looks_like_equation`, and drop the band
requirement from the date-chip rule. A bare ISO date is a date wherever it sits.

```python
_DATELIKE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\s*$")
```

### 8d — Header misread as a figure

A running header merging with its underline rule forms a wide, short, sparse block
that scores 0.55 chars/kpx and reclassifies as a drawing.

**Fix:** `min_h = 45`. A short wide strip is a rule or a header line, never a
drawing.

### 8e — OCR line grouping (if you touch the OCR backend)

Backend line indices are unreliable. In sparse-text mode an engine reports each of
an Aadhaar-style grouped number's chunks as its own block, so multi-part values
never form. **Group lines geometrically by vertical overlap.** Geometry is the only
signal every backend agrees on.

---

## 9. Thresholds

Every one of these is currently a literal. Extract them to a frozen `Config`
dataclass in `types.py`, threaded as a parameter with a module-level default.

| Name | Value | Note |
| --- | --- | --- |
| `work_height` | 1600 | px; larger wastes OCR, smaller loses small type |
| `dilate_h` | 25 | joins words into lines |
| `dilate_v` | 14 | joins lines into paragraphs; too high merges heading into body |
| `fig_close_kernel` | 35 | joins scattered strokes |
| `fig_min_area` | 9000 | px² |
| `merge_iou` | 0.30 | |
| `sparse_min_area` | 12000 | px² |
| `sparse_min_h` | 45 | **guards 8d** |
| `sparse_max_density` | 0.25 | chars/kpx²; body scores 3–8, a line drawing scored 0.03 |
| `sparse_conf_floor` | 40 | |
| `band_top` / `band_bottom` | 0.10 / 0.92 | page fraction |
| `small_ratio` | 0.88 | × body **line** height — see 8a |
| `faint_delta` | 42 | grey levels above body median |
| `heading_ratio` | 1.15 | × body line height. **Known too strict** — fix at step 8 with data |
| `heading_dark_delta` | 10 | |
| `chrome_max_chars` | 90 | |
| `repeat_band` | 0.12 | |
| `repeat_min_pages` | 2 | |
| `y_band` | 12 | px, sort quantisation |
| `col_min_width` | 0.12 | |
| `col_separation` | 0.18 | |
| `margin_max_width` | 0.16 | |
| `caption_max_gap` | 90 | px |
| `Y_BUCKET` | 0.02 | page fraction — see 8b |

The 0.25 density threshold is not delicate: the separation between body text and
line drawings is three orders of magnitude. Do not agonise over it.

---

## 10. Anti-patterns

Do not:

- **Merge `nature` and `kind`.** §4.
- **Use min/max for the text-column span.** One horizontal rule or a page-wide
  footer stretches it to the full page, after which no margin note is ever
  detected. Use `np.median` over blocks wider than 25% of the page.
- **Filter OCR output by confidence.** It silently drops real content. For a
  reader, unreadable text is *more* dangerous than readable text.
- **Trust a backend's line numbers.** §8e.
- **Silently skip anything.** Suppression is always recorded with a reason. A page
  that produces no utterances must say "page N, could not be read" — a listener
  cannot distinguish silence-because-blank from silence-because-broken.
- **Let the VLM invent.** Keep the `UNCLEAR` escape and write a test that feeds it
  noise and asserts the fallback fires. A fabricated chart description is worse
  than no description, because the listener cannot detect it.
- **Cache on the image hash alone.** Key on `phash(crop) + model_id +
  prompt_version`, or a prompt change silently serves stale descriptions and
  presents as "the model ignored my edit".
- **Tune against the holdout.** Twenty-five tunables, one hundred pages.
  Overfitting is the default outcome, not a risk.
- **Repair on a tie.** A 1-1 disagreement is noise. Require a 2/3 majority over at
  least three observations, or report and change nothing.
- **Let a repair loop optimise its own consistency.** Uniform classification is the
  global optimum of cross-page agreement and it destroys the product.
- **Report chrome recall without content precision.** A system that suppresses
  everything scores perfect recall. Always the pair, and lead with precision.

---

## 11. Self-correction

The system has no labels at runtime. It cannot know it is *right*. It can know it
is **inconsistent**, and inconsistency is a reliable proxy for error.

Three signals need no ground truth:

| Signal | Meaning | Cost |
| --- | --- | --- |
| **Contradiction** | output violates a stated invariant | free |
| **Disagreement** | two independent paths compute different answers | free |
| **Incoherence** | the same thing classified differently across pages | free |

Build `winnow/selfcheck.py` with these, and run it on every document. Four tiers,
in increasing cost and decreasing autonomy.

### Tier 1 — invariant assertions (always on, free)

Run after every page. Each maps directly to an invariant in §3.

| Check | Severity | Fires when |
| --- | --- | --- |
| `suppressed_without_reason` | error | chrome region with empty `meta["why"]` |
| `caption_spoken_alone` | error | caption emitted despite a bound figure |
| `over_suppression` | error | >60% of text regions suppressed |
| `silent_page_with_ink` | error | text regions present, nothing spoken |
| `chrome_too_long` | warn | furniture over `chrome_max_chars` |
| `orphan_caption` | warn | caption bound to no figure |
| `order_inversion` | warn | reading order steps backwards within a column |

An `error` fails the run. Do not ship output that trips one.

### Tier 2 — cross-page reconciliation

**The document is its own consistency check.** A string classified as furniture on
one page and content on another is a contradiction detectable without ever being
told the truth. Key on `(normalised_text, y_bucket)`, collect the `kind` each
occurrence received, and flag any key with more than one.

Also track body-character yield per page: a page yielding under 15% of the
document median has probably failed segmentation, not gone blank.

### Tier 3 — uncertainty-triggered escalation

Spend compute only where the system is unsure. Re-OCR a low-confidence region at
higher resolution; ask the VLM whether an ambiguous region is a figure; re-run
segmentation with a different `dilate_v` and compare. Escalation is per-region and
bounded — never a whole-document retry loop.

### Tier 4 — offline threshold learning (NOT at runtime)

This is where the RL framing is actually honest:

| RL concept | Here |
| --- | --- |
| Environment | the labelled benchmark, §7 step 7 |
| Policy | the `Config` thresholds |
| Reward | content precision, then chrome recall |
| Episode | one evaluation run over the tuning split |
| Deployment gate | a human, plus the untouched holdout |

Log every tier 1–3 finding with the config hash that produced it. Offline, propose
threshold changes, evaluate on the tuning split, and promote **only** if the
holdout does not regress. Twenty-five parameters over a hundred pages is small
enough that coordinate descent or a light random search is plenty; you do not need
gradients and you should not pretend to have them.

### The failure that makes this dangerous

I built tier 2 with the repair rule *"a conflict between suppressed and spoken
resolves toward spoken"*, on the reasoning that over-reading is recoverable and
silent suppression is not. That reasoning is right and the rule was still wrong.

On a two-page document with one injected fault, the repair took a **correctly
suppressed** running header and un-suppressed it, because one page disagreed and a
1–1 tie resolved toward "speak". The self-correction amplified the fault instead of
fixing it.

**A tie is not a signal.** Repair requires a clear majority over enough
observations:

```python
MIN_PAGES_FOR_REPAIR = 3
MIN_MAJORITY = 0.66

if len(pages) < MIN_PAGES_FOR_REPAIR:
    report_only()                 # not enough evidence to act
for key, hits in by_key.items():
    if len(hits) < MIN_PAGES_FOR_REPAIR:
        continue
    winner, n = Counter(k for _, _, k in hits).most_common(1)[0]
    if n / len(hits) < MIN_MAJORITY:
        continue                  # report the conflict, change nothing
    repair_minority_toward(winner)
```

Everything below the bar is reported and left alone. Detection is cheap and always
safe; repair is expensive to get wrong and must be earned.

### Reward hacking — read this before tuning anything

Every one of these scores well and is useless. If an automated loop is optimising
thresholds, it will find them.

- **Suppress everything.** Chrome recall 1.0. This is why precision and recall are
  always reported as a pair, and why precision leads.
- **Suppress nothing.** Content precision 1.0, and the product does not exist.
- **Never bind captions.** Zero `caption_spoken_alone` errors, because captions are
  simply read aloud like furniture.
- **Emit one region per page.** Zero order inversions.
- **Tier 2 consensus collapse.** Uniform classification maximises cross-page
  agreement. A repair loop that rewards its own consistency will converge on
  classifying every region the same and reporting perfect coherence.

That last one is the real hazard, because it is self-reinforcing and looks healthy
from the inside. Guard it structurally: **tier 2 never repairs toward `chrome`
unless the majority is chrome across at least three pages**, and the tier 1
`over_suppression` error runs *after* repair, not before. A self-correcting system
must be able to fail its own check.

### Reproducibility

Self-correction must never make a run unreproducible.

- Log config hash, every finding, and every repair applied.
- `--no-repair` must reproduce the raw, unrepaired output exactly.
- A repaired region carries `meta["repaired"]` with the reason and the vote count.
- Repairs apply in a deterministic order; no repair reads another repair's output
  within the same pass.

---

## 12. The prototype

One window. Drag a document onto it.

```
┌─────────────────────────────────┬──────────────────────┐
│                                 │  TRANSCRIPT          │
│   [page, rendered]              │  Balanced Binary…    │
│   ░░ faint header ░░  greyed    │ ▶h equals big O of…  │
│   ██ title ██  speaking now     │  Margin note: ask…   │
│   body text…                    │  ──────────────      │
│   [ figure ]  green outline     │  Skipped (4)    ▼    │
│   ░░ Scanned by CamScanner ░░   │  · CSE 2026 | Unit 3 │
│                                 │    repeats on 4 of 4 │
├─────────────────────────────────┴──────────────────────┤
│  ▶ ‖ ⏮ ⏭   1.0×   tier: announce ─○── full             │
│  [ compare: naive ⟷ winnow ]   EasyOCR·NPU·180ms  net 0│
└────────────────────────────────────────────────────────┘
```

The region being spoken highlights and scrolls itself into view. Furniture is
visibly greyed, so the listener *sees* it being passed over. The `compare`
toggle plays naive read-aloud against Winnow on the same file — fifteen seconds
each makes the case without a slide.

`Skipped (n)` lists every suppression with its reason. This is
announce-never-skip made visible, and it answers the first objection a careful
reviewer has: that the tool is quietly deleting things.

### The five upgrades, in priority order

A passive demo proves the product works. It does not prove the system is
intelligent. These do.

**1. Ask a region a question — tier 3.** Click a figure, type *"how many
levels?"*, hear the answer. Nothing shipping does this. It moves the demo from
"reads better" to "understands", which is the difference between a 4 and a 5 on
innovation. `Describer.ask(crop, question, context)` is already in the interface
and `Utterance.region` already indexes back to the crop. **Gated on the VLM.**

**2. Hover any region → the rule and its numbers.** Not "chrome" but
`chrome — repeats at this position on 4 of 4 pages`, `size_ratio 0.69`,
`ink_delta +89`, `band top`. Engineers trust what they can inspect. **Free** —
`meta["evidence"]` carries every measured value the decision was made on.

**3. Compute strip.** `EasyOCR · NPU · 180ms   Qwen3-VL · NPU · 412ms · 2.1W`.
The on-device claim is otherwise taken on faith, and this is the one thing a
Qualcomm reviewer specifically wants to see. **Gated on the NPU backend**;
`telemetry.Telemetry.record()` already collects it.

**4. Network counter reading 0.** Pull the wifi mid-playback; nothing changes.
Five seconds, unforgettable, and it evidences the product's central claim.
**Free** — `telemetry.assert_offline()` raises on any socket attempt, so the
counter is enforced rather than decorative. Run the whole pipeline inside it in
CI.

**5. Tier slider, live.** Drag announce → full and the same page gets deeper:
*"heading. paragraph. equation."* → the full text. **Free** —
`plan(regions, img, tier=n)`. The tier is applied after planning, so dragging
the slider filters an existing plan instead of re-analysing the document.

### Two things that are not features

**Show a failure on purpose.** Leave one figure the VLM cannot read, so the
reviewer hears *"figure 4, content not recognised"* and watches it fall back to
the caption. A system that only ever works looks cherry-picked. A system that
visibly knows what it does not know reads as real — and it exercises the
`UNCLEAR` path, which is the honest reason it exists.

**Surface the self-correction.** A badge: *"2 cross-page conflicts found, 1
repaired (4/5 consensus), 1 left alone — insufficient evidence"*. The last
clause is the impressive part. Most systems that self-correct cannot tell you
when they chose not to.

### Scope discipline

A reviewer gives you ninety seconds. Build **2, 4 and 5** — all free, all
supported by code that exists. Add **1** only if the latency probe comes back
under 3 s. Leave **3** until the NPU backend is real; a fabricated telemetry
strip is worse than none.

A demo that tries to show eight things shows none of them.

---

## 13. Runtime

**Streaming vs. document memory is a real tension.** Repeated-chrome detection
needs every page before judging any page; a listener needs page 1 within seconds.

Resolve with a **warm-up window**: process 5 pages to collect candidate furniture,
begin speaking, refine the repeated set as later pages arrive. A string that only
reveals itself as furniture on page 40 is suppressed from page 40 on; earlier
pages already spoke it. That is acceptable — mild over-reading beats a silent
two-minute wait, and it fails in the safe direction.

**Concurrency:** page analysis is CPU-bound and parallel — `ProcessPoolExecutor`,
2–4 workers, with an ordering queue after it because workers finish out of order
and speech must not. OCR and the VLM are **not** parallelised across pages; the
NPU is one device and queueing two models against it causes contention.

**Errors: degrade, never crash, never go silent without saying so.** Encrypted
PDF → say so, stop cleanly. Blank page → "page N, blank". OCR returns nothing on
a page with ink → "page N, could not be read". VLM timeout → caption, then
"figure, not described". Library code raises typed errors; only the CLI catches
broadly.

---

## 14. Done

| Step | Done when |
| --- | --- |
| 0 | Six assertions pass |
| 1 | Debug render shows correct classes on the fixture |
| 2 | `docs/measurements.md` has real ms and NPU layer counts |
| 3 | A real PDF produces correct utterances with OCR disabled |
| 4 | §8 fixes in, §0 gate still green |
| 5 | A document plays end to end with no manual steps |
| 5b | Whole pipeline runs inside `assert_offline()` with zero attempts |
| 6 | `grep` finds no literal thresholds outside `types.py` |
| 7 | 100 pages labelled, agreement reported on a 20-page overlap |
| 8 | Recall and precision printed on the 20-page holdout |
| 4b | Injected faults caught; two-page doc refuses to repair |
| 9 | A figure produces a description that is not its caption |
| 10 | README opens with the A/B, limits stated before a reader finds them |

The six assertions, which are the contract:

```
test_chrome_suppressed              header, page number, watermark all silent
test_title_not_suppressed           the dangerous case — see 8a and 8b
test_caption_becomes_figure_context no caption emitted alone
test_equation_spoken_as_words       h = O(log n) -> "h equals big O of log n"
test_margin_note_not_mid_sentence   announced separately, never spliced
test_nothing_silently_dropped       every suppression recoverable, with a reason
```

---

## 15. Stop and ask

Do not guess on these. Ask.

- The AI Hub latency number comes back over 3 s. The architecture changes; confirm
  before building the precompute path.
- A benchmark bucket cannot be collected (no handwritten pages available). The
  evaluation claim narrows and that should be a decision, not a silent omission.
- A threshold change improves recall while lowering content precision. That is a
  regression regardless of the headline number; surface it.
- Table handling turns out to need its own segmentation pass. Scope decision.

---

## 16. Every bug becomes a test

All four bugs in §8 were one-line assertions after the fact. When you find a fifth
on a real page: add the page to `tests/corpus/`, label it, assert the specific
behaviour. That is what stops it coming back, and it is the difference between a
system that improves and one that oscillates.
