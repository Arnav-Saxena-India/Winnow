# Design

The rules Winnow is built on. Code comments refer to this page: "invariant 2" and
"8a" below are the names used there.

## Two behaviours define it

- **Substitution.** A caption is never spoken alone. "Fig 3." carries no information:
  what the listener needs is the figure's description, and the caption becomes context
  for producing it.
- **Announce, never silently skip.** Suppressed content stays reachable, with a stated
  reason. A reader that removes content without a record is deciding what the listener
  is allowed to know.

Getting characters off a page is the easy half. Deciding what each region *is* is the
product.

## Invariants

Each has a test (`tests/test_invariants.py` and others).

1. The speech planner is pure: same regions in, same utterances out.
2. Only `pipeline` and `cli` import `backends`. Backends (voice, OCR, vision model) are
   passed in, never constructed inside the logic.
3. Classification never moves boxes; reading order never changes kinds.
4. Every region classified as `chrome` (furniture) carries a reason, `meta["why"]`.
5. A caption bound to a figure is never spoken as its own utterance.
6. Suppressed content is kept, at tier 0, with its reason. Nothing is deleted.
7. A repaired run is reproducible: `--no-repair` gives the raw output exactly, and every
   repair is logged with its reason and vote count.
8. **The title is never suppressed.** A false suppression is invisible to a listener,
   while a false read is merely annoying, so the two failures are not treated alike.

`nature` (physical: text pixels or drawing pixels) and `kind` (semantic: heading,
caption, furniture) are separate fields and stay separate: sparse reclassification
changes `nature` after OCR, before any `kind` exists.

## The classes

| kind | signals | spoken as |
| --- | --- | --- |
| `chrome` | repeats at the same position across pages; watermark words; page number in a margin band; date stamps and handles; small or faint text in a band | skipped, with its reason |
| `heading` | taller lines than the body, or darker; short; no final full stop | "heading" at tier 0, the text above that |
| `body` | the default | verbatim |
| `caption` | "Fig", "Figure", "Table" or "Chart" and a number | never alone: context for its figure or table |
| `figure` | a drawing, chart or image | a description from the vision model, after its label |
| `equation` | operator symbols, short, few long words, not a date | spoken as words |
| `annotation` | narrow, outside the text column, beside body text | "Margin note: …", between paragraphs |
| `table` | ruled rows under a caption | each row named by its first cell, each value by its header |

Chrome rules run in priority order and the first match wins: repeated at the same
position, watermark, sideways text in the margin, page number in a band, date or
handle anywhere, small or faint in a band.

## Known failure modes

Each was found, fixed, and is now a test.

- **8a: the title suppressed as "small".** Size was measured against the median
  *block* height; a three-line paragraph is ~90 px and a one-line 26 pt title ~36 px,
  so every one-line region looked small. Size is now compared on *line* height: block
  height divided by the line count from a horizontal projection.
- **8b: the title suppressed as a repeated header.** A title that repeats the running
  header's words matched it by text alone. Position is part of the repeat key: furniture
  repeats at the same place on every page; a title echoing it does not.
- **8c: a date read as arithmetic.** "2026-09-19" became "2026 minus 09 minus 19".
  Date-like strings are never equations, and the date rule needs no margin band, since a
  byline often sits below the top 10%.
- **8d: a header read as a figure.** A running header joined to its underline formed a
  wide, short, sparse block that looked like a drawing. A strip under 45 px high is
  never a drawing.
- **8e: OCR lines taken from the engine.** Engines' line numbers are unreliable. Lines
  are grouped geometrically, by vertical overlap: geometry is the one signal every
  engine agrees on.

## Never

- Use min/max for the text column's edges: one page-wide footer stretches it to the
  whole page and margin notes vanish. Medians over wide blocks, per column.
- Filter OCR output by confidence: it silently drops real content. Unreadable text is
  reported, never dropped.
- Skip anything silently. A page that produces nothing says so ("page 4, could not be
  read"): a listener cannot tell blank from broken.
- Let the vision model invent. An `UNCLEAR` answer, an admission, or a parroted
  caption falls back to the caption, audibly. A made-up chart description is worse
  than none, because the listener cannot detect it.
- Cache descriptions on the image alone: the key is image hash, model and prompt
  version, or a prompt change silently serves stale descriptions.
- Tune against the holdout, repair on a tie, or report chrome recall without content
  precision (suppressing everything scores perfect recall).

## Self-correction

At run time there are no labels, so Winnow cannot know it is right. It can know it is
*inconsistent*, and inconsistency is a good proxy for error.

- **Tier 1: invariant checks on every page.** A reason missing from a suppression, a
  caption spoken alone, more than 60% of text suppressed, a page with ink that says
  nothing: each is an error and fails the run.
- **Tier 2: the document checks itself.** The same text at the same position classified
  differently on different pages is a contradiction found without ground truth. It is
  repaired only with at least three pages and a two-thirds majority, never on a tie, and
  never toward suppression without a clear majority. An early version repaired a
  one-against-one tie and un-suppressed a correct header: the self-correction amplified
  the fault. Below the bar, the conflict is reported and nothing changes.
- **Tier 3: spend compute where unsure.** An unreadable region gets one re-read at
  higher resolution; bounded, never a retry loop.
- **Tier 4: offline threshold tuning.** Coordinate descent over a few thresholds on a
  labelled tuning split; a change is promoted only if the untouched holdout does not
  regress, and a change that buys recall with precision is shown, never taken.

The traps an automatic tuner falls into, each guarded by a test: suppress everything
(perfect recall), suppress nothing (perfect precision), never bind captions (no
"caption spoken alone" errors), one region per page (no order inversions), and uniform
classification (perfect cross-page agreement).
