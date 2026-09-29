# Winnow

Every text-to-speech tool can read. None knows what is worth reading.

Same page of `tests/corpus/lecture.pdf`, read two ways:

**Naive read-aloud** (the text layer, top to bottom — what Edge does):

> Balanced Binary Search Trees Balanced Binary Search Trees 2026-09-19 A binary search
> tree keeps its keys in sorted order, so lookup, insertion and deletion can skip half of
> the remaining tree at every step. That promise only **ask about AVL** holds while the tree
> stays balanced. […] h = O(log n) **8 4 12 2 6 10 14** Fig 3. A balanced binary search
> tree of height 2 **Page 1**

**Winnow:**

> Balanced Binary Search Trees
> A binary search tree keeps its keys in sorted order, so lookup, insertion and deletion
> can skip half of the remaining tree at every step. That promise only holds while the
> tree stays balanced. […]
> Margin note: ask about AVL
> h equals big O of log n
> Figure 3: The figure is a binary tree diagram showing a balanced binary search tree with
> a height of 2. The root node is 8, which has two children: 4 and 12. The node 4 has two
> children: 2 and 6. The node 12 has two children: 10 and 14. […]
>
> *Skipped: "Balanced Binary Search Trees" — repeats at this position on 3 of 3 pages ·
> "2026-09-19" — date stamp · "Page 1" — repeats at this position on 3 of 3 pages*

The naive reader says the tree's labels, "8 4 12 2 6 10 14", and moves on. Winnow
describes the figure itself, using the caption as context, never reading the caption
alone. The description comes from Qwen3-VL-2B running on this machine; with no model
it falls back audibly to "Figure 3: A balanced binary search tree of height 2. Figure,
not described." On page 2 the naive reader says "x2 + y2 = r2"; Winnow says "x squared
plus y squared equals r squared". Reproduce with
`python -m winnow compare tests/corpus/lecture.pdf --describer llamacpp`.

## Limits — read these first

- **Speed on this laptop** (i5-12500H, Iris Xe, no NPU), measured:
  a text PDF is ready to speak in 0.5 s and a scanned page in 2.0 s. Figures are
  described behind the reader, 3.9-4.7 s each (image encoder on the Iris Xe, language
  model on the CPU). Reading waits only if it reaches a figure still being described.
- **The NPU, measured on Qualcomm AI Hub** (a hosted Snapdragon X Elite, not this
  laptop): EasyOCR's recogniser takes 19.1 ms per text line with all 3,121 layers on
  the NPU, against 140 ms on this CPU, and reads a real line identically. Winnow
  cannot use it here: this laptop has no NPU. The describer's language model was not
  run on the NPU by us; Qualcomm's published profile gives 3.1-6.6 s per figure there,
  no faster than this laptop, because writing the description one token at a time
  dominates. "Ask a figure a question" (§12 upgrade 1) is not built: the spec gates it
  on latency under 3 s.
- **No independent benchmark yet.** The labels in `tests/corpus/` were bootstrapped from
  Winnow's own output and checked by eye, so scoring against them is a regression check.
  Winnow has also been run on five real documents (four arXiv papers and one long page
  of notes), and every bug found is now a test. But the figure and table results below
  are measured against my own labels (46 graphics clusters, 21 captioned tables), not a
  second person's. Steps 7, 8 and 8b need people. The tools are ready: a planner that
  draws the holdout, a labelling window with one keypress per box, scoring, agreement
  and the tuner. The protocol is in `docs/benchmark.md`.
- **Tables are read only from a PDF's text layer, and only with a caption** ("Table N",
  or "Figure N"). Tables in scans are read row by row as plain text. Columns closer than
  16 px merge: a 20-column table, and Transformer Table 3's eight tight numeric columns,
  come out with several values in one cell. A symbol the PDF's font does not map (the
  "×" in "13.3 × 10⁶" in one paper) comes out as "�".
- **OCR is slow on long pages.** EasyOCR on CPU reads the 13,445 px `Combinatorics.pdf`
  page in 89 s (it took about 2 minutes before the switch to full precision). A normal
  scanned page takes about 2 s. On that page's handwriting-style font, full precision
  misreads about 5 words in 890 more than 8-bit ("progranming"); on printed scans it
  is as good (253 vs 252 of 263 words). On scans it loses superscripts: `x² + y² = r²`
  comes back as "x2 + Y2 =".
- Once, straight after a module split, the test suite took 133 s and one test failed.
  It has not recurred in six full runs since, and I could not find the cause.

What to do next, in order, is in `docs/next-steps.md`.

## Run it

```
pip install -r requirements.txt
python -m winnow fetch-models                      # once: EasyOCR weights (~100 MB)
python -m winnow ui tests/corpus/lecture.pdf       # the window; or drag a PDF onto it
python -m winnow read tests/corpus/lecture.pdf     # transcript (+ --speak, --tier 0|1|2)
python -m winnow read tests/corpus/scan_p1.png     # a scan: OCR, no text layer
python -m winnow render tests/corpus/lecture.pdf out/   # debug render of every page
python -m winnow read DOC --no-repair --json       # raw output, reproducible exactly
python -m winnow label DOC OUT.labels.json --labeller NAME   # a label file to fill in
python -m winnow annotate OUT.labels.json          # fill it in: one keypress per box
python -m winnow eval tests/corpus/*.labels.json   # content precision, then chrome recall
python -m pytest tests                             # 110 tests
python tools/ocr_compare.py DOC TRUTH.txt --box X0 Y0 X1 Y1   # word recall per OCR engine
python tools/prompt_check.py                       # describe prompts: length vs noise
python tools/bench_plan.py FOLDER                  # benchmark: draw the holdout, list commands
python tools/aihub_ocr.py                          # OCR recogniser on a Snapdragon X Elite NPU (AI Hub)
```

Without EasyOCR's weights, scans fall back to Windows' built-in OCR, which reports no
confidence (`conf = -1`) and read 62% of the words on the handwriting-style page that
EasyOCR read at 97%.

**Figure descriptions** need a local vision model. With llama.cpp (a release from
github.com/ggml-org/llama.cpp) and Qwen's GGUF files
(huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF: `Qwen3VL-2B-Instruct-Q4_K_M.gguf` and
`mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`, 1.55 GB together). Use the **Vulkan** build
with `-ngl 0`: the image encoder then runs on the integrated GPU and the language
model on the CPU, 25-30% faster per figure than the CPU build (all layers on the
Iris Xe was no faster: it writes more slowly):

```
llama-server -m Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf --host 127.0.0.1 --port 8080 -c 4096 -ngl 0
python -m winnow read tests/corpus/lecture.pdf --describer llamacpp
python tools/describe_bench.py tests/corpus/lecture.pdf --backend llamacpp   # tokens, timing
```

`--describer ollama:qwen3-vl:2b` also works with Ollama. Importing these GGUF files into
Ollama is reported to crash on the first image, so use its own `ollama pull qwen3-vl:2b`.
Both servers listen on loopback only, so the offline guard still holds.

Every document run happens inside `telemetry.assert_offline()`. Any socket to a
non-loopback address raises and is counted, and the window's **net 0** reads that
counter. Speech is Windows SAPI with local voices.

## The window

![](docs/ui.png)

Furniture is greyed out on the page. The region being spoken is outlined and scrolled
into view. Hovering any region shows the rule and the numbers it was decided on
(`chrome — repeats at this position on 3 of 3 pages · size_ratio 0.82 · band top ·
ink_delta +89`). **Skipped (n)** lists every suppression with its reason. The tier
slider (announce → summarise → full) filters the existing plan without re-analysing
anything. **compare** plays the naive reading of the same file. The status line
carries the self-check badge and the config hash.

## How it decides

```
PDF text layer ─┐                                   ┌─ figures only ─> describer ─┐
                ├─> regions ─> measure ─> classify ─┤                             ├─> plan ─> TTS
scan ─> segment ┘   (boxes,    (line h,   (+ doc    └─> reconcile ─> order ─> bind┘   (pure)
         + OCR       nature)    ink)      memory)       (tier 2)
```

| module | stage |
| --- | --- |
| `types.py` | `Region`, `Utterance`, frozen `Config` (every threshold), typed errors |
| `pdf_adapter.py` | text layer → regions; sideways text; column gutters |
| `pdf_figures.py` | figures (sparse clusters, or ones with a plotted line), their labels and panels; ruled, captioned tables |
| `tables.py` | **pure**: rows, columns, headers and row groups from a table's words; header-qualified speech |
| `segment.py` | pixels → boxes; work scale, polarity, small-text zoom; line count (8a); ink |
| `lines.py` | **pure**: words → lines, geometrically (8e); never across a column gutter |
| `ocr.py` | OCR stage: line boxes, strips for tall pages, one bounded re-read (tier 3) |
| `chrome.py` | the furniture rules in priority order (§6), document memory (8b), margin bands |
| `classify.py` | measures, runs the chrome rules, then kinds; `meta["why"]` and `meta["evidence"]` |
| `reading_order.py` | columns, spanning blocks, margin notes slotted between paragraphs |
| `caption_binding.py` | caption → figure or table, one each, like with like first |
| `describe.py` | the VLM branch; `UNCLEAR`, parroted captions, timeouts → fallback |
| `speech.py` | **pure** planner; one plan, three tiers |
| `selfcheck.py` | tier 1 invariants, tier 2 cross-page reconciliation and gated repair |
| `pipeline.py` | orchestration, process pool with in-order results, warm-up streaming |
| `backends.py` | SAPI, describers (llama.cpp, Ollama); re-exports `ocr_backends` (EasyOCR, Windows OCR) — imported only by `pipeline` and `cli` |
| `evaluate.py` | precision/recall pair, Cohen's kappa, tier 4 tuner with holdout gate |
| `label_ui.py` | the labelling window for the benchmark (`winnow annotate`) |

Things that are enforced rather than hoped for (each has a test):

- **The title is never suppressed.** Three mechanisms protect it. Size is compared on
  line height, not block height (8a). The repeat key includes position (8b). The
  largest text on a page can't be suppressed by a small, faint, date or page-number
  rule, and tier 2 won't repair it toward chrome (`title_suppressed` is a tier 1 error).
- **Nothing is silently dropped.** Chrome is kept at tier 0 with its reason. Unreadable
  ink is announced ("Text here could not be read."). Blank, unreadable and
  furniture-only pages say so.
- **Repair is earned.** It needs 3+ pages and a 2/3 majority, never happens on a tie,
  is decided from a snapshot and applied in a fixed order, and is logged with its vote
  count. `--no-repair` reproduces the raw output exactly. A two-page document refuses
  to repair.
- **No literal thresholds** in decision modules. An AST test checks this.

## Bugs found while building (all now tests)

- Faint grey furniture disappeared from scans entirely, because page-wide Otsu filed
  it under background. Anything clearly darker than the paper now counts as ink.
- "Red-black trees" was read as an equation, because the intra-word hyphen counted as
  an operator.
- Short footers outvoted paragraphs when computing the body ink/size reference. That
  reference is now a character-weighted median.
- A page with a figure and unreadable text spoke only the figure; the unread text
  vanished without a word.
- OCR at work resolution misread 3 words on `scan_p1.png`. At 2× it misread none
  (`ocr_scale`). Unread regions get one bounded re-read at 4× (tier 3).

Found by the first real model and the first real document:

- The describe prompt ("...if you cannot read the figure with confidence, reply with
  exactly UNCLEAR") made Qwen3-VL-2B answer UNCLEAR for every figure, including a clear
  tree it describes perfectly when asked plainly. Prompt v3 names what counts as
  unreadable first.
- The UNCLEAR check only looked at the start of the reply. The model wrote "The image
  is UNCLEAR.", or described noise and ended with UNCLEAR, or said "not a clear or
  readable image" with no UNCLEAR at all. Each would have been read aloud as a
  description. Winnow now treats UNCLEAR anywhere, or an explicit admission, as unclear.
- On a LaTeX paper every sentence came out glued: "presentaresiduallearningframework".
  Words are placed with ~2 pt gaps and no space character, under pdfplumber's fixed
  3 pt tolerance. The gap is now relative to font size (142 glued words -> 2 URLs).
- A sideways arXiv stamp in the margin became about 20 "margin notes", read backwards
  ("5102 ceD"). Sideways text is now read in its own direction, and in the margin it is
  furniture: "arXiv:1512.03385v1 [cs.CV] 10 Dec 2015 — sideways text in the page margin".
- On a two-column page, one median over all blocks landed on the left column, so every
  narrow block in the right column became a "margin note".
- A centred title and an author line, one block each at odd positions, chained two
  columns into one, and the title was read after the whole left column. Columns now
  need two blocks sharing a left edge, and anything above the column body is read first.

Found on real documents on the second day (each rebuilt as a synthetic test):

- A long page of dark-mode notes (`Combinatorics.pdf`, 17 × 103 inches) was read as
  "Figure: No caption. Figure, not described." Light text on black made the background
  "ink"; scaled to a fixed height the page was 262 px wide; and Windows OCR refuses
  images over 10000 px. Pages are now inverted when dark, kept at least 1000 px wide,
  rendered larger when letters come out under 8 px, and read in strips.
- EasyOCR's own text detector took 6 minutes on that page. It now recognises Winnow's
  line boxes instead: 97% of words (from 93%) in about a sixth of the time. Line boxes
  widened by the dilation read a leading "~" or "'" on each line ("'Page 1" then
  escaped the page-number rule); they are now tight plus a 25% margin.
- "Top 10% of the page" was a whole screen on a long page, so content was skipped as
  margin. Bands are now fractions of an A4-proportioned page at most.
- Labels alone on their line ("Example", "Answer:") became margin notes; a note must
  now sit beside body text. List dashes were read as "minus".
- A code block's "}" and a drawn rule were each announced as "Text here could not be
  read." Lone marks and drawn lines are now skipped with a reason.
- Charts were not detected as figures, so tick labels were read out. A cluster with a
  diagonal stroke (a plotted line) and less text than body is a figure; a table has only
  horizontal and vertical rules. On 46 real clusters: 23/23 figures, 0/23 false.
- A chart's invisible white background joined a page header, the chart and its caption.
- Panels over one caption were two figures, one uncaptioned; a caption inside a
  graphics cluster was swallowed as labels; a figure label on the caption's line ("F
  Figure 2.") hid the caption; two captions bound to one chart and the table's caption
  was never spoken. On the four papers, all 18 detected figures now carry their caption.
- One paper's 20 px column gutter was under the 25 px word-joining gap, so its two
  columns were read interleaved, row by row. Lines no longer join across a gutter.
- "ResNet-152" was read "ResNet minus 152".
- A scripted edit twice turned a regex's `\b` into a backspace character; the pattern
  matched nothing and every test still passed. A test now rejects control characters.

Found on the third day:

- The Batch Normalization paper captions its tables "Figure N" and draws every table
  rule as a 0.48 pt embedded image, so its three tables had no rules and were read as
  streams of numbers. Hairline images now count as rules, and a ruled grid under any
  caption is a table. Boxed prose under a "Figure" caption is not a grid and stays text.
  On the four papers: 21 tables, from 18, and no new false ones.
- A group label centred on its rows ("(A)" on a line of its own, beside four rows)
  gave its first two rows to the group above and was read as an empty row. When rules
  separate the groups, every row between two rules now takes the one label there.
- The window said nothing until every figure in the document had been described,
  about 5 s each: a minute of silence for ten figures. It now shows and reads the
  document at once, and describes figures behind the reader.
- EasyOCR's default on CPU is an 8-bit recogniser; on this laptop it was twice as slow
  as the full-precision one (285 vs 140 ms per line). Winnow now uses full precision:
  as accurate on printed scans, about 5 words in 890 worse on a handwriting-style font.
- The figure timings reported earlier (2.0-4.3 s) were measured with the model server's
  cache warm from repeated runs. A figure the server has not seen took 5-7 s on the CPU.
