# Measurements

Only numbers that were actually measured go here. Where the measurement is someone
else's, or a number is derived from measurements, the section says so.

## Describer latency on Snapdragon X Elite

**Verdict: precompute at document open.** With token counts measured from the real
model and prompt v5, an average figure takes 3.1 s at best and 6.6 s typically; the
longest takes 4.5-9.8 s. The pipeline already works this way (`pipeline._finish` describes
a page's figures before planning its speech), so no architecture change was needed.

A description is not one forward pass: it is vision encoder + prefill + one decode step
per output token. Decoding is 85-95% of the total, which is why timing the encoder
alone (what the probe originally did) would have pointed at the wrong architecture.

**Source of the numbers.** These are Qualcomm's own AI Hub profiles on a hosted
Snapdragon X Elite CRD, published in `qai-hub-models` v0.63.0 (`perf.yaml`); Qualcomm
ran them, not us: our AI Hub account is configured and lists the device, but the model
weights to export could not be downloaded on this network. The job IDs are Qualcomm's.
The totals below are derived from those numbers, not measured end to end.

| component | model | runtime | NPU numbers on X Elite CRD |
| --- | --- | --- | --- |
| language model | Qwen3-VL-2B-Instruct, Q4_0 | GenieX llama.cpp | 512 ctx: prefill 964 tok/s, decode 26.4 tok/s |
| | | | 4096 ctx: prefill 799 tok/s, decode 11.8 tok/s |
| vision encoder (stand-in) | OpenAI-CLIP image encoder, w8a16 | ONNX | 25.2 ms, 1054/1054 layers on NPU (job `jgly7w725`) |
| | OpenAI-CLIP image encoder, float | ONNX | 32.6 ms, 997/997 layers on NPU (job `jgolmd3kg`) |

Qwen3-VL's LLM metrics exclude its vision encoder, so CLIP's encoder stands in for it.
CLIP's is smaller, but the encoder is under 1% of the total either way.

Token counts measured with the real model (Qwen3-VL-2B Q4_K_M via llama.cpp,
`tools/describe_bench.py`) on the four corpus figures, prompt v5: prompt (image + text)
206-223 tokens, mean 213; description 41-112 tokens, mean 74. Output varies a little
between runs even at temperature 0.

`python tools/aihub_probe.py --encoder-ms 33 --prompt-tokens 213 --output-tokens 74`:

| context | encoder | prefill | decode | total | verdict |
| --- | --- | --- | --- | --- | --- |
| 512 | 33 | 221 | 2803 | **3057 ms** | precompute |
| 4096 | 33 | 267 | 6271 | **6571 ms** | precompute |

The longest description (223 + 112 tokens) takes 4507 / 9804 ms. Prompt v3 averaged
87-93 output tokens (3550 / 7674 ms).

### Describe prompt (`tools/prompt_check.py`)

Each prompt on the 4 corpus figures and 9 noise/blank images (6 random-noise seeds,
blank, blurred noise, salt-and-pepper). A prompt only counts if every noise image
still falls back to UNCLEAR: a made-up description is worse than none.

| prompt | noise caught | figures described | mean output tokens |
| --- | --- | --- | --- |
| v3 "two or three plain sentences" | 9/9 | 4/4 | 93 |
| v4 "at most two short sentences" | 8/9 | 4/4 | 63 |
| **v5 "two sentences: kind, then what it shows"** | **9/9** | **4/4** | **74** |
| v6 "one sentence each" | 9/9 | 4/4 | 64 (one reply ran to 142) |
| v7 "be brief" | 9/9 | 3/4 | 92 |

v7's miss was Winnow's check, not the model: it wrote "The chart is not random noise
and is not unreadable", and the admission check matched "unreadable". Denials are now
removed before the check.

## CPU timings (this machine, Windows 11, Python 3.14)

| run | ms |
| --- | --- |
| `lecture.pdf`, 3 pages, text layer, serial (render + extract + classify + plan) | 338 |
| `scan_p1.png`, 1 page: segmentation + measurement | 129 |
| `scan_p1.png`, 1 page: Windows OCR at 2× (`ocr_scale`) | 180 |
| one figure description, Qwen3-VL-2B Q4_K_M, llama.cpp b11200 on CPU (i5-12500H), prompt v5, 4 figures, server cache warm (superseded below) | 1971 - 4287 |
| `Combinatorics.pdf`: 1 page, 2200 × 13445 px after zoom, EasyOCR on line boxes + describer | 125 000 |

## OCR resolution (why `ocr_scale = 2.0`)

Windows OCR on `tests/corpus/scan_p1.png` (page 1 rasterised with grey paper and
noise). Counted: words of 3+ letters that do not appear in the text layer.

| scale | words | wrong |
| --- | --- | --- |
| 1.0× | 103 | 3 (`Seamh`, `atter`, `baiow`) |
| 1.5× | 105 | 3 |
| 2.0× | 105 | 0 |
| 2.5× | 105 | 0 |

## Evaluation (tiny, see README limits)

| label files | content precision | chrome recall | content lost | kind accuracy |
| --- | --- | --- | --- | --- |
| `lecture.labels.json` (bootstrap, circular) | 1.000 | 1.000 | 0/15 | 25/25 |
| `scan_p{1,2,3}.labels.json` (OCR path vs text-layer labels) | 1.000 | 1.000 | 0/15 | 25/25 |

## First real document: a two-column paper (arXiv 1512.03385, 12 pages)

Not in `tests/corpus/` (its licence does not allow redistribution); each bug it
exposed is rebuilt as a synthetic test. Text layer, no OCR: 16 s for 12 pages.

**Word gaps (`word_gap_ratio`).** LaTeX places words with no space glyph and ~2 pt
gaps, under pdfplumber's fixed 3 pt tolerance. Pages 1-3, counting "words" of 20+
letters:

| extraction | words | glued (20+ letters) |
| --- | --- | --- |
| fixed 3 pt (before) | 1394 | 142 |
| fixed 2 pt | 2382 | 11 |
| 0.15 × char size (now) | 2454 | 2 (both real URLs) |

`lecture.pdf` is identical under every setting. Single-letter words rose 62 -> 103,
almost all the word "a" (17 -> 45) coming unglued, not words being split.

**Figure text density (superseded below).** chars per 1000 px² inside each vector
cluster, against `sparse_max_density` = 0.25:

| cluster | density | is a |
| --- | --- | --- |
| `lecture.pdf` tree | 0.10 | figure |
| `lecture.pdf` bar chart | 0.00 | figure |
| paper Fig. 5 (diagram) | 1.03 | figure |
| paper Fig. 1 (labelled line chart) | 1.37 | figure |
| paper Tables 3-5 (ruled) | 1.88 - 2.37 | table |
| paper Fig. 2 architecture columns | 2.20 - 3.78 | figure |
| body text | 3 - 8 | text |

The 0.25 cut was measured on scanned line drawings. Labelled charts in a real paper
sit at 1-4, overlapping tables and approaching body text, so density alone cannot
separate them. A cut near 1.5 would fix this paper's two charts and exclude its
tables, but that is one document; it was left for the benchmark, and
tables were handled separately (below).

## Charts, tables and diagrams on four real papers

arXiv 1512.03385 (ResNet), 1412.6980 (Adam), 1502.03167 (Batch Norm), 1706.03762
(Transformer) and `lecture.pdf`: every graphics cluster the figure detector considers
(46), labelled by eye as chart, diagram, table or framed algorithm.

| label | clusters | text density | diagonal strokes |
| --- | --- | --- | --- |
| chart | 10 | 0.48 - 1.70 | 1 - 269 |
| diagram / visualisation | 13 | 0 - 3.78 | 8+, or 0 at density 0 (embedded images, a bar chart of rectangles) |
| ruled table | 21 | 1.07 - 5.07 | **0 in every one** |
| framed algorithm | 2 | 1.31 - 2.13 | 0 |

Rule (`stroke_fig_max_density`): a figure has density under 0.25 (as before), or a
diagonal stroke and density under 4. Result: **23/23 charts and diagrams found, 0/23
tables or algorithms taken.** Also needed: white fills that never stroke are invisible
(one was a chart's background joining a page header, the chart and its caption), and
short labels just outside the plotted area (ticks, axis titles, panel titles) belong
to the figure. After caption fixes, all 18 detected figures carry their own caption.

## Column gutters

Batch Norm's column gutter is 10 pt (20 px at work scale), under the 25 px word-joining
gap (`dilate_h`), so its two columns were read interleaved. A gutter (`_gutters`) is a
vertical strip at least 8 px wide in the middle of the page that at most 2% of words
cross, with justified text ending and starting at its edges:

| page kind | words crossing | rows aligned each side |
| --- | --- | --- |
| Batch Norm, pp. 1-11 (real gutter) | 1 - 6 | 15 - 52 |
| ResNet, pp. 1-12 (real gutter, 44 px) | 0 - 17 | 9 - 54 |
| Adam p. 14, a page of proofs (no gutter) | 18 | 15 - 20 |

A gutter needs aligned rows at least twice the crossings: Batch Norm keeps 10/11 pages,
Adam's proofs none. ResNet loses a few pages whose wide tables cross it; its gutter is
wider than 25 px, so those pages never joined columns.

## Tables (header-qualified reading)

A table is two or more aligned horizontal rules (drawn edges, or hairline images of at
most `rule_image_max_pt`) with a caption above or below. Found: **ResNet 14/14,
Transformer 4/4, Batch Norm 3/3** (it captions its tables "Figure N" and draws their
rules as 0.48 pt images); Adam has none. Allowing "Figure N" captions added no false
tables on the four papers: a ruled block must still form a grid. Clean on the common
booktabs layouts, for example ResNet Table 3: "VGG-16 [41]: top-1 err. 28.07, top-5 err.
9.33. …" and Transformer Table 2 with spanning headers: "GNMT + RL [38]: BLEU EN-DE
24.6, BLEU EN-FR 39.92, …". Transformer Table 3's group labels ("(A)" centred on
four rows, groups separated by rules) now name every row of their group. Imperfect on
ResNet Table 1 (architecture, multi-line cells), the VOC per-class tables (20 tight
columns) and Transformer Table 3's eight numeric columns, whose gaps are under the
16 px `table_col_gap`, so "d model d ff h d k d v P drop ϵ ls" is one header cell.

## OCR engines on a real page (`tools/ocr_compare.py`)

`Combinatorics.pdf`: typed notes in a handwriting-style font, light on black, one page
of 17 × 103 inches. Truth: 15 lines typed by hand (`tests/corpus/combinatorics_excerpt.txt`).

| engine, as Winnow runs it | word recall | time for the excerpt |
| --- | --- | --- |
| Windows OCR, page at 2× | 43/69 (62%) | 0.2 s |
| EasyOCR, its own detector, 2× | 64/69 (93%) | 20.6 s |
| EasyOCR, its own detector, 1× | 63/69 | 7.8 s |
| **EasyOCR recogniser on Winnow's line boxes** | **67/69 (97%)** | **2.5 s** |

Whole page: EasyOCR's detector 356 s, line boxes 61 s. Line-box margin (word recall on
3 corpus scans + the excerpt, 332 words with exact truth): 0 → 84.6%, 0.1 → 93.4%,
**0.25 → 96.1%**, 0.35 → 94.6%, 0.5 → 88.9%; Windows OCR 91.3% (it is perfect on the
printed scans and weak on this font). Before line boxes, 24 lines started with a stray
"~" or "'", which broke the page-number and date rules.

Recogniser settings, whole page (2200 × 13445 px, 320 line boxes, 12 CPU threads);
excerpt recall was 67/69 under every setting:

| setting | whole page |
| --- | --- |
| default (`batch_size` 1, second pass on low-contrast lines) | 65.5 s |
| `batch_size` 8 | 62.4 s |
| `batch_size` 32 | 61.9 s |
| no low-contrast second pass | 61.0 s |
| both | 59.7 s |

At most 9%, and the first row includes warm-up, so the defaults stay. The cost is the
recogniser itself, about 0.19 s per line on this CPU, which is why the next step is
an NPU or DirectML build, not tuning.

Letter height at work scale: 11 px on every corpus page, 5 px on this page before
`text_zoom` (it re-renders so letters are `body_glyph_h` = 11 px).

## Speed on this laptop (2026-09-29)

i5-12500H (12 cores, 16 threads), Intel Iris Xe (8 GB shared), 15.7 GB RAM, no NPU.

**Time until reading starts** (`pipeline.run(..., describe=False)`, then
`describe_pages` behind the reader; EasyOCR full precision; real model):

| document | ready to speak | figures, described behind the reader |
| --- | --- | --- |
| `lecture.pdf`, 3 pages, text layer | 0.5 s | 2 figures, 7.1 s |
| `scan_p1.png`, 1 scanned page | 2.0 s | 1 figure, 4.4 s |

Before, the window spoke nothing until every figure was described.

**One figure description**, lecture figures (194 and 206 prompt tokens, 57-87 output
tokens), prompt cache off, two rounds each:

| llama.cpp b11200 build | reading image + prompt | writing | total |
| --- | --- | --- | --- |
| CPU | 2547-3410 ms | 2221-3934 ms (22-26 tok/s) | 5.5-6.2 s |
| Vulkan, all layers on Iris Xe (`-ngl 99`) | 1320-1538 ms (first call 8548) | 3637-4943 ms (12-18 tok/s) | 5.2-5.9 s |
| **Vulkan, `-ngl 0`: encoder on Iris Xe, LLM on CPU** | 1830-2336 ms | 1956-3232 ms (25-29 tok/s) | **3.9-4.7 s** |

**EasyOCR recogniser, one 64 × 1024 px line** (`tools/aihub_ocr.py`):

| where | ms per line |
| --- | --- |
| this CPU, EasyOCR's default 8-bit dynamic quantisation | 284.6 |
| this CPU, full precision (Winnow's setting now) | 140.4 |
| **Snapdragon X Elite CRD NPU, QNN (AI Hub profile job `jg9z7mowp`)** | **19.1**, 3,121/3,121 layers on NPU, 16 MB peak |

The NPU's output on a real corpus line matched the CPU's exactly (inference job
`jp8edjjkp`: same text, same character at every position, largest logit difference
0.19). Compile job `j5wl07y3p`. Uploaded to AI Hub: the recogniser's weights and that
one line; no user document. Full precision on the hard Combinatorics excerpt: 66/69
words (8-bit: 67/69), 1.5 s instead of 2.5 s.

8-bit against full precision, recogniser on Winnow's line boxes:

| text | 8-bit | full precision |
| --- | --- | --- |
| `scan_p1-3.png`, printed (truth: the text layer, 263 words) | 252 (95.8%), 8.8 s | 253 (96.2%), 4.7 s |
| `Combinatorics.pdf`, whole page | about 2 min | 89 s |

On the Combinatorics page full precision misread five words the 8-bit model read
("binomial", "combination", "programming", "remaining", "problem", each with m as n),
out of about 890.

