# Benchmark protocol

This is the one part of the project that needs people. Every tool is built and tested;
what is missing is about 100 labelled pages and a second labeller for 20 of them.
Until this is done, every accuracy figure in this repository is measured against one
person's labels.

## 1. Collect about 100 pages

Put the documents in one folder (for example `bench/docs/`). Aim for a spread like this,
and write down where each document came from:

| bucket | pages | examples |
| --- | --- | --- |
| papers and articles, text layer | 30 | two-column papers, reports, textbook chapters |
| slides exported to PDF | 20 | lecture decks with running headers and page numbers |
| scans and phone photos | 20 | CamScanner or Adobe Scan output, photocopies, watermarks |
| notes, including handwriting | 15 | typed notes, handwritten pages, dark-mode screenshots |
| anything else people read | 15 | forms, handouts, pages with margin notes |

If a bucket cannot be filled (no handwritten pages, say), stop and decide whether
the claim narrows, and say so in the results. Do not quietly leave it out.

Keep documents whose licence allows sharing if the benchmark is to be published.
Otherwise keep only the label files and a list of sources.

## 2. Draw the holdout before anyone looks

```
python tools/bench_plan.py bench/docs --overlap-pages 20 --seed 0 > bench/plan.txt
```

This draws whole documents at random until they hold 20 pages. Both people label
those. They give the agreement figure and they are the holdout: never used for
tuning, scored once at the end. `plan.txt` lists every command below, with paths
filled in.

## 3. Label

For each document: `python -m winnow label DOC OUT.labels.json --labeller NAME` writes a
file of Winnow's boxes with every kind left blank. Winnow's guesses are not shown, so
nobody is nudged towards the classifier's answer. Then:

```
python -m winnow annotate OUT.labels.json
```

The page appears with every box drawn; the current one is thick. Press a number to
set its kind, and the window moves on to the next box without one. Every change is
saved at once.

| key | kind | means |
| --- | --- | --- |
| 1 | `chrome` | furniture a listener does not want: running headers and footers, page numbers, watermarks ("Scanned by CamScanner"), date stamps, handles, arXiv side stamps, drawn lines, stray marks |
| 2 | `heading` | titles and section headings, including a title that repeats the running header's words |
| 3 | `body` | ordinary text, including list items, references and footnotes |
| 4 | `caption` | "Figure 3. …", "Table 2: …", the caption only |
| 5 | `figure` | a chart, diagram, photo or drawing, with its own labels and ticks |
| 6 | `equation` | displayed maths; a date is never an equation |
| 7 | `annotation` | a margin note beside the body text |
| 8 | `table` | a whole table: the grid, its header and its cells |

Other keys: → and ← move between boxes, PgDn and PgUp between pages. Delete removes
a box that is not a region at all (noise). **Drag** on the page to add a region Winnow
missed. Missed content is what "content lost" counts, so do not skip this.

When unsure, choose what a listener would want: would they want to hear it? Content
that is hard to classify is still content, never chrome.

## 4. Score

```
python -m winnow eval A.labels.json --agree A.labels.json B.labels.json   # per overlap doc
python -m winnow tune --tuning <tuning files> --holdout <holdout files>
python -m winnow eval <holdout files>                                      # the reported line
```

Report, in `docs/measurements.md`:

1. Cohen's kappa on the overlap. If it is low, fix the label guide above and relabel.
   A low kappa means the task is ill-defined, not that Winnow is wrong.
2. The holdout line: content precision first, then chrome recall, content lost and
   kind accuracy. Precision is never quoted alone.
3. What `tune` proposed and whether it was promoted. A change that buys recall with
   precision is shown, never taken. The holdout is scored once. Running it again
   after each change turns it into a second tuning split.
