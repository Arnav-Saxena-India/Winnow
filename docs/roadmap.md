# Roadmap

What is done, and what comes next, in order.

## Done

- Furniture detection with a reason for every skip; the title protected by three
  separate mechanisms; cross-page self-checks with evidence-gated repair.
- Text-layer PDFs (including two-column papers, sideways margin stamps and tight LaTeX
  spacing) and scans or photos through OCR, including very long and dark-mode pages.
- Figures detected (charts, diagrams, panels over one caption) and described by a local
  vision model, with an `UNCLEAR` fallback tested against noise images.
- Tables read row by row with their headers, spanning headers and row groups.
- Equations spoken as words; margin notes announced between paragraphs.
- The window: reading starts at once, figures are described behind the reader, every
  skip is listed with its reason, and a network counter proves it stays offline.
- Benchmark tooling: holdout planner, labelling window, scoring, agreement and a gated
  threshold tuner (`docs/benchmark.md`).

## Next

1. **Benchmark.** About 100 labelled pages, a second labeller on 20 of them, then
   agreement, tuning and a single holdout score. Until then, accuracy figures are
   measured against one person's labels.
2. **Tables in scans.** Tables are read from a PDF's text layer only. Scans need a
   pixel-level ruling detector feeding the same table reader.
3. **Tight table columns.** Columns closer than 16 px merge. A gap relative to the
   table's own word spacing is the likely fix.
4. **OCR on the NPU.** The recogniser is measured on a Snapdragon X Elite NPU at 19 ms
   per line with identical output (`tools/aihub_ocr.py`). Wiring it in needs a
   Snapdragon machine: the compiled model run through ONNX Runtime's QNN provider.
5. **The describer's language model on the NPU.** Needs Qualcomm's export recipe and
   the original weights. Qualcomm's published profile shows it no faster than a laptop
   CPU per figure, so shorter descriptions matter more than the device.
6. **Unmapped font glyphs** ("�" where a PDF's font gives no Unicode for "×") could be
   recovered from the glyph's name where the font provides one.
