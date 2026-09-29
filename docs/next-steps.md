# Next steps

Where the build stopped on 2026-09-29, and what to do first next time.

## State

- All tests pass (`python -m pytest tests`). The two real-model tests run only while
  `llama-server` is up; otherwise they skip.
- Build steps done (prompt.md §7/§14): 0, 1, 2 (from Qualcomm's published profiles, not
  our own run), 3, 4, 4b, 5, 5b, 6, 9, 10. Steps 7, 8 and 8b need people; their tools
  are built and tested (`docs/benchmark.md`).
- Beyond the steps: header-qualified tables (the owner chose this over "announce only"),
  including "Figure N"-captioned tables, rules drawn as images and row groups; chart
  detection; column gutters; long and dark pages; EasyOCR on Winnow's own line boxes.
  Each was found on a real document; numbers are in `docs/measurements.md`.
- Every module is under the ~250-line guideline (the largest, `pdf_figures.py`, is 248).
- Vision model: `C:\Users\ASUS\models\qwen3-vl-2b\` (checksums verified), served by
  `C:\Users\ASUS\models\llama.cpp\llama-server.exe`. The start command is in the README.
  EasyOCR weights are in `~/.EasyOCR/model`.

## To do, in order

1. **Benchmark (steps 7-8b).** Follow `docs/benchmark.md`: collect ~100 pages, draw the
   holdout with `tools/bench_plan.py`, label with `winnow annotate`, have a second person
   label the 20 holdout pages, then `eval --agree`, `tune`, and score the holdout once.
2. **Tables in scans** are read as plain rows. They need a pixel-level ruling detector
   (long thin runs of ink) feeding the same `tables.structure`.
3. **Tight table columns** (gaps under 16 px) merge. A gap relative to the table's own
   word spacing, rather than a fixed pixel width, is the likely fix; check it on the
   ResNet VOC tables and Transformer Table 3.
4. **OCR on the NPU in Winnow itself.** The recogniser is measured on the X Elite NPU
   (19 ms per line, identical text; `tools/aihub_ocr.py`). A Snapdragon machine is
   needed to wire it in: download the compiled model from the AI Hub compile job, and
   add an OCR backend running it through ONNX Runtime's QNN provider.
5. **The describer's language model on the NPU (step 2)**: exporting it needs
   Qualcomm's export recipe, the original ~4 GB weights and a lot of memory. Their
   published profile already shows it is no faster than this laptop per figure.
6. **Unmapped font glyphs** ("�" for a maths "×") could be recovered from the glyph's
   name in the font where the PDF gives one.

## Housekeeping

- The model server now running is the Vulkan build in `C:\Users\ASUS\models\llama.cpp-vulkan\`
  (`-ngl 0`, port 8080). Stop `llama-server.exe` (Task Manager) when not demoing; it holds ~2 GB.
- `tools/.env` holds the AI Hub token. It is in `.gitignore` and left out of the
  submission zip; never commit or share it.
- Real papers used for testing live only in the session scratchpad; they are not
  redistributable, so each bug they exposed is rebuilt as a synthetic test instead.
