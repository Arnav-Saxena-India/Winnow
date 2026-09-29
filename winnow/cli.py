"""Command line. The only place that catches broadly, and — with pipeline —
the only place that constructs backends.

  python -m winnow read DOC [--tier 0|1|2] [--no-repair] [--speak] [--json]
  python -m winnow compare DOC          naive read-aloud vs Winnow, side by side
  python -m winnow render DOC OUTDIR    debug render of every page
  python -m winnow ui [DOC]             the prototype window
  python -m winnow label DOC OUT --labeller NAME   label file to fill in (kinds blank)
  python -m winnow eval LABELS...       content precision + chrome recall
  python -m winnow tune --tuning A --holdout B    tier 4, offline only
  python -m winnow fetch-models         one-time EasyOCR weight download (needs network)

Every document run happens inside telemetry.assert_offline().
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import asdict

from winnow import backends, pipeline, telemetry
from winnow.types import DEFAULT, WinnowError


def _describer(spec: str | None):
    if not spec:
        return backends.NullDescriber()
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        return backends.OllamaDescriber(model or backends.DEFAULT_VLM)
    if kind == "llamacpp":      # llama-server on loopback; ``model`` is its host:port
        return backends.LlamaCppDescriber(host=f"http://{model or '127.0.0.1:8080'}")
    raise WinnowError(f"unknown describer {spec!r}; try ollama:<model> or llamacpp")


def _run(args):
    with telemetry.assert_offline():
        return pipeline.run(args.doc, DEFAULT, describer=_describer(args.describer),
                            ocr=backends.make_ocr(), repair=not args.no_repair)


def _print_findings(doc) -> None:
    for f in doc.all_findings():
        print(f"  [{f.severity}] tier {f.tier} p{f.page} {f.check}: {f.detail}", file=sys.stderr)
    print(f"  self-check: {doc.badge()}   config {doc.config_hash}   "
          f"network attempts {telemetry.attempts()}", file=sys.stderr)


def cmd_read(args) -> int:
    doc = _run(args)
    if args.json:
        print(json.dumps({
            "config": doc.config_hash, "repair": not args.no_repair,
            "repairs": [asdict(r) for r in doc.repairs],
            "findings": [asdict(f) for f in doc.all_findings()],
            "utterances": [dict(asdict(u), page=p.page_no) for p, u in doc.transcript(args.tier)],
        }, indent=1, ensure_ascii=False))
        return 0 if doc.ok else 2
    _print_findings(doc)
    if not doc.ok and not args.allow_errors:
        print("self-check errors: output withheld (use --allow-errors to see it)", file=sys.stderr)
        return 2
    speaker = backends.make_speaker() if args.speak else None
    page = None
    for p, u in doc.transcript(args.tier):
        if p.page_no != page:
            page = p.page_no
            print(f"\n— page {page} —")
        if u.suppressed:
            print(f"   (skipped: {u.text[:50]!r} — {u.why})")
            continue
        print(textwrap.fill(u.text, 88, initial_indent="  ", subsequent_indent="  "))
        if speaker:
            speaker.say(u.text)
            speaker.wait()
    return 0


def cmd_compare(args) -> int:
    doc = _run(args)
    for p in doc.pages:
        print(f"\n=== page {p.page_no} — naive read-aloud ===")
        print(textwrap.fill(" ".join(p.naive.split()), 88))
        print(f"=== page {p.page_no} — Winnow ===")
        for u in pipeline.at_tier(p.utterances, 2):
            if not u.suppressed:
                print(textwrap.fill(u.text, 88))
        skipped = [u for u in p.utterances if u.suppressed]
        if skipped:
            print("  skipped: " + "; ".join(f"{u.text[:30]!r} ({u.why})" for u in skipped))
    return 0 if doc.ok else 2


def cmd_render(args) -> int:
    from winnow.debug_render import render
    doc = _run(args)
    os.makedirs(args.outdir, exist_ok=True)
    for p in doc.pages:
        out = os.path.join(args.outdir, f"page{p.page_no:03d}.png")
        render(p.gray, p.regions, out)
        print(out)
    _print_findings(doc)
    return 0


def cmd_ui(args) -> int:
    from winnow import ui
    return ui.main(args.doc, speaker=backends.make_speaker(), ocr=backends.make_ocr(),
                   describer=_describer(args.describer))


def cmd_label(args) -> int:
    from winnow import evaluate
    if os.path.exists(args.out):
        raise WinnowError(f"{args.out} exists; not overwriting someone's labels")
    args.no_repair, args.describer = True, None
    doc = _run(args)
    ref = os.path.relpath(os.path.abspath(args.doc), os.path.dirname(os.path.abspath(args.out)))
    lab = evaluate.template(doc, ref.replace(os.sep, "/"), args.labeller)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(lab, f, indent=1, ensure_ascii=False)
    n = sum(len(v) for v in lab["pages"].values())
    print(f"{args.out}: {n} boxes on {len(lab['pages'])} pages, every kind blank. Fill each "
          f"with one of {', '.join(evaluate.KINDS)}: `winnow annotate {args.out}`.")
    return 0


def cmd_annotate(args) -> int:
    from winnow import label_ui
    return label_ui.main(args.labels)


def cmd_eval(args) -> int:
    from winnow import evaluate
    labs = [evaluate.load(p) for p in args.labels]
    with telemetry.assert_offline():
        s = evaluate.evaluate(labs, ocr=backends.make_ocr())
    print(f"{len(labs)} label files, config {DEFAULT.hash()}")
    print(s.line())
    if args.agree:
        k, n = evaluate.agreement(evaluate.load(args.agree[0]), evaluate.load(args.agree[1]))
        print(f"inter-labeller agreement: Cohen's kappa {k:.3f} over {n} shared boxes")
    return 0


def cmd_tune(args) -> int:
    from winnow import evaluate
    with telemetry.assert_offline():
        cfg, log = evaluate.tune([evaluate.load(p) for p in args.tuning],
                                 [evaluate.load(p) for p in args.holdout],
                                 ocr=backends.make_ocr())
    print(chr(10).join(log))
    changed = {k: v for k, v in vars(cfg).items() if v != getattr(DEFAULT, k)}
    print(f"proposed Config changes: {changed or 'none'}")
    return 0


def cmd_fetch(args) -> int:
    print("Downloading EasyOCR weights (one time, needs network)…")
    backends.EasyOcr(download=True)
    print("done. For figure descriptions: `ollama pull qwen3-vl:2b`, then "
          "--describer ollama:qwen3-vl:2b")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="winnow", description="Decides what is worth reading.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("read", cmd_read), ("compare", cmd_compare), ("render", cmd_render),
                     ("ui", cmd_ui)):
        sp = sub.add_parser(name)
        sp.add_argument("doc", nargs="?" if name == "ui" else None)
        if name == "render":
            sp.add_argument("outdir")
        sp.add_argument("--tier", type=int, default=2, choices=(0, 1, 2))
        sp.add_argument("--no-repair", action="store_true",
                        help="reproduce the raw, unrepaired output exactly")
        sp.add_argument("--describer", help="ollama:qwen3-vl:2b or llamacpp[:host:port] (default: none)")
        sp.add_argument("--speak", action="store_true")
        sp.add_argument("--json", action="store_true")
        sp.add_argument("--allow-errors", action="store_true")
        sp.set_defaults(fn=fn)
    sub.add_parser("fetch-models").set_defaults(fn=cmd_fetch)
    la = sub.add_parser("label", help="write a label file for a person to fill in")
    la.add_argument("doc")
    la.add_argument("out")
    la.add_argument("--labeller", required=True)
    la.set_defaults(fn=cmd_label)
    an = sub.add_parser("annotate", help="fill in a label file: one keypress per box")
    an.add_argument("labels")
    an.set_defaults(fn=cmd_annotate)
    ev = sub.add_parser("eval", help="score against labels: precision first, then recall")
    ev.add_argument("labels", nargs="+")
    ev.add_argument("--agree", nargs=2, metavar=("LABELS_A", "LABELS_B"))
    ev.set_defaults(fn=cmd_eval)
    tu = sub.add_parser("tune", help="tier 4: propose Config changes, gated on the holdout")
    tu.add_argument("--tuning", nargs="+", required=True)
    tu.add_argument("--holdout", nargs="+", required=True)
    tu.set_defaults(fn=cmd_tune)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        return args.fn(args)
    except WinnowError as e:
        print(f"winnow: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # the one broad catch in the codebase
        print(f"winnow: internal error: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
