"""§7 step 9 / step 2: run the real describer on every figure of a document.

Prints, per figure: the caption, what the model said, what the listener hears,
the token counts Ollama reports, and this machine's wall time. The token counts
are what ``tools/aihub_probe.py --prompt-tokens/--output-tokens`` needs.

    python tools/describe_bench.py tests/corpus/lecture.pdf [--backend ollama|llamacpp]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from winnow import backends, pipeline, telemetry  # noqa: E402


class Recording:
    """Wraps a describer and keeps the stats of every call."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []
        self.name, self.device, self.model_id = inner.name, inner.device, inner.model_id

    def describe(self, crop, context):
        out = self.inner.describe(crop, context)
        self.calls.append(dict(self.inner.last_stats, raw=out, context=context,
                               shape=crop.shape))
        return out

    def ask(self, crop, question, context):
        return self.inner.ask(crop, question, context)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc")
    ap.add_argument("--backend", choices=("ollama", "llamacpp"), default="ollama")
    ap.add_argument("--model", default=backends.DEFAULT_VLM)
    a = ap.parse_args()
    cls = backends.LlamaCppDescriber if a.backend == "llamacpp" else backends.OllamaDescriber
    rec = Recording(cls(a.model))
    with telemetry.assert_offline():
        doc = pipeline.run(a.doc, describer=rec, ocr=backends.make_ocr(), workers=1)
    spoken = [(p.page_no, u.text) for p, u in doc.transcript(2) if u.kind == "figure"]
    for (page, heard), c in zip(spoken, rec.calls):
        ms = c.get("total_duration", 0) / 1e6
        print(f"— page {page}, crop {c['shape'][1]}x{c['shape'][0]} px")
        print(f"  context : {c['context']}")
        print(f"  model   : {c['raw']}")
        print(f"  heard   : {heard}")
        print(f"  tokens  : prompt {c.get('prompt_eval_count')}  output {c.get('eval_count')}"
              f"   wall {ms:.0f} ms on this machine ({a.backend}, {a.model})")
    print(f"self-check: {doc.badge()}   network attempts {telemetry.attempts()}")


if __name__ == "__main__":
    main()
