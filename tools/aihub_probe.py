"""§7 step 2: ms per figure for the describer on a Snapdragon X Elite NPU.

A figure description is not one forward pass. It is

    vision encoder (once) + prefill (image + prompt tokens) + decode (one step per token)

so this measures the encoder on a hosted device and adds prefill and decode
from token counts and throughputs. Decoding usually dominates.

    python tools/aihub_probe.py --model encoder.onnx --input-shape 1,3,448,448
    python tools/aihub_probe.py --encoder-ms 35.2 --output-tokens 71

``--encoder-ms`` skips the compile (e.g. the encoder was profiled with
``qai_hub_models ... export``). Take ``--prompt-tokens`` / ``--output-tokens``
from a real describer run (Ollama reports ``prompt_eval_count`` / ``eval_count``).
The default throughputs are Qualcomm's published Snapdragon X Elite CRD NPU numbers
for Qwen3-VL-2B-Instruct, Q4_0, GenieX llama.cpp (qai-hub-models v0.63.0 perf.yaml):
512-token context 964 prefill / 26.4 decode tok/s; 4096-token context 799 / 11.8.
Their LLM metrics exclude the vision encoder; OpenAI-CLIP's image encoder, profiled on
the same device, takes 25-33 ms with every layer on the NPU (a stand-in, not Qwen's).

Under 1.5 s/figure: describe on demand. Over 3 s: precompute at document open.
"""
import argparse
from collections import Counter

ON_DEMAND_MS, PRECOMPUTE_MS = 1_500, 3_000


def encoder_ms(model: str, input_shape: str, input_name: str, device_name: str) -> float:
    import qai_hub as hub
    device = hub.Device(device_name)
    shape = tuple(int(x) for x in input_shape.split(","))
    compiled = hub.submit_compile_job(
        model=model, device=device, input_specs={input_name: shape},
        options="--target_runtime qnn_dlc").get_target_model()
    prof = hub.submit_profile_job(model=compiled, device=device).download_profile()
    units = Counter(layer.get("compute_unit", "?") for layer in prof.get("execution_detail", []))
    print("encoder layers by compute unit:", dict(units))
    return prof["execution_summary"]["estimated_inference_time"] / 1000


def per_figure_ms(enc_ms, prompt_tokens, output_tokens, prefill_tps, decode_tps) -> dict:
    parts = {"encoder": enc_ms,
             "prefill": 1000 * prompt_tokens / prefill_tps,
             "decode": 1000 * output_tokens / decode_tps}
    return dict(parts, total=sum(parts.values()))


def verdict(total_ms: float) -> str:
    if total_ms < ON_DEMAND_MS:
        return "on demand"
    return "precompute at document open" if total_ms > PRECOMPUTE_MS else "borderline"


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="exported vision encoder (ONNX or TorchScript)")
    src.add_argument("--encoder-ms", type=float, help="encoder latency already measured")
    ap.add_argument("--input-shape", default="1,3,448,448")
    ap.add_argument("--input-name", default="pixel_values")
    ap.add_argument("--device", default="Snapdragon X Elite CRD")
    ap.add_argument("--prompt-tokens", type=int, default=300, help="image + text prompt")
    ap.add_argument("--output-tokens", type=int, default=70, help="a 2-3 sentence description")
    ap.add_argument("--prefill-tps", type=float, nargs="+", default=[964.0, 799.0])
    ap.add_argument("--decode-tps", type=float, nargs="+", default=[26.4, 11.8])
    a = ap.parse_args()
    enc = a.encoder_ms if a.encoder_ms is not None else encoder_ms(
        a.model, a.input_shape, a.input_name, a.device)
    print(f"{a.device}: encoder {enc:.1f} ms; {a.prompt_tokens} prompt + "
          f"{a.output_tokens} output tokens")
    for pre, dec in zip(a.prefill_tps, a.decode_tps):
        t = per_figure_ms(enc, a.prompt_tokens, a.output_tokens, pre, dec)
        print(f"  prefill {pre:g} tok/s, decode {dec:g} tok/s -> encoder {t['encoder']:.0f} + "
              f"prefill {t['prefill']:.0f} + decode {t['decode']:.0f} = {t['total']:.0f} ms"
              f"  => {verdict(t['total'])}")


if __name__ == "__main__":
    main()
