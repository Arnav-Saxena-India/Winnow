"""EasyOCR's recogniser on a Snapdragon X Elite NPU, via Qualcomm AI Hub.

    python tools/aihub_ocr.py [--width 1024] [--device "Snapdragon X Elite CRD"]

Winnow finds its own line boxes and only runs EasyOCR's recogniser (a VGG
feature extractor, two bidirectional LSTMs, a linear layer: CTC output). This
compiles that network for the NPU at one fixed line size (64 px high, as EasyOCR
feeds it; ``--width`` wide), profiles it on a hosted device, then runs one real
text line from the corpus through it and compares the decoded text with the CPU.

What is uploaded to AI Hub: the recogniser's weights and one line cropped from
``tests/corpus/scan_p1.png`` (a page generated for this repository). No user
document is sent. This is a development measurement: at read time Winnow stays
offline, and the NPU path needs the compiled model on a Snapdragon machine.
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Recogniser(torch.nn.Module):
    """EasyOCR's model takes (image, text) and ignores text; the NPU wants one input."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image, None)


def cpu_ms(net, x, runs=10) -> float:
    with torch.no_grad():
        for _ in range(2):
            net(x)
        t = time.perf_counter()
        for _ in range(runs):
            net(x)
    return 1000 * (time.perf_counter() - t) / runs


def line_tensor(width: int) -> torch.Tensor:
    """The first text line of the corpus scan, prepared the way EasyOCR does
    (grey, 64 px high, scaled to [-1, 1], padded on the right to ``width``)."""
    import cv2
    from winnow.ocr import line_boxes
    gray = cv2.imread(str(ROOT / "tests" / "corpus" / "scan_p1.png"), cv2.IMREAD_GRAYSCALE)
    x0, y0, x1, y1 = max(line_boxes(gray), key=lambda b: b[2] - b[0])
    crop = gray[int(y0):int(y1), int(x0):int(x1)]
    w = min(width, int(np.ceil(64 * crop.shape[1] / crop.shape[0])))
    crop = cv2.resize(crop, (w, 64), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    out = np.zeros((64, width), np.float32)
    out[:, :w] = (crop - 0.5) / 0.5
    out[:, w:] = out[:, w - 1:w]                   # EasyOCR pads with the last column
    return torch.from_numpy(out)[None, None]


def decode(converter, logits: np.ndarray) -> str:
    idx = logits.argmax(axis=2).reshape(-1)
    return converter.decode_greedy(idx, np.array([logits.shape[1]]))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--device", default="Snapdragon X Elite CRD")
    ap.add_argument("--runtime", default="qnn_dlc",
                    help="qnn_dlc (QNN on the NPU) or onnx (ONNX Runtime + QNN)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")      # AI Hub's client prints an hourglass

    import easyocr
    import qai_hub as hub
    reader = easyocr.Reader(["en"], gpu=False, verbose=False, download_enabled=False,
                            quantize=False)
    net = Recogniser(reader.recognizer).eval()
    quant = Recogniser(easyocr.Reader(["en"], gpu=False, verbose=False,
                                      download_enabled=False).recognizer).eval()
    x = line_tensor(a.width)
    with torch.no_grad():
        cpu_text = decode(reader.converter, net(x).numpy())
    print(f"line 64 x {a.width}: CPU text {cpu_text!r}")
    print(f"CPU, as Winnow runs it (int8 dynamic): {cpu_ms(quant, x):.1f} ms per line")
    print(f"CPU, float: {cpu_ms(net, x):.1f} ms per line")

    traced = torch.jit.trace(net, x)
    device = hub.Device(a.device)
    compile_job = hub.submit_compile_job(
        model=traced, device=device, input_specs={"image": tuple(x.shape)},
        options=f"--target_runtime {a.runtime}")
    print(f"compile job {compile_job.job_id}")
    target = compile_job.get_target_model()
    if target is None:
        print("compile failed:", compile_job.get_status().message)
        return 1
    profile_job = hub.submit_profile_job(model=target, device=device)
    print(f"profile job {profile_job.job_id}")
    prof = profile_job.download_profile()
    summ = prof["execution_summary"]
    units = Counter(layer.get("compute_unit", "?") for layer in prof.get("execution_detail", []))
    print(f"{a.device} NPU: {summ['estimated_inference_time'] / 1000:.2f} ms per line; "
          f"layers by compute unit {dict(units)}; peak memory "
          f"{summ.get('estimated_inference_peak_memory', 0) / 1e6:.1f} MB")

    inference_job = hub.submit_inference_job(model=target, device=device,
                                             inputs={"image": [x.numpy()]})
    print(f"inference job {inference_job.job_id}")
    out = inference_job.download_output_data()
    npu_text = decode(reader.converter, np.asarray(next(iter(out.values()))[0]))
    print(f"NPU text {npu_text!r}  ({'same as' if npu_text == cpu_text else 'DIFFERS from'} CPU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
