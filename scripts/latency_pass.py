#!/usr/bin/env python3
"""Re-measure one model's latency, reusing its own bench.py adapter.

Why this exists: the accuracy runs happened in separate batches, and this box is
a shared 4-core CPU whose throughput drifts by ~25% between batches. Measured
back to back, MobileSAM and TinySAM (byte-identical encoders, same FLOPs) come
out equal; measured in different batches they differed by 25%. Accuracy is
deterministic and unaffected, but latency is only comparable across models when
every model is timed in one uninterrupted sequence.

Driving each model through the same adapter its accuracy run used keeps the
timed code path identical to the measured one.

    uv run --project <dir> python scripts/latency_pass.py <dir> [variant]

`scripts/run_latency.sh` drives all of them in order.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_bench(kind: str):
    """Import <kind>/bench.py as a module, with its own sys.path set up."""
    d = ROOT / kind
    sys.path.insert(0, str(d))
    sys.path.insert(0, str(ROOT / "common"))
    spec = importlib.util.spec_from_file_location(f"{kind}_bench", d / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return d, mod


def make_adapter(kind: str, variant: str | None):
    d, m = load_bench(kind)
    w = d / "weights"
    if kind == "fastsam":
        v = variant or "x"
        return m.FastSAMAdapter(str(w / f"FastSAM-{v}.pt")), 1024
    if kind == "mobilesam":
        return m.MobileSAMAdapter(str(w / "mobile_sam.pt")), 1024
    if kind == "edgesam":
        return m.EdgeSAMAdapter(str(w / "edge_sam_3x.pth")), 1024
    if kind == "efficientsam":
        v = variant or "vitt"
        return m.EfficientSAMAdapter(v, str(w / f"efficient_sam_{v}.pt")), 1024
    if kind == "efficientsam3":
        b = variant or "tinyvit"
        return m.EfficientSAM3Adapter(b, str(w / f"efficientsam3_{b}.pt")), 1008
    if kind in ("tinysam", "repvitsam"):
        ck = "tinysam_42.3.pth" if kind == "tinysam" else "repvit_sam.pt"
        return m.build(str(w / ck)), 1024
    if kind == "efficientvitsam":
        t = variant or "l0"
        a = m.build(f"efficientvit-sam-{t}", str(w / f"efficientvit_sam_{t}.pt"))
        return a, a.encoder_input_size
    raise SystemExit(f"unknown model dir: {kind}")


def main() -> int:
    sys.path.insert(0, str(ROOT / "common"))
    from pbs_bench.runner import measure_latency

    kind = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else None
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)

    adapter, res = make_adapter(kind, variant)

    # A real image, not a blank canvas: FastSAM detects nothing in a blank one,
    # so its box_prompt would return instantly through the empty-result path and
    # report a decode time that has nothing to do with real use. Every model
    # resizes this to its own input resolution internally, exactly as in the
    # accuracy runs.
    import cv2
    img = cv2.imread(str(ROOT / "assets" / "dogs.jpg"))
    if img is None:
        raise SystemExit("assets/dogs.jpg missing")
    h, w = img.shape[:2]
    box = np.array([w * 0.40, h * 0.28, w * 0.72, h * 0.97], dtype=np.float32)

    # The decoder call is short enough (tens of ms) that scheduler jitter
    # dominates a small sample -- most of these models share an identical 4.06M
    # mask decoder, so their decode times should agree, and only a larger sample
    # shows that. The encoder call is ~1s, so fewer repeats suffice there.
    enc = measure_latency(lambda: adapter.set_image(img), warmup=2, repeat=6)
    adapter.set_image(img)
    dec = measure_latency(lambda: adapter.predict_box(box), warmup=3, repeat=25)

    out = {"model": adapter.name, "variant": adapter.variant,
           "encoder_input": res, "encode": enc, "decode": dec}
    print(f"{adapter.name:22} encode={enc['median_ms']:8.1f} ms   "
          f"decode={dec['median_ms']:8.2f} ms", flush=True)
    suffix = f"_{variant}" if variant else ""
    (ROOT / kind / "outputs" / f"latency{suffix}.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
