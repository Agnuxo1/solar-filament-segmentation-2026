#!/usr/bin/env python3
"""P101: bounded original-source F1 H3 inference with one frozen CuDNN flag intervention.

Both variants keep the original four-view AMP/TTA inference, byte rounding and
released checkpoint untouched. GPU0 safety checks run before CUDA initialization.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

import cv2
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import replay_h3_from_original_images as original  # noqa: E402

FOLDS = ("F1",)
GPU = "GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14"
VARIANTS = {"benchmark_off": False, "deterministic": True}


def self_test() -> None:
    """Ensure variant mapping is fixed and no unregistered mode is accepted."""
    assert len(VARIANTS) == 2
    assert VARIANTS["benchmark_off"] is False
    assert VARIANTS["deterministic"] is True
    assert (PROJECT / "release/source/native_model.py").is_file()
    print("P101_SELF_TEST_PASS", flush=True)


def main() -> None:
    """Recompute only the four frozen original F1 maps using assigned GPU0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--images", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if args.variant is None or args.images is None or args.out is None:
        parser.error("--variant, --images and --out are required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU:
        raise RuntimeError("GPU0 must be the only CUDA-visible physical GPU")
    original.ensure_exclusive_gpu(GPU)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("GPU device count is not exactly one")

    files = sorted(args.images.glob("*.jpeg"))
    if len(files) != 180 or len({x.stem for x in files}) != 180:
        raise RuntimeError("Expected exactly 180 unique Kaggle test JPEGs")

    weight_name = "F1_resnet34_s0_fp16.pt"
    weight = PROJECT / "release/models" / weight_name
    expected_sha = original.manifest_weights()[weight_name]
    if original.sha256(weight) != expected_sha:
        raise RuntimeError("F1 public checkpoint SHA-256 mismatch")

    out_dir = args.out / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out / f"{args.variant}_receipt.json").exists():
        raise FileExistsError("Existing complete variant receipt: refuse repetition")

    # These two flags are the ONLY planned interventions. All other backend
    # defaults, autocast, D4 TTA and source implementation remain unchanged.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = VARIANTS[args.variant]
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    device = torch.device("cuda:0")
    model = original.native_model.NativeUNet("resnet34", pretrained=False)
    state = torch.load(weight, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    started = time.monotonic()

    rows = []
    for image in files[:4]:
        native = original.validate_image(image)
        pred = original.native_model.predict(model, native, device)
        if pred.shape != (2048, 2048) or pred.dtype != "uint8":
            raise RuntimeError("Unexpected output map format")
        output = out_dir / f"{image.stem}.png"
        original.atomic_png(output, pred)
        rows.append({
            "image": image.name, "input_sha256": original.sha256(image),
            "map_sha256": original.sha256(output),
        })
        print("P101_MAP_COMPLETE", args.variant, len(rows), 4, flush=True)

    torch.cuda.synchronize()
    payload = {
        "status": "P101_VARIANT_COMPLETE", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "variant": args.variant, "fold": "F1",
        "images": rows,
        "checkpoint_sha256": expected_sha,
        "source_sha256": original.sha256(PROJECT / "release/source/native_model.py"),
        "runner_sha256": original.sha256(Path(__file__)),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "elapsed_s": round(time.monotonic() - started, 3),
        "max_cuda_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "accessed_test_labels": False, "official_submission_made": False,
        "end_to_end_csv_verified": False,
    }
    original.atomic_json(args.out / f"{args.variant}_receipt.json", payload)
    print("P101_VARIANT_VERIFIED", args.variant, flush=True)


if __name__ == "__main__":
    main()
