#!/usr/bin/env python3
"""P87: Verify public FP32 detector source on CPU against frozen P69 reference.

Preregistered pilot: five lexically first, original unlabeled test JPEGs in each
of F1-F4. Compare both views, sorted proposal counts, coordinates, scores and
individual mask crops, at the original source image resolution. No Kaggle
submissions, training, labels, or leaderboard optimization are performed.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import pathlib
import pickle
import time

from replay_public_detector_p81 import unreflect, verify_proposals, sha256

EXPECTED = {
    "F1": "399b4bd790c5acf88f32da926dd00a9ac4b704040241b8b0899117c687756820",
    "F2": "9a50cc6d23d1689f8d9a69ac90d25c086d128d98682c77669366ce59caca1115",
    "F3": "e7f9ed980842c2a04ee5ff21fe3018d00eef0ed742aace9f9bb0e08af77f8952",
    "F4": "59e942afc969318681c7de5c748ed58bd29b79448243389e34db93a8fe54827f",
}

class OneImage:
    """Single-image, read-only adapter for the original detector implementation."""

    def __init__(self, array):
        self.array = array

    def __getitem__(self, position):
        if position != 0:
            raise IndexError(position)
        return self.array


def summarize_view(current, baseline):
    """Return exact agreement diagnostics without matching instances post hoc."""
    import numpy as np

    exact_boxes = exact_scores = exact_masks = 0
    exact_instances = 0
    for left, right in zip(current, baseline):
        box = left[1:5] == right[1:5]
        score = float(left[0]) == float(right[0])
        pixels = box and np.array_equal(left[5], right[5])
        exact_boxes += int(box)
        exact_scores += int(score)
        exact_masks += int(pixels)
        exact_instances += int(box and score and pixels)
    full = len(current) == len(baseline) and exact_instances == len(current)
    return {
        "current_count": len(current),
        "reference_count": len(baseline),
        "same_index_exact_boxes": exact_boxes,
        "same_index_exact_scores": exact_scores,
        "same_index_exact_masks": exact_masks,
        "same_index_complete_instances": exact_instances,
        "entire_view_exact": full,
    }


def self_test():
    """Reject a count mismatch even if the shared prefix matches exactly."""
    import numpy as np

    sample = (0.7, 1, 3, 2, 4, np.ones((2, 2), dtype=np.uint8))
    same = summarize_view([sample], [sample])
    different = summarize_view([sample], [sample, sample])
    assert same["entire_view_exact"] is True
    assert different["entire_view_exact"] is False
    assert different["same_index_complete_instances"] == 1


def save_json_atomic(path, payload):
    """Write immutable checked reports without overwriting a prior experiment."""
    if path.exists():
        raise FileExistsError(f"Prior immutable report already exists: {path}")
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as writer:
        json.dump(payload, writer, sort_keys=True, indent=2)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=pathlib.Path, required=True)
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--pilot-count", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("P87_SELFTEST_OK", flush=True)
        return
    if args.pilot_count != 5:
        raise ValueError("Frozen preregistration requires exactly five images per fold.")
    if args.out.exists():
        raise FileExistsError("P87 output directory already exists; do not repeat work.")
    images = sorted(args.images.glob("*.jpeg"))
    if len(images) != 180 or len({p.name for p in images}) != 180:
        raise ValueError("Original Kaggle test inventory must have 180 unique JPEGs.")
    for fold, original_hash in EXPECTED.items():
        path = args.weights / f"maskrcnn_{fold}_fp32.pt"
        if sha256(path) != original_hash:
            raise ValueError(f"Released FP32 checkpoint hash differs for {fold}.")
        manifest = args.reference / fold / "manifest.json"
        entry = json.loads(manifest.read_text(encoding="utf-8"))
        if entry["status"] != "CPU_BYTE_EXACT_PASS" or entry["images"] != 180:
            raise ValueError(f"Untrusted or incomplete historical reference: {fold}.")
        for image in images[:5]:
            if not (args.reference / fold / f"{image.stem}.pkl").is_file():
                raise FileNotFoundError(image.name)

    import numpy as np
    import torch
    import filseg.train_maskrcnn as detector
    from filseg.data import RES, read_gray

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("P87 must be CPU-only with CUDA_VISIBLE_DEVICES empty.")
    info = "CPU-only preflight: CUDA_VISIBLE_DEVICES is empty"
    detector.autocast = lambda device: contextlib.nullcontext()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        raise RuntimeError("P87 must not initialize CUDA.")

    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {
        "experiment": "P87_public_FP32_CPU_vs_original_CPU_pilot",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": "cpu",
        "cpu_preflight": info,
        "source_sha256": sha256(pathlib.Path(__file__)),
        "original_unlabeled_images": 5,
        "folds": {},
        "used_test_labels": False,
        "created_kaggle_submission": False,
        "no_leaderboard_tuning": True,
    }
    all_exact = True
    device = torch.device("cpu")
    for fold, original_hash in EXPECTED.items():
        model = detector.build_model(size=RES, pretrained=False)
        model.load_state_dict(torch.load(args.weights / f"maskrcnn_{fold}_fp32.pt",
                                         map_location="cpu", weights_only=True), strict=True)
        model.eval().to(device)
        entries = {}
        for image in images[:5]:
            gray = read_gray(image, RES)
            if gray.shape != (RES, RES):
                raise RuntimeError("Unexpected image geometry.")
            with torch.inference_mode():
                first = detector.predict_files(model, OneImage(gray),
                                               {image.name: 0}, [image.name], device)[image.name]
                flipped = np.ascontiguousarray(gray[:, ::-1])
                second = detector.predict_files(model, OneImage(flipped),
                                                {image.name: 0}, [image.name], device)[image.name]
            views = [first, [unreflect(det, RES) for det in second]]
            verify_proposals(views, RES)
            frozen = pickle.loads((args.reference / fold / f"{image.stem}.pkl").read_bytes())
            pairs = [summarize_view(views[0], frozen[0]),
                     summarize_view(views[1], frozen[1])]
            is_exact = all(p["entire_view_exact"] for p in pairs)
            all_exact = all_exact and is_exact
            entries[image.name] = {
                "input_sha256": sha256(image), "both_views_exact": is_exact,
                "views": pairs,
            }
        report["folds"][fold] = {
            "source_weight_sha256": original_hash,
            "exact_images": sum(x["both_views_exact"] for x in entries.values()),
            "pilot_images": len(entries), "entries": entries,
        }
        print("P87_PILOT", fold, report["folds"][fold]["exact_images"], "/5", flush=True)
        del model

    report["all_pilot_images_bit_exact"] = all_exact
    report["full_domain_gpu_followup_allowed"] = all_exact
    report["elapsed_seconds"] = round(time.monotonic() - start, 3)
    report["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save_json_atomic(args.out / "receipt.json", report)
    print("P87_PILOT_COMPLETE", all_exact, report["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
