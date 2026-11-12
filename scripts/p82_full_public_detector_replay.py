#!/usr/bin/env python3
"""Resume-safe full-domain public Mask R-CNN proposal reconstruction.

Runs all four published FP16 checkpoints on all 180 original unlabelled test
JPEGs. This is an intermediate reproducibility artifact, never a Kaggle CSV
or a Panoptic Quality claim. External lock and single-GPU isolation required.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import pathlib
import pickle
import shutil
import time

from replay_public_detector_p81 import (
    FOLDS, GPU_UUID, compare_to_reference, gpu_preflight,
    self_test, sha256, signed_manifest, unreflect, verify_proposals,
)


def atomic_json(path: pathlib.Path, value: dict) -> None:
    """Preserve immutable receipts by flushing a unique intermediate file."""
    if path.exists():
        raise FileExistsError(f"Existing immutable receipt: {path}")
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def journal(root: pathlib.Path, kind: str, **kwargs) -> None:
    """Append physically timestamped, ordered progress events on the HDD."""
    payload = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "kind": kind, **kwargs}
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()


class OneImage:
    """Read-only single-image store for the original public detector API."""

    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, key):
        if key != 0:
            raise IndexError(key)
        return self.arr


def main() -> None:
    """Reconstruct two proposal views per test image, independently per fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=pathlib.Path)
    parser.add_argument("--images", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--reference", type=pathlib.Path,
                        help="Optional trusted local P69 pickle directory")
    args = parser.parse_args()
    self_test()
    images = sorted(args.images.glob("*.jpeg"))
    if len(images) != 180 or len(set(p.name for p in images)) != 180:
        raise RuntimeError("Expected 180 original Kaggle test JPEGs.")
    signatures = signed_manifest(args.release)
    snapshot = gpu_preflight()
    args.out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out).free < 20 * (1024 ** 3):
        raise RuntimeError("HDD lacks the required free-space reserve")
    final_receipt = args.out / "receipt.json"
    if final_receipt.exists():
        raise FileExistsError("P82 has already completed; never repeat it.")
    import numpy as np
    import torch
    import filseg.train_maskrcnn as detector
    from filseg.data import RES, read_gray

    detector.autocast = lambda device: contextlib.nullcontext()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Only designated CUDA GPU0 is allowed.")
    device = torch.device("cuda:0")
    started = time.monotonic()
    runs = {}
    journal(args.out, "start", gpu=GPU_UUID, images=180)
    for fold in FOLDS:
        fold_out = args.out / fold
        fold_out.mkdir(exist_ok=True)
        fold_receipt = fold_out / "manifest.json"
        if fold_receipt.exists():
            completed = json.loads(fold_receipt.read_text(encoding="utf-8"))
            if (completed.get("status") != "P82_FOLD_FULLY_REPLAYED"
                    or completed.get("images") != 180):
                raise RuntimeError(f"Inconsistent existing manifest for {fold}")
            if set(completed["entries"]) != set(p.name for p in images):
                raise RuntimeError("Old fold receipt does not match frozen images")
            for name, entry in completed["entries"].items():
                if sha256(fold_out / (pathlib.Path(name).stem + ".pkl")) != entry["sha256"]:
                    raise RuntimeError("Previously verified proposal file changed")
            runs[fold] = {key: completed[key] for key in
                          ("images", "models", "two_view_proposals",
                           "identical_boxes", "identical_crops", "elapsed_seconds")}
            journal(args.out, "fold_skipped_verified_receipt", fold=fold)
            continue

        weight = args.release / "models" / f"maskrcnn_{fold}_fp16.pt"
        model = detector.build_model(size=RES, pretrained=False)
        model.load_state_dict(
            torch.load(weight, map_location="cpu", weights_only=True), strict=True
        )
        model.eval().to(device)
        entries = {}
        proposals = 0
        identical_boxes = 0
        identical_crops = 0
        fold_start = time.monotonic()
        for i, image in enumerate(images, 1):
            if time.monotonic() - started > 1300:
                raise TimeoutError("P82 budget reached: resumable per-image outputs kept")
            output = fold_out / (image.stem + ".pkl")
            if output.exists():
                views = pickle.loads(output.read_bytes())
            else:
                gray = read_gray(image, RES)
                if gray.shape != (RES, RES):
                    raise RuntimeError("Unexpected original image size")
                with torch.inference_mode():
                    first = detector.predict_files(
                        model, OneImage(gray), {image.name: 0}, [image.name], device
                    )[image.name]
                    mirrored = np.ascontiguousarray(gray[:, ::-1])
                    second = detector.predict_files(
                        model, OneImage(mirrored), {image.name: 0}, [image.name], device
                    )[image.name]
                views = [first, [unreflect(d, RES) for d in second]]
                temp = output.with_name(output.name + f".{os.getpid()}.tmp")
                if temp.exists():
                    raise RuntimeError(f"Interrupted temporary output requires audit: {temp}")
                with temp.open("xb") as stream:
                    pickle.dump(views, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, output)
            if len(views) != 2:
                raise RuntimeError("Unexpected number of reference views")
            counts = verify_proposals(views, RES)
            comparisons = None
            if args.reference is not None:
                refpath = args.reference / fold / (image.stem + ".pkl")
                if not refpath.is_file():
                    raise FileNotFoundError(refpath)
                comparisons = compare_to_reference(
                    views, pickle.loads(refpath.read_bytes())
                )
                for item in comparisons["views"]:
                    identical_boxes += item["same_index_exact_boxes"]
                    identical_crops += item["same_index_exact_crop_masks"]
            proposals += sum(counts)
            entries[image.name] = {
                "input_sha256": sha256(image),
                "sha256": sha256(output),
                "counts": counts,
                "reference": comparisons,
            }
            if i == 1 or i % 20 == 0 or i == 180:
                journal(args.out, "fold_progress", fold=fold, done=i, images=180,
                        proposals=proposals)
                print("P82_PROGRESS", fold, i, 180, proposals, flush=True)

        torch.cuda.synchronize(device)
        payload = {
            "status": "P82_FOLD_FULLY_REPLAYED",
            "fold": fold, "images": 180, "models": 1,
            "two_view_proposals": proposals,
            "identical_boxes": identical_boxes,
            "identical_crops": identical_crops,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(0)),
            "weight_sha256": signatures[weight.name],
            "elapsed_seconds": round(time.monotonic() - fold_start, 2),
            "entries": entries,
        }
        atomic_json(fold_receipt, payload)
        runs[fold] = {key: payload[key] for key in
                      ("images", "models", "two_view_proposals",
                       "identical_boxes", "identical_crops", "elapsed_seconds")}
        journal(args.out, "fold_complete", fold=fold, seconds=payload["elapsed_seconds"])
        del model, views
        torch.cuda.empty_cache()

    final = {
        "status": "P82_FULL_DOMAIN_PUBLIC_DETECTOR_REPLAY_COMPLETE",
        "source_sha256": sha256(pathlib.Path(__file__)),
        "gpu": GPU_UUID, "nvidia_smi_preflight": snapshot,
        "public_model_hashes": signatures,
        "images": 180, "folds": runs,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "no_test_annotations_used": True, "no_pq_or_kaggle_submission": True,
        "not_a_final_h9b_csv": True,
    }
    atomic_json(final_receipt, final)
    journal(args.out, "completed", elapsed_seconds=final["elapsed_seconds"])
    print("P82_ALL_FOLDS_COMPLETE", json.dumps(runs), flush=True)


if __name__ == "__main__":
    main()
