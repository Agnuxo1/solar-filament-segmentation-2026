#!/usr/bin/env python3
"""Auditable pilot inference with public Solar Filament Mask R-CNN detectors.

Reads only original unlabeled competition test JPEGs and public checkpoint files.
Each fold processes the first lexically ordered original test image, with both
original and horizontal-reflection views. This is not a final CSV or PQ scorer.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import pickle
import subprocess
import sys
import time

GPU_UUID = "GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14"
FOLDS = ("F1", "F2", "F3", "F4")


def sha256(path: pathlib.Path) -> str:
    """Hash arbitrary-sized files with bounded host memory."""
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unreflect(det: tuple, size: int) -> tuple:
    """Invert the historical left-right image reflection for crop proposals."""
    import numpy as np
    score, y0, y1, x0, x1, crop = det
    return score, y0, y1, size - x1, size - x0, np.ascontiguousarray(crop[:, ::-1])


def self_test() -> None:
    """Check reflection involution and array ownership before any GPU work."""
    import numpy as np
    sample = (0.8, 2, 4, 3, 6, np.arange(6, dtype=np.uint8).reshape(2, 3))
    back = unreflect(unreflect(sample, 10), 10)
    assert sample[:5] == back[:5]
    assert np.array_equal(sample[5], back[5])
    assert back[5].flags.c_contiguous


def signed_manifest(release: pathlib.Path) -> dict[str, str]:
    """Parse a checked-in SHA-256 manifest and reject missing or altered weights."""
    path = release / "model_metadata" / "WEIGHTS_SHA256SUMS.txt"
    manifest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        signature, name = line.strip().split(maxsplit=1)
        name = name.removeprefix("*")
        if len(signature) != 64 or name in manifest:
            raise ValueError("Invalid or duplicated signed manifest entry")
        manifest[name] = signature.lower()
    for fold in FOLDS:
        name = f"maskrcnn_{fold}_fp16.pt"
        weight = release / "models" / name
        if manifest.get(name) != sha256(weight):
            raise ValueError(f"Public weight hash mismatch: {name}")
    return manifest


def gpu_preflight() -> str:
    """Require exclusive physical GPU0 before initializing any CUDA context."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("Unexpected CUDA visibility; only assigned GPU0 is authorized.")
    info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"], text=True
    )
    if GPU_UUID not in info:
        raise RuntimeError("GPU0 UUID not present in nvidia-smi.")
    pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
         "--format=csv,noheader,nounits"], text=True
    )
    if any(GPU_UUID in line for line in pids.splitlines()):
        raise RuntimeError("Assigned GPU0 already has an active compute process.")
    return info


def verify_proposals(views: list, size: int) -> list[int]:
    """Verify that every detection crop is in range and has consistent shape."""
    counts = []
    for detections in views:
        counts.append(len(detections))
        for item in detections:
            if len(item) != 6:
                raise AssertionError("Malformed detector proposal tuple")
            score, y0, y1, x0, x1, crop = item
            if not (0 <= float(score) <= 1 and 0 <= y0 < y1 <= size
                    and 0 <= x0 < x1 <= size):
                raise AssertionError("Invalid proposal geometry")
            if crop.shape != (y1 - y0, x1 - x0):
                raise AssertionError("Detector mask crop does not fit its box")
    return counts


def compare_to_reference(current: list, historic: list) -> dict:
    """Report mismatches descriptively, without promoting or optimizing a model."""
    import numpy as np
    if len(historic) != 2:
        raise ValueError("Expected two historical view arrays.")
    comparisons = []
    for i in range(2):
        now, old = current[i], historic[i]
        matched = sum(a[1:5] == b[1:5] for a, b in zip(now, old))
        pixel_exact = sum(
            a[1:5] == b[1:5] and np.array_equal(a[5], b[5])
            for a, b in zip(now, old)
        )
        comparisons.append({
            "view": i,
            "public_proposal_count": len(now),
            "historical_proposal_count": len(old),
            "same_index_exact_boxes": matched,
            "same_index_exact_crop_masks": pixel_exact,
        })
    return {"comparison_type": "descriptive_fp16_vs_original_fp32", "views": comparisons}


def main() -> None:
    """Run one evidence-bounded pilot and retain atomic per-fold receipts on HDD."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=pathlib.Path)
    parser.add_argument("--images", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--reference", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("P81_SELFTEST_OK", flush=True)
        return

    release = args.release.resolve()
    images = sorted(args.images.glob("*.jpeg"))
    if len(images) != 180 or len({p.name for p in images}) != 180:
        raise RuntimeError("Expected exactly 180 distinct official test JPEGs.")
    signatures = signed_manifest(release)
    gpu_snapshot = gpu_preflight()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    import numpy as np
    import torch
    import filseg.train_maskrcnn as detector
    from filseg.data import RES, read_gray

    detector.autocast = lambda device: contextlib.nullcontext()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA did not expose exactly one assigned GPU.")
    device = torch.device("cuda:0")
    image = images[0]
    gray = read_gray(image, RES)
    if gray.shape != (RES, RES):
        raise ValueError("Unexpected preprocessed official image geometry.")

    class OneImage:
        """Single-item read-only store compatible with public predict_files."""

        def __init__(self, value):
            self.value = value

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return self.value

    results = {"status": "P81_PILOT_COMPLETE", "source_sha256": sha256(pathlib.Path(__file__)),
               "input_name": image.name, "input_sha256": sha256(image),
               "gpu_uuid": GPU_UUID, "nvidia_smi_before_cuda": gpu_snapshot,
               "no_test_labels_used": True, "no_pq_or_submission": True,
               "folds": {}, "elapsed_seconds": 0}
    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "receipt.json").exists():
        raise FileExistsError("P81 pilot is already complete; do not repeat it.")
    for fold in FOLDS:
        weight = release / "models" / f"maskrcnn_{fold}_fp16.pt"
        model = detector.build_model(size=RES, pretrained=False)
        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True),
                              strict=True)
        model = model.eval().to(device)
        with torch.inference_mode():
            first = detector.predict_files(model, OneImage(gray), {image.name: 0},
                                           [image.name], device)[image.name]
            mirrored = np.ascontiguousarray(gray[:, ::-1])
            second = detector.predict_files(model, OneImage(mirrored), {image.name: 0},
                                            [image.name], device)[image.name]
        views = [first, [unreflect(item, RES) for item in second]]
        counts = verify_proposals(views, RES)
        outfile = args.out / f"{fold}_{image.stem}_two_views.pkl"
        temporary = outfile.with_name(outfile.name + f".{os.getpid()}.tmp")
        if outfile.exists() or temporary.exists():
            raise FileExistsError("Unreconciled pilot output exists; inspect before retry.")
        with temporary.open("xb") as handle:
            pickle.dump(views, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, outfile)
        entry = {"weight_sha256": signatures[weight.name],
                 "file_sha256": sha256(outfile), "proposal_counts": counts,
                 "peak_vram_bytes": int(torch.cuda.max_memory_allocated(0))}
        if args.reference:
            reference_file = args.reference / fold / (image.stem + ".pkl")
            if not reference_file.is_file():
                raise FileNotFoundError(reference_file)
            historic = pickle.loads(reference_file.read_bytes())
            entry["historical_comparison"] = compare_to_reference(views, historic)
        results["folds"][fold] = entry
        del model, views
        torch.cuda.empty_cache()
        print("P81_FOLD", fold, "PROPOSALS", counts, flush=True)
    results["elapsed_seconds"] = round(time.monotonic() - start, 2)
    temp_receipt = args.out / f"receipt.json.{os.getpid()}.tmp"
    with temp_receipt.open("x", encoding="utf-8") as output:
        json.dump(results, output, sort_keys=True, indent=2)
        output.flush()
        os.fsync(output.fileno())
    temp_receipt.replace(args.out / "receipt.json")
    print("P81_PILOT_COMPLETE", json.dumps({
        "elapsed_seconds": results["elapsed_seconds"],
        "counts": {fold: data["proposal_counts"]
                   for fold, data in results["folds"].items()}
    }), flush=True)


if __name__ == "__main__":
    main()
