#!/usr/bin/env python3
"""Reproduce H9b stage-one H3 maps from original competition test pixels.

This is a source-image-to-semantic-map replay, not a full H9b CSV generator.
It intentionally uses the exact released native_model.NativeUNet implementation
and the four SHA256-verified owned weights. No test labels are accessed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "release" / "source"
sys.path.insert(0, str(SOURCE))
import native_model  # noqa: E402

NATIVE = (2048, 2048)
GPU_UUID_DEFAULT = "GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14"


def sha256(path: Path) -> str:
    """Hash a file without buffering large checkpoints or images in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def manifest_weights() -> dict[str, str]:
    """Read exact public weight hashes, rejecting malformed SHA-256 entries."""
    path = PROJECT / "release" / "model_metadata" / "WEIGHTS_SHA256SUMS.txt"
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError("Malformed public checkpoint manifest line.")
        signature, name = fields
        if len(signature) != 64 or any(c not in "0123456789abcdef" for c in signature):
            raise ValueError(f"Invalid SHA-256 signature for {name}.")
        if name in result:
            raise ValueError(f"Duplicate model entry: {name}.")
        result[name] = signature
    return result


def ensure_exclusive_gpu(uuid: str) -> None:
    """Check physical GPU occupancy before creating any CUDA context."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != uuid:
        raise RuntimeError("Requested GPU UUID must be the only visible GPU.")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    if any(row.strip().startswith(uuid) for row in result.stdout.splitlines()):
        raise RuntimeError("GPU0 already hosts a CUDA process; refusing overlap.")


def validate_image(path: Path) -> np.ndarray:
    """Read a native-resolution H-alpha test image, with no annotation access."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.dtype != np.uint8 or image.shape != NATIVE:
        raise ValueError(f"Invalid 2048x2048 uint8 grayscale image: {path.name}")
    return image


def atomic_png(path: Path, data: np.ndarray) -> None:
    """Store one prediction atomically for interrupted-run recovery."""
    temp = path.with_name(path.stem + ".partial.png")
    if temp.exists() or path.exists():
        raise FileExistsError(f"Refusing to overwrite a completed map: {path}")
    if not cv2.imwrite(str(temp), data, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise OSError(f"Failed to encode PNG {path.name}.")
    os.replace(temp, path)


def atomic_json(path: Path, payload: dict) -> None:
    """Publish an immutable run receipt only after all images are validated."""
    if path.exists():
        raise FileExistsError(f"Receipt exists, use a different output directory: {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    """Rebuild one frozen model's maps, reserving GPU0 for one process only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="Original Kaggle test/test_images directory.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Separate writeable output directory, ideally on HDD.")
    parser.add_argument("--fold", choices=("F1", "F2", "F3", "F4"), required=True)
    parser.add_argument("--limit", type=int,
                        help="Audit-only prefix count; omit for all 180 official images.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-uuid", default=GPU_UUID_DEFAULT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and not 1 <= args.limit <= 180:
        parser.error("--limit must be 1..180.")

    files = sorted(args.images.glob("*.jpeg"))
    if len(files) != 180 or len({path.stem for path in files}) != 180:
        raise ValueError("Expected 180 unique original Kaggle test JPEG images.")
    selected = files[:args.limit] if args.limit else files

    expected = manifest_weights()
    model_name = f"{args.fold}_resnet34_s0_fp16.pt"
    weight = PROJECT / "release" / "models" / model_name
    if sha256(weight) != expected[model_name]:
        raise RuntimeError(f"Public weight SHA-256 mismatch: {model_name}")
    if args.device == "cuda":
        ensure_exclusive_gpu(args.gpu_uuid)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("The process must see precisely one CUDA GPU.")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.75, 0)

    model = native_model.NativeUNet("resnet34", pretrained=False)
    state = torch.load(weight, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    start = time.monotonic()

    out_folder = args.out / args.fold
    out_folder.mkdir(parents=True, exist_ok=True)
    results = []
    for index, path in enumerate(selected, 1):
        output = out_folder / f"{path.stem}.png"
        if output.exists():
            if not args.resume:
                raise FileExistsError(f"Existing map requires --resume: {output}")
            decoded = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
            if decoded is None or decoded.shape != NATIVE or decoded.dtype != np.uint8:
                raise ValueError(f"Corrupt restart map: {output}")
        else:
            decoded = native_model.predict(model, validate_image(path), device)
            if decoded.dtype != np.uint8 or decoded.shape != NATIVE:
                raise RuntimeError("Unexpected model output shape or dtype.")
            atomic_png(output, decoded)
        results.append({"file": path.name, "input_sha256": sha256(path),
                        "map_sha256": sha256(output)})
        if index == 1 or index % 20 == 0 or index == len(selected):
            print("H3_RAW_REPLAY", args.fold, index, len(selected), flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    receipt = {
        "status": "H3_STAGE_ONE_REPLAY_COMPLETE_NOT_FINAL_CSV",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fold": args.fold, "images_processed": len(results),
        "full_180_images": len(results) == 180,
        "device": args.device, "gpu_uuid": args.gpu_uuid if args.device == "cuda" else None,
        "weight_sha256": expected[model_name],
        "model_source_sha256": sha256(SOURCE / "native_model.py"),
        "script_sha256": sha256(Path(__file__)),
        "runtime_seconds": round(time.monotonic() - start, 3),
        "peak_cuda_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
                             if device.type == "cuda" else None,
        "image_sha256": results,
        "test_labels_read": False, "official_kaggle_submission_made": False,
        "end_to_end_submission_csv_reproduced": False,
    }
    atomic_json(args.out / f"{args.fold}_h3_replay_receipt.json", receipt)
    print("H3_REPLAY_VERIFIED", args.fold, len(results), flush=True)


if __name__ == "__main__":
    main()

