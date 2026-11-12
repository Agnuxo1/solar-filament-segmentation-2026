#!/usr/bin/env python3
"""Audit frozen stage-one H-alpha map parity without reading any labels."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    """Compute the binary SHA-256 of an immutable image."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_native(path: Path) -> np.ndarray:
    """Decode a native 2048-square grayscale image with strict validation."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (2048, 2048) or image.dtype != np.uint8:
        raise ValueError(f"Invalid native probability image: {path}")
    return image


def compare(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """Quantify pixel parity, retaining all reference-independent diagnostics."""
    absolute = cv2.absdiff(candidate, reference)
    histogram = np.bincount(absolute.ravel(), minlength=256)
    return {
        "pixel_equal_fraction": float(histogram[0] / absolute.size),
        "different_pixels": int(absolute.size - histogram[0]),
        "total_pixels": int(absolute.size),
        "max_absolute_byte_error": int(absolute.max()),
        "mean_absolute_byte_error": float(absolute.mean(dtype=np.float64)),
        "within_one_byte_fraction": float(histogram[:2].sum() / absolute.size),
        "exact_image": bool(histogram[0] == absolute.size),
    }


def atomic_write(destination: Path, value: dict) -> None:
    """Write a non-replaceable receipt with a durable atomic rename."""
    if destination.exists():
        raise FileExistsError("Refusing to overwrite a scientific receipt.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def main() -> None:
    """Audit one public replay against two independently archived maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-map", type=Path, required=True)
    parser.add_argument("--prior-replay", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "public_source_replay": args.public_map,
        "private_p63_replay": args.prior_replay,
        "historical_pre_submission": args.historical,
    }
    if len({path.name for path in paths.values()}) != 1:
        raise ValueError("All three files must represent the same image stem.")
    maps = {name: read_native(path) for name, path in paths.items()}
    current = maps["public_source_replay"]
    details = {
        "public_vs_p63": compare(current, maps["private_p63_replay"]),
        "public_vs_historical": compare(current, maps["historical_pre_submission"]),
        "p63_vs_historical": compare(maps["private_p63_replay"], maps["historical_pre_submission"]),
    }
    receipt = {
        "status": "P76_ONE_IMAGE_MAP_PARITY_AUDITED",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_image_stem": args.public_map.stem,
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256(path) for name, path in paths.items()},
        "comparisons": details,
        "count_images": 1,
        "neither_ground_truth_nor_labels_read": True,
        "does_not_establish_end_to_end_submission_csv": True,
        "no_new_kaggle_submission": True,
    }
    atomic_write(args.receipt, receipt)
    for name, metrics in details.items():
        print(name, json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
