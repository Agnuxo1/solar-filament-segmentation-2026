#!/usr/bin/env python3
"""Audit byte-level parity for all 180 unlabelled public H3 probability maps.

This is a stage-one source reproducibility check, not panoptic scoring or
reconstruction of the accepted final competition CSV.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

EXPECTED_SHAPE = (2048, 2048)
EXPECTED_IMAGES = 180
MIN_EQUAL_FRACTION = 0.9999
MAX_ALLOWED_BYTE_DIFFERENCE = 1


def file_sha256(path: Path) -> str:
    """Calculate a streaming SHA256 digest without loading images into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_arrays(public: np.ndarray, reference: np.ndarray) -> dict:
    """Measure exact equality and absolute quantized-map deviation."""
    if public.shape != reference.shape or public.dtype != reference.dtype:
        raise ValueError("Images have different dimensions or dtypes.")
    deviation = cv2.absdiff(public, reference)
    return {
        "pixels": int(deviation.size),
        "different_pixels": int(np.count_nonzero(deviation)),
        "absolute_difference_sum": int(deviation.sum(dtype=np.uint64)),
        "max_absolute_byte_difference": int(deviation.max()),
    }


def self_test() -> None:
    """Check parity arithmetic on exact and single-byte perturbed arrays."""
    reference = np.arange(64, dtype=np.uint8).reshape(8, 8)
    same = compare_arrays(reference, reference.copy())
    assert same["different_pixels"] == 0 and same["absolute_difference_sum"] == 0
    changed = reference.copy()
    changed[0, 0] += 1
    measured = compare_arrays(changed, reference)
    assert measured["pixels"] == 64 and measured["different_pixels"] == 1
    assert measured["max_absolute_byte_difference"] == 1
    assert measured["absolute_difference_sum"] == 1
    print("SELF_TEST_PASS")


def main() -> None:
    """Verify input provenance, compare all maps, and issue an immutable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-receipt", type=Path)
    parser.add_argument("--public-maps", type=Path)
    parser.add_argument("--reference-maps", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    required = [args.public_receipt, args.public_maps, args.reference_maps, args.output]
    if any(item is None for item in required):
        parser.error("All four paths are required for the full-domain audit.")
    if args.output.exists():
        raise FileExistsError("Immutable audit already exists.")

    cv2.setNumThreads(1)
    receipt = json.loads(args.public_receipt.read_text(encoding="utf8"))
    if (receipt.get("images_processed") != EXPECTED_IMAGES or
        not receipt.get("full_180_images") or
        receipt.get("test_labels_read") is not False):
        raise ValueError("Unverified or unexpectedly labelled public replay.")
    item_by_stem = {Path(record["file"]).stem: record
                    for record in receipt.get("image_sha256", [])}
    if len(item_by_stem) != EXPECTED_IMAGES:
        raise ValueError("Duplicate or missing filenames in public replay receipt.")
    public_files = {p.stem: p for p in args.public_maps.glob("*.png")}
    reference_files = {p.stem: p for p in args.reference_maps.glob("*.png")}
    if set(public_files) != set(reference_files) or set(public_files) != set(item_by_stem):
        raise ValueError("All 180 maps must have one-to-one filenames in both sets.")

    pairs = []
    total_pixels = 0
    different_pixels = 0
    absolute_error_sum = 0
    worst_difference = 0
    for index, stem in enumerate(sorted(public_files), 1):
        pub_path, ref_path = public_files[stem], reference_files[stem]
        pub_digest = file_sha256(pub_path)
        if pub_digest != item_by_stem[stem]["map_sha256"]:
            raise ValueError("Public receipt SHA mismatch: " + stem)
        public = cv2.imread(str(pub_path), cv2.IMREAD_UNCHANGED)
        reference = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
        if (public is None or reference is None or
            public.shape != EXPECTED_SHAPE or reference.shape != EXPECTED_SHAPE or
            public.dtype != np.uint8 or reference.dtype != np.uint8):
            raise ValueError("Non-native grayscale map encountered: " + stem)
        result = compare_arrays(public, reference)
        result["image_id"] = stem
        result["public_png_sha256"] = pub_digest
        result["reference_png_sha256"] = file_sha256(ref_path)
        pairs.append(result)
        total_pixels += result["pixels"]
        different_pixels += result["different_pixels"]
        absolute_error_sum += result["absolute_difference_sum"]
        worst_difference = max(worst_difference, result["max_absolute_byte_difference"])
        if index % 30 == 0:
            print("P78_COMPARE", index, EXPECTED_IMAGES, flush=True)

    equal_fraction = 1 - different_pixels / total_pixels
    accepted = (equal_fraction >= MIN_EQUAL_FRACTION and
                worst_difference <= MAX_ALLOWED_BYTE_DIFFERENCE)
    payload = {
        "protocol": "P78_F1_FULL_180_STAGE_ONE_BYTE_PARITY",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS_STAGE_ONE_NEAR_PARITY" if accepted else "INVESTIGATE_MAP_DIFFERENCES",
        "input_public_receipt_sha256": file_sha256(args.public_receipt),
        "audit_script_sha256": file_sha256(Path(__file__)),
        "all_images": EXPECTED_IMAGES,
        "total_pixels": total_pixels,
        "different_pixels": different_pixels,
        "exactly_equal_fraction": equal_fraction,
        "absolute_byte_error_sum": absolute_error_sum,
        "mean_absolute_byte_error": absolute_error_sum / total_pixels,
        "max_absolute_byte_error": worst_difference,
        "preregistered_gate": {
            "min_exact_equal_fraction": MIN_EQUAL_FRACTION,
            "max_absolute_byte_error": MAX_ALLOWED_BYTE_DIFFERENCE,
        },
        "top_five_differences": sorted(
            [{"image_id": p["image_id"],
              "different_pixels": p["different_pixels"],
              "max_absolute_byte_error": p["max_absolute_byte_difference"]}
             for p in pairs],
            key=lambda p: -p["different_pixels"])[:5],
        "by_image": pairs,
        "uses_ground_truth_labels": False,
        "official_kaggle_score": None,
        "accepted_final_csv_reproduced": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("x", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print("P78_RESULT", payload["status"], "equal_fraction", equal_fraction,
          "different_pixels", different_pixels,
          "max_difference", worst_difference, flush=True)


if __name__ == "__main__":
    main()
