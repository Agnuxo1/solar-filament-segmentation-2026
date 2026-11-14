#!/usr/bin/env python3
"""P113: independently validate the compressed COCO RLE streams of two frozen CSVs.

This deterministic integrity audit reads JPEG headers and prediction encodings.
It does not use labels, evaluate PQ, train models, or contact Kaggle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from PIL import Image
from pycocotools import mask as coco_mask

ROOT = Path("/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112")
OUT = ROOT / "p113_rle_cross_decoder_20261114"
JPEG_DIR = Path(
    "/home/ai/pc/d/PROJECTS/196_NeuroPixel/kaggle/filament/data/"
    "MAGFiLO_1.0_Kaggle_2026/test/test_images"
)
SOURCES = {
    "accepted_h9b": (
        ROOT / "p74_accepted_artifact_audit/accepted_h9b_nosite.csv",
        "24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60",
    ),
    "p106_public_rebuild": (
        ROOT / "p106_public_clean_6843362_20261113/"
        "inference_cuda/submission_public_rebuilt.csv",
        "16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3",
    ),
}


def sha256_file(path: Path) -> str:
    """Hash file bytes in bounded chunks, without modifying the original."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_coco_counts_ascii(value: str) -> list[int]:
    """Independently decode COCO's signed base-32/differential run-length stream."""
    if not value:
        raise ValueError("Empty compressed COCO RLE string")
    counts: list[int] = []
    position = 0
    while position < len(value):
        number = 0
        shift = 0
        while True:
            if position >= len(value):
                raise ValueError("Truncated COCO variable-length integer")
            digit = ord(value[position]) - 48
            position += 1
            if not 0 <= digit <= 63:
                raise ValueError("Non-COCO ASCII character")
            number |= (digit & 31) << shift
            shift += 5
            if shift > 70:
                raise ValueError("Unreasonably long COCO integer")
            if digit & 32:
                continue
            if digit & 16:
                number |= -(1 << shift)
            break
        if len(counts) > 2:
            number += counts[-2]
        if number < 0:
            raise ValueError("Negative decoded run length")
        counts.append(number)
    if not counts:
        raise ValueError("No RLE counts")
    return counts


def image_inventory() -> dict[str, tuple[int, int]]:
    """Read only JPEG headers to establish native mask dimensions."""
    images: dict[str, tuple[int, int]] = {}
    for path in sorted(JPEG_DIR.glob("*.jpeg")):
        if path.stem in images:
            raise ValueError("Duplicate JPEG stem: " + path.stem)
        with Image.open(path) as photo:
            images[path.stem] = (photo.height, photo.width)
    if len(images) != 180:
        raise ValueError("Expected 180 original JPEGs, found " + str(len(images)))
    return images


def validate_csv(
    path: Path, expected_hash: str, inventory: dict[str, tuple[int, int]]
) -> tuple[dict[str, tuple[int, str]], dict]:
    """Validate every original row by two independent COCO area decoders."""
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError("Frozen source SHA-256 mismatch: " + str(path))
    rows: dict[str, tuple[int, str]] = {}
    image_counts: Counter[str] = Counter()
    areas: list[int] = []
    total_integer_runs = 0
    empty_masks = 0
    max_run = 0

    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Unexpected CSV header: " + str(reader.fieldnames))
        for row_index, row in enumerate(reader, start=2):
            filament_id = row["filament_id"]
            stem, separator, suffix = filament_id.rpartition("_")
            if not separator or not suffix.isdecimal() or int(suffix) < 1:
                raise ValueError("Invalid filament ID at line " + str(row_index))
            if filament_id in rows:
                raise ValueError("Duplicate filament ID: " + filament_id)
            if stem not in inventory:
                raise ValueError("Unknown original image stem: " + stem)
            height, width = inventory[stem]
            rle = row["segmentation_rle"]
            runs = decode_coco_counts_ascii(rle)
            if sum(runs) != height * width:
                raise ValueError("Incorrect RLE pixel total for " + filament_id)
            independent_area = sum(runs[1::2])
            library_area = int(
                coco_mask.area(
                    {"size": [height, width], "counts": rle.encode("ascii")}
                )
            )
            if independent_area != library_area:
                raise ValueError("Independent area mismatch for " + filament_id)
            if independent_area > height * width:
                raise ValueError("Area exceeds image for " + filament_id)
            rows[filament_id] = (independent_area, rle)
            image_counts[stem] += 1
            areas.append(independent_area)
            total_integer_runs += len(runs)
            max_run = max(max_run, max(runs))
            empty_masks += independent_area == 0

    if len(rows) != 1241:
        raise ValueError("Expected 1,241 instances, found " + str(len(rows)))
    if set(image_counts) != set(inventory):
        raise ValueError("CSV does not cover the complete 180-JPEG inventory")
    summary = {
        "path": str(path),
        "sha256": actual_hash,
        "rows_validated": len(rows),
        "images_covered": len(image_counts),
        "total_decoded_integer_runs": total_integer_runs,
        "all_area_checks_agree": True,
        "total_foreground_pixel_instances": sum(areas),
        "smallest_instance_area": min(areas),
        "largest_instance_area": max(areas),
        "empty_masks": empty_masks,
        "largest_run": max_run,
        "image_instance_count_min": min(image_counts.values()),
        "image_instance_count_max": max(image_counts.values()),
    }
    return rows, summary


def main() -> None:
    """Run the locked audit and persist a receipt only after every check passes."""
    start = time.monotonic()
    images = image_inventory()
    validated = {}
    summaries = {}
    for name, (path, expected_hash) in SOURCES.items():
        validated[name], summaries[name] = validate_csv(path, expected_hash, images)
    accepted = validated["accepted_h9b"]
    replay = validated["p106_public_rebuild"]
    if set(accepted) != set(replay):
        raise ValueError("H9b and P106 have different per-instance ID sets")
    by_image_h9b = Counter(key.rpartition("_")[0] for key in accepted)
    by_image_p106 = Counter(key.rpartition("_")[0] for key in replay)
    if by_image_h9b != by_image_p106:
        raise ValueError("H9b and P106 image instance counts differ")

    differing_rle_ids = [
        key for key in accepted if accepted[key][1] != replay[key][1]
    ]
    differing_area_ids = [
        key for key in accepted if accepted[key][0] != replay[key][0]
    ]
    absolute_area_delta = sum(
        abs(accepted[key][0] - replay[key][0]) for key in accepted
    )
    prereg = OUT / "P113_PREREG.md"
    receipt = {
        "experiment": "P113",
        "audit_type": "Independent compressed COCO-RLE geometry and area validation",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256_file(prereg),
        "source_script_sha256": sha256_file(Path(__file__)),
        "test_labels_used": False,
        "training_performed": False,
        "gpu_used": False,
        "kaggle_submission_made": False,
        "pq_measured": False,
        "final_h9b_selection_unchanged": True,
        "original_jpeg_count": len(images),
        "jpeg_dimensions": sorted({tuple(dim) for dim in images.values()}),
        "inputs": summaries,
        "cross_csv": {
            "same_filament_ids": True,
            "same_instance_count_per_image": True,
            "identical_rles_at_same_id": len(accepted) - len(differing_rle_ids),
            "different_rles_at_same_id": len(differing_rle_ids),
            "different_areas_at_same_id": len(differing_area_ids),
            "sum_absolute_area_differences_at_same_id": absolute_area_delta,
            "scope": (
                "Pure format/area validation of predictions. Same-ID statistics "
                "can include instance reordering and do not measure geometry "
                "matching, official PQ, or accuracy."
            ),
        },
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "status": "PASS_ALL_ROWS_BOTH_INDEPENDENT_AREA_DECODERS",
    }
    if receipt["elapsed_seconds"] > 180:
        raise RuntimeError("Exceeded 180-second preregistered CPU budget")
    receipt_path = OUT / "P113_FINAL_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    print("RECEIPT_SHA256", sha256_file(receipt_path))


if __name__ == "__main__":
    main()

