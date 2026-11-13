#!/usr/bin/env python3
"""P107: independently audit P106 mask correspondence without competition labels.

This is a diagnostic comparison of participant-generated CSV predictions only.
It does not calculate Panoptic Quality, estimate leaderboard performance, or
modify checkpoints, thresholds, or any submission artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from scipy.optimize import linear_sum_assignment


def sha256_file(path: Path) -> str:
    """Stream a file into SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_instances(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    """Load the two-column competition CSV and group instances by source image."""
    csv.field_size_limit(50_000_000)
    images: dict[str, list[dict[str, str]]] = defaultdict(list)
    encountered: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Incorrect CSV schema: {reader.fieldnames}")
        for item in reader:
            filament_id = item["filament_id"]
            match = re.fullmatch(r"(.+)_([1-9][0-9]*)", filament_id)
            if not match or filament_id in encountered:
                raise ValueError(f"Invalid or duplicated instance: {filament_id}")
            if not item["segmentation_rle"]:
                raise ValueError(f"Empty encoded mask: {filament_id}")
            encountered.add(filament_id)
            images[match.group(1)].append(item)
    return dict(images)


def image_dimensions(image_directory: Path, image_id: str) -> tuple[int, int]:
    """Read only original JPEG dimensions, never an annotation or target mask."""
    matches = [
        path for suffix in (".jpeg", ".jpg", ".JPEG", ".JPG")
        if (path := image_directory / f"{image_id}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source JPEG for {image_id}; found {matches}")
    with Image.open(matches[0]) as image:
        width, height = image.size
    return height, width


def convert_rles(rows: list[dict[str, str]], shape: tuple[int, int]) -> list[dict]:
    """Prepare valid COCO compressed RLE objects for independent geometry checks."""
    return [{"counts": item["segmentation_rle"].encode("ascii"), "size": list(shape)}
            for item in rows]


def analyze_pair(reference: Path, candidate: Path, image_directory: Path) -> dict:
    """Match candidate masks to reference within each image by optimal IoU assignment."""
    ref_images = read_instances(reference)
    new_images = read_instances(candidate)
    if set(ref_images) != set(new_images):
        raise AssertionError("Image inventories differ; cannot claim full correspondence.")
    totals = {
        "images": 0, "instances": 0, "same_position_exact": 0,
        "same_position_changed": 0, "matched_exact": 0, "matched_changed": 0,
        "reordered_assignments": 0, "reordered_exact": 0,
        "images_with_exact_mask_set": 0, "images_with_exact_sequence": 0,
        "changed_pair_pixels": 0, "largest_pair_pixel_difference": 0,
    }
    changed_pixels: list[int] = []
    changed_ious: list[float] = []
    image_reports = []
    for image_id in sorted(ref_images):
        originals = ref_images[image_id]
        rebuilt = new_images[image_id]
        if len(originals) != len(rebuilt):
            raise AssertionError(f"Instance count differs in {image_id}")
        height, width = image_dimensions(image_directory, image_id)
        ref_rles = convert_rles(originals, (height, width))
        new_rles = convert_rles(rebuilt, (height, width))
        count = len(originals)
        matrix = coco_mask.iou(new_rles, ref_rles, [0] * count)
        if matrix.shape != (count, count):
            raise AssertionError(f"Unexpected IoU matrix in {image_id}: {matrix.shape}")
        candidate_indices, reference_indices = linear_sum_assignment(-matrix)
        if sorted(candidate_indices.tolist()) != list(range(count)):
            raise AssertionError("Non-bijective candidate assignment")
        if sorted(reference_indices.tolist()) != list(range(count)):
            raise AssertionError("Non-bijective reference assignment")
        masks_exact = 0
        reorderings = 0
        image_pixel_difference = 0
        for candidate_index, reference_index in zip(candidate_indices, reference_indices):
            is_same_position = int(candidate_index) == int(reference_index)
            is_same_rle = (rebuilt[int(candidate_index)]["segmentation_rle"] ==
                           originals[int(reference_index)]["segmentation_rle"])
            if not is_same_position:
                reorderings += 1
                totals["reordered_assignments"] += 1
                if is_same_rle:
                    totals["reordered_exact"] += 1
            if is_same_rle:
                masks_exact += 1
                totals["matched_exact"] += 1
                if abs(float(matrix[candidate_index, reference_index]) - 1.0) > 1e-12:
                    raise AssertionError("Exact RLE unexpectedly has IoU less than one")
                continue
            candidate_mask = coco_mask.decode(new_rles[int(candidate_index)]).astype(bool)
            original_mask = coco_mask.decode(ref_rles[int(reference_index)]).astype(bool)
            xor_pixels = int(np.count_nonzero(candidate_mask != original_mask))
            if xor_pixels == 0:
                raise AssertionError("Different RLE strings decode to identical masks; update accounting")
            intersection = int(np.count_nonzero(candidate_mask & original_mask))
            union = int(np.count_nonzero(candidate_mask | original_mask))
            independent_iou = intersection / union if union else 1.0
            if abs(independent_iou - float(matrix[candidate_index, reference_index])) > 1e-9:
                raise AssertionError(f"COCO and NumPy IoU disagree in {image_id}")
            changed_pixels.append(xor_pixels)
            changed_ious.append(independent_iou)
            image_pixel_difference += xor_pixels
            totals["matched_changed"] += 1
            totals["changed_pair_pixels"] += xor_pixels
            totals["largest_pair_pixel_difference"] = max(
                totals["largest_pair_pixel_difference"], xor_pixels)
        same_pos = sum(
            a["segmentation_rle"] == b["segmentation_rle"]
            for a, b in zip(originals, rebuilt))
        totals["same_position_exact"] += same_pos
        totals["same_position_changed"] += count - same_pos
        totals["images"] += 1
        totals["instances"] += count
        if same_pos == count:
            totals["images_with_exact_sequence"] += 1
        if masks_exact == count:
            totals["images_with_exact_mask_set"] += 1
        image_reports.append({
            "image_id": image_id, "instances": count,
            "same_position_exact": same_pos,
            "matched_exact": masks_exact,
            "matched_changed": count - masks_exact,
            "reordered_assignments": reorderings,
            "symmetric_difference_pixels": image_pixel_difference,
        })
    if totals["same_position_exact"] + totals["same_position_changed"] != totals["instances"]:
        raise AssertionError("Row count reconciliation failed")
    if totals["matched_exact"] + totals["matched_changed"] != totals["instances"]:
        raise AssertionError("Optimal assignment reconciliation failed")
    totals["median_changed_pair_pixels"] = (
        float(np.median(changed_pixels)) if changed_pixels else 0.0)
    totals["min_changed_pair_iou"] = (
        float(min(changed_ious)) if changed_ious else 1.0)
    totals["median_changed_pair_iou"] = (
        float(np.median(changed_ious)) if changed_ious else 1.0)
    return {
        "reference_csv_sha256": sha256_file(reference),
        "candidate_csv_sha256": sha256_file(candidate),
        "summary": totals,
        "per_image": image_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--p103", type=Path, required=True)
    parser.add_argument("--p106", type=Path, required=True)
    parser.add_argument("--test-jpeg-dir", type=Path, required=True)
    parser.add_argument("--p106-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    receipt = json.loads(args.p106_receipt.read_text(encoding="utf-8"))
    if receipt["status"] != "P106_ALL_EIGHT_NOTEBOOK_CELLS_COMPLETED":
        raise AssertionError("P106 has not completed the eight notebook code cells")
    if len(receipt["code_cells"]) != 8 or receipt["test_annotations_used"] is not False:
        raise AssertionError("P106 notebook or no-test-label contract failed")
    if receipt["csv_sha256"] != sha256_file(args.p106):
        raise AssertionError("P106 CSV does not match its immutable execution receipt")
    audit = {
        "experiment": "P107",
        "audit_type": "read-only geometric correspondence audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": args.protocol_sha256,
        "p106_receipt_sha256": sha256_file(args.p106_receipt),
        "p106_executed_code_cells": len(receipt["code_cells"]),
        "p106_completed_utc": receipt["completed_utc"],
        "p106_commit": receipt["commit"],
        "labels_used": False,
        "pq_evaluation": False,
        "gpu_used": False,
        "comparisons": {
            "p106_vs_accepted": analyze_pair(args.accepted, args.p106, args.test_jpeg_dir),
            "p106_vs_p103": analyze_pair(args.p103, args.p106, args.test_jpeg_dir),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(args.output)
    for name, comparison in audit["comparisons"].items():
        print(name, json.dumps(comparison["summary"], sort_keys=True))
    print("P107_AUDIT_COMPLETE", args.output)


if __name__ == "__main__":
    main()
