#!/usr/bin/env python3
"""P109: preregistered, CPU-only topology audit of two frozen COCO-RLE predictions.

The audit measures replay stability, not prediction accuracy. It never reads
image pixels, annotation labels, training manifests or Kaggle credentials.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as coco
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment

HEIGHT = WIDTH = 2048
IMAGE_PIXELS = HEIGHT * WIDTH
EXPECTED = {
    "accepted": "24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60",
    "replay": "16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3",
}
PREREG_COMMIT = "413248db9a12534399d12afbd68b7599d9909f0d"
EIGHT_NEIGHBOURS = np.ones((3, 3), dtype=bool)


def sha256_file(path: Path) -> str:
    """Hash a complete file without retaining its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_intervals(encoded: str) -> tuple[tuple[tuple[int, int], ...], int]:
    """Decode compressed COCO RLE into intervals in Fortran pixel order."""
    counts: list[int] = []
    pos = 0
    while pos < len(encoded):
        value = shift = 0
        while True:
            if pos >= len(encoded):
                raise ValueError("Truncated compressed COCO RLE")
            digit = ord(encoded[pos]) - 48
            pos += 1
            value |= (digit & 31) << shift
            shift += 5
            if not digit & 32:
                if digit & 16:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError("Negative RLE run length")
        counts.append(value)
    if sum(counts) != IMAGE_PIXELS:
        raise ValueError(f"RLE pixels do not equal {IMAGE_PIXELS}")
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for index, length in enumerate(counts):
        if index & 1 and length:
            intervals.append((cursor, cursor + length))
        cursor += length
    area = sum(end - start for start, end in intervals)
    if not area:
        raise ValueError("Empty instance mask")
    return tuple(intervals), area


def overlap(a: tuple, b: tuple) -> int:
    """Count intersection pixels using independent interval arithmetic."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def read_predictions(path: Path) -> dict[str, list[dict]]:
    """Validate exact two-column submission schema and unique instance IDs."""
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Invalid columns: {reader.fieldnames}")
        for source in reader:
            identifier = source["filament_id"]
            encoded = source["segmentation_rle"]
            if not identifier or identifier in ids or "_" not in identifier:
                raise ValueError(f"Duplicate/malformed filament_id: {identifier}")
            if not encoded or not encoded.isascii():
                raise ValueError(f"Invalid COCO-RLE for {identifier}")
            ids.add(identifier)
            intervals, area = decode_intervals(encoded)
            mask = {"size": [HEIGHT, WIDTH], "counts": encoded.encode("ascii")}
            if int(coco.area(mask)) != area:
                raise AssertionError(f"Independent mask.area disagrees: {identifier}")
            groups[identifier.rsplit("_", 1)[0]].append(
                {"id": identifier, "encoded": encoded, "rle": mask,
                 "intervals": intervals, "area": area}
            )
    if len(ids) != 1241 or len(groups) != 180:
        raise ValueError(f"Unexpected rows/images: {len(ids)}/{len(groups)}")
    return groups


def components(binary: np.ndarray) -> int:
    """Count 8-connected foreground regions, excluding the background."""
    if binary.dtype != np.uint8 or binary.ndim != 2:
        raise ValueError("Connected-component input must be a 2D uint8 mask")
    return int(label(binary, structure=EIGHT_NEIGHBOURS)[1])


def synthetic_tests() -> dict:
    """Verify a changed boundary pixel and a newly isolated component."""
    initial = np.zeros((10, 10), dtype=np.uint8)
    initial[4, 4:6] = 1
    boundary = initial.copy()
    boundary[4, 6] = 1
    isolated = initial.copy()
    isolated[8, 8] = 1
    assert components(initial) == components(boundary) == 1
    assert components(isolated) == 2
    assert int(np.count_nonzero(initial ^ boundary)) == 1
    assert int(np.count_nonzero(initial ^ isolated)) == 1
    return {"boundary_pixel": "PASS", "isolated_component": "PASS"}


def summarize(values: list[int]) -> dict:
    """Return exact descriptive statistics without fitting a model."""
    return {"count": len(values), "sum": sum(values),
            "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None}


def audit(accepted: Path, replay: Path, max_seconds: float) -> dict:
    """Perform frozen bijective matching and inspect only changed masks."""
    started = time.monotonic()
    synthetic = synthetic_tests()
    verified_hashes = {}
    for name, path in (("accepted", accepted), ("replay", replay)):
        actual = sha256_file(path)
        if actual != EXPECTED[name]:
            raise ValueError(f"{name} SHA-256 mismatch: {actual}")
        verified_hashes[name] = actual

    reference = read_predictions(accepted)
    candidate = read_predictions(replay)
    if set(reference) != set(candidate):
        raise ValueError("Image sets differ")
    if {x["id"] for group in reference.values() for x in group} != {
        x["id"] for group in candidate.values() for x in group
    }:
        raise ValueError("Filament ID sets differ")
    expected_exact = 1108
    expected_changed = 133
    expected_pixels = 270
    total_exact = total_changed = total_pixels = 0
    changed_bboxes = changed_components = changed_areas = 0
    reordered_assignments = 0
    pair_differences: list[int] = []
    changed_pair_data: list[dict] = []
    per_image: list[dict] = []
    for image_id in sorted(reference):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("Preregistered runtime cap exceeded")
        ref, cur = reference[image_id], candidate[image_id]
        if len(ref) != len(cur):
            raise ValueError(f"Instance count mismatch: {image_id}")
        scores = []
        for current in cur:
            row = []
            for original in ref:
                intersect = overlap(current["intervals"], original["intervals"])
                union = current["area"] + original["area"] - intersect
                row.append(1.0 - intersect / union)
            scores.append(row)
        rows, columns = linear_sum_assignment(np.asarray(scores))
        image_changed = image_pixels = image_bbox = image_components = 0
        for i, j in zip(rows, columns):
            a, b = cur[int(i)], ref[int(j)]
            if a["id"] != b["id"]:
                reordered_assignments += 1
            intersect = overlap(a["intervals"], b["intervals"])
            symmetric = a["area"] + b["area"] - 2 * intersect
            if symmetric == 0:
                total_exact += 1
                continue
            total_changed += 1
            total_pixels += symmetric
            image_changed += 1
            image_pixels += symmetric
            pair_differences.append(symmetric)

            # Independent COCO primitives must agree with interval counting.
            merged = coco.merge([a["rle"], b["rle"]], intersect=True)
            official_intersect = int(coco.area(merged))
            if official_intersect != intersect:
                raise AssertionError(f"COCO/interval disagreement: {image_id}")
            left = np.asarray(coco.decode(a["rle"]), dtype=np.uint8)
            right = np.asarray(coco.decode(b["rle"]), dtype=np.uint8)
            if left.shape != (HEIGHT, WIDTH) or right.shape != (HEIGHT, WIDTH):
                raise AssertionError("Wrong decoded mask shape")
            if int(np.count_nonzero(left != right)) != symmetric:
                raise AssertionError(f"Pixel difference disagreement: {image_id}")

            bbox_a = [int(v) for v in coco.toBbox(a["rle"]).tolist()]
            bbox_b = [int(v) for v in coco.toBbox(b["rle"]).tolist()]
            bbox_changed = bbox_a != bbox_b
            ncomp_a, ncomp_b = components(left), components(right)
            topology_changed = ncomp_a != ncomp_b
            area_delta = a["area"] - b["area"]
            if bbox_changed:
                changed_bboxes += 1
                image_bbox += 1
            if topology_changed:
                changed_components += 1
                image_components += 1
            if area_delta:
                changed_areas += 1
            changed_pair_data.append({
                "image_id": image_id, "candidate_id": a["id"],
                "reference_id": b["id"], "symmetric_difference_pixels": symmetric,
                "area_delta": area_delta, "bbox_changed": bbox_changed,
                "candidate_bbox": bbox_a, "accepted_bbox": bbox_b,
                "component_count_changed": topology_changed,
                "candidate_components": ncomp_a, "accepted_components": ncomp_b,
            })
        per_image.append({
            "image_id": image_id, "instances": len(ref),
            "changed_pairs": image_changed, "changed_pixels": image_pixels,
            "changed_bounding_boxes": image_bbox,
            "changed_component_counts": image_components,
        })
    if (total_exact, total_changed, total_pixels) != (
        expected_exact, expected_changed, expected_pixels
    ):
        raise AssertionError(
            f"P107 control differs: exact={total_exact}, changed={total_changed}, "
            f"pixels={total_pixels}"
        )
    return {
        "protocol_commit": PREREG_COMMIT, "status": "COMPLETE",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_sha256": verified_hashes,
        "prereg_synthetic_tests": synthetic,
        "images": len(reference), "matched_masks": total_exact + total_changed,
        "identical_masks": total_exact, "changed_masks": total_changed,
        "symmetric_difference_pixels": total_pixels,
        "reordered_assignments": reordered_assignments,
        "changed_bounding_boxes": changed_bboxes,
        "changed_component_counts": changed_components,
        "changed_areas": changed_areas,
        "changed_mask_difference_distribution": summarize(pair_differences),
        "images_with_changed_masks": sum(x["changed_pairs"] > 0 for x in per_image),
        "hypothesis_falsified": bool(changed_bboxes or changed_components),
        "changed_pairs_detail": changed_pair_data, "per_image": per_image,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "scope": "Prediction-to-prediction topology only; no labels, PQ or leaderboard inference",
        "promotion_allowed": False, "kaggle_submission_made": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=850.0)
    args = parser.parse_args()
    if args.max_seconds > 900 or args.max_seconds <= 0:
        raise ValueError("Runtime budget must be in (0,900] seconds")
    result = audit(args.accepted, args.replay, args.max_seconds)
    args.out.mkdir(parents=True, exist_ok=True)
    final = args.out / "P109_FINAL_RECEIPT.json"
    temporary = args.out / "P109_FINAL_RECEIPT.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    if temporary.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Output budget exceeded")
    temporary.replace(final)
    print(json.dumps({
        "receipt": str(final), "receipt_sha256": sha256_file(final),
        "images": result["images"], "matched_masks": result["matched_masks"],
        "identical_masks": result["identical_masks"],
        "changed_masks": result["changed_masks"],
        "symmetric_difference_pixels": result["symmetric_difference_pixels"],
        "changed_bounding_boxes": result["changed_bounding_boxes"],
        "changed_component_counts": result["changed_component_counts"],
        "hypothesis_falsified": result["hypothesis_falsified"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
