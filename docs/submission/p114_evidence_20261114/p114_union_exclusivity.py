#!/usr/bin/env python3
"""P114: independent image-level COCO-RLE mask union/exclusivity audit.

Read only frozen predictions and JPEG headers. Never use labels, GPU or a
competition endpoint. The pure-Python union algorithm is checked against
pycocotools for every image to detect serialization/implementation mistakes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from PIL import Image
from pycocotools import mask as coco_mask

ROOT = Path("/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112")
OUTPUT = ROOT / "p114_union_exclusivity_20261114"
JPEG_DIR = Path("/home/ai/pc/d/PROJECTS/196_NeuroPixel/kaggle/filament/data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
SOURCES = {
    "h9b": (
        ROOT / "p74_accepted_artifact_audit/accepted_h9b_nosite.csv",
        "24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60",
    ),
    "p106": (
        ROOT / "p106_public_clean_6843362_20261113/inference_cuda/submission_public_rebuilt.csv",
        "16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3",
    ),
}
MAX_SECONDS = 240


def sha256_file(path: Path) -> str:
    """Stream the full file into SHA-256 without modifying its bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded_counts(value: str) -> list[int]:
    """Decode compressed COCO counts in original Fortran-order linear indexing."""
    if not value:
        raise ValueError("Empty COCO RLE")
    numbers: list[int] = []
    offset = 0
    while offset < len(value):
        number = 0
        shift = 0
        while True:
            if offset >= len(value):
                raise ValueError("Truncated compressed COCO RLE integer")
            c = ord(value[offset]) - 48
            offset += 1
            if not (0 <= c <= 63):
                raise ValueError("Invalid COCO RLE ASCII")
            number |= (c & 31) << shift
            shift += 5
            if shift > 70:
                raise ValueError("Excessive integer length")
            if c & 32:
                continue
            if c & 16:
                number |= -(1 << shift)
            break
        if len(numbers) > 2:
            number += numbers[-2]
        if number < 0:
            raise ValueError("Negative COCO RLE run length")
        numbers.append(number)
    return numbers


def intervals_from_counts(counts: list[int], total_pixels: int) -> list[tuple[int, int]]:
    """Emit nonempty foreground intervals from alternating background/foreground runs."""
    intervals: list[tuple[int, int]] = []
    pos = 0
    for index, count in enumerate(counts):
        if index & 1 and count:
            intervals.append((pos, pos + count))
        pos += count
    if pos != total_pixels:
        raise ValueError(f"RLE dimensions mismatched: {pos} != {total_pixels}")
    return intervals


def union_intervals(all_intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Coalesce overlapping or adjacent 1D foreground spans into a pixel union."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(all_intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        elif end > merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
    return merged


def interval_length(spans: list[tuple[int, int]]) -> int:
    """Count occupied integer pixels in disjoint intervals."""
    return sum(end - start for start, end in spans)


def intersection_length(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Count pixels present in both disjoint interval unions."""
    i = j = shared = 0
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        shared += max(0, min(a_end, b_end) - max(a_start, b_start))
        if a_end <= b_end:
            i += 1
        else:
            j += 1
    return shared


def jpeg_inventory() -> dict[str, tuple[int, int]]:
    """Use original JPEG file headers exclusively to obtain native mask size."""
    dimensions = {}
    for path in sorted(JPEG_DIR.glob("*.jpeg")):
        if path.stem in dimensions:
            raise ValueError("Duplicate JPEG filename stem")
        with Image.open(path) as image:
            dimensions[path.stem] = (image.height, image.width)
    if len(dimensions) != 180:
        raise ValueError(f"Expected 180 JPEG files, got {len(dimensions)}")
    return dimensions


def load_frozen_csv(
    filename: Path, expected_sha256: str, dimensions: dict[str, tuple[int, int]]
) -> dict[str, dict]:
    """Hash, parse and audit every instance without a single decoded image array."""
    if sha256_file(filename) != expected_sha256:
        raise ValueError("Frozen CSV hash mismatch: " + str(filename))
    groups: dict[str, dict] = defaultdict(
        lambda: {"ids": set(), "intervals": [], "rles": [], "sum": 0, "n": 0}
    )
    global_ids: set[str] = set()
    with filename.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")
        for row in reader:
            filament_id = row["filament_id"]
            stem, sep, suffix = filament_id.rpartition("_")
            if not sep or not suffix.isdecimal() or int(suffix) < 1:
                raise ValueError("Invalid instance identifier")
            if filament_id in global_ids:
                raise ValueError("Duplicate instance identifier")
            global_ids.add(filament_id)
            if stem not in dimensions:
                raise ValueError("Instance image does not occur in original JPEG set")
            h, w = dimensions[stem]
            encoding = row["segmentation_rle"]
            counts = decoded_counts(encoding)
            spans = intervals_from_counts(counts, h * w)
            area = interval_length(spans)
            if area == 0:
                raise ValueError("Empty foreground instance")
            g = groups[stem]
            g["ids"].add(filament_id)
            g["intervals"].extend(spans)
            g["rles"].append({"size": [h, w], "counts": encoding.encode("ascii")})
            g["sum"] += area
            g["n"] += 1
    if len(global_ids) != 1241 or set(groups) != set(dimensions):
        raise ValueError("Wrong row count or incomplete image coverage")
    for stem in groups:
        g = groups[stem]
        g["union_spans"] = union_intervals(g["intervals"])
        g["pure_union"] = interval_length(g["union_spans"])
        pycoco_union = coco_mask.merge(g["rles"], intersect=False)
        g["coco_union"] = int(coco_mask.area(pycoco_union))
        if g["pure_union"] != g["coco_union"]:
            raise AssertionError("Independent union-area disagreement: " + stem)
        g["overlap_excess"] = g["sum"] - g["pure_union"]
        if g["overlap_excess"] < 0:
            raise AssertionError("Union area exceeds sum of instance areas")
        del g["intervals"], g["rles"]
    return groups


def main() -> None:
    """Run the frozen P114 analysis and emit deterministic per-image evidence."""
    start = time.monotonic()
    dims = jpeg_inventory()
    audits = {
        name: load_frozen_csv(path, sha, dims)
        for name, (path, sha) in SOURCES.items()
    }
    a = audits["h9b"]
    b = audits["p106"]
    if set(a) != set(b):
        raise AssertionError("Mismatched JPEG inventories")
    image_records = []
    for stem in sorted(dims):
        aa, bb = a[stem], b[stem]
        if aa["ids"] != bb["ids"] or aa["n"] != bb["n"]:
            raise AssertionError("Mismatched instance identifier sets")
        shared = intersection_length(aa["union_spans"], bb["union_spans"])
        symmetric = aa["pure_union"] + bb["pure_union"] - 2 * shared
        if symmetric < 0:
            raise AssertionError("Negative set symmetric difference")
        image_records.append({
            "image_stem": stem, "instances": aa["n"],
            "h9b_instance_sum": aa["sum"], "h9b_union_area": aa["pure_union"],
            "h9b_overlap_excess": aa["overlap_excess"],
            "p106_instance_sum": bb["sum"], "p106_union_area": bb["pure_union"],
            "p106_overlap_excess": bb["overlap_excess"],
            "union_symmetric_difference_pixels": symmetric,
        })
    elapsed = time.monotonic() - start
    if elapsed > MAX_SECONDS:
        raise RuntimeError(f"CPU budget exceeded: {elapsed:.2f} seconds")
    summary = {}
    for name, groups in audits.items():
        summary[name] = {
            "rows": sum(group["n"] for group in groups.values()),
            "images": len(groups),
            "instance_area_sum": sum(group["sum"] for group in groups.values()),
            "union_area_sum": sum(group["pure_union"] for group in groups.values()),
            "overlap_excess_sum": sum(group["overlap_excess"] for group in groups.values()),
            "images_with_overlaps": sum(group["overlap_excess"] > 0 for group in groups.values()),
            "max_image_overlap_excess": max(group["overlap_excess"] for group in groups.values()),
            "all_180_union_areas_agree_with_pycocotools": True,
        }
    changed_union_pixels = sum(r["union_symmetric_difference_pixels"] for r in image_records)
    receipt = {
        "experiment": "P114",
        "title": "Independent mask-union and pixel exclusivity audit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv_sha256": {name: sha for name, (_, sha) in SOURCES.items()},
        "preregistration_sha256": sha256_file(OUTPUT / "P114_PREREG.md"),
        "source_script_sha256": sha256_file(Path(__file__)),
        "images": len(image_records),
        "summaries": summary,
        "cross_csv": {
            "union_symmetric_difference_pixels": changed_union_pixels,
            "images_with_different_foreground_union": sum(
                r["union_symmetric_difference_pixels"] > 0 for r in image_records
            ),
            "max_image_union_symmetric_difference": max(
                r["union_symmetric_difference_pixels"] for r in image_records
            ),
            "net_union_area_change_p106_minus_h9b":
                summary["p106"]["union_area_sum"] - summary["h9b"]["union_area_sum"],
        },
        "scope": "Predicted per-image pixel-union integrity only; no labels, PQ, or accuracy.",
        "seed": "not applicable: deterministic",
        "elapsed_seconds": round(elapsed, 3),
        "gpu_used": False, "test_annotations_used": False,
        "training_performed": False, "kaggle_submission_made": False,
        "final_h9b_selection_unchanged": True,
        "status": "PASS_ALL_CROSSCHECKS" if all(
            s["images_with_overlaps"] == 0 for s in summary.values()
        ) else "COMPLETE_WITH_EXCLUSIVITY_VIOLATIONS",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = OUTPUT / "P114_BY_IMAGE.csv"
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_records[0]))
        writer.writeheader()
        writer.writerows(image_records)
    receipt["per_image_csv_sha256"] = sha256_file(table)
    out = OUTPUT / "P114_FINAL_RECEIPT.json"
    out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    print("RECEIPT_SHA256", sha256_file(out))


if __name__ == "__main__":
    main()
