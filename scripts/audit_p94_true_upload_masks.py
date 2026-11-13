#!/usr/bin/env python3
"""Audit genuine uploaded RLE masks against a reproducibility candidate.

This tool does not read reference annotations, optimize predictions or query
Kaggle. It operates only on two complete prediction CSV files.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import tempfile
from collections import Counter

import numpy as np
from pycocotools import mask as coco

ACCEPTED_SHA256 = "24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60"


def sha256(path: pathlib.Path) -> str:
    """Compute a streaming SHA-256 digest without loading the file into RAM."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: pathlib.Path) -> dict[str, str]:
    """Read a prediction CSV and reject malformed headers or repeated IDs."""
    result: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Unexpected prediction CSV columns")
        for row in reader:
            identifier = row["filament_id"]
            text = row["segmentation_rle"]
            if not identifier or not text or identifier in result:
                raise ValueError("Empty field or duplicate filament identifier")
            text.encode("ascii", errors="strict")
            result[identifier] = text
    if not result:
        raise ValueError("Empty prediction CSV")
    return result


def quantify(reference: pathlib.Path, candidate: pathlib.Path, size: int = 2048) -> dict:
    """Compare paired masks directly through COCO RLE without ground truth."""
    a, b = read_rows(reference), read_rows(candidate)
    if set(a) != set(b):
        raise ValueError("The prediction sets have different filament IDs")
    if size <= 0:
        raise ValueError("Invalid image size")
    overlap = []
    area_changes = []
    different_images = Counter()
    same_encoded = 0
    equal_masks_different_encoding = 0
    for identifier in sorted(a):
        ra = {"size": [size, size], "counts": a[identifier].encode("ascii")}
        rb = {"size": [size, size], "counts": b[identifier].encode("ascii")}
        ia = int(coco.area(ra))
        ib = int(coco.area(rb))
        if ia < 0 or ib < 0 or ia > size * size or ib > size * size:
            raise ValueError("Invalid decoded mask area")
        iou = float(coco.iou([ra], [rb], [0])[0, 0])
        if not (0.0 <= iou <= 1.0):
            raise ValueError("Invalid paired IoU")
        if a[identifier] == b[identifier]:
            if iou != 1.0:
                raise ValueError("Identical RLE strings decoded differently")
            same_encoded += 1
            continue
        overlap.append(iou)
        area_changes.append(ib - ia)
        different_images[identifier.rsplit("_", 1)[0]] += 1
        if iou == 1.0:
            equal_masks_different_encoding += 1
    percentiles = [0, 5, 25, 50, 75, 95, 100]
    result = {
        "reference_sha256": sha256(reference),
        "candidate_sha256": sha256(candidate),
        "rows": len(a),
        "exact_rle_rows": same_encoded,
        "different_rle_rows": len(overlap),
        "different_images": len(different_images),
        "different_encoding_same_binary_mask": equal_masks_different_encoding,
        "mismatched_mask_iou_percentiles": {
            str(p): (float(np.percentile(overlap, p)) if overlap else None)
            for p in percentiles
        },
        "mismatched_mask_iou_below_050": sum(x < 0.5 for x in overlap),
        "mismatched_mask_iou_below_090": sum(x < 0.9 for x in overlap),
        "mismatched_mask_iou_below_099": sum(x < 0.99 for x in overlap),
        "mismatched_mask_iou_below_0999": sum(x < 0.999 for x in overlap),
        "area_difference_sum": sum(area_changes),
        "area_difference_abs_sum": sum(abs(x) for x in area_changes),
        "max_area_difference": max((abs(x) for x in area_changes), default=0),
        "largest_image_row_difference_count": max(different_images.values(), default=0),
        "ground_truth_used": False,
        "kaggle_submissions": 0,
        "cuda_used": False,
    }
    return result


def self_test() -> None:
    """Verify complete identity, small geometry changes and duplicate guards."""
    def encoded(coords: list[tuple[int, int]]) -> str:
        x = np.zeros((5, 5), np.uint8, order="F")
        for i, j in coords:
            x[i, j] = 1
        return coco.encode(np.asfortranarray(x))["counts"].decode("ascii")

    def output(path: pathlib.Path, rows: list[tuple[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["filament_id", "segmentation_rle"])
            writer.writerows(rows)

    with tempfile.TemporaryDirectory() as directory:
        a, b = pathlib.Path(directory) / "a.csv", pathlib.Path(directory) / "b.csv"
        original = encoded([(0, 0), (1, 1)])
        changed = encoded([(0, 0), (1, 1), (3, 3)])
        output(a, [("x_1", original), ("x_2", original)])
        output(b, [("x_1", original), ("x_2", changed)])
        result = quantify(a, b, size=5)
        assert result["rows"] == 2
        assert result["exact_rle_rows"] == 1
        assert result["different_rle_rows"] == 1
        assert result["different_images"] == 1
        assert abs(result["mismatched_mask_iou_percentiles"]["50"] - 2 / 3) < 1e-8
        output(b, [("x_1", original), ("x_1", changed)])
        try:
            quantify(a, b, size=5)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate IDs not rejected")
    print("P94_SELF_TEST_PASS", flush=True)


def main() -> None:
    """Run a self-test or audited comparison, verifying the true-upload hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--reference", type=pathlib.Path)
    parser.add_argument("--candidate", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--require-upload-hash", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all([args.reference, args.candidate, args.output]):
        parser.error("--reference, --candidate and --output are required")
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing audit receipt")
    if args.require_upload_hash and sha256(args.reference) != ACCEPTED_SHA256:
        raise ValueError("Reference is NOT the genuine Kaggle upload")
    result = quantify(args.reference, args.candidate)
    result.update({
        "experiment": "P94_TRUE_UPLOAD_MASK_PARITY",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "comparison_scope": "prediction provenance only; not ground-truth PQ",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(args.output.suffix + ".partial")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
    pending.replace(args.output)
    print("P94_COMPLETE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
