"""Validate a submission CSV before it can be sent to Kaggle."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

EXPECTED_SHAPE = (2048, 2048)


def audit_submission(submission: Path, test_images: Path) -> dict[str, Any]:
    expected = {p.stem for p in test_images.glob("*.jpeg")}
    rows: list[tuple[str, str]] = []
    with submission.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("CSV must contain exactly filament_id,segmentation_rle")
        for row in reader:
            filament_id = (row["filament_id"] or "").strip()
            rle = row["segmentation_rle"] or ""
            if not filament_id or "_" not in filament_id or not rle:
                raise ValueError(f"invalid row for filament_id={filament_id!r}")
            rows.append((filament_id, rle))

    ids = [item[0] for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("filament_id values must be unique")
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for filament_id, rle in rows:
        physical_id = filament_id.split("_", 1)[0]
        if physical_id not in expected:
            raise ValueError(f"prediction references non-test image: {physical_id}")
        decoded = mask_utils.decode({"size": list(EXPECTED_SHAPE), "counts": rle})
        if decoded.shape != EXPECTED_SHAPE:
            raise ValueError(f"wrong mask shape for {filament_id}: {decoded.shape}")
        if int(decoded.sum()) <= 0:
            raise ValueError(f"empty mask for {filament_id}")
        grouped[physical_id].append(decoded.astype(np.uint8, copy=False))

    overlap_pixels = 0
    for physical_id, masks in grouped.items():
        stack = np.stack(masks, axis=2)
        overlap_pixels += int((stack.sum(axis=2) > 1).sum())
    if overlap_pixels:
        raise ValueError(f"submission masks overlap at {overlap_pixels} pixels")

    return {
        "rows": len(rows),
        "test_images": len(expected),
        "images_with_predictions": len(grouped),
        "images_without_predictions": len(expected - set(grouped)),
        "overlap_pixels": overlap_pixels,
        "shape": list(EXPECTED_SHAPE),
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--test-images", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_submission(args.submission, args.test_images), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
