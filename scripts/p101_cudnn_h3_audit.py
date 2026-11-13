#!/usr/bin/env python3
"""P101: independent CPU-only comparison of preregistered CuDNN variant maps.

Compares exact uint8 pixels; does not compute PQ, read segmentation labels,
make a leaderboard submission, or alter the public source checkpoint.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import p100_h3_repeatability_audit as audit

VARIANTS = ("benchmark_off", "deterministic")


def main() -> None:
    """Validate complete model/PNG hashes and write immutable comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    audit.self_test()
    if args.self_test:
        print("P101_AUDIT_SELF_TEST_PASS", flush=True)
        return
    images = sorted(args.images.glob("*.jpeg"))
    if len(images) != 180 or len({p.stem for p in images}) != 180:
        raise RuntimeError("Original test image inventory must contain 180 unique JPEGs")
    records = []
    for variant in VARIANTS:
        p = args.out / f"{variant}_receipt.json"
        run = json.loads(p.read_text(encoding="utf-8"))
        if run["status"] != "P101_VARIANT_COMPLETE" or run["variant"] != variant:
            raise RuntimeError("Incorrect variant receipt")
        if run["fold"] != "F1" or len(run["images"]) != 4:
            raise RuntimeError("Unexpected experiment inventory")
        for source, item in zip(images[:4], run["images"], strict=True):
            if item["image"] != source.name or item["input_sha256"] != audit.sha256(source):
                raise RuntimeError("Wrong image order or input hash")
            filename = source.stem + ".png"
            target = args.out / variant / filename
            if item["map_sha256"] != audit.sha256(target):
                raise RuntimeError("Variant PNG hash mismatch")
            paths = {
                "intervention": target,
                "public_baseline": args.baseline / filename,
                "historical_p63": args.historical / filename,
            }
            arrays = {k: audit.read_map(v) for k, v in paths.items()}
            records.append({
                "variant": variant, "image_id": source.stem,
                "sha256": {k: audit.sha256(v) for k, v in paths.items()},
                "vs_public": audit.compare(arrays["intervention"], arrays["public_baseline"]),
                "vs_historical": audit.compare(arrays["intervention"], arrays["historical_p63"]),
                "baseline_vs_historical": audit.compare(arrays["public_baseline"], arrays["historical_p63"]),
            })

    summaries = {}
    for variant in VARIANTS:
        subset = [row for row in records if row["variant"] == variant]
        summaries[variant] = {
            field: {
                "exact_images": sum(int(row[field]["exact"]) for row in subset),
                "changed_pixels": sum(row[field]["changed_pixels"] for row in subset),
                "max_absolute_byte_difference": max(row[field]["max_absolute_byte_difference"] for row in subset),
                "sum_absolute_byte_difference": sum(row[field]["sum_absolute_byte_difference"] for row in subset),
            }
            for field in ("vs_public", "vs_historical", "baseline_vs_historical")
        }
    receipt = {
        "status": "P101_COMPLETE", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "P101_CUDNN_H3_PRECISION_PREREG_20261113.md",
        "fold": "F1", "images_per_variant": 4, "variants": list(VARIANTS),
        "summary": summaries, "records": records,
        "self_test": "passed", "test_labels_read": False,
        "new_PQ_computed": False, "Kaggle_submission_made": False,
        "end_to_end_CSV_parity_verified": False,
        "audit_sha256": audit.sha256(Path(__file__)),
    }
    audit.atomic_receipt(args.receipt, receipt)
    print("P101_COMPARISON_COMPLETE", json.dumps(summaries, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
