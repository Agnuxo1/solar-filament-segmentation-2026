#!/usr/bin/env python3
"""Audit two independent GPU0 H3 replays against earlier public map baselines.

This program never opens labels, trains models, or touches CUDA; it validates
the preregistered four-by-four original-image subset and writes one immutable
machine-readable receipt to the research HDD.
"""
from __future__ import annotations

import argparse
import datetime as datetime
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

FOLDS = ("F1", "F2", "F3", "F4")
EXPECTED_SHAPE = (2048, 2048)


def sha256(path: Path) -> str:
    """Return a streaming binary checksum without trusting a filename."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_map(path: Path) -> np.ndarray:
    """Reject missing, incorrectly shaped, or non-byte map images."""
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None or value.shape != EXPECTED_SHAPE or value.dtype != np.uint8:
        raise ValueError(f"Invalid native uint8 probability map: {path}")
    return value


def compare(left: np.ndarray, right: np.ndarray) -> dict:
    """Quantify exact byte differences without lossy image comparison."""
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Map shape or dtype mismatch")
    delta = cv2.absdiff(left, right)
    changed = np.count_nonzero(delta)
    return {
        "exact": changed == 0,
        "changed_pixels": int(changed),
        "total_pixels": int(delta.size),
        "max_absolute_byte_difference": int(delta.max()),
        "sum_absolute_byte_difference": int(delta.sum(dtype=np.int64)),
    }


def self_test() -> None:
    """Prove that one modified pixel and malformed shapes cause failure."""
    a = np.zeros((3, 5), dtype=np.uint8)
    b = a.copy()
    assert compare(a, b)["exact"]
    b[1, 2] = 1
    check = compare(a, b)
    assert not check["exact"] and check["changed_pixels"] == 1
    assert check["sum_absolute_byte_difference"] == 1
    try:
        compare(a, b[:, :4])
    except ValueError:
        pass
    else:
        raise AssertionError("Shape mismatch was not rejected")


def atomic_receipt(destination: Path, payload: dict) -> None:
    """Write an immutable fsynced JSON report, never overwrite an old study."""
    if destination.exists():
        raise FileExistsError(f"Existing scientific receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".json.partial")
    with partial.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, destination)


def main() -> None:
    """Audit all predefined images and folds after both inferences complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ref-f1", required=True, type=Path)
    parser.add_argument("--ref-other", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("P100_SELF_TEST_PASS", flush=True)
        return

    images = sorted(args.images.glob("*.jpeg"))
    if len(images) != 180 or len({p.stem for p in images}) != 180:
        raise ValueError("Expected exactly 180 unique official test JPEGs")
    selected = images[:4]
    rows = []
    for fold in FOLDS:
        for condition in ("a", "b"):
            receipt = args.out / condition / f"{fold}_h3_replay_receipt.json"
            data = json.loads(receipt.read_text(encoding="utf-8"))
            if data["fold"] != fold or data["images_processed"] != 4:
                raise ValueError(f"Unverified partial fold receipt: {receipt}")
            if data["device"] != "cuda" or data["test_labels_read"]:
                raise ValueError("Invalid device or data policy in source receipt")
            if [row["file"] for row in data["image_sha256"]] != [p.name for p in selected]:
                raise ValueError("Unexpected image selection or order")
            for item in data["image_sha256"]:
                path = args.out / condition / fold / (Path(item["file"]).stem + ".png")
                if sha256(path) != item["map_sha256"]:
                    raise ValueError(f"Replay receipt hash mismatch: {path}")

        reference_dir = (args.ref_f1 if fold == "F1" else args.ref_other) / fold
        for image in selected:
            filename = image.stem + ".png"
            paths = {
                "repeat_a": args.out / "a" / fold / filename,
                "repeat_b": args.out / "b" / fold / filename,
                "earlier_public": reference_dir / filename,
            }
            maps = {name: read_map(path) for name, path in paths.items()}
            rows.append({
                "fold": fold,
                "image": image.name,
                "original_image_sha256": sha256(image),
                "map_sha256": {name: sha256(path) for name, path in paths.items()},
                "a_vs_b": compare(maps["repeat_a"], maps["repeat_b"]),
                "a_vs_earlier_public": compare(maps["repeat_a"], maps["earlier_public"]),
                "b_vs_earlier_public": compare(maps["repeat_b"], maps["earlier_public"]),
            })

    fields = ("a_vs_b", "a_vs_earlier_public", "b_vs_earlier_public")
    aggregates = {field: {
        "exact_pairs": sum(int(row[field]["exact"]) for row in rows),
        "changed_pixels": sum(row[field]["changed_pixels"] for row in rows),
        "max_absolute_byte_difference": max(row[field]["max_absolute_byte_difference"] for row in rows),
    } for field in fields}

    payload = {
        "study": "P100", "status": "COMPLETE", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": "P100_H3_REPEATABILITY_PREREG_20261113.md",
        "experiment": "two independent original-source CUDA replays of F1-F4 on the same four original test JPEGs",
        "images_per_fold": 4, "folds": list(FOLDS),
        "audited_image_model_pairs": len(rows),
        "self_test": "passed",
        "summary": aggregates, "per_pair": rows,
        "used_test_labels": False, "computed_new_PQ": False,
        "official_submission_made": False,
        "full_raw_jpeg_to_csv_parity": False,
        "audit_script_sha256": sha256(Path(__file__)),
    }
    atomic_receipt(args.receipt, payload)
    print("P100_COMPLETE", json.dumps(aggregates, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
