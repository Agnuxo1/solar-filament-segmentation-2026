#!/usr/bin/env python3
"""P112: frozen, CPU-only audit of large pixel differences between public replay crops.

The experiment measures numerical reproducibility of already-created predictions.
It never loads segmentation labels, trains a model, or changes a competition entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT_SHA = {
    "p103_csv": "8ae3160af9bddd5ac3ef914bc807da71e3e9cbad0444cc055d2ff0056cdfe2ba",
    "p106_csv": "16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3",
    "p111_receipt": "7b0ad18bebf50fb436cd89e447c02c632d04ed7f098fe3306372cf8ee4a278fc",
}
BINS = ((1, 1, "1"), (2, 4, "2-4"), (5, 31, "5-31"),
        (32, 127, "32-127"), (128, 254, "128-254"), (255, 255, "255"))


def hash_file(path: Path) -> str:
    """Compute a streaming SHA-256 digest from an immutable input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    """Reject missing or changed evidence before performing the analysis."""
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = hash_file(path)
    if actual != expected:
        raise RuntimeError(f"Source digest mismatch: {path}: {actual} != {expected}")


def classify(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute fixed magnitude bands and endpoint swaps for one aligned uint8 pair."""
    if a.dtype != np.uint8 or b.dtype != np.uint8 or a.shape != b.shape:
        raise ValueError(f"Expected equally shaped uint8 crops; received {a.shape}/{a.dtype}, {b.shape}/{b.dtype}")
    mask = a != b
    old = a[mask].astype(np.int16)
    new = b[mask].astype(np.int16)
    delta = np.abs(new - old)
    bands = {label: int(np.count_nonzero((delta >= lower) & (delta <= upper)))
             for lower, upper, label in BINS}
    high = delta >= 128
    endpoints = ((old == 0) & (new == 255)) | ((old == 255) & (new == 0))
    nonend = np.flatnonzero(high & ~endpoints)
    unexpected = [[int(old[i]), int(new[i]), int(delta[i])] for i in nonend[:10]]
    result = {
        "elements": int(a.size),
        "different_elements": int(mask.sum()),
        "max_magnitude": int(delta.max()) if delta.size else 0,
        "magnitude_bins": bands,
        "high_ge128": int(high.sum()),
        "high_endpoint_swaps": int(np.count_nonzero(high & endpoints)),
        "high_nonendpoint": int(nonend.size),
        "zero_to_255": int(np.count_nonzero((old == 0) & (new == 255))),
        "255_to_zero": int(np.count_nonzero((old == 255) & (new == 0))),
        "high_nonendpoint_examples": unexpected,
    }
    if sum(bands.values()) != result["different_elements"]:
        raise AssertionError("Difference magnitude accounting failure")
    if result["high_ge128"] != result["high_endpoint_swaps"] + result["high_nonendpoint"]:
        raise AssertionError("High-magnitude accounting failure")
    return result


def synthetic_controls() -> dict:
    """Test endpoint-positive, non-endpoint-negative and exact-pair controls."""
    p = classify(np.array([0, 255, 8], dtype=np.uint8),
                 np.array([255, 0, 8], dtype=np.uint8))
    assert p["high_ge128"] == p["high_endpoint_swaps"] == 2
    n = classify(np.array([10, 42], dtype=np.uint8),
                 np.array([220, 43], dtype=np.uint8))
    assert n["high_ge128"] == n["high_nonendpoint"] == 1
    assert n["magnitude_bins"]["1"] == 1
    z = classify(np.array([1, 2], dtype=np.uint8), np.array([1, 2], dtype=np.uint8))
    assert z["different_elements"] == 0
    failed = False
    try:
        classify(np.array([0], dtype=np.float32), np.array([0], dtype=np.float32))
    except ValueError:
        failed = True
    assert failed
    return {
        "endpoint_swap_positive": True,
        "non_endpoint_negative": True,
        "identical_control": True,
        "non_uint8_rejected": True,
    }


def main() -> None:
    """Validate frozen evidence, execute the preregistered test, and write an atomic receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prereg-sha256", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("This audit requires CUDA_VISIBLE_DEVICES=-1")
    root = args.root.resolve(strict=True)
    out = args.out.resolve(strict=True)
    if not out.is_relative_to(root):
        raise RuntimeError("Output must reside on project HDD")
    prereg = out / "P112_PREREG.md"
    verify(prereg, args.prereg_sha256)
    p103 = root / "p103_full_public_stage_coupling_20261113"
    p106 = root / "p106_public_clean_6843362_20261113" / "inference_cuda"
    p107_path = root / "p107_clean_replay_geometry_20261113" / "P107_VS_P103_AUDIT.json"
    p111_path = root / "p111_stage_drift_20261113" / "P111_FINAL_RECEIPT.json"
    verify(p103 / "p103_180_reconstructed.csv", ROOT_SHA["p103_csv"])
    verify(p106 / "submission_public_rebuilt.csv", ROOT_SHA["p106_csv"])
    verify(p111_path, ROOT_SHA["p111_receipt"])
    if not p107_path.is_file():
        raise FileNotFoundError(p107_path)
    prior = json.loads(p111_path.read_text(encoding="utf-8"))
    p107 = json.loads(p107_path.read_text(encoding="utf-8"))
    changed = {entry["image"] for entry in p107["image_summaries"]
               if entry["different_masks_after_matching"] > 0}
    a_files = {p.stem: p for p in (p103 / "crops").glob("*.npz")}
    b_files = {p.stem: p for p in (p106 / "crops").glob("*.npz")}
    expected = {r["image"] for r in prior["per_image"]}
    if len(expected) != 180 or set(a_files) != expected or set(b_files) != expected or len(changed) != 11:
        raise RuntimeError("Image-ID or prior-audit coverage is incompatible")
    controls = synthetic_controls()
    cumulative = Counter()
    bins = Counter()
    strata = {"final_changed": Counter(), "final_unchanged": Counter()}
    image_records = []
    examples = []
    total_elements = 0
    for image in sorted(expected):
        with np.load(a_files[image], allow_pickle=False) as source, np.load(b_files[image], allow_pickle=False) as rebuilt:
            if set(source.files) != {"X", "M", "comp", "score"} or set(rebuilt.files) != set(source.files):
                raise RuntimeError(f"Crop archive schema mismatch: {image}")
            result = classify(source["X"], rebuilt["X"])
        total_elements += result["elements"]
        for key in ("different_elements", "high_ge128", "high_endpoint_swaps", "high_nonendpoint",
                    "zero_to_255", "255_to_zero"):
            cumulative[key] += result[key]
        for key, value in result["magnitude_bins"].items():
            bins[key] += value
        group = "final_changed" if image in changed else "final_unchanged"
        strata[group]["images"] += 1
        strata[group]["images_with_high"] += int(result["high_ge128"] > 0)
        strata[group]["different_elements"] += result["different_elements"]
        strata[group]["high_ge128"] += result["high_ge128"]
        if len(examples) < 20:
            examples += [{"image": image, "old_new_delta": pair}
                         for pair in result["high_nonendpoint_examples"][:20-len(examples)]]
        image_records.append({"image": image, "changed_final_image": group == "final_changed",
                              "different_elements": result["different_elements"],
                              "high_ge128": result["high_ge128"],
                              "high_nonendpoint": result["high_nonendpoint"],
                              "max_magnitude": result["max_magnitude"]})
    if total_elements != 244_580_352 or cumulative["different_elements"] != 1577:
        raise RuntimeError(f"Aborted: P111 numeric controls disagree ({total_elements}, {cumulative['different_elements']})")
    for record in prior["per_image"]:
        image = record["image"]
        earlier = record["crop_comparisons"]["X"]["different_elements"]
        observed = next(r["different_elements"] for r in image_records if r["image"] == image)
        if earlier != observed:
            raise RuntimeError(f"P111 per-image mismatch: {image}")
    receipt = {
        "experiment": "P112",
        "status": "COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "numerical prediction reproducibility, not segmentation accuracy",
        "hypothesis": "Every |delta| >= 128 is an exact 0<->255 swap",
        "hypothesis_refuted": bool(cumulative["high_nonendpoint"]),
        "prereg_sha256": args.prereg_sha256,
        "script_sha256": hash_file(Path(__file__)),
        "inputs_sha256": {**ROOT_SHA, "p107": hash_file(p107_path)},
        "controls": controls,
        "images": len(image_records),
        "elements_compared": total_elements,
        "counts": dict(cumulative),
        "fixed_magnitude_bins": dict(bins),
        "final_change_strata": {key: dict(value) for key, value in strata.items()},
        "high_nonendpoint_examples_first20": examples,
        "per_image": image_records,
        "used_labels": False,
        "used_cuda": False,
        "trained_model": False,
        "submitted_to_kaggle": False,
        "new_pq": None,
        "jev_goldfish_consulted": False,
    }
    temporary = out / "P112_FINAL_RECEIPT.json.tmp"
    destination = out / "P112_FINAL_RECEIPT.json"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    print(json.dumps({"experiment": receipt["experiment"],
                      "hypothesis_refuted": receipt["hypothesis_refuted"],
                      "counts": receipt["counts"], "bins": receipt["fixed_magnitude_bins"],
                      "strata": receipt["final_change_strata"],
                      "receipt_sha256": hash_file(destination)}, sort_keys=True))


if __name__ == "__main__":
    main()
