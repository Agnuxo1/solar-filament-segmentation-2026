#!/usr/bin/env python3
"""P111: independently localize numerical drift between P103 and P106.

This reproducibility audit only reads model predictions and intermediate arrays.
It does not read competition labels, calculate PQ, train or submit a model.
All outputs are written to the designated NAS experiment directory.
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


EXPECTED_HASHES = {
    "p103_csv": "8ae3160af9bddd5ac3ef914bc807da71e3e9cbad0444cc055d2ff0056cdfe2ba",
    "p106_csv": "16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3",
    "prereg": "cd8a2716c6293e1550ca3cb6e256e14f4f1ad98c9d8f55b89ba6a873eeec468b",
}
ARRAY_KEYS = ("X", "M", "comp", "score")
MODEL_NAMES = ("resnet34", "efficientnet_v2_s")


def sha256_file(path: Path) -> str:
    """Stream a SHA-256 hash without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_hash(path: Path, expected: str) -> None:
    """Abort rather than silently proceeding with an unexpected source."""
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Frozen input hash mismatch for {path}: {actual} != {expected}")


def compare_arrays(first: np.ndarray, second: np.ndarray) -> dict:
    """Compare shape, dtype, exact values and finite numerical drift."""
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape:
        raise ValueError(f"Incompatible array shapes: {first.shape} vs {second.shape}")
    if first.dtype.kind not in "biuf" or second.dtype.kind not in "biuf":
        raise TypeError(f"Unsupported array dtypes: {first.dtype} vs {second.dtype}")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Non-finite source element")
    different = int(np.count_nonzero(first != second))
    max_abs = float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64)))) if first.size else 0.0
    crossings = int(np.count_nonzero((first >= 0.5) != (second >= 0.5)))
    return {
        "shape": list(first.shape),
        "dtype_p103": str(first.dtype),
        "dtype_p106": str(second.dtype),
        "elements": int(first.size),
        "different_elements": different,
        "exact": bool(different == 0 and first.dtype == second.dtype),
        "max_absolute_difference": max_abs,
        "threshold_half_crossings_descriptive_only": crossings,
    }


def synthetic_controls() -> dict:
    """Independent small-array controls to detect comparison regressions."""
    a = np.array([[0.3, 0.5], [0.7, 0.9]], dtype=np.float32)
    identical = compare_arrays(a, a.copy())
    assert identical["exact"] and identical["different_elements"] == 0
    b = a.copy()
    b[0, 1] = np.nextafter(b[0, 1], np.float32(0.0))
    changed = compare_arrays(a, b)
    assert changed["different_elements"] == 1
    assert changed["threshold_half_crossings_descriptive_only"] == 1
    mismatch_raised = False
    try:
        compare_arrays(a, b.ravel())
    except ValueError:
        mismatch_raised = True
    assert mismatch_raised
    return {"identical": True, "single_element_and_half_threshold": True, "shape_rejection": True}


def array_keys_from_npz(path: Path) -> set[str]:
    """Inspect a compressed archive and reject object/pickle arrays."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return set(archive.files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Paired-small run root")
    parser.add_argument("--out", required=True, type=Path, help="P111 output directory")
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("P111 requires CUDA_VISIBLE_DEVICES to be empty or -1")

    root = args.root.resolve(strict=True)
    out = args.out.resolve(strict=True)
    assert out.is_relative_to(root), "Outputs must remain on the project HDD"

    p103 = root / "p103_full_public_stage_coupling_20261113"
    p106 = root / "p106_public_clean_6843362_20261113" / "inference_cuda"
    p107_path = root / "p107_clean_replay_geometry_20261113" / "P107_VS_P103_AUDIT.json"

    checked_hash(p103 / "p103_180_reconstructed.csv", EXPECTED_HASHES["p103_csv"])
    checked_hash(p106 / "submission_public_rebuilt.csv", EXPECTED_HASHES["p106_csv"])
    checked_hash(out / "P111_PREREG.md", EXPECTED_HASHES["prereg"])
    if not p107_path.is_file():
        raise FileNotFoundError(p107_path)

    p107 = json.loads(p107_path.read_text(encoding="utf-8"))
    if p107["images"] != 180 or p107["different_masks_after_assignment"] != 12:
        raise RuntimeError("Unexpected P107 baseline; do not compare incompatible revisions")
    baseline = {entry["image"]: int(entry["different_masks_after_matching"])
                for entry in p107["image_summaries"]}
    if len(baseline) != 180 or sum(baseline.values()) != 12:
        raise RuntimeError("P107 image-level evidence is incomplete")
    changed_images = {image for image, count in baseline.items() if count > 0}
    if len(changed_images) != 11:
        raise RuntimeError("Unexpected P107 changed-image count")

    crop103 = p103 / "crops"
    crop106 = p106 / "crops"
    files103 = {file.stem: file for file in crop103.glob("*.npz")}
    files106 = {file.stem: file for file in crop106.glob("*.npz")}
    if set(files103) != set(baseline) or set(files106) != set(baseline):
        raise RuntimeError("Crop image-ID mismatch against the frozen P107 180-image list")

    controls = synthetic_controls()
    crop_evidence = {}
    crop_different_images = Counter()
    for image in sorted(baseline):
        with np.load(files103[image], allow_pickle=False) as a, np.load(files106[image], allow_pickle=False) as b:
            if set(a.files) != set(ARRAY_KEYS) or set(b.files) != set(ARRAY_KEYS):
                raise RuntimeError(f"Unexpected crop fields for {image}")
            comparison = {key: compare_arrays(a[key], b[key]) for key in ARRAY_KEYS}
        crop_evidence[image] = comparison
        for key, data in comparison.items():
            if not data["exact"]:
                crop_different_images[key] += 1

    probability_evidence = {image: {} for image in baseline}
    refiner_changed_images = Counter()
    pilot_images = {}
    for model in MODEL_NAMES:
        path103 = p103 / f"{model}_new176.npz"
        path106 = p106 / "refiners" / f"{model}_probabilities.npz"
        ids103 = array_keys_from_npz(path103)
        ids106 = array_keys_from_npz(path106)
        if len(ids103) != 176 or ids106 != set(baseline) or not ids103.issubset(ids106):
            raise RuntimeError(f"Inconsistent 176/180 refiner keys for {model}")
        pilot_images[model] = sorted(ids106 - ids103)
        with np.load(path103, allow_pickle=False) as a, np.load(path106, allow_pickle=False) as b:
            for image in sorted(ids103):
                stats = compare_arrays(a[image], b[image])
                probability_evidence[image][model] = stats
                if not stats["exact"]:
                    refiner_changed_images[model] += 1

    if pilot_images[MODEL_NAMES[0]] != pilot_images[MODEL_NAMES[1]]:
        raise RuntimeError("Different pilot exclusions between the two refiners")
    if len(pilot_images[MODEL_NAMES[0]]) != 4:
        raise RuntimeError("Expected precisely four pilot cases omitted from P103 fresh refiner outputs")

    evidence_rows = []
    unexplained = []
    unknown_pilot = []
    for image in sorted(baseline):
        c = crop_evidence[image]
        crop_changed = any(not c[key]["exact"] for key in ARRAY_KEYS)
        model_comparisons = probability_evidence[image]
        model_coverage = len(model_comparisons) == len(MODEL_NAMES)
        refinement_changed = any(not record["exact"] for record in model_comparisons.values())
        final_changed = baseline[image] > 0
        if final_changed and not crop_changed and not refinement_changed:
            if model_coverage:
                unexplained.append(image)
            else:
                unknown_pilot.append(image)
        evidence_rows.append({
            "image": image,
            "different_final_masks": baseline[image],
            "crop_changed": crop_changed,
            "refiner_covered_176": model_coverage,
            "refiner_changed": refinement_changed if model_coverage else None,
            "crop_comparisons": c,
            "refiner_comparisons": model_comparisons,
        })

    counts = {
        "images": len(baseline),
        "changed_final_images": len(changed_images),
        "changed_final_masks": sum(baseline.values()),
        "crop_different_images_by_key": dict(crop_different_images),
        "refiner_different_images_by_model": dict(refiner_changed_images),
        "pilot_images": pilot_images[MODEL_NAMES[0]],
        "changed_final_images_among_pilots": sorted(changed_images.intersection(pilot_images[MODEL_NAMES[0]])),
        "unexplained_with_full_176_model_coverage": unexplained,
        "unassessable_changed_pilot_images": unknown_pilot,
        "falsified": bool(unexplained),
        "changed_final_with_exact_crops": sum(
            1 for r in evidence_rows if r["different_final_masks"] and not r["crop_changed"]),
    }
    result = {
        "experiment": "P111",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "scope": "prediction-to-prediction stage reproducibility; no ground-truth or PQ",
        "prereg_sha256": EXPECTED_HASHES["prereg"],
        "source_sha256": sha256_file(Path(__file__)),
        "frozen_inputs_sha256": {
            "p103_csv": EXPECTED_HASHES["p103_csv"],
            "p106_csv": EXPECTED_HASHES["p106_csv"],
            "p107_audit": sha256_file(p107_path),
        },
        "synthetic_controls": controls,
        "summary": counts,
        "per_image": evidence_rows,
        "used_labels": False,
        "used_cuda": False,
        "trained_model": False,
        "made_kaggle_submission": False,
        "official_pq_new": None,
        "jev_goldfish_consulted": False,
    }
    result_path = out / "P111_FINAL_RECEIPT.json"
    tmp_path = out / "P111_FINAL_RECEIPT.json.tmp"
    tmp_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, result_path)
    print(json.dumps({"status": result["status"], "summary": counts, "receipt_sha256": sha256_file(result_path)}, indent=2))


if __name__ == "__main__":
    main()
