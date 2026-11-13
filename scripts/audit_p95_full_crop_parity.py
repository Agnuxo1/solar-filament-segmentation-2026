#!/usr/bin/env python3
"""Independent, read-only P95 receipt audit for the 180 official test JPEGs.

This script never evaluates test labels, runs CUDA, or optimizes the Kaggle score.
It rehashes all source JPEGs and frozen crop references, validates all per-image
P95 receipts, checks selector provenance, and writes a machine-checkable report.
Reuses historical H3 and Mask R-CNN caches: NOT end-to-end model inference.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys

FIELDS = ("X", "M", "comp", "score")
EXPERIMENT = "P95_JPEG_TO_NOSITE_CROP_INDEPENDENT_PARITY"


def sha256(path: pathlib.Path) -> str:
    """Hash a file in bounded memory without interpreting its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_record(record: dict, stem: str, source_sha: str,
                  ref_sha: str, selector_sha: str) -> list[str]:
    """Reject any missing, modified, mismatched, or non-exact field."""
    problems = []
    expected = {
        "experiment": EXPERIMENT,
        "image": stem + ".jpeg",
        "image_sha256": source_sha,
        "reference_sha256": ref_sha,
        "selector_sha256": selector_sha,
        "cached_h3_and_detector_inputs": True,
        "gpu_used": False,
        "test_annotations_used": False,
        "exact_all_fields": True,
    }
    for field, value in expected.items():
        if record.get(field) != value or type(record.get(field)) is not type(value):
            problems.append(field + ": unexpected value")
    field_results = record.get("per_field")
    if not isinstance(field_results, dict) or set(field_results) != set(FIELDS):
        problems.append("per_field: incorrect inventory")
    else:
        for field in FIELDS:
            item = field_results[field]
            if (not isinstance(item, dict) or
                item.get("exact") is not True or
                item.get("different") != 0 or
                not isinstance(item.get("elements"), int) or
                item["elements"] < 0):
                problems.append("per_field." + field + ": mismatched")
    return problems


def test_verification() -> None:
    """Prove this checker rejects each of the four key scientific mistakes."""
    record = {
        "experiment": EXPERIMENT, "image": "sample.jpeg",
        "image_sha256": "a", "reference_sha256": "b",
        "selector_sha256": "c", "cached_h3_and_detector_inputs": True,
        "gpu_used": False, "test_annotations_used": False,
        "exact_all_fields": True,
        "per_field": {name: {"exact": True, "different": 0, "elements": 2}
                      for name in FIELDS}
    }
    assert verify_record(record, "sample", "a", "b", "c") == []
    for field, wrong in (("image_sha256", "x"), ("reference_sha256", "x"),
                         ("selector_sha256", "x"), ("gpu_used", True),
                         ("test_annotations_used", True),
                         ("cached_h3_and_detector_inputs", False),
                         ("exact_all_fields", False)):
        sample = json.loads(json.dumps(record))
        sample[field] = wrong
        assert verify_record(sample, "sample", "a", "b", "c"), field
    for field in FIELDS:
        sample = json.loads(json.dumps(record))
        sample["per_field"][field]["different"] = 1
        assert verify_record(sample, "sample", "a", "b", "c"), field
    print("P95_AGGREGATE_SELF_TEST_PASS")


def analyze(root: pathlib.Path, images: pathlib.Path,
            reference: pathlib.Path, repo: pathlib.Path) -> dict:
    """Audit exactly one immutable receipt per source JPEG and archived crop."""
    receipts = root / "p95_nosite_jpeg_crop_audit_20261113"
    selector = repo / "release/model_metadata/h9b_nosite_selector.pkl"
    manifest = repo / "release/model_metadata/h9b_nosite_selector.json"
    prereg = receipts / "P95_PREREGISTRATION.txt"
    program = repo / "scripts/p95_nosite_jpeg_crop_audit.py"

    for path in (selector, manifest, prereg, program):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_data = json.loads(manifest.read_text(encoding="utf8"))
    selector_sha = sha256(selector)
    if selector_sha != manifest_data["weight_sha256"]:
        raise ValueError("TRAIN-only selector SHA-256 does not match manifest")
    sources = {p.stem: p for p in images.glob("*.jpeg")}
    refs = {p.stem: p for p in reference.glob("*.npz")}
    reports = {p.stem: p for p in receipts.glob("*.json")}
    errors = []
    if len(sources) != 180 or len(refs) != 180 or len(reports) != 180:
        errors.append("INVENTORY_COUNT_FAILURE")
    if set(sources) != set(refs) or set(sources) != set(reports):
        errors.append("INVENTORY_STEM_MISMATCH")
    checked = 0
    field_totals = {field: 0 for field in FIELDS}
    file_hashes = {}
    for stem in sorted(set(sources) & set(refs) & set(reports)):
        try:
            image_sha = sha256(sources[stem])
            ref_sha = sha256(refs[stem])
            record = json.loads(reports[stem].read_text(encoding="utf8"))
            issues = verify_record(record, stem, image_sha, ref_sha, selector_sha)
            if issues:
                errors.append({"stem": stem, "issues": issues})
                continue
            checked += 1
            for field in FIELDS:
                field_totals[field] += record["per_field"][field]["elements"]
            file_hashes[stem] = sha256(reports[stem])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            errors.append({"stem": stem, "error": str(exc)})
    return {
        "experiment": "P95_FULL_180_INDEPENDENT_SHA_AND_FIELD_RECEIPT_AUDIT",
        "audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "audit_status": "PASS" if checked == 180 and not errors else "FAIL",
        "images_expected": 180, "images_checked_exact": checked,
        "images_with_errors": len(errors), "errors": errors,
        "per_field_element_totals": field_totals,
        "selector_sha256": selector_sha, "selector_manifest_sha256": sha256(manifest),
        "preregistration_sha256": sha256(prereg),
        "p95_source_sha256": sha256(program),
        "receipt_sha256_by_image": file_hashes,
        "used_test_ground_truth": False, "used_cuda": False,
        "historical_h3_and_detector_caches_required": True,
        "scope": "JPEG plus frozen earlier predictions to H6 no-site crop only",
        "fully_independent_end_to_end_JPEG_to_CSV": False
    }


def main() -> int:
    """Run the audit with immutable source files and atomic JSON output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=False,
                        default=pathlib.Path("/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112"))
    parser.add_argument("--images", type=pathlib.Path, required=False,
                        default=pathlib.Path("/home/ai/pc/d/PROJECTS/196_NeuroPixel/kaggle/filament/data/MAGFiLO_1.0_Kaggle_2026/test/test_images"))
    parser.add_argument("--reference", type=pathlib.Path, required=False,
                        default=pathlib.Path("/srv/almacen/nas-2tb-1/solar-filament-2026/final-release-staging-20261112/h9/official/crops"))
    parser.add_argument("--repo", type=pathlib.Path, required=False, default=None)
    parser.add_argument("--output", type=pathlib.Path, required=False, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    test_verification()
    if args.self_test:
        return 0
    repo = args.repo or args.root / "release_checkout_20261112"
    result = analyze(args.root, args.images, args.reference, repo)
    output = args.output or repo / "docs/submission/P95_FULL_180_INDEPENDENT_RECEIPT_20261113.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Refusing to replace an existing audit artifact: " + str(output))
    temp = output.with_suffix(".tmp")
    with temp.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temp.replace(output)
    print(json.dumps({key: result[key] for key in (
        "audit_status", "images_checked_exact", "images_with_errors",
        "per_field_element_totals", "historical_h3_and_detector_caches_required"
    )}, indent=2))
    print("RECEIPT", output, sha256(output))
    return 0 if result["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
