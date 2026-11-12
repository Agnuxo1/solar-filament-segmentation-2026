#!/usr/bin/env python3
"""Read-only integrity audit of preregistered solar temporal folds.

Checks that labelled calibration/test holdouts are disjoint from training,
their physical images match frozen split manifests, and every holdout NPZ
has the expected cryptographic hash. No neural-network inference, model
selection, or Kaggle API access occurs in this program.

The checks validate artifacts and declared provenance, not the complete
training program's runtime behavior or the competition's hidden test labels.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile

SPLIT_SHA256 = {
    1: "4f30e3b8f6d84e03bee1cfc6a65c03261116f725a0ff621ddfbd4de49e88cbda",
    2: "654b31f3edf0b553d651e54ba4e7cc3fccc4ee22403777564b459dc7d95e1609",
    3: "82783558e625fbc72bbf5397e903deb57133bb29952610e81f7ff1321428b90d",
    4: "c6c44d745b454848316d773b6125daddd0658e2b189f7ed539020908bbc98db6",
}
EXPECTED_TRAIN_COUNTS = {1: 446, 2: 449, 3: 449, 4: 454}


def file_sha256(path: Path) -> str:
    """Hash each file incrementally, limiting transient RAM use."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    """Load one complete JSON object or raise on malformed artifacts."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def validate_manifest(directory: Path, expected_names: set[str]) -> dict:
    """Match all declared held-out physical images and independently hash NPZs."""
    manifest_path = directory / "manifest.json"
    manifest = load_json(manifest_path)
    catalog = manifest.get("files")
    if not isinstance(catalog, dict) or set(catalog) != expected_names:
        raise ValueError(f"Holdout manifest images do not match frozen split: {directory}")
    if any(not isinstance(n, str) or not n.endswith(".jpeg") for n in catalog):
        raise ValueError(f"Invalid original physical-image key: {directory}")
    bytes_read = 0
    for physical_name, expected_hash in sorted(catalog.items()):
        file = directory / (Path(physical_name).stem + ".npz")
        if not file.is_file():
            raise FileNotFoundError(f"Missing held-out annotation cache: {file}")
        if file_sha256(file) != expected_hash:
            raise ValueError(f"Holdout cache hash mismatch: {file}")
        bytes_read += file.stat().st_size
    return {
        "physical_images": len(catalog),
        "annotation_records": manifest.get("annotation_records"),
        "manifest_sha256": file_sha256(manifest_path),
        "cache_bytes_hashed": bytes_read,
        "all_files_hashed": True,
    }


def validate_fold(root: Path, fold: int) -> dict:
    """Verify one preregistered TRAIN/calibration/test split without label leakage."""
    split_path = root / ("split_F1.json" if fold == 1 else f"folds/split_F{fold}.json")
    actual_sha = file_sha256(split_path)
    if actual_sha != SPLIT_SHA256[fold]:
        raise ValueError(f"Frozen F{fold} split SHA-256 mismatch")
    obj = load_json(split_path)
    if obj.get("fold") != fold or obj.get("selection_policy") != "checkpoint and thresholds on calibration only":
        raise ValueError(f"Wrong fold identity or calibration-only policy: F{fold}")
    files = obj.get("files", {})
    if not isinstance(files, dict) or set(files) != {"train", "calibration", "test"}:
        raise ValueError(f"Malformed F{fold} partitions")
    groups: dict[str, set[str]] = {}
    for label in ("train", "calibration", "test"):
        names = files[label]
        if not isinstance(names, list) or len(set(names)) != len(names):
            raise ValueError(f"Duplicate or malformed F{fold} {label} names")
        if any(not isinstance(name, str) or not name.endswith(".jpeg") for name in names):
            raise ValueError(f"Invalid F{fold} {label} filename")
        groups[label] = set(names)
    if len(groups["train"]) != EXPECTED_TRAIN_COUNTS[fold]:
        raise ValueError(f"Unexpected F{fold} training-image count")
    for first, second in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        intersection = groups[first] & groups[second]
        if intersection:
            raise ValueError(f"F{fold} leakage: {first} intersects {second}")
    holdout = root / ("gt_holdout" if fold == 1 else f"folds/F{fold}/gt_holdout")
    verified = {
        label: validate_manifest(holdout / label, groups[label])
        for label in ("calibration", "test")
    }
    pack_provenance = None
    if fold > 1:
        pack = load_json(root / f"folds/F{fold}_train/manifest.json")
        if (pack.get("fold") != fold or pack.get("images") != len(groups["train"])
                or pack.get("split_sha256") != actual_sha
                or pack.get("status") != "FOLD_SOFT_TARGETS_TRAIN_ONLY"
                or pack.get("no_calibration_or_test_labels_loaded_for_training") is not True):
            raise ValueError(f"F{fold} TRAIN-pack provenance failed")
        pack_provenance = {
            "manifest_sha256": file_sha256(root / f"folds/F{fold}_train/manifest.json"),
            "train_only_provenance_flag_present": True,
            "declared_train_images": pack["images"],
        }
    return {
        "fold": fold,
        "split_sha256": actual_sha,
        "train_images": len(groups["train"]),
        "calibration": verified["calibration"],
        "test": verified["test"],
        "pairwise_disjoint": True,
        "train_pack": pack_provenance,
    }


def self_test() -> None:
    """Exercise hash mismatch and filename-set rejection without competition data."""
    with tempfile.TemporaryDirectory(prefix="solar-fold-audit-") as temp:
        directory = Path(temp)
        cache = directory / "example.npz"
        cache.write_bytes(b"synthetic-test-data")
        receipt = {
            "files": {"example.jpeg": file_sha256(cache)},
            "annotation_records": 1,
        }
        (directory / "manifest.json").write_text(json.dumps(receipt), encoding="utf-8")
        assert validate_manifest(directory, {"example.jpeg"})["all_files_hashed"]
        try:
            validate_manifest(directory, {"different.jpeg"})
            raise AssertionError("Different physical-image set was not rejected")
        except ValueError:
            pass
        cache.write_bytes(b"tampered")
        try:
            validate_manifest(directory, {"example.jpeg"})
            raise AssertionError("Altered annotation cache was not rejected")
        except ValueError:
            pass


def main() -> None:
    """Run self-tests and all independent fold checks; preserve an immutable receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Existing research root on HDD")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("SPLIT_AUDIT_SELFTEST_OK")
        return
    if args.root is None:
        parser.error("--root is required except with --self-test")
    root = args.root.resolve(strict=True)
    results = [validate_fold(root, fold) for fold in (1, 2, 3, 4)]
    receipt = {
        "schema": "SOLAR_FOLD_SPLIT_AUDIT_V1",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "FOUR_FOLD_SPLIT_AND_CACHE_INTEGRITY_VERIFIED",
        "source_script_sha256": file_sha256(Path(__file__).resolve()),
        "folds": results,
        "notes": [
            "Only original TRAIN-derived calibration and held-out annotations were inspected.",
            "No competition hidden TEST labels were accessed or available.",
            "A train-only manifest flag is provenance, not a proof of all prior runtime I/O.",
            "This audit does not calculate PQ, prove model quality or submit to Kaggle.",
        ],
    }
    destination = root / "audit"
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = destination / f"split_integrity_{timestamp}.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
    print(json.dumps({
        "status": receipt["status"],
        "receipt": str(output),
        "fold_summary": [
            {"fold": x["fold"], "train": x["train_images"],
             "calibration": x["calibration"]["physical_images"],
             "test": x["test"]["physical_images"]}
            for x in results
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
