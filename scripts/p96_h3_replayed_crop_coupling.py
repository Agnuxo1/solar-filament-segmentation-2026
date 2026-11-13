#!/usr/bin/env python3
"""P96: quantify H9b crop sensitivity to independently replayed H3 maps.

Uses original unlabeled test JPEGs, independently reexecuted public H3
predictions, frozen TRAIN-only selector, and immutable historical detector
proposals. Compares all four crop archive fields with the accepted H9b
stage's original reference crops. A failure to match is evidence, not a
condition to tune or retry model thresholds. CPU-only and restartable.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
STAGING = WORK.parent / "final-release-staging-20261112"
OUT = WORK / "p96_h3_replayed_crop_coupling_20261113"
ASSETS = OUT / "assets"
IMAGES = Path("/home/ai/pc/d/PROJECTS/196_NeuroPixel/kaggle/filament/data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
SELECTOR = REPO / "release/model_metadata/h9b_nosite_selector.pkl"
SELECTOR_MANIFEST = REPO / "release/model_metadata/h9b_nosite_selector.json"
REFERENCE = STAGING / "h9/official/crops"
P88 = WORK / "p88_public_fp32_cpu_full_20261112"
FOLDS = ("F1", "F2", "F3", "F4")
MAPS = {
    "F1": WORK / "p77_full_h3_public_20261112/F1",
    "F2": WORK / "p79_full_h3_public_20261112/F2",
    "F3": WORK / "p79_full_h3_public_20261112/F3",
    "F4": WORK / "p79_full_h3_public_20261112/F4",
}
RECEIPTS = {
    "F1": WORK / "p77_full_h3_public_20261112/F1_h3_replay_receipt.json",
    **{f: WORK / f"p79_full_h3_public_20261112/{f}_h3_replay_receipt.json"
       for f in ("F2", "F3", "F4")},
}
sys.path.insert(0, str(REPO / "scripts"))
import p95_nosite_jpeg_crop_audit as P95  # noqa: E402


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    """Publish a write-once receipt through a fully flushed temporary file."""
    if path.exists():
        raise FileExistsError(f"Immutable audit receipt exists: {path}")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def self_test() -> None:
    """Exercise the four-array exactness gates and negative mismatch cases."""
    P95.self_test()
    x = {"X": np.zeros((1, 4, 2, 2), np.uint8),
         "M": np.zeros((1, 2, 3), np.float32),
         "comp": np.zeros(1, np.int64),
         "score": np.zeros(1, np.float64)}
    assert all(v["exact"] for v in P95.compare_arrays(x, x).values())
    for key in x:
        y = {k: val.copy() for k, val in x.items()}
        y[key].flat[0] += 1
        assert not P95.compare_arrays(x, y)[key]["exact"]
    print("P96_SELFTEST_PASS", flush=True)


def verify_upstream(images: list[Path]) -> dict:
    """Verify original-image and published map manifests before comparisons."""
    expected = {p.name for p in images}
    manifest = json.loads(SELECTOR_MANIFEST.read_text(encoding="utf-8"))
    selector_hash = sha256(SELECTOR)
    if selector_hash != manifest["weight_sha256"]:
        raise RuntimeError("Frozen TRAIN-only selector SHA-256 changed")
    if len(manifest["train_feature_columns"]) != 15:
        raise RuntimeError("No-site selector feature contract changed")
    detector = json.loads((P88 / "receipt.json").read_text(encoding="utf-8"))
    if detector.get("status") != "P88_PUBLIC_SOURCE_CPU_720_EXACT_PASS":
        raise RuntimeError("Public detector replay is not independently verified")
    for fold in FOLDS:
        if detector["folds"][fold]["exact_images"] != 180:
            raise RuntimeError(f"Incomplete public detector verification: {fold}")
        payload = json.loads(RECEIPTS[fold].read_text(encoding="utf-8"))
        items = payload["image_sha256"]
        if payload.get("full_180_images") is not True or {r["file"] for r in items} != expected:
            raise RuntimeError(f"Unexpected original JPEG inventory in {fold}")
        if len(items) != 180:
            raise RuntimeError(f"Duplicate original JPEG entry in {fold}")
        for r in items:
            if sha256(MAPS[fold] / (Path(r["file"]).stem + ".png")) != r["map_sha256"]:
                raise RuntimeError(f"Public H3 map SHA-256 mismatch: {fold}, {r['file']}")
    return {"selector": selector_hash,
            "h3_receipts": {fold: sha256(RECEIPTS[fold]) for fold in FOLDS},
            "detector_receipt": sha256(P88 / "receipt.json")}


def setup_assets() -> None:
    """Build an isolated, read-only source overlay without modifying caches."""
    for fold in FOLDS:
        directory = ASSETS / f"h3/maps/{fold}_resnet34_s0"
        directory.mkdir(parents=True, exist_ok=True)
        link = directory / "official"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(MAPS[fold], target_is_directory=True)
        if not link.is_symlink() or link.resolve() != MAPS[fold].resolve():
            raise RuntimeError(f"Unexpected H3 overlay target: {link}")
    target = STAGING / "dets_official.pkl"
    link = ASSETS / "dets_official.pkl"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(target)
    if not link.is_symlink() or link.resolve() != target.resolve():
        raise RuntimeError("Historical detector cache overlay was modified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 = full 180 images")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the empty string: CPU-only run")
    if args.limit < 0 or args.limit > 180:
        parser.error("--limit must be between 0 and 180")
    images = sorted(IMAGES.glob("*.jpeg"))
    if len(images) != 180 or len({p.name for p in images}) != 180:
        raise RuntimeError("Incomplete official raw JPEG inventory")
    if {p.stem for p in images} != {p.stem for p in REFERENCE.glob("*.npz")}:
        raise RuntimeError("Missing or extra frozen crop reference")
    if os.statvfs(OUT.parent).f_bavail * os.statvfs(OUT.parent).f_frsize < 20 * 1024**3:
        raise RuntimeError("HDD below safety reserve")
    prereg = OUT / "P96_PREREGISTRATION.md"
    if not prereg.is_file():
        raise RuntimeError("Immutable P96 preregistration is required before comparisons")
    upstream = verify_upstream(images)
    setup_assets()
    os.environ["SOLARFIL_ASSET_ROOT"] = str(ASSETS)
    sys.path.insert(0, str(REPO / "release/source"))
    with SELECTOR.open("rb") as stream:
        selector = pickle.load(stream)
    metadata = json.loads(SELECTOR_MANIFEST.read_text(encoding="utf-8"))
    count = len(images) if args.limit == 0 else args.limit
    exact = 0
    field_mismatch = {k: 0 for k in ("X", "M", "comp", "score")}
    for index, path in enumerate(images[:count], start=1):
        receipt_path = OUT / f"{path.stem}.json"
        input_hash = sha256(path)
        reference_path = REFERENCE / f"{path.stem}.npz"
        ref_hash = sha256(reference_path)
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("image_sha256") != input_hash or
                    receipt.get("reference_sha256") != ref_hash or
                    receipt.get("upstream_receipts") != upstream):
                raise RuntimeError(f"Previously verified evidence changed: {path.name}")
        else:
            started = time.monotonic()
            actual = P95.rebuild_one(path, selector, metadata)
            with np.load(reference_path, allow_pickle=False) as archive:
                original = {key: archive[key] for key in ("X", "M", "comp", "score")}
            fields = P95.compare_arrays(actual, original)
            receipt = {
                "experiment": "P96_PUBLIC_H3_TO_H6_COUPLING",
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "image": path.name, "image_sha256": input_hash,
                "reference_sha256": ref_hash,
                "upstream_receipts": upstream,
                "new_h3_replay_used": True, "historical_detector_cache_used": True,
                "test_labels_used": False, "gpu_used": False,
                "field_comparison": fields,
                "exact_all_fields": all(v["exact"] for v in fields.values()),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            atomic_json(receipt_path, receipt)
        exact += bool(receipt["exact_all_fields"])
        for key, val in receipt["field_comparison"].items():
            field_mismatch[key] += not val["exact"]
        if index == 1 or index % 10 == 0 or index == count:
            print(f"P96_PROGRESS {index}/{count} exact={exact} mismatches={field_mismatch}",
                  flush=True)
    if count == 180:
        final_path = OUT / "P96_FINAL_RECEIPT.json"
        if not final_path.exists():
            atomic_json(final_path, {
                "experiment": "P96_PUBLIC_H3_TO_H6_COUPLING",
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "images": count, "exact_crop_images": int(exact),
                "different_crop_images": count - int(exact),
                "field_mismatched_images": field_mismatch,
                "upstream_receipts": upstream,
                "historical_detector_cache_used": True,
                "not_full_raw_to_csv": True, "test_labels_used": False,
                "cpu_only": True,
            })
    print(f"P96_COMPLETE checked={count} exact={exact} different={count-exact}", flush=True)


if __name__ == "__main__":
    main()
