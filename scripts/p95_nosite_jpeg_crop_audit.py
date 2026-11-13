#!/usr/bin/env python3
"""P95: independently rebuild H9b no-site crops from original Kaggle test JPEGs.

This strictly CPU-only verification uses the public source code, a SHA-256-pinned
TRAIN-only frozen selector, and historical H3 maps / Mask R-CNN proposal caches.
It never loads competition test annotations, retrains models, or queries Kaggle.
Proves only the JPEG -> H6 crop stage; not raw-image -> detector/map provenance.
Each image is independently verified and atomically checkpointed for recovery.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import pickle
import sys
import time

import cv2
import numpy as np

WORK = pathlib.Path("/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112")
REPO = WORK / "release_checkout_20261112"
STAGING = WORK.parent / "final-release-staging-20261112"
IMAGES = pathlib.Path("/home/ai/pc/d/PROJECTS/196_NeuroPixel/kaggle/filament/data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
OUT = WORK / "p95_nosite_jpeg_crop_audit_20261113"
SELECTOR = REPO / "release/model_metadata/h9b_nosite_selector.pkl"
MANIFEST = REPO / "release/model_metadata/h9b_nosite_selector.json"
SOURCE = REPO / "release/source"
REFERENCE = STAGING / "h9/official/crops"
PRE_REG = "P95 preregistered 2026-11-13 UTC before image comparisons. Objective: rebuild all 180 JPEG-to-H6 no-site crops with SHA-256 frozen TRAIN-only selector and original cached model stages. Measure exact agreement for X,M,comp,score, record negative results. No test annotations, no Kaggle feedback, no optimization, no GPU. The original JPEGs, historical H3 maps, historical detector cache, frozen selector and archived crops are read-only. CPU <= 150%, RAM <= 6GiB, HDD <= 2GiB, runtime <= 6h. Pilot first two sorted filenames; proceed with full 180 only if unit tests and inventory checks pass. A discrepancy is reported, never silently fixed. This is not a fully independent JPEG-to-CSV replay because cached H3/detector outputs are inputs.\n"

def sha256(path: pathlib.Path) -> str:
    """Hash arbitrary files in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()

def save_new_json(path: pathlib.Path, data: dict) -> None:
    """Create a receipt atomically and refuse to overwrite prior evidence."""
    if path.exists():
        raise FileExistsError(f"Immutable receipt exists: {path}")
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)

def compare_arrays(actual: dict, reference: dict) -> dict:
    """Compare every crop field including types, geometry and all values."""
    result = {}
    for key in ("X", "M", "comp", "score"):
        a, b = actual[key], reference[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            result[key] = {
                "exact": False, "shape_actual": list(a.shape),
                "shape_reference": list(b.shape),
                "dtype_actual": str(a.dtype), "dtype_reference": str(b.dtype),
                "different": None,
            }
        else:
            different = int(np.count_nonzero(a != b))
            result[key] = {"exact": different == 0, "different": different,
                           "elements": int(a.size)}
    return result

def self_test() -> None:
    """Prove that changes in all four fields fail the exactness gate."""
    original = {
        "X": np.ones((1, 4, 2, 2), np.uint8),
        "M": np.ones((1, 2, 3), np.float32),
        "comp": np.array([2], np.int64),
        "score": np.array([0.77], np.float64),
    }
    assert all(v["exact"] for v in compare_arrays(original, original).values())
    for key in original:
        changed = {k: value.copy() for k, value in original.items()}
        changed[key].flat[0] += 1 if key != "score" else .01
        result = compare_arrays(changed, original)
        assert not result[key]["exact"] and result[key]["different"] == 1
    changed = {k: v.copy() for k, v in original.items()}
    changed["X"] = changed["X"].astype(np.float32)
    assert not compare_arrays(changed, original)["X"]["exact"]
    print("P95_SELF_TEST_PASS", flush=True)

def preregister() -> None:
    """Store frozen hypotheses and immutable budgets before image evaluation."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "P95_PREREGISTRATION.txt"
    if not path.exists():
        with path.open("x", encoding="utf8") as stream:
            stream.write(PRE_REG)
            stream.flush()
            os.fsync(stream.fileno())
    if path.read_text(encoding="utf8") != PRE_REG:
        raise RuntimeError("Preregistration changed unexpectedly.")

def rebuild_one(path: pathlib.Path, selector, metadata: dict) -> dict:
    """Reconstruct a complete H6 frozen crop from the original JPEG pixels."""
    import h9_official as H
    import h9_crops as C
    from h6_features import MIN_AREA, table
    from h6_official import FALLBACK_AREA, TAU
    name, stem = path.name, path.stem
    prob, unet, mr, lab, area, peak = H.components(name, metadata["w"], metadata["t"])
    sites = metadata["sites"]
    if stem[-2] not in sites:
        raise ValueError(f"Unknown site character for image {name}")
    feats, _ = table(prob, unet, mr, [], metadata["t"], sites.index(stem[-2]))
    big = np.nonzero(area >= MIN_AREA)[0]
    features = feats[:, :-1]
    pred = selector.predict(features) if len(feats) else np.zeros(0, np.float64)
    keep, score = big[pred > TAU], pred[pred > TAU]
    if not keep.size and np.any(area >= FALLBACK_AREA):
        keep = np.array([int(np.argmax(np.where(area >= FALLBACK_AREA, peak, -1)))])
        score = np.ones(1)
    k = int(keep.size)
    X = np.zeros((k, 4, C.CH, C.CW), np.uint8)
    M = np.zeros((k, 2, 3), np.float32)
    if k:
        original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if original is None or original.dtype != np.uint8 or original.shape != (2048, 2048):
            raise ValueError(f"Bad original JPEG image: {path}")
        flat = C.flat_field(original)
        p8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        all_big = np.isin(lab, big + 1).astype(np.uint8) * 255
        for i, component in enumerate(keep):
            seed = lab == component + 1
            ys, xs = np.nonzero(seed)
            affine = C.frame(xs, ys)
            M[i] = affine
            seed8 = cv2.warpAffine(seed.astype(np.uint8) * 255, affine,
                                  (C.CW, C.CH), flags=cv2.INTER_NEAREST)
            X[i, 0] = cv2.warpAffine(flat, affine, (C.CW, C.CH),
                                     flags=cv2.INTER_LINEAR, borderValue=127)
            X[i, 1] = cv2.warpAffine(p8, affine, (C.CW, C.CH), flags=cv2.INTER_LINEAR)
            X[i, 2] = seed8
            X[i, 3] = np.where(seed8 > 0, 0,
                               cv2.warpAffine(all_big, affine, (C.CW, C.CH),
                                              flags=cv2.INTER_NEAREST))
    return {"X": X, "M": M, "comp": keep.astype(np.int64), "score": score}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="Audit a deterministic leading subset, 0 for all.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("P95 requires CUDA_VISIBLE_DEVICES to be empty.")
    if args.limit < 0:
        raise ValueError("Non-negative image limit required.")
    preregister()
    metadata = json.loads(MANIFEST.read_text(encoding="utf8"))
    if sha256(SELECTOR) != metadata["weight_sha256"]:
        raise RuntimeError("TRAIN-only selector does not match its recorded hash.")
    if metadata["train_feature_columns"][-1] == "site" or len(metadata["train_feature_columns"]) != 15:
        raise RuntimeError("H6 no-site feature contract is violated.")
    images = sorted(IMAGES.glob("*.jpeg"))
    references = sorted(REFERENCE.glob("*.npz"))
    if len(images) != 180 or len(references) != 180:
        raise RuntimeError("The official 180-image inventory is incomplete.")
    if {p.stem for p in images} != {p.stem for p in references}:
        raise RuntimeError("Historical reference inventory does not equal raw JPEG inventory.")
    os.environ["SOLARFIL_ASSET_ROOT"] = str(STAGING)
    sys.path.insert(0, str(SOURCE))
    cv2.setNumThreads(1)
    with SELECTOR.open("rb") as stream:
        selector = pickle.load(stream)
    selected = images[:args.limit] if args.limit else images
    all_exact = 0
    for index, path in enumerate(selected, 1):
        receipt = OUT / (path.stem + ".json")
        original_reference = REFERENCE / (path.stem + ".npz")
        image_hash, reference_hash = sha256(path), sha256(original_reference)
        if receipt.exists():
            old = json.loads(receipt.read_text(encoding="utf8"))
            if (old.get("image_sha256") != image_hash
                    or old.get("reference_sha256") != reference_hash
                    or old.get("selector_sha256") != metadata["weight_sha256"]):
                raise RuntimeError(f"Previously verified source changed for {path.name}")
            all_exact += int(old.get("exact_all_fields", False))
            continue
        started = time.monotonic()
        actual = rebuild_one(path, selector, metadata)
        with np.load(original_reference, allow_pickle=False) as archive:
            historical = {key: archive[key] for key in ("X", "M", "comp", "score")}
        comparison = compare_arrays(actual, historical)
        success = all(item["exact"] for item in comparison.values())
        save_new_json(receipt, {
            "experiment": "P95_JPEG_TO_NOSITE_CROP_INDEPENDENT_PARITY",
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "image": path.name, "image_sha256": image_hash,
            "reference_sha256": reference_hash,
            "selector_sha256": metadata["weight_sha256"],
            "cached_h3_and_detector_inputs": True,
            "test_annotations_used": False,
            "gpu_used": False,
            "exact_all_fields": success,
            "per_field": comparison, "elapsed_seconds": round(time.monotonic()-started, 3),
        })
        all_exact += int(success)
        print(f"P95_IMAGE {index}/{len(selected)} {path.stem} exact={success}", flush=True)
    print(f"P95_COMPLETE checked={len(selected)} exact={all_exact} mismatched={len(selected)-all_exact}", flush=True)

if __name__ == "__main__":
    main()
