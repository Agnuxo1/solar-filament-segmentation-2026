#!/usr/bin/env python3
"""P97: audit the deterministic fourfold Mask R-CNN proposal aggregation.

Rebuild candidate official proposal lists using independently reproduced,
public-source, bit-exact FP32 CPU model outputs from P88. Compare directly
against the original frozen aggregate for equality of order, coordinates,
scores and instance-mask pixel values. No inference, labels, CUDA or tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
P88 = WORK / "p88_public_fp32_cpu_full_20261112"
STAGING = WORK.parent / "final-release-staging-20261112"
OUT = WORK / "p97_public_detector_aggregation_20261113"
FOLDS = ("F1", "F2", "F3", "F4")


def same_proposal(a, b):
    """Require exact ordered scalar values, array shape/dtype and mask pixels."""
    return (len(a) == len(b) == 6 and a[:5] == b[:5] and
            a[5].shape == b[5].shape and a[5].dtype == b[5].dtype and
            np.array_equal(a[5], b[5]))


def compare_list(candidate, historical):
    """Return strict per-index array and scalar parity, with no reordering."""
    exact_items = 0
    if len(candidate) != len(historical):
        return {"same_length": False, "length_candidate": len(candidate),
                "length_original": len(historical), "exact": False}
    for newer, old in zip(candidate, historical):
        if not same_proposal(newer, old):
            break
        exact_items += 1
    return {"same_length": True, "length_candidate": len(candidate),
            "length_original": len(historical), "matched_prefix": exact_items,
            "exact": exact_items == len(candidate)}


def all_orders(replayed: dict):
    """Compare two fixed aggregation conventions without learning ordering."""
    a = [item for fold in FOLDS for view in replayed[fold] for item in view]
    b = [item for view in (0, 1) for fold in FOLDS for item in replayed[fold][view]]
    return {"fold_then_view": a, "view_then_fold": b}


def self_test():
    """Verify that a one-byte mask change or scalar change breaks parity."""
    q = (.2, 1, 2, 3, 4, np.ones((2, 2), np.uint8))
    z = (.3, 1, 2, 3, 4, q[-1])
    assert compare_list([q], [q])["exact"]
    assert not compare_list([q], [z])["exact"]
    assert not compare_list([q], [])["exact"]
    changed = (.2, 1, 2, 3, 4, np.zeros((2, 2), np.uint8))
    assert not compare_list([q], [changed])["exact"]
    print("P97_SELFTEST_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("P97 is CPU-only; CUDA_VISIBLE_DEVICES must be empty")
    if not (1 <= args.limit <= 180):
        parser.error("Expected --limit in 1..180")
    if not (OUT / "P97_PREREGISTRATION.md").is_file():
        raise RuntimeError("P97 scientific preregistration must exist before audit")
    p88_manifest = json.loads((P88 / "receipt.json").read_text())
    if p88_manifest.get("status") != "P88_PUBLIC_SOURCE_CPU_720_EXACT_PASS":
        raise RuntimeError("P88 public source reconstruction not verified")
    for fold in FOLDS:
        if p88_manifest["folds"][fold]["exact_images"] != 180:
            raise RuntimeError("Incomplete public detector source fold")
    with (STAGING / "dets_official.pkl").open("rb") as fh:
        historical = pickle.load(fh)
    names = sorted(historical)
    if len(names) != 180:
        raise RuntimeError("Unexpected historical proposal inventory")
    results = {}
    for index, name in enumerate(names[:args.limit], 1):
        replayed = {}
        for fold in FOLDS:
            filename = P88 / fold / f"{Path(name).stem}.pkl"
            with filename.open("rb") as fh:
                replayed[fold] = pickle.load(fh)
            if len(replayed[fold]) != 2:
                raise RuntimeError(f"Missing detector views: {fold}, {name}")
        results[name] = {order: compare_list(proposals, historical[name])
                         for order, proposals in all_orders(replayed).items()}
        if index == 1 or index % 10 == 0 or index == args.limit:
            exact = {order: sum(r[order]["exact"] for r in results.values())
                     for order in ("fold_then_view", "view_then_fold")}
            print(f"P97_PROGRESS {index}/{args.limit} exact={exact}", flush=True)
    OUT.mkdir(exist_ok=True, parents=True)
    out = OUT / f"P97_PILOT_{args.limit}_AUDIT.json"
    result = {
        "test_annotations_used": False,
        "cpu_only": True,
        "historical_detector_comparison": True,
        "input_public_P88": str(P88 / "receipt.json"),
        "images": args.limit,
        "counts": {order: sum(r[order]["exact"] for r in results.values())
                   for order in ("fold_then_view", "view_then_fold")},
        "per_image": results,
    }
    if out.exists():
        if json.loads(out.read_text()) != result:
            raise RuntimeError("An existing immutable P97 report conflicts with new results")
    else:
        tmp = out.with_suffix(".partial")
        with tmp.open("x") as file:
            json.dump(result, file, indent=2, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, out)
    print("P97_COMPLETE", json.dumps(result["counts"]), flush=True)


if __name__ == "__main__":
    main()
