#!/usr/bin/env python3
"""P98: source-faithful, CPU-only audit of the historical detector consensus.

This script uses the archived, frozen consensus implementation without modifying it.
It compares predictions reproduced independently by P88 with the original
unlabelled detector aggregate, requiring exact proposal order and mask bytes.
No calibration/test labels, Kaggle feedback, training, or CUDA are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
P88 = WORK / "p88_public_fp32_cpu_full_20261112"
STAGING = WORK.parent / "final-release-staging-20261112"
OUT = WORK / "p98_source_consensus_parity_20261113"
SOURCE = Path("/home/ai/pc/d/PROJECTS/.cognition/solar-filament-20261106/testinf/make_submission.py")
PREREG = REPO / "docs/submission/P98_SOURCE_CONSENSUS_PREREG_20261113.md"
FOLDS = ("F1", "F2", "F3", "F4")

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_consensus():
    """Load the original, immutable algorithm, not a guessed replacement."""
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Archived source missing: {SOURCE}")
    sys.path.insert(0, str(SOURCE.parent))
    from make_submission import consensus
    return consensus

def equal_proposal(a, b) -> bool:
    """Compare score/boxes, geometry, dtype and each mask byte in order."""
    import numpy as np
    return (len(a) == len(b) == 6 and a[:5] == b[:5]
            and a[5].shape == b[5].shape and a[5].dtype == b[5].dtype
            and bool(np.array_equal(a[5], b[5])))

def compare(candidate, historical):
    if len(candidate) != len(historical):
        return {"exact": False, "new_count": len(candidate),
                "old_count": len(historical), "matching_prefix": 0}
    prefix = 0
    for newer, older in zip(candidate, historical):
        if not equal_proposal(newer, older):
            break
        prefix += 1
    return {"exact": prefix == len(candidate), "new_count": len(candidate),
            "old_count": len(historical), "matching_prefix": prefix}

def self_test() -> None:
    import numpy as np
    m = np.arange(9, dtype=np.uint8).reshape(3, 3)
    a = (0.9, 0, 3, 0, 3, m)
    b = (0.8, 0, 3, 0, 3, m)
    assert compare([a], [a])["exact"]
    assert not compare([a], [b])["exact"]
    assert not compare([a], [])["exact"]
    assert not compare([a], [(0.9, 0, 3, 0, 3, m.copy() + 1)])["exact"]
    assert not compare([a], [(0.9, 0, 3, 0, 3, m.astype(np.int16))])["exact"]
    print("P98_SELF_TEST_PASS", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Refusing CUDA: set CUDA_VISIBLE_DEVICES=''")
    if not PREREG.is_file():
        raise RuntimeError("Missing preregistration before execution")
    if not 1 <= args.limit <= 180:
        parser.error("--limit must be 1..180")
    manifest_path = P88 / "receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "P88_PUBLIC_SOURCE_CPU_720_EXACT_PASS":
        raise RuntimeError("P88 complete exact-parity manifest unavailable")
    if any(manifest["folds"][fold]["exact_images"] != 180 for fold in FOLDS):
        raise RuntimeError("P88 incomplete fold")
    source_sha = digest(SOURCE)
    consensus = load_consensus()
    with (STAGING / "dets_official.pkl").open("rb") as stream:
        original = pickle.load(stream)
    names = sorted(original)
    if len(names) != 180:
        raise RuntimeError("Expected 180 unlabelled official test images")

    records = {}
    for count, name in enumerate(names[:args.limit], 1):
        views = []
        for fold in FOLDS:
            with (P88 / fold / f"{Path(name).stem}.pkl").open("rb") as stream:
                fold_views = pickle.load(stream)
            if len(fold_views) != 2:
                raise RuntimeError(f"Missing two CPU views: {fold}/{name}")
            views.extend(fold_views)
        candidate = consensus(views)
        records[name] = compare(candidate, original[name])
        if count == 1 or count % 10 == 0 or count == args.limit:
            print(f"P98_PROGRESS {count}/{args.limit} exact={sum(v['exact'] for v in records.values())}", flush=True)

    result = {
        "experiment": "P98", "source_file": str(SOURCE),
        "source_sha256": source_sha, "prereg_sha256": digest(PREREG),
        "p88_receipt_sha256": digest(manifest_path),
        "original_aggregate_sha256": digest(STAGING / "dets_official.pkl"),
        "cpu_only": True, "test_annotations_used": False,
        "method": "archived make_submission.consensus; F1..F4 each with two views",
        "total": args.limit, "exact": sum(v["exact"] for v in records.values()),
        "per_image": records
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"P98_{args.limit}_AUDIT.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != result:
            raise RuntimeError("Existing immutable receipt differs")
    else:
        temporary = destination.with_suffix(".partial")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    print(f"P98_DONE exact={result['exact']}/{args.limit} receipt={destination}", flush=True)

if __name__ == "__main__":
    main()
