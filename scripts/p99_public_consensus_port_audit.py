#!/usr/bin/env python3
"""P99: verify the public standalone consensus port against the archived source."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
ORIGINAL_SOURCE = Path("/home/ai/pc/d/PROJECTS/.cognition/solar-filament-20261106/testinf/make_submission.py")
STAGING = WORK.parent / "final-release-staging-20261112"
P88 = WORK / "p88_public_fp32_cpu_full_20261112"
OUT = WORK / "p99_public_consensus_port_audit_20261113"
PREREG = REPO / "docs/submission/P99_PUBLIC_CONSENSUS_PORT_PREREG_20261113.md"
FOLDS = ("F1", "F2", "F3", "F4")

from official_consensus import consensus as published_consensus
from p98_source_consensus_parity import compare, digest


def self_test() -> None:
    import numpy as np
    from official_consensus import mask_iou_ge, masked, MIN_SCORE
    mask = np.full((3, 3), 255, dtype=np.uint8)
    d = (.9, 2, 5, 3, 6, mask)
    q = (.8, 2, 5, 3, 6, mask.copy())
    assert abs(MIN_SCORE - .3) < 1e-12
    assert len(masked([d])) == 1
    assert not masked([(.1, 2, 5, 3, 6, mask)])
    assert mask_iou_ge(masked([d])[0], masked([q])[0])
    assert len(published_consensus([[d], [q]])) == 1
    assert compare(published_consensus([[d], []]), [(.675, 2, 5, 3, 6, mask)])["exact"]
    assert not compare([d], [q])["exact"]
    print("P99_SELFTEST_PASS", flush=True)


def run(limit: int) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")
    if not PREREG.is_file():
        raise RuntimeError("Preregistration not found")
    if not ORIGINAL_SOURCE.is_file():
        raise RuntimeError("Archived baseline source not found")
    receipt_path = P88 / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "P88_PUBLIC_SOURCE_CPU_720_EXACT_PASS":
        raise RuntimeError("P88 exact source-stage attestation missing")
    if any(receipt["folds"][f]["exact_images"] != 180 for f in FOLDS):
        raise RuntimeError("P88 incomplete")
    sys.path.insert(0, str(ORIGINAL_SOURCE.parent))
    from make_submission import consensus as archived_consensus
    with (STAGING / "dets_official.pkl").open("rb") as stream:
        historical = pickle.load(stream)
    names = sorted(historical)
    if len(names) != 180 or not 1 <= limit <= 180:
        raise ValueError("Expected exactly 180 images and limit 1..180")
    results = {}
    for index, name in enumerate(names[:limit], 1):
        views = []
        for fold in FOLDS:
            with (P88 / fold / f"{Path(name).stem}.pkl").open("rb") as stream:
                two_views = pickle.load(stream)
            if len(two_views) != 2:
                raise ValueError(f"Invalid two-view evidence {fold}/{name}")
            views.extend(two_views)
        newer = published_consensus(views)
        archived = archived_consensus(views)
        results[name] = {
            "public_vs_archived": compare(newer, archived),
            "public_vs_historical": compare(newer, historical[name])
        }
        if index == 1 or index % 10 == 0 or index == limit:
            exact = sum(all(v["exact"] for v in pair.values()) for pair in results.values())
            print(f"P99_PROGRESS {index}/{limit} exact={exact}", flush=True)
    exact_count = sum(all(v["exact"] for v in pair.values()) for pair in results.values())
    result = {
        "experiment": "P99", "images": limit, "exact_images": exact_count,
        "public_port_sha256": digest(REPO / "scripts/official_consensus.py"),
        "original_source_sha256": digest(ORIGINAL_SOURCE),
        "reference_p88_receipt_sha256": digest(receipt_path),
        "original_detector_cache_sha256": digest(STAGING / "dets_official.pkl"),
        "prereg_sha256": digest(PREREG),
        "cpu_only": True, "test_ground_truth_used": False,
        "per_image": results
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"P99_{limit}_AUDIT.json"
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise RuntimeError("Immutable receipt conflict")
    else:
        temporary = output.with_suffix(".partial")
        with temporary.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    print(f"P99_FINISHED exact={exact_count}/{limit} output={output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        run(args.limit)
