#!/usr/bin/env python3
"""P110: audit uniqueness of per-image optimal COCO-IoU instance assignments.

This CPU-only audit is a frozen reproducibility diagnostic. It never accesses
annotations, model weights, the GPU, or Kaggle credentials.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as coco
from scipy.optimize import linear_sum_assignment

EXPECTED = {
    "accepted": "24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60",
    "replay": "16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3",
}
TIE = 1e-12
NEAR = 1e-8
SIZE = [2048, 2048]


def sha256(path: Path) -> str:
    """Compute an entire-file SHA-256 in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv(path: Path) -> dict[str, list[tuple[str, dict]]]:
    """Validate all rows and group compressed COCO-RLE by physical image."""
    groups = defaultdict(list)
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected CSV schema in {path}: {reader.fieldnames}")
        for row in reader:
            ident, encoded = row["filament_id"], row["segmentation_rle"]
            if not ident or "_" not in ident or ident in seen:
                raise ValueError(f"Invalid or duplicate filament ID: {ident!r}")
            if not encoded or not encoded.isascii():
                raise ValueError(f"Missing or non-ASCII RLE: {ident}")
            seen.add(ident)
            encoded_rle = {"size": SIZE, "counts": encoded.encode("ascii")}
            if int(coco.area(encoded_rle)) <= 0:
                raise ValueError(f"Empty mask: {ident}")
            groups[ident.rsplit("_", 1)[0]].append((ident, encoded_rle))
    if len(groups) != 180 or len(seen) != 1241:
        raise ValueError(f"Unexpected image/row counts: {len(groups)}/{len(seen)}")
    return {key: sorted(value, key=lambda t: t[0]) for key, value in groups.items()}


def optimum(score: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return maximum total IoU and a deterministic Hungarian assignment."""
    rows, cols = linear_sum_assignment(-score)
    return float(score[rows, cols].sum()), rows, cols


def alternative_margin(score: np.ndarray) -> tuple[float | None, float, np.ndarray, np.ndarray]:
    """Find a globally optimal alternative by forbidding every chosen edge in turn."""
    best, rows, cols = optimum(score)
    n = len(rows)
    if n == 1:
        return None, best, rows, cols
    alternative = float("-inf")
    for row, col in zip(rows, cols):
        changed = score.copy()
        changed[int(row), int(col)] = -1e6
        runner_up, r2, c2 = optimum(changed)
        if any(int(r) == int(row) and int(c) == int(col) for r, c in zip(r2, c2)):
            raise AssertionError("Forbidden assignment edge was reused")
        alternative = max(alternative, runner_up)
    margin = best - alternative
    if margin < -TIE:
        raise AssertionError("Alternative improved upon optimum")
    return max(0.0, margin), best, rows, cols


def synthetic_tests() -> dict:
    """Catch any loss of sensitivity to a true assignment tie."""
    unique = np.array([[1.0, 0.0], [0.0, 1.0]])
    tied = np.ones((2, 2))
    single = np.array([[0.75]])
    assert math.isclose(alternative_margin(unique)[0], 2.0, abs_tol=1e-12)
    assert math.isclose(alternative_margin(tied)[0], 0.0, abs_tol=1e-12)
    assert alternative_margin(single)[0] is None
    return {"unique_2x2": "PASS", "tied_2x2": "PASS", "single_1x1": "PASS"}


def run(args: argparse.Namespace) -> dict:
    """Enforce input controls and compute all assignment margins without fitting."""
    started = time.monotonic()
    tests = synthetic_tests()
    if sha256(args.protocol) != args.protocol_sha:
        raise ValueError("Preregistration SHA-256 mismatch")
    source_hashes = {}
    for name, path in (("accepted", args.accepted), ("replay", args.replay)):
        actual = sha256(path)
        if actual != EXPECTED[name]:
            raise ValueError(f"{name} input SHA-256 mismatch: {actual}")
        source_hashes[name] = actual
    ref, replay = parse_csv(args.accepted), parse_csv(args.replay)
    if set(ref) != set(replay):
        raise ValueError("Different physical image sets")
    if {ident for rows in ref.values() for ident, _ in rows} != {
        ident for rows in replay.values() for ident, _ in rows
    }:
        raise ValueError("Different filament ID sets")

    per_image = []
    exact = changed = n_reordered = 0
    for image_id in sorted(ref):
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("Preregistered CPU wall-time budget exceeded")
        reference, candidate = ref[image_id], replay[image_id]
        n = len(reference)
        if n != len(candidate):
            raise ValueError(f"Different instance counts in image {image_id}")
        scores = np.asarray(coco.iou(
            [item[1] for item in candidate],
            [item[1] for item in reference], [0] * n,
        ), dtype=np.float64)
        if scores.shape != (n, n) or not np.isfinite(scores).all():
            raise AssertionError(f"Invalid IoU matrix: {image_id}")
        margin, best, row_idx, col_idx = alternative_margin(scores)
        reverse_best, _, _ = optimum(scores.T)
        if not math.isclose(best, reverse_best, abs_tol=1e-10):
            raise AssertionError(f"Reverse Hungarian optimum differs: {image_id}")

        image_exact = 0
        for i, j in zip(row_idx, col_idx):
            if scores[i, j] == 1.0:
                image_exact += 1
            if candidate[int(i)][0] != reference[int(j)][0]:
                n_reordered += 1
        exact += image_exact
        changed += n - image_exact
        per_image.append({
            "image_id": image_id, "instances": n,
            "best_total_iou": best, "unique_forced_single": n == 1,
            "optimality_margin": margin,
            "exact_pairs": image_exact, "changed_pairs": n - image_exact,
            "ties_at_1e12": margin is not None and margin <= TIE,
            "near_ties_at_1e8": margin is not None and margin <= NEAR,
        })
    if (exact, changed) != (1108, 133):
        raise AssertionError(f"Frozen P107/P109 control failed: {exact}/{changed}")
    margins = [v["optimality_margin"] for v in per_image
               if v["optimality_margin"] is not None]
    result = {
        "experiment": "P110", "status": "COMPLETE",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": args.protocol_sha,
        "source_sha256": sha256(Path(__file__)),
        "frozen_inputs_sha256": source_hashes,
        "synthetic_tests": tests,
        "image_count": len(per_image), "matched_pairs": exact + changed,
        "exact_pairs": exact, "changed_pairs": changed,
        "reordered_pairs_by_id": n_reordered,
        "forced_single_image_count": sum(v["unique_forced_single"] for v in per_image),
        "nonunique_image_count": sum(v["ties_at_1e12"] for v in per_image),
        "near_tied_image_count": sum(v["near_ties_at_1e8"] for v in per_image),
        "minimum_nontrivial_margin": min(margins) if margins else None,
        "hypothesis_falsified": any(v["ties_at_1e12"] for v in per_image),
        "per_image": per_image,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "promotion_allowed": False,
        "local_pq_new": None, "official_pq_new": None,
        "kaggle_submission_made": False,
        "interpretation": "Prediction-to-prediction matching identifiability; NOT a PQ or accuracy measurement",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "P110_FINAL_RECEIPT.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if temporary.stat().st_size > 5 * 1024 * 1024:
        raise RuntimeError("Preregistered disk output budget exceeded")
    temporary.replace(result_path)
    print(json.dumps({
        "receipt_sha256": sha256(result_path),
        "nonunique_images": result["nonunique_image_count"],
        "near_tied_images": result["near_tied_image_count"],
        "minimum_margin": result["minimum_nontrivial_margin"],
        "exact_pairs": exact, "changed_pairs": changed,
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=850)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 900:
        raise ValueError("Invalid CPU runtime budget")
    run(args)


if __name__ == "__main__":
    main()
