#!/usr/bin/env python3
"""Post-hoc paired physical-image bootstrap of frozen, TRAIN-derived PQ reports.

Do not select Kaggle submissions with these already examined local test groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ARMS = ("3600_control", "3600_treatment", "7200_control", "7200_treatment")
IMAGE_COUNTS = {1: 139, 2: 146, 3: 139, 4: 143}
N_BOOTSTRAP = 4096
TASKS = [
    (1, "P59_original", "evaluations/test_pq_report.json"),
    *[(1, str(seed), f"seeds/seed{seed}/evaluations/test_pq_report.json")
      for seed in (20261113, 20261114, 20261115)],
    *[(fold, str(seed), f"folds/runs/F{fold}_seed{seed}/evaluations/test_pq_report.json")
      for fold in (2, 3, 4) for seed in (20261112, 20261113)],
]


def pq(counts: np.ndarray) -> np.ndarray:
    """Compute the official count-based PQ on one or more image aggregates."""
    numerator = counts[..., 0]
    denominator = counts[..., 1] + .5 * (counts[..., 2] + counts[..., 3])
    return np.divide(numerator, denominator,
                     out=np.zeros_like(numerator, dtype=np.float64),
                     where=denominator > 0)


def parse_report(path: Path, fold: int) -> tuple[dict, dict[str, np.ndarray]]:
    """Validate the native metric, arm schema, image count and frozen policy."""
    content = path.read_bytes()
    data = json.loads(content)
    if data.get("fold") != fold or data.get("status") != f"P59_F{fold}_HELDOUT_TEST_SCORED_NOT_PROMOTED":
        raise ValueError(f"Wrong fold or status: {path}")
    if not isinstance(data.get("selected_policy"), dict):
        raise ValueError(f"Missing frozen policy: {path}")
    if set(data.get("comparisons", {})) != set(ARMS):
        raise ValueError(f"Wrong arms: {path}")
    out = {}
    for arm in ARMS:
        item = data["comparisons"][arm]
        values = np.asarray(item["counts_per_physical_image"], dtype=np.float64)
        if values.shape != (IMAGE_COUNTS[fold], 4):
            raise ValueError(f"Wrong physical image count: {fold}/{arm}")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid physical image counts: {fold}/{arm}")
        expected = float(item["summary"]["PQ"])
        actual = float(pq(values.sum(axis=0)))
        if not math.isclose(actual, expected, abs_tol=1e-11):
            raise ValueError(f"Official metric mismatch: {fold}/{arm}")
        out[arm] = values
    return {"sha256": hashlib.sha256(content).hexdigest(),
            "policy": data["selected_policy"],
            "scores": {a: float(pq(v.sum(axis=0))) for a, v in out.items()},
            "completed_utc": data.get("completed_utc")}, out


def paired_bootstrap(control: np.ndarray, treatment: np.ndarray,
                     seed: int, replicates: int = N_BOOTSTRAP) -> dict:
    """Resample physical images identically across both arms."""
    if control.shape != treatment.shape or control.ndim != 2 or control.shape[1] != 4:
        raise ValueError("Expected aligned per-image counting tuples")
    n = control.shape[0]
    rng = np.random.default_rng(seed)
    deltas = []
    for begin in range(0, replicates, 256):
        k = min(256, replicates - begin)
        indices = rng.integers(0, n, size=(k, n))
        a = pq(control[indices].sum(axis=1))
        b = pq(treatment[indices].sum(axis=1))
        deltas.extend((b - a).tolist())
    lower, upper = np.quantile(deltas, [.025, .975])
    measured = float(pq(treatment.sum(axis=0)) - pq(control.sum(axis=0)))
    return {"delta": measured, "ci95_within_image": [float(lower), float(upper)],
            "bootstrap_physical_images": n, "bootstrap_repetitions": replicates}


def audit(root: Path) -> dict:
    """Audit all ten planned reports; abort on missing or inconsistent evidence."""
    rows, policies = [], {}
    for fold, label, relative in TASKS:
        path = root / relative
        meta, counts = parse_report(path, fold)
        if fold in policies and meta["policy"] != policies[fold]:
            raise ValueError(f"Policy changed across seed evaluations in fold F{fold}")
        policies[fold] = meta["policy"]
        bootstrap = paired_bootstrap(counts["7200_control"],
                                     counts["7200_treatment"],
                                     seed=20261112 + 100 * fold + len(rows))
        rows.append({"fold": fold, "seed": label, "relative_source": relative,
                     **meta, **bootstrap})
    positive = sum(x["delta"] > 0 for x in rows)
    by_fold = {}
    for fold in IMAGE_COUNTS:
        current = [x for x in rows if x["fold"] == fold]
        by_fold[f"F{fold}"] = {
            "replicates": len(current),
            "mean_matched_step_delta_descriptive": float(
                np.mean([x["delta"] for x in current])),
            "positive_replicates": sum(x["delta"] > 0 for x in current),
            "image_count": IMAGE_COUNTS[fold],
        }
    return {
        "status": "POST_HOC_CROSSFOLD_AUDIT_COMPLETE_NO_PROMOTION",
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "study": "P84",
        "samples": "Kaggle TRAIN-derived held-out images only; already examined",
        "n_reports": len(rows), "positive_matched_step_replicates": positive,
        "negative_matched_step_replicates": len(rows) - positive,
        "folds": by_fold, "reports": rows, "frozen_policies": policies,
        "interpretation": ("Descriptive only: repeated F1 test, shared image groups, "
                           "two seeds per F2-F4, no inferential pooling or Kaggle promotion."),
    }


def self_test() -> None:
    """Reject misaligned inputs and prove deterministic paired uncertainty."""
    a = np.array([[2., 3., 1., 0.], [1., 2., 1., 1.]], dtype=float)
    b = np.array([[1., 2., 1., 1.], [2., 3., 0., 1.]], dtype=float)
    assert math.isclose(float(pq(a.sum(axis=0))), 3 / 6.5, abs_tol=1e-12)
    r1 = paired_bootstrap(a, b, 42, 512)
    r2 = paired_bootstrap(a, b, 42, 512)
    assert r1 == r2 and r1["bootstrap_physical_images"] == 2
    try:
        paired_bootstrap(a, b[:1], 42)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched physical-image count was accepted")
    print("P84_SELF_TEST_PASSED")


def main() -> None:
    """Run tests or emit an explicit, no-promotion audit with source hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None or args.out is None:
        parser.error("--root and --out are required")
    result = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps({key: result[key] for key in
                      ("status", "n_reports", "positive_matched_step_replicates",
                       "negative_matched_step_replicates", "folds")}, indent=2))


if __name__ == "__main__":
    main()
