#!/usr/bin/env python3
"""Audit preregistered F1 seed replication without access to hidden test labels.

The input files are the local, held-out TRAIN-derived F1 native-PQ reports.
This program never loads an image, checkpoint, CUDA device or Kaggle credentials.
Reports are *descriptive*. F1 test was previously examined, so no reported
result should be used to select a new Kaggle submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

SEEDS = (20261113, 20261114, 20261115)
ARMS = ("3600_control", "3600_treatment", "7200_control", "7200_treatment")
POLICY = {"thr": 0.25, "min_area": 200, "peak": 0.95, "mean": 0.7}
REFERENCE = "3600_control"
TEST_IMAGE_COUNT = 139


def read_report(path: Path, seed: int) -> dict:
    """Validate one completed native-PQ report and retain only aggregate data."""
    original = path.read_bytes()
    data = json.loads(original)
    if data.get("fold") != 1:
        raise ValueError(f"Wrong fold in {path}")
    if data.get("selected_policy") != POLICY:
        raise ValueError(f"Unfrozen postprocessing policy in {path}")
    if data.get("status") != "P59_F1_HELDOUT_TEST_SCORED_NOT_PROMOTED":
        raise ValueError(f"Unexpected evaluation status in {path}")
    arms = data["comparisons"]
    if set(ARMS) != set(arms):
        raise ValueError(f"Missing or unexpected model arm in {path}")
    scores = {}
    for arm in ARMS:
        entry = arms[arm]
        pq = float(entry["summary"]["PQ"])
        if not math.isfinite(pq) or not 0.0 <= pq <= 1.0:
            raise ValueError(f"Invalid PQ for {seed}, {arm}")
        if len(entry["counts_per_physical_image"]) != TEST_IMAGE_COUNT:
            raise ValueError(f"Wrong physical-image count for {seed}, {arm}")
        scores[arm] = pq

    effects = {}
    reported = data["deltas_vs_frozen_reference"]
    if set(reported) != set(ARMS) - {REFERENCE}:
        raise ValueError(f"Wrong comparison set in {path}")
    for arm in ARMS[1:]:
        item = reported[arm]
        if item["bootstrap_physical_images"] != TEST_IMAGE_COUNT:
            raise ValueError("Incorrect bootstrap sampling unit/count")
        if item["repetitions"] < 2000:
            raise ValueError("Insufficient bootstrap resamples")
        delta = float(item["delta"])
        expected = scores[arm] - scores[REFERENCE]
        if not math.isclose(delta, expected, abs_tol=1e-10):
            raise ValueError(f"Difference does not reconcile: {seed}, {arm}")
        lower, upper = map(float, item["CI95"])
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            raise ValueError(f"Invalid CI: {seed}, {arm}")
        effects[arm] = {
            "delta_vs_3600_control": delta,
            "ci95": [lower, upper],
            "ci_excludes_zero": lower > 0 or upper < 0,
            "direction": "positive" if lower > 0 else "negative" if upper < 0 else "inconclusive",
        }
    return {
        "seed": seed,
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "completed_utc": data["completed_utc"],
        "pq": scores,
        "effects": effects,
        "matched_step_treatment_minus_control": scores["7200_treatment"] - scores["7200_control"],
        "policy": data["selected_policy"],
        "policy_file_sha256": data.get("policy_sha256"),
    }


def summarize(root: Path, require_all: bool = False) -> dict:
    """Aggregate available independent seeds, explicitly recording missing ones."""
    completed, pending = [], []
    for seed in SEEDS:
        path = root / "seeds" / f"seed{seed}" / "evaluations" / "test_pq_report.json"
        if not path.is_file():
            pending.append(seed)
            continue
        completed.append(read_report(path, seed))
    if require_all and pending:
        raise RuntimeError(f"Replication incomplete: missing {pending}")
    mean_scores = (
        {arm: sum(e["pq"][arm] for e in completed) / len(completed) for arm in ARMS}
        if completed else {}
    )
    return {
        "experiment": "P83_F1_INDEPENDENT_SEED_REPLICATION_EVALUATION",
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_DESCRIPTIVE_NO_PROMOTION" if not pending else "PARTIAL_NO_PROMOTION",
        "seeds_completed": len(completed),
        "seeds_expected": len(SEEDS),
        "missing_seeds": pending,
        "calibration_count": 117,
        "heldout_train_derived_test_count": TEST_IMAGE_COUNT,
        "frozen_selected_policy": POLICY,
        "replications": completed,
        "descriptive_mean_pq": mean_scores,
        "scientific_caution": (
            "F1 test has already been examined in P59. This is exploratory "
            "replication, not new independent blind validation. The native "
            "TRAIN-derived test PQ is not an official Kaggle score. "
            "No automatic promotion or Kaggle submission is authorized."
        ),
    }


def to_markdown(result: dict) -> str:
    """Render a compact, reviewer-readable receipt with explicit uncertainty."""
    lines = [
        "# P83 independent-seed F1 native-PQ replication audit",
        "",
        f"Status: **{result['status']}**. Completed {result['seeds_completed']}/{result['seeds_expected']}; "
        f"pending: {result['missing_seeds']}.",
        "",
        "The frozen policy is threshold 0.25, minimum area 200, peak 0.95, mean 0.7. "
        "The 139-image F1 test partition derives exclusively from competition TRAIN. "
        "It has already been examined in P59; these are exploratory repeated "
        "holdout evaluations, not new blind validation or Kaggle scores.",
        "",
        "| Seed | 3600 control | 3600 treatment | 7200 control | 7200 treatment | "
        "Treatment-control at 7200 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for e in result["replications"]:
        a = e["pq"]
        lines.append(
            f"| {e['seed']} | {a['3600_control']:.6f} | {a['3600_treatment']:.6f} | "
            f"{a['7200_control']:.6f} | {a['7200_treatment']:.6f} | "
            f"{e['matched_step_treatment_minus_control']:+.6f} |"
        )
    if result["descriptive_mean_pq"]:
        a = result["descriptive_mean_pq"]
        lines.append(
            f"| Descriptive mean | {a['3600_control']:.6f} | {a['3600_treatment']:.6f} | "
            f"{a['7200_control']:.6f} | {a['7200_treatment']:.6f} | — |"
        )
    lines.extend([
        "",
        "Each native evaluator additionally reports image-cluster bootstrap intervals "
        "versus its own 3600-step control. Full per-seed CIs, source file SHA-256, "
        "and selected policies are in the companion JSON receipt; the small number "
        "of seeds does not justify a confident cross-seed improvement claim.",
        "",
        "**Decision: HOLD/NO PROMOTION.** Do not claim a Kaggle improvement, "
        "change the accepted H9b submission, or optimize the previously examined "
        "F1 test. No hidden test labels were used.",
        "",
    ])
    return "\n".join(lines)


def self_test() -> None:
    """Exercise fold, policy, physical-image and strict-completeness gates."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert summarize(root)["status"] == "PARTIAL_NO_PROMOTION"
        try:
            summarize(root, require_all=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Incomplete report accepted in strict mode")

        seed = SEEDS[0]
        report_path = root / "seeds" / f"seed{seed}" / "evaluations" / "test_pq_report.json"
        report_path.parent.mkdir(parents=True)
        values = dict(zip(ARMS, (0.4, 0.39, 0.41, 0.405)))
        fixture = {
            "fold": 1,
            "status": "P59_F1_HELDOUT_TEST_SCORED_NOT_PROMOTED",
            "selected_policy": dict(POLICY),
            "policy_sha256": "a" * 64,
            "completed_utc": "2026-11-12T00:00:00+00:00",
            "comparisons": {
                arm: {
                    "summary": {"PQ": values[arm]},
                    "counts_per_physical_image": [[0.1, 1, 1, 1]] * TEST_IMAGE_COUNT,
                }
                for arm in ARMS
            },
            "deltas_vs_frozen_reference": {
                arm: {
                    "bootstrap_physical_images": TEST_IMAGE_COUNT,
                    "repetitions": 2000,
                    "CI95": [-0.02, 0.02],
                    "delta": values[arm] - values[REFERENCE],
                }
                for arm in ARMS if arm != REFERENCE
            },
        }

        def write_fixture() -> None:
            report_path.write_text(json.dumps(fixture), encoding="utf8")

        write_fixture()
        assert read_report(report_path, seed)["pq"]["7200_control"] == 0.41
        assert summarize(root)["seeds_completed"] == 1
        assert "NO PROMOTION" in to_markdown(summarize(root))
        for key, altered in (
            ("fold", 2),
            ("selected_policy", {**POLICY, "thr": 0.9}),
        ):
            before = fixture[key]
            fixture[key] = altered
            write_fixture()
            try:
                read_report(report_path, seed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Unsafe {key} accepted")
            fixture[key] = before
        fixture["comparisons"]["7200_control"]["counts_per_physical_image"] = [[0.1, 1, 1, 1]]
        write_fixture()
        try:
            read_report(report_path, seed)
        except ValueError:
            pass
        else:
            raise AssertionError("Incorrect test image count accepted")

    print("P83_AUDIT_SELF_TEST_PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Research HDD root")
    parser.add_argument("--json-out", type=Path, help="Write aggregate JSON")
    parser.add_argument("--md-out", type=Path, help="Write Markdown summary")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None:
        parser.error("--root is required unless --self-test is used")
    result = summarize(args.root, require_all=args.require_all)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(to_markdown(result), encoding="utf8")
    print(json.dumps({
        "status": result["status"],
        "seeds_completed": result["seeds_completed"],
        "missing_seeds": result["missing_seeds"],
        "descriptive_mean_pq": result["descriptive_mean_pq"],
    }, indent=2))


if __name__ == "__main__":
    main()
