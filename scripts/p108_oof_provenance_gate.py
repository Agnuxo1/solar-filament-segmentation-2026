#!/usr/bin/env python3
"""P108 read-only provenance gate for a proposed TRAIN-only context ablation.

All findings are metadata-only. This script deliberately never computes a PQ
score, loads image/annotation tensors, invokes CUDA, or modifies a submission.
A missing exclusion manifest is a failed precondition, not evidence of leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_P106_SHA = "16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3"
EXPECTED_H9B_SHA = "24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60"
EXPECTED_REF_POINTS = (
    "refiner_ALL.pt",
    "refiner_ALL_efficientnet_v2_s.pt",
)
NEEDED_IMAGE_EXCLUSION_ROLES = (
    "H3 semantic predictors",
    "Mask R-CNN proposal models",
    "no-site selector",
    "ResNet34 crop refiner",
    "EfficientNetV2-S crop refiner",
)


def file_sha256(path: Path) -> str:
    """Stream a SHA-256 digest without retaining large files in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    """Load an on-disk JSON dictionary and reject malformed contracts."""
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object JSON: {path.name}")
    return obj


def analyze(root: Path) -> dict:
    """Check immutable prerequisites without using labels or fitted models."""
    split_path = root / "split_F1.json"
    receipt_path = root / "p106_public_clean_6843362_20261113" / "P106_FINAL_RECEIPT.json"
    selector_path = root / "p106_checkout_6843362" / "release" / "model_metadata" / "h9b_nosite_selector.json"
    checkout = root / "p106_checkout_6843362"
    split = load_json(split_path)
    receipt = load_json(receipt_path)
    selector = load_json(selector_path)

    partitions = {}
    for key in ("train", "calibration", "test"):
        raw = split.get("files", {}).get(key)
        if not isinstance(raw, list) or any(not isinstance(name, str) for name in raw):
            raise ValueError(f"Missing or invalid F1 split partition: {key}")
        partitions[key] = set(raw)
    overlaps = {
        "train_calibration": len(partitions["train"] & partitions["calibration"]),
        "train_test": len(partitions["train"] & partitions["test"]),
        "calibration_test": len(partitions["calibration"] & partitions["test"]),
    }
    partition_sizes = {k: len(v) for k, v in partitions.items()}
    duplicates = {key: len(split["files"][key]) - len(value)
                  for key, value in partitions.items()}

    p106_complete = (
        receipt.get("status") == "P106_ALL_EIGHT_NOTEBOOK_CELLS_COMPLETED"
        and len(receipt.get("code_cells", [])) == 8
        and receipt.get("csv_sha256") == EXPECTED_P106_SHA
        and receipt.get("test_annotations_used") is False
        and receipt.get("kaggle_submission_made") is False
        and receipt.get("accepted_comparison", {}).get("reference_sha256") == EXPECTED_H9B_SHA
    )

    # The released ALL refiners are legitimate for competition TEST inference.
    # They must not be reused to evaluate labelled TRAIN images.
    release_refiners = sorted(p.name for p in (checkout / "release" / "models").glob("*refiner*.pt"))
    available_fold_specific_refiners = sorted({
        p.name for d in (root / "folds", root / "seeds", checkout / "release" / "models")
        if d.is_dir() for p in d.rglob("*refiner*.pt")
        if "ALL" not in p.name.upper()
    })

    # Check whether a frozen per-image exclusion certificate is directly
    # attached to each checkpoint and selector. The selector metadata has
    # feature-family exclusions, not TRAIN-membership exclusions.
    exclusion_fields = ("per_image_exclusions", "training_image_ids",
                        "held_out_image_ids", "out_of_fold_manifest",
                        "source_split_manifest")
    selector_exclusion_fields = [key for key in exclusion_fields if key in selector]
    selector_oof_certified = False
    # A list of fitted images alone is insufficient without matching upstream
    # maps and refiner exclusions; consequently this audit stays fail-closed.
    refiner_oof_certified = bool(available_fold_specific_refiners) and False
    upstream_provenance_certified = False
    oof_evidence = {role: False for role in NEEDED_IMAGE_EXCLUSION_ROLES}

    # Ancillary inventory only. NPZ counts do NOT certify out-of-sample status.
    p59_cache = root / "p59_train_inference"
    cache_stems = {p.stem for p in p59_cache.glob("*.npz")} if p59_cache.exists() else set()
    map_presence = {k: len({Path(name).stem for name in names} & cache_stems)
                    for k, names in partitions.items()}

    gates = {
        "F1_physical_filename_partitions_disjoint":
            all(value == 0 for value in overlaps.values())
            and all(value == 0 for value in duplicates.values()),
        "P106_eight_cell_receipt_integrity": p106_complete,
        "F1_cohort_sizes_match_frozen_protocol":
            partition_sizes["calibration"] == 117 and partition_sizes["test"] == 139,
        "per_image_OOF_provenance_for_all_upstream_models":
            upstream_provenance_certified,
        "per_image_OOF_provenance_for_both_refiners":
            refiner_oof_certified,
        "per_image_OOF_provenance_for_fitted_selector":
            selector_oof_certified,
        "frozen_OOF_baseline_and_1p15_context_inputs_proven":
            False,
    }

    return {
        "protocol": "P108_PREREG_PROVENANCE_GATE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only F1 TRAIN provenance assessment; NOT a PQ evaluation",
        "input_sha256": {
            "split_F1.json": file_sha256(split_path),
            "P106_FINAL_RECEIPT.json": file_sha256(receipt_path),
            "h9b_nosite_selector.json": file_sha256(selector_path),
        },
        "frozen_cohort_counts": partition_sizes,
        "within_partition_duplicate_counts": duplicates,
        "between_partition_overlap_counts": overlaps,
        "existing_p59_train_inference_npz_count": len(cache_stems),
        "p59_npz_filename_matches_per_F1_cohort_not_OOF_proof": map_presence,
        "release_refiner_checkpoint_names": release_refiners,
        "non_ALL_refiner_checkpoints_found_in_scoped_folds_seeds_release": available_fold_specific_refiners,
        "selector_feature_rows_used_for_training": selector.get("train_feature_rows"),
        "selector_metadata_oof_exclusion_fields_found": selector_exclusion_fields,
        "model_role_image_exclusion_evidence": oof_evidence,
        "gates": gates,
        "status": "ELIGIBLE" if all(gates.values()) else "BLOCKED_NO_VERIFIED_OOF_PROVENANCE",
        "run_1p15_context_ablation": all(gates.values()),
        "official_PQ_new": None,
        "local_PQ_new": None,
        "kaggle_submission_made": False,
        "limitations": [
            "Scope is the frozen project and two released model families, not every storage device.",
            "An unverified or absent OOF manifest is not proof of label leakage.",
            "F1 cohorts have been examined previously and cannot constitute an unseen blind test.",
            "The accepted H9b submission is not modified by this audit.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root.resolve())
    if args.output.resolve().is_relative_to(args.root.resolve()) is False:
        raise ValueError("Output must remain inside the project HDD directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "gates": result["gates"],
        "frozen_cohort_counts": result["frozen_cohort_counts"],
        "input_sha256": result["input_sha256"],
        "receipt": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
