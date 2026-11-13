#!/usr/bin/env python3
"""Audit RLE-row provenance against the Kaggle-accepted submission.

This is a read-only, CPU-only forensic comparison of two prediction CSVs.
It never loads annotations, optimizes a model, or calls Kaggle services.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_FIELDS = ("filament_id", "segmentation_rle")


def sha256_file(path: Path) -> str:
    """Hash a file without loading the entire CSV into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_submission(path: Path) -> tuple[list[str], dict[str, str]]:
    """Validate a submission's schema and unique, nonempty row identifiers."""
    if not path.is_file():
        raise FileNotFoundError(path)
    identifiers: list[str] = []
    predictions: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or tuple(reader.fieldnames) != EXPECTED_FIELDS:
            raise ValueError(f"Incorrect schema in {path}: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            if None in row or row.get("filament_id") is None:
                raise ValueError(f"Malformed CSV at row {row_number}")
            key = row["filament_id"].strip()
            value = row["segmentation_rle"]
            if not key or not value:
                raise ValueError(f"Empty id or RLE at row {row_number}")
            if key in predictions:
                raise ValueError(f"Duplicate filament_id: {key}")
            predictions[key] = value
            identifiers.append(key)
    if not identifiers:
        raise ValueError(f"Submission has no rows: {path}")
    return identifiers, predictions


def image_id(filament_id: str) -> str:
    """Return the original observation identifier, without its instance suffix."""
    return filament_id.rsplit("_", 1)[0]


def compare_submissions(accepted: Path, candidate: Path) -> dict:
    """Compare CSV bytes, identifiers, ordering and RLE payloads independently."""
    reference_order, reference = load_submission(accepted)
    candidate_order, proposed = load_submission(candidate)
    common = sorted(reference.keys() & proposed.keys())
    missing = sorted(reference.keys() - proposed.keys())
    unexpected = sorted(proposed.keys() - reference.keys())
    different = [key for key in common if reference[key] != proposed[key]]
    identical = len(common) - len(different)
    counts = Counter(image_id(key) for key in different)
    payload = {
        "accepted_file": str(accepted),
        "candidate_file": str(candidate),
        "accepted_sha256": sha256_file(accepted),
        "candidate_sha256": sha256_file(candidate),
        "byte_identical": sha256_file(accepted) == sha256_file(candidate),
        "same_row_order": reference_order == candidate_order,
        "accepted_rows": len(reference_order),
        "candidate_rows": len(candidate_order),
        "common_ids": len(common),
        "exact_rle_rows": identical,
        "different_rle_rows": len(different),
        "missing_ids_count": len(missing),
        "unexpected_ids_count": len(unexpected),
        "affected_images": len(counts),
        "largest_image_mismatch_counts": [
            {"image_id": key, "mismatching_instances": value}
            for key, value in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        ],
        "first_mismatched_rows": [
            {
                "filament_id": key,
                "accepted_rle_sha256": hashlib.sha256(reference[key].encode()).hexdigest(),
                "candidate_rle_sha256": hashlib.sha256(proposed[key].encode()).hexdigest(),
            }
            for key in different[:20]
        ],
        "first_missing_ids": missing[:20],
        "first_unexpected_ids": unexpected[:20],
    }
    return payload


def self_test() -> None:
    """Exercise exact, changed, missing and duplicate-row cases without external data."""
    with tempfile.TemporaryDirectory() as directory:
        original = Path(directory) / "original.csv"
        second = Path(directory) / "second.csv"
        original.write_text(
            "filament_id,segmentation_rle\na_1,aa\na_2,bb\nb_1,cc\n",
            encoding="utf-8",
        )
        second.write_bytes(original.read_bytes())
        assert compare_submissions(original, second)["byte_identical"]
        second.write_text(
            "filament_id,segmentation_rle\na_1,aa\na_2,changed\nc_1,dd\n",
            encoding="utf-8",
        )
        result = compare_submissions(original, second)
        assert result["different_rle_rows"] == 1
        assert result["missing_ids_count"] == 1
        assert result["unexpected_ids_count"] == 1
        assert result["affected_images"] == 1
        second.write_text(
            "filament_id,segmentation_rle\na_1,aa\na_1,bb\n",
            encoding="utf-8",
        )
        try:
            load_submission(second)
        except ValueError as exc:
            assert "Duplicate" in str(exc)
        else:
            raise AssertionError("Duplicate identifiers must be rejected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, help="Historically accepted submission CSV")
    parser.add_argument("--candidate", type=Path, help="Independently regenerated candidate CSV")
    parser.add_argument("--output", type=Path, help="HDD JSON evidence receipt")
    parser.add_argument("--expected-accepted-sha256", default="", help="Optional hard integrity gate")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic correctness tests")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        print("P90_SELF_TEST_PASS")
        if not arguments.accepted and not arguments.candidate:
            return 0
    if not (arguments.accepted and arguments.candidate and arguments.output):
        parser.error("--accepted, --candidate and --output are all required")
    result = compare_submissions(arguments.accepted, arguments.candidate)
    if arguments.expected_accepted_sha256 and result["accepted_sha256"] != arguments.expected_accepted_sha256:
        raise ValueError("Historical accepted CSV checksum mismatch; audit stopped")
    result.update({
        "protocol": "P90_SCIENTIFIC_REPLAY_GAP",
        "utc": datetime.now(timezone.utc).isoformat(),
        "input_labels_used": False,
        "competition_test_annotations_used": False,
        "new_cuda_processes": 0,
        "new_kaggle_submissions": 0,
        "scope": "CSV RLE fingerprint and row provenance only; no PQ inference",
    })
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.output)
    print(json.dumps({
        "status": "P90_CSV_COMPARISON_COMPLETE",
        "byte_identical": result["byte_identical"],
        "different_rle_rows": result["different_rle_rows"],
        "missing_ids_count": result["missing_ids_count"],
        "unexpected_ids_count": result["unexpected_ids_count"],
        "affected_images": result["affected_images"],
        "receipt": str(arguments.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
