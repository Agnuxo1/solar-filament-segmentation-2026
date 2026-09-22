from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.audit_submission import audit_submission
from scripts.evaluate_pq import evaluate, polygon_to_rle


def _write_prediction(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)


def test_official_pq_is_one_for_an_identical_mask(tmp_path: Path) -> None:
    polygon = [[0, 0, 10, 0, 10, 10, 0, 10]]
    image_id = "040301-20200101000000Ch"
    ground_truth = {
        "images": [{"id": image_id, "file_name": "20200101000000Ch.jpeg", "height": 2048, "width": 2048}],
        "annotations": [{"id": "a", "image_id": image_id, "segmentation": polygon}],
        "categories": [],
    }
    gt_path = tmp_path / "annotations.json"
    gt_path.write_text(json.dumps(ground_truth), encoding="utf-8")
    rle = polygon_to_rle(polygon, 2048, 2048)
    prediction = tmp_path / "prediction.csv"
    _write_prediction(prediction, [("20200101000000Ch_1", rle)])

    result = evaluate(gt_path, prediction)

    assert result["pq"] == pytest.approx(1.0)
    assert result["true_positive_pairs"] == 1
    assert result["false_positive_predictions"] == 0
    assert result["false_negative_ground_truth"] == 0


def test_submission_audit_rejects_overlapping_instances(tmp_path: Path) -> None:
    test_images = tmp_path / "test_images"
    test_images.mkdir()
    (test_images / "20200101000000Ch.jpeg").touch()
    polygon = [[0, 0, 10, 0, 10, 10, 0, 10]]
    rle = polygon_to_rle(polygon, 2048, 2048)
    prediction = tmp_path / "prediction.csv"
    _write_prediction(
        prediction,
        [("20200101000000Ch_1", rle), ("20200101000000Ch_2", rle)],
    )

    with pytest.raises(ValueError, match="overlap"):
        audit_submission(prediction, test_images)
