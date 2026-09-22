"""Official Panoptic Quality evaluator for the 2026 filament challenge.

The matching rule intentionally mirrors the shared Kaggle notebook: every
ground-truth/prediction pair with IoU > 0.5 contributes a true positive. This
is not a one-to-one Hungarian matching metric.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

HEIGHT = 2048
WIDTH = 2048
IOU_THRESHOLD = 0.5


def _counts_text(rle: dict[str, Any]) -> str:
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def polygon_to_rle(segmentation: list[list[float]], height: int, width: int) -> str:
    """Rasterize one COCO polygon list and return compressed COCO RLE text."""
    encoded = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    return _counts_text(encoded)


def decode_rles(rles: list[str], height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """Return masks as ``(n, height, width)`` uint8 arrays."""
    if not rles:
        return np.zeros((0, height, width), dtype=np.uint8)
    encoded = [{"size": [height, width], "counts": rle} for rle in rles]
    decoded = mask_utils.decode(encoded)
    if decoded.ndim == 2:
        decoded = decoded[:, :, None]
    return np.moveaxis(decoded, 2, 0).astype(np.uint8, copy=False)


def _read_predictions(path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("prediction CSV must contain exactly filament_id,segmentation_rle")
        for row in rows:
            filament_id = (row["filament_id"] or "").strip()
            if "_" not in filament_id:
                raise ValueError(f"invalid filament_id: {filament_id!r}")
            physical_id = filament_id.split("_", 1)[0]
            grouped[physical_id].append(row["segmentation_rle"] or "")
    return grouped


def _read_ground_truth(path: Path) -> dict[str, tuple[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {str(item["id"]): item for item in payload["images"]}
    grouped: dict[str, list[str]] = defaultdict(list)
    for annotation in payload["annotations"]:
        image = images[str(annotation["image_id"])]
        grouped[str(annotation["image_id"])].append(
            polygon_to_rle(annotation["segmentation"], image["height"], image["width"])
        )
    return {
        image_id: (Path(str(images[image_id]["file_name"])).stem, rles)
        for image_id, rles in grouped.items()
    }


def evaluate(ground_truth: Path, prediction: Path) -> dict[str, Any]:
    gt_by_annotator_image = _read_ground_truth(ground_truth)
    pred_by_physical_image = _read_predictions(prediction)
    iou_sum = 0.0
    true_positive = false_positive = false_negative = 0
    one_to_many = many_to_one = 0

    for _annotator_image, (physical_id, gt_rles) in gt_by_annotator_image.items():
        gt = decode_rles(gt_rles)
        pred = decode_rles(pred_by_physical_image.get(physical_id, []))
        if not len(gt):
            false_positive += len(pred)
            continue
        if not len(pred):
            false_negative += len(gt)
            continue
        gt_flat = gt.reshape(len(gt), -1).astype(np.float32)
        pred_flat = pred.reshape(len(pred), -1).astype(np.float32)
        intersection = gt_flat @ pred_flat.T
        gt_area = gt_flat.sum(axis=1, keepdims=True)
        pred_area = pred_flat.sum(axis=1, keepdims=True).T
        union = gt_area + pred_area - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        hits = iou > IOU_THRESHOLD
        true_positive += int(hits.sum())
        false_positive += int((hits.sum(axis=0) == 0).sum())
        false_negative += int((hits.sum(axis=1) == 0).sum())
        iou_sum += float(iou[hits].sum())
        one_to_many += int((hits.sum(axis=1) > 1).sum())
        many_to_one += int((hits.sum(axis=0) > 1).sum())

    denominator = true_positive + 0.5 * false_positive + 0.5 * false_negative
    return {
        "pq": iou_sum / denominator if denominator else 0.0,
        "iou_threshold": IOU_THRESHOLD,
        "iou_sum_true_positives": iou_sum,
        "true_positive_pairs": true_positive,
        "false_positive_predictions": false_positive,
        "false_negative_ground_truth": false_negative,
        "one_to_many_ground_truth": one_to_many,
        "many_to_one_predictions": many_to_one,
        "annotator_images_evaluated": len(gt_by_annotator_image),
        "physical_images_with_predictions": len(pred_by_physical_image),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = evaluate(args.ground_truth, args.prediction)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
