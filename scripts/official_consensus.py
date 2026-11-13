#!/usr/bin/env python3
"""Reproducible, CPU-only cross-view Mask R-CNN proposal consensus.

This clean-checkout implementation ports the frozen 6 November 2026
`make_submission.py` inference rule. The port was independently audited
against the archived source and historical aggregate in experiments P98–P99.

Each detection is (score, y0, y1, x0, x1, uint8_soft_mask_crop), with
exclusive upper bounds. Inputs are eight views (F1–F4, normal then unflipped).
No training, CUDA, private caches or competition labels are required.
"""
from __future__ import annotations

import numpy as np

MASK_IOU = 0.5
UNSUPPORTED_FACTOR = 0.75
MIN_SCORE = 0.3


def masked(detections: list) -> list:
    """Discard subthreshold or empty proposals; threshold mask bytes at 127."""
    output = []
    for detection in detections:
        if detection[0] < MIN_SCORE:
            continue
        binary_mask = detection[-1] > 127
        area = int(np.count_nonzero(binary_mask))
        if area:
            output.append((detection, binary_mask, area))
    return output


def mask_iou_ge(a: tuple, b: tuple) -> bool:
    """Check intersection over union without a costly full-image allocation."""
    (_, y0, y1, x0, x1, _), mask, area = a
    (_, oy0, oy1, ox0, ox1, _), other_mask, other_area = b
    iy0, iy1 = max(y0, oy0), min(y1, oy1)
    ix0, ix1 = max(x0, ox0), min(x1, ox1)
    if iy0 >= iy1 or ix0 >= ix1:
        return False
    intersection = int(np.count_nonzero(
        mask[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] &
        other_mask[iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0]
    ))
    return bool(intersection and intersection >= MASK_IOU * (area + other_area - intersection))


def consensus(views: list[list]) -> list:
    """Rescore proposals by cross-view support, then greedy mask-IoU NMS.

    A proposal supported by any *different* view keeps its score; otherwise
    the score is multiplied by the preregistered factor 0.75. Stable Python
    sorting preserves the original relative order for equal scores.
    """
    sides = [masked(view) for view in views]
    pooled = []
    for index, side in enumerate(sides):
        for proposal in side:
            supported = any(
                mask_iou_ge(proposal, other)
                for other_index, opposite in enumerate(sides)
                if other_index != index
                for other in opposite
            )
            detection = proposal[0]
            factor = 1.0 if supported else UNSUPPORTED_FACTOR
            pooled.append(((detection[0] * factor, *detection[1:]),
                           proposal[1], proposal[2]))
    kept = []
    for detection, mask, area in sorted(pooled, key=lambda item: -item[0][0]):
        if detection[0] < MIN_SCORE:
            break
        if not any(mask_iou_ge((detection, mask, area), earlier) for earlier in kept):
            kept.append((detection, mask, area))
    return [detection for detection, _, _ in kept]


__all__ = ["consensus", "masked", "mask_iou_ge",
           "MASK_IOU", "UNSUPPORTED_FACTOR", "MIN_SCORE"]
