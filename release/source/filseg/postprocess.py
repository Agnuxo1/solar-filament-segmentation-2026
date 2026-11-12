"""From probability maps / instance proposals to non-overlapping filament instances."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .data import NATIVE

_EIGHT = np.ones((3, 3), bool)


def _relabel_by_area(lab: np.ndarray, min_area: int) -> np.ndarray:
    n = int(lab.max())
    if not n:
        return lab.astype(np.int32)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    keep = area >= min_area
    keep[0] = False
    new = np.zeros(n + 1, np.int32)
    new[keep] = np.arange(1, int(keep.sum()) + 1)
    return new[lab]


def instances_from_prob(prob: np.ndarray, thr: float, min_area: int, close: int = 0, link: int = 0) -> np.ndarray:
    """Threshold, optional closing, connected components (8-neighbour) and an area floor.

    ``link`` groups fragments whose gap is at most ~2*link pixels into one instance
    without adding the gap pixels to the mask (filaments are often broken by noise).
    """
    m = prob > thr
    if close:
        m = ndi.binary_closing(m, structure=_EIGHT, iterations=close)
    if link:
        grp, _ = ndi.label(ndi.binary_dilation(m, structure=_EIGHT, iterations=link), structure=_EIGHT)
        lab = np.where(m, grp, 0)
    else:
        lab, _ = ndi.label(m, structure=_EIGHT)
    return _relabel_by_area(lab, min_area)


def to_native(lab: np.ndarray, prob: np.ndarray, thr: float) -> np.ndarray:
    """Refine a working-resolution instance map to 2048: bilinear probability, nearest instance id."""
    if lab.shape[0] == NATIVE:
        return lab.astype(np.int32)
    up_prob = cv2.resize(prob.astype(np.float32), (NATIVE, NATIVE), interpolation=cv2.INTER_LINEAR)
    grown = ndi.grey_dilation(lab, size=(3, 3))
    up_lab = cv2.resize(grown.astype(np.int32).astype(np.float32), (NATIVE, NATIVE),
                        interpolation=cv2.INTER_NEAREST).astype(np.int32)
    out = np.where(up_prob > thr, up_lab, 0)
    return _relabel_by_area(out, 1)


def paint_instances(dets, size: int, score_thr: float, mask_thr: float,
                    min_area: int) -> tuple[np.ndarray, np.ndarray]:
    """Instance proposals -> exclusive label map (higher score wins) and a probability map.

    ``dets`` is a list of ``(score, y0, y1, x0, x1, crop)`` with ``crop`` a uint8 soft mask
    (0..255) covering rows y0:y1 and columns x0:x1 of a ``size`` x ``size`` image.
    """
    lab = np.zeros((size, size), np.int32)
    prob = np.zeros((size, size), np.float32)
    k = 0
    for score, y0, y1, x0, x1, crop in sorted(dets, key=lambda d: -d[0]):
        if score < score_thr:
            break
        soft = crop.astype(np.float32) / 255.0
        m = (soft > mask_thr) & (lab[y0:y1, x0:x1] == 0)
        if m.sum() < min_area:
            continue
        k += 1
        lab[y0:y1, x0:x1][m] = k
        prob[y0:y1, x0:x1] = np.maximum(prob[y0:y1, x0:x1], np.where(m, soft, 0))
    return lab, prob


def rle_counts(mask: np.ndarray) -> str:
    from pycocotools import mask as mask_utils
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return r["counts"].decode("ascii")
