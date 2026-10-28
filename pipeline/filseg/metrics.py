"""Panoptic Quality as used by the challenge, at two resolutions.

* ``pq_counts_1024`` is a line-for-line port of the NeuroPixel control's
  evaluator (1024 label maps, greedy IoU > 0.5 matching). It is used only for
  like-for-like comparison with that control.
* ``native_counts`` follows the host's self-evaluation notebook at 2048x2048:
  every ground-truth/prediction pair with IoU > 0.5 is a true positive, an
  unmatched prediction is a false positive and an unmatched ground truth is a
  false negative. Ground-truth masks come straight from the COCO polygons.
"""
from __future__ import annotations

import numpy as np

from .data import NATIVE, polygon_mask

IOU_THR = 0.5


def pq_counts_1024(gt: np.ndarray, pr: np.ndarray):
    g_ids = [i for i in np.unique(gt) if i]
    p_ids = [i for i in np.unique(pr) if i]
    iou_sum, tp, matched_p = 0.0, 0, set()
    if g_ids and p_ids:
        ng, npr = max(g_ids) + 1, max(p_ids) + 1      # same table as np.add.at, computed with bincount
        inter = np.bincount(gt.ravel().astype(np.int64) * npr + pr.ravel(),
                            minlength=ng * npr).reshape(ng, npr).astype(np.float64)
        ga, pa = inter.sum(1), inter.sum(0)
        for g in g_ids:
            for p in p_ids:
                if inter[g, p] == 0 or p in matched_p:
                    continue
                iou = inter[g, p] / (ga[g] + pa[p] - inter[g, p])
                if iou > IOU_THR:
                    iou_sum += iou
                    tp += 1
                    matched_p.add(p)
                    break
    return iou_sum, tp, len(p_ids) - len(matched_p), len(g_ids) - tp


def gt_native(polygons) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Ground-truth masks as (bbox, cropped bool mask) to keep memory small."""
    out = []
    for poly in polygons:
        m = polygon_mask(poly, NATIVE).astype(bool)
        ys, xs = np.nonzero(m)
        if not len(ys):
            out.append(((0, 0, 0, 0), np.zeros((0, 0), bool)))
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        # A view keeps the entire 2048x2048 parent allocation alive for every
        # polygon. Own the small crop so retained memory matches the bbox.
        out.append(((y0, y1, x0, x1), m[y0:y1, x0:x1].copy()))
    return out


def native_counts(gts, pred_lab: np.ndarray):
    """Official-style counts for one annotation record. ``pred_lab`` is an int label map at 2048."""
    n_pred = int(pred_lab.max())
    pred_area = np.bincount(pred_lab.ravel(), minlength=n_pred + 1).astype(np.float64)
    hit_pred = np.zeros(n_pred + 1, bool)
    iou_sum, tp, fn, one_to_many = 0.0, 0, 0, 0
    for (y0, y1, x0, x1), m in gts:
        g_area = float(m.sum())
        if g_area == 0:
            fn += 1
            continue
        inter = np.bincount(pred_lab[y0:y1, x0:x1][m], minlength=n_pred + 1).astype(np.float64)
        inter[0] = 0
        iou = inter / np.maximum(g_area + pred_area - inter, 1)
        hits = np.nonzero(iou > IOU_THR)[0]
        if len(hits):
            tp += len(hits)
            iou_sum += float(iou[hits].sum())
            hit_pred[hits] = True
            one_to_many += int(len(hits) > 1)
        else:
            fn += 1
    present = pred_area > 0
    present[0] = False
    fp = int((present & ~hit_pred).sum())
    return iou_sum, tp, fp, fn


def summarize(counts) -> dict:
    s = np.asarray(counts, dtype=np.float64).reshape(-1, 4).sum(0)
    iou_sum, tp, fp, fn = s
    den = tp + 0.5 * fp + 0.5 * fn
    return {"PQ": float(iou_sum / den) if den else 0.0,
            "SQ": float(iou_sum / tp) if tp else 0.0,
            "RQ": float(tp / den) if den else 0.0,
            "TP": int(tp), "FP": int(fp), "FN": int(fn)}
