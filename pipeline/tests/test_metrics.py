"""Native PQ must agree with a dense re-implementation of the host's self-evaluation rule."""
import numpy as np

from filseg.metrics import native_counts, pq_counts_1024, summarize
from filseg.postprocess import instances_from_prob, paint_instances, to_native


def dense_official(gt_masks, pred_lab):
    preds = [pred_lab == k for k in range(1, int(pred_lab.max()) + 1) if (pred_lab == k).any()]
    if not preds:
        return 0.0, 0, 0, len(gt_masks)
    g = np.stack([m.ravel() for m in gt_masks]).astype(np.float64)
    p = np.stack([m.ravel() for m in preds]).astype(np.float64)
    inter = g @ p.T
    union = g.sum(1, keepdims=True) + p.sum(1)[None] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0)
    hits = iou > 0.5
    return float(iou[hits].sum()), int(hits.sum()), int((hits.sum(0) == 0).sum()), int((hits.sum(1) == 0).sum())


def as_gts(masks):
    out = []
    for m in masks:
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        out.append(((y0, y1, x0, x1), m[y0:y1, x0:x1]))
    return out


def test_native_matches_dense_rule():
    rng = np.random.default_rng(0)
    for _ in range(20):
        size = 64
        gts = []
        for _ in range(rng.integers(1, 5)):
            m = np.zeros((size, size), bool)
            y, x = rng.integers(0, size - 12, 2)
            m[y:y + rng.integers(3, 12), x:x + rng.integers(3, 12)] = True
            gts.append(m)
        prob = rng.random((size, size)) * 0.4
        for m in gts:
            prob[m] += rng.uniform(0.2, 0.8)
        lab = instances_from_prob(prob, 0.5, 2)
        got = native_counts(as_gts(gts), lab)
        want = dense_official(gts, lab)
        assert np.isclose(got[0], want[0]) and got[1:] == want[1:]


def test_perfect_prediction_scores_one():
    gt = np.zeros((32, 32), np.uint8)
    gt[2:8, 2:20] = 1
    gt[20:30, 5:9] = 2
    s = summarize([pq_counts_1024(gt, gt.astype(np.int32))])
    assert s["PQ"] == 1.0 and s["FP"] == 0 and s["FN"] == 0


def test_paint_is_exclusive_and_ordered():
    crop = np.full((10, 10), 255, np.uint8)
    lab, _ = paint_instances([(0.9, 0, 10, 0, 10, crop), (0.8, 5, 15, 5, 15, crop)], 20, 0.5, 0.5, 1)
    assert set(np.unique(lab)) == {0, 1, 2}
    assert (lab[0:10, 0:10] == 1).all()


def test_to_native_keeps_instances():
    prob = np.zeros((1024, 1024), np.float32)
    prob[100:110, 100:300] = 0.9
    prob[500:700, 600:605] = 0.9
    lab = instances_from_prob(prob, 0.5, 10)
    nat = to_native(lab, prob, 0.5)
    assert nat.shape == (2048, 2048) and nat.max() == 2
