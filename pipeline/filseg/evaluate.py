"""Post-processing selection and scoring on a record group (calibration or test)."""
from __future__ import annotations

from collections import Counter

import itertools

import numpy as np

from .metrics import gt_native, native_counts, pq_counts_1024, summarize
from .postprocess import instances_from_prob, paint_instances, to_native

SEMANTIC_GRID = {"thr": [0.3, 0.4, 0.5, 0.6, 0.7], "min_area": [20, 60, 120, 240],
                 "close": [0, 2], "link": [0, 3, 6]}
INSTANCE_GRID = {"score_thr": [0.3, 0.5, 0.7, 0.85], "mask_thr": [0.4, 0.5, 0.6], "min_area": [20, 60, 120]}


def grid(spec: dict) -> list[dict]:
    keys = list(spec)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(spec[k] for k in keys))]


def make_labels(pred, cfg: dict, size: int = 1024):
    """Return (label map at working resolution, probability map, native threshold)."""
    if isinstance(pred, np.ndarray):                                   # semantic probability map
        prob = pred.astype(np.float32)
        lab = instances_from_prob(prob, cfg["thr"], cfg["min_area"], cfg["close"], cfg["link"])
        return lab, prob, cfg["thr"]
    lab, prob = paint_instances(pred, size, cfg["score_thr"], cfg["mask_thr"], cfg["min_area"])
    return lab, prob, cfg["mask_thr"]


class GroupScorer:
    """Scores per-file predictions against every annotation record of a group."""

    def __init__(self, record_idx, meta, labs, records_full=None, native=True):
        self.records = [meta["records"][i] for i in record_idx]
        self.labs = labs
        self.idx = list(record_idx)
        self.native = native and records_full is not None
        # Keep polygons, not every full-resolution rasterised mask in RAM.
        self.native_polygons = [records_full[i].polygons for i in record_idx] if self.native else None

    def files(self):
        return sorted({r["file"] for r in self.records})

    def score(self, preds: dict, cfg: dict, native: bool = False) -> dict:
        cache, counts, ncounts = {}, [], []
        remaining = Counter(r['file'] for r in self.records)
        for k, (i, r) in enumerate(zip(self.idx, self.records)):
            f = r["file"]
            if f not in cache:
                cache[f] = make_labels(preds[f], cfg)
            lab, prob, thr = cache[f]
            counts.append(pq_counts_1024(np.asarray(self.labs[i]), lab))
            if native:
                key = f + "#native"
                if key not in cache:
                    cache[key] = to_native(lab, prob, thr)
                ncounts.append(native_counts(gt_native(self.native_polygons[k]), cache[key]))
            remaining[f] -= 1
            if remaining[f] == 0:
                cache.pop(f, None)
                cache.pop(f + '#native', None)
        out = {"r1024": summarize(counts)}
        if native:
            out["native"] = summarize(ncounts)
        return out

    def select(self, preds: dict, spec: dict, top_native: int = 6) -> tuple[dict, dict, list]:
        """Grid on 1024 PQ, confirm the best configurations at native resolution, pick by native PQ."""
        rows = []
        for cfg in grid(spec):
            rows.append((self.score(preds, cfg)["r1024"]["PQ"], cfg))
        rows.sort(key=lambda r: -r[0])
        use_native = self.native and top_native > 0
        best, table = None, []
        for _, cfg in rows[:top_native if use_native else 1]:
            s = self.score(preds, cfg, native=use_native)
            table.append({"cfg": cfg, **s})
            key = s["native"]["PQ"] if use_native else s["r1024"]["PQ"]
            if best is None or key > best[0]:
                best = (key, cfg, s)
        return best[1], best[2], table
