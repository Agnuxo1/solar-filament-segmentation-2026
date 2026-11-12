"""H6, step one: component table of the U-Net and Mask R-CNN fusion on every held-out group (CPU only).

See experiments/PROTOCOL_20261106.md, section P16. Per component it stores features that need no labels, and per
annotator record the hit pairs, so step two can score any subset of components with the official counting.

    python scripts/h6_features.py --maps DIR --dets DIR --w 0.7 --t 0.35 --out work/h6-20261106/components.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

import score_maps as S
from filseg.data import data_root, load_records
from filseg.metrics import gt_native

BLOCKS = ("F1", "F2", "F3", "F4")
MIN_AREA = 100
NAMES = ("log_area", "mean_p", "peak_p", "mean_u", "peak_u", "mean_m", "peak_m", "share_m", "extent", "aspect",
         "major", "minor", "radius", "n_comps", "peak_rank", "site")
YY, XX = np.mgrid[0:2048, 0:2048].astype(np.float32)


def mr_map(dets) -> np.ndarray:
    out = np.zeros((2048, 2048), np.float32)
    for score, y0, y1, x0, x1, crop in dets:
        soft = cv2.resize(crop.astype(np.float32) / 255.0, (2 * (x1 - x0), 2 * (y1 - y0)),
                          interpolation=cv2.INTER_LINEAR)
        view = out[2 * y0:2 * y1, 2 * x0:2 * x1]
        np.maximum(view, soft * float(score), out=view)
    return out


def table(prob, unet, mr, gts, t, site):
    lab, area, peak, mean = S.components(prob, t)
    big = np.nonzero(area >= MIN_AREA)[0]
    if big.size == 0:
        return np.zeros((0, len(NAMES)), np.float32), [(len(g), np.zeros(0, np.int64), np.zeros(0)) for g in gts]
    n = area.size
    flat = lab.ravel()

    def total(w):
        return np.bincount(flat, weights=w.ravel(), minlength=n + 1)[1:]

    idx = np.arange(1, n + 1)
    a = np.maximum(area, 1)
    cy, cx = total(YY) / a, total(XX) / a
    vy = total(YY * YY) / a - cy ** 2
    vx = total(XX * XX) / a - cx ** 2
    cov = total(YY * XX) / a - cy * cx
    root = np.sqrt(np.maximum((vy - vx) ** 2 / 4 + cov ** 2, 0))
    major = np.sqrt(np.maximum((vy + vx) / 2 + root, 0))
    minor = np.sqrt(np.maximum((vy + vx) / 2 - root, 0))
    boxes = ndi.find_objects(lab)
    height = np.array([b[0].stop - b[0].start for b in boxes], float)
    width = np.array([b[1].stop - b[1].start for b in boxes], float)
    order = np.argsort(np.argsort(-peak[big]))
    feats = np.stack([
        np.log(area), mean, peak, total(unet) / a, np.asarray(ndi.maximum(unet, lab, idx)), total(mr) / a,
        np.asarray(ndi.maximum(mr, lab, idx)), total((mr > 0.3).astype(np.float32)) / a, area / (height * width),
        np.maximum(height, width) / np.minimum(height, width), major, minor,
        np.hypot(cy - 1024, cx - 1024) / 1024], axis=1)[big]
    feats = np.column_stack([feats, np.full(big.size, big.size), order, np.full(big.size, site)]).astype(np.float32)
    new = np.full(n, -1, np.int64)
    new[big] = np.arange(big.size)
    records = []
    for g in gts:
        comp, ious = [], []
        for (y0, y1, x0, x1), m in g:
            g_area = float(m.sum())
            if g_area == 0:
                continue
            inter = np.bincount(lab[y0:y1, x0:x1][m], minlength=n + 1).astype(np.float64)[1:]
            iou = inter / np.maximum(g_area + area - inter, 1)
            for c in np.nonzero(iou > 0.5)[0]:
                if new[c] >= 0:
                    comp.append(int(new[c]))
                    ious.append(float(iou[c]))
        records.append((len(g), np.array(comp, np.int64), np.array(ious)))
    return feats, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--dets", type=Path, required=True)
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--member", default="resnet34_s0")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    _, records = load_records(data_root())
    polys: dict = {}
    for r in records:
        polys.setdefault(r.file, []).append(r.polygons)
    sites = sorted({Path(f).stem[-2] for f in polys})
    rows, done = {}, 0
    for b in BLOCKS:
        with (a.dets / f"dets_{b}.pkl").open("rb") as fh:       # own cache from dump_dets.py
            groups = pickle.load(fh)["groups"]
        for group in ("calibration", "test"):
            for f in sorted(groups[group]):
                gts = [gt_native(p) for p in polys[f]]
                stem = Path(f).stem
                unet = S.load_map(a.maps, b, a.member.split(","), group, stem)
                mr = mr_map(groups[group][f])
                prob = a.w * unet + (1 - a.w) * mr
                rows[(b, group, f)] = table(prob, unet, mr, gts, a.t, sites.index(stem[-2]))
                done += 1
                if done % 50 == 0:
                    print(done, flush=True)
    a.out.write_bytes(pickle.dumps({"names": NAMES, "sites": sites, "w": a.w, "t": a.t, "rows": rows}))
    print("done", done, flush=True)


if __name__ == "__main__":
    main()
