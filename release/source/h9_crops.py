"""H9, step one: rotated native crops of every H6 component, with the consensus of the records that drew it (CPU only).

See experiments/PROTOCOL_20261106.md, section P20. One compressed file per held-out row, so the job can resume.

    python scripts/h9_crops.py --maps DIR --dets DIR --out DIR [--w 0.7 --t 0.35 --workers 4]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
from pathlib import Path

import cv2
import numpy as np

import score_maps as S
from filseg.data import data_root, load_records
from filseg.metrics import gt_native
from h6_features import BLOCKS, MIN_AREA, mr_map

CW, CH, PAD, MATCH = 384, 128, 1.5, 0.2
CENTRE, RADIUS, SIGMA = (1023.1, 1024.1), 905.0, 60.0
WORK: dict = {}
YY, XX = np.mgrid[0:2048, 0:2048].astype(np.float32)
DISK = (np.hypot(XX - CENTRE[0], YY - CENTRE[1]) < RADIUS).astype(np.float32)
WEIGHT = np.maximum(cv2.GaussianBlur(DISK, (0, 0), SIGMA), 1e-3)


def flat_field(img: np.ndarray) -> np.ndarray:
    """Image over its slowly varying background inside the disk, as uint8 on [0.5, 1.5]."""
    x = img.astype(np.float32)
    back = cv2.GaussianBlur(x * DISK, (0, 0), SIGMA) / WEIGHT
    ratio = np.where(DISK > 0, x / np.maximum(back, 1.0), 1.0)
    return (np.clip(ratio - 0.5, 0, 1) * 255).astype(np.uint8)


def frame(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Affine map from the image to a crop with the long axis horizontal; scales down only when it does not fit."""
    (cx, cy), (w, h), angle = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    if h > w:
        w, h, angle = h, w, angle + 90
    scale = min(1.0, CW / (PAD * max(w, 1.0)), CH / (PAD * max(h, 1.0) + 16))
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle), float(scale))
    M[0, 2] += CW / 2 - cx
    M[1, 2] += CH / 2 - cy
    return M


def row(task):
    block, group, name = task
    stem = Path(name).stem
    dst = WORK["out"] / f"{block}_{group}_{stem}.npz"
    if dst.exists():
        return task
    cv2.setNumThreads(1)
    unet = S.load_map(WORK["maps"], block, WORK["member"].split(","), group, stem)
    prob = WORK["w"] * unet + (1 - WORK["w"]) * mr_map(WORK["dets"][block][group][name])
    lab, area, _, _ = S.components(prob, WORK["t"])
    big = np.nonzero(area >= MIN_AREA)[0]
    n = big.size
    X = np.zeros((n, 4, CH, CW), np.uint8)
    Y = np.zeros((n, CH, CW), np.uint8)
    has = np.zeros(n, bool)
    Ms = np.zeros((n, 2, 3), np.float32)
    if n:
        flat = flat_field(cv2.imread(str(WORK["images"] / name), cv2.IMREAD_GRAYSCALE))
        p8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        all_big = np.isin(lab, big + 1).astype(np.uint8) * 255
        gts = [gt_native(p) for p in WORK["polys"][name]]
        best = []
        for g in gts:                                    # per record: the filament of largest IoU with each component
            top, which = np.zeros(area.size), np.full(area.size, -1)
            for i, ((y0, y1, x0, x1), m) in enumerate(g):
                g_area = float(m.sum())
                if g_area == 0:
                    continue
                inter = np.bincount(lab[y0:y1, x0:x1][m], minlength=area.size + 1).astype(np.float64)[1:]
                iou = inter / np.maximum(g_area + area - inter, 1)
                better = iou > top
                top[better], which[better] = iou[better], i
            best.append((top, which))
        acc = np.zeros((2048, 2048), np.float32)
        for k, c in enumerate(big):
            seed = lab == c + 1
            ys, xs = np.nonzero(seed)
            M = frame(xs, ys)
            Ms[k] = M
            size = (CW, CH)
            s8 = cv2.warpAffine(seed.astype(np.uint8) * 255, M, size, flags=cv2.INTER_NEAREST)
            X[k, 0] = cv2.warpAffine(flat, M, size, flags=cv2.INTER_LINEAR, borderValue=127)
            X[k, 1] = cv2.warpAffine(p8, M, size, flags=cv2.INTER_LINEAR)
            X[k, 2] = s8
            X[k, 3] = np.where(s8 > 0, 0, cv2.warpAffine(all_big, M, size, flags=cv2.INTER_NEAREST))
            drawn = [(g, w[c]) for g, (t, w) in zip(gts, best) if t[c] >= MATCH]
            if drawn:
                for g, i in drawn:
                    (y0, y1, x0, x1), m = g[i]
                    acc[y0:y1, x0:x1] += m
                Y[k] = np.clip(cv2.warpAffine(acc, M, size, flags=cv2.INTER_LINEAR) / len(drawn) * 255 + 0.5, 0, 255)
                has[k] = True
                for g, i in drawn:
                    (y0, y1, x0, x1), _ = g[i]
                    acc[y0:y1, x0:x1] = 0
    np.savez_compressed(dst, X=X, Y=Y, has=has, M=Ms)
    return task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--dets", type=Path, required=True)
    ap.add_argument("--w", type=float, default=0.7)
    ap.add_argument("--t", type=float, default=0.35)
    ap.add_argument("--member", default="resnet34_s0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _, records = load_records(data_root())
    polys: dict = {}
    for r in records:
        polys.setdefault(r.file, []).append(r.polygons)
    dets, tasks = {}, []
    for b in BLOCKS:
        with (a.dets / f"dets_{b}.pkl").open("rb") as fh:       # own cache from dump_dets.py
            dets[b] = pickle.load(fh)["groups"]
        tasks += [(b, g, f) for g in ("calibration", "test") for f in sorted(dets[b][g])]
    WORK.update(polys=polys, dets=dets, maps=a.maps, member=a.member, w=a.w, t=a.t, out=a.out,
                images=data_root() / "train" / "train_images")
    tasks = tasks[:a.limit or None]
    with mp.get_context("fork").Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(row, tasks), 1):
            if n % 50 == 0:
                print(n, len(tasks), flush=True)
    print("done", len(tasks), flush=True)


if __name__ == "__main__":
    main()
