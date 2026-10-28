"""Inference on the official test images and Kaggle submission CSV.

    python -m filseg.predict --images DIR --unet ckpt.pt [--unet ckpt2.pt ...] --cfg cfg.json --out submission.csv

Each row is one predicted filament: ``filament_id`` = ``<image stem>_<k>``, ``segmentation_rle`` =
compressed COCO RLE counts of a 2048x2048 mask. Masks are mutually exclusive.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import autocast, to_input
from .data import read_gray
from .evaluate import make_labels
from .postprocess import rle_counts, to_native


def load_unets(paths, arch, encoder, device):
    from .train_unet import build_model
    models = []
    for p in paths:
        m = build_model(arch, encoder, None)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        models.append(m.to(device).eval())
    return models


@torch.no_grad()
def unet_prob(models, img1024: np.ndarray, device, tta: bool) -> np.ndarray:
    x = to_input(torch.from_numpy(img1024)[None]).to(device)
    flips = [(), (-1,), (-2,), (-1, -2)] if tta else [()]
    acc = 0
    with autocast(device):
        for m in models:
            for d in flips:
                xi = x.flip(d) if d else x
                y = torch.sigmoid(m(xi).float())
                acc = acc + (y.flip(d) if d else y)
    return (acc / (len(models) * len(flips)))[0, 0].float().cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--unet", type=Path, nargs="+", required=True)
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--cfg", type=Path, required=True, help="JSON with thr/min_area/close/link")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads(a.cfg.read_text(encoding="utf-8"))
    models = load_unets(a.unet, a.arch, a.encoder, device)
    paths = sorted(a.images.glob("*.jpeg"))
    t0, rows = time.time(), []
    for p in paths:
        prob = unet_prob(models, read_gray(p), device, a.tta)
        lab, prob, thr = make_labels(prob.astype(np.float16), cfg)
        native = to_native(lab, prob, thr)
        for k in range(1, int(native.max()) + 1):
            m = native == k
            if m.any():
                rows.append((f"{p.stem}_{k}", rle_counts(m)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)
    print(json.dumps({"images": len(paths), "instances": len(rows), "seconds": round(time.time() - t0, 1),
                      "out": str(a.out)}))


if __name__ == "__main__":
    main()
