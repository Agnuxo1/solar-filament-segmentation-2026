"""Candidate H6: U-Net and Mask R-CNN fusion with the learned keep rule on the 180 official images (CPU only).

See experiments/PROTOCOL_20261106.md, section P16. The final model is fit on every row of the component table with the
frozen hyperparameters, and a component stays when its predicted mean sum of IoU clears tau = PQ / 2 of P11. An image
with no surviving component keeps its strongest component of area >= 60, and an image with none takes the P1 rows.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

import score_maps as S
from filseg.postprocess import rle_counts
from h6_features import MIN_AREA, mr_map, table
from h6_rule import make_model, target

import os
O = Path(os.environ.get("SOLARFIL_ASSET_ROOT", "/mnt/datos/solar-filament-20261106"))
TAU, FALLBACK_AREA = 0.2217106230425525, 60
BLOCKS = ("F1", "F2", "F3", "F4")


def unet_map(stem: str) -> np.ndarray:
    acc = np.zeros((2048, 2048), np.float32)
    for b in BLOCKS:
        p = cv2.imread(str(O / "h3/maps" / f"{b}_resnet34_s0/official/{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if p is None:
            raise IOError(f"missing map {b}/{stem}")
        acc += p
    return acc / len(BLOCKS) / 255.0


def main() -> None:
    data = pickle.loads((O / "h6/components.pkl").read_bytes())  # own cache from h6_features.py
    names, sites, w, t = list(data["names"]), data["sites"], data["w"], data["t"]
    X, y, sw = [], [], []
    for feats, records in data["rows"].values():
        if len(feats):
            X.append(feats)
            y.append(target(len(feats), records))
            sw.append(np.full(len(feats), len(records), float))
    X, y, sw = np.concatenate(X), np.concatenate(y), np.concatenate(sw)
    model = make_model(names).fit(X, y, sample_weight=sw)
    print("fit", X.shape, flush=True)
    with (O / "dets_official.pkl").open("rb") as fh:              # own cache from dump_official.py
        dets = pickle.load(fh)
    control: dict = {}
    with (O / "submission.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            control.setdefault(row["filament_id"].rsplit("_", 1)[0], []).append(row["segmentation_rle"])
    rows, report = [], []
    for name in sorted(dets):
        stem = Path(name).stem
        unet, mr = unet_map(stem), mr_map(dets[name])
        prob = w * unet + (1 - w) * mr
        feats, _ = table(prob, unet, mr, [], t, sites.index(stem[-2]))
        lab, area, peak, _ = S.components(prob, t)
        big = np.nonzero(area >= MIN_AREA)[0]
        pred = model.predict(feats) if len(feats) else np.zeros(0)
        keep = big[pred > TAU]
        source = "rule"
        if keep.size == 0 and (area >= FALLBACK_AREA).any():
            keep = np.array([int(np.argmax(np.where(area >= FALLBACK_AREA, peak, -1)))])
            source = "strongest"
        masks = [rle_counts(lab == c + 1) for c in keep]
        if not masks:
            masks, source = control.get(stem, []), "control"
        rows += [(f"{stem}_{k}", m) for k, m in enumerate(masks, 1)]
        report.append({"image": stem, "instances": len(masks), "candidates": int(big.size), "source": source})
    sub = O / "submission_h6.csv"
    with sub.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["filament_id", "segmentation_rle"])
        wr.writerows(rows)
    summary = {"rows": len(rows), "images": len(report), "sha256": hashlib.sha256(sub.read_bytes()).hexdigest(),
               "sources": {s: sum(r["source"] == s for r in report) for s in ("rule", "strongest", "control")}}
    (O / "submission_h6_report.json").write_text(json.dumps({**summary, "per_image": report}, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
