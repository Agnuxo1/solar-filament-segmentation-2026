"""Candidate H9b on the 180 official images: H6 selection, then the mean of the two refiners (CPU only).

See experiments/PROTOCOL_20261106.md, sections P16, P20 and P21. Two steps around the GPU:

    python scripts/h9_official.py crops --out DIR [--no-site]     # H6 decision and rotated crops of the kept instances
    python scripts/h9_official.py paste --out DIR --probs FILE [FILE ...]   # refined instances back, CSV and report

The H6 rule is fit on every row of the component table with its frozen hyperparameters. An image with no surviving
component keeps its strongest component of area >= 60, and an image with none takes the P1 rows, as in h6_official.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import h9_crops as C
import h9_score as H
import score_maps as S
from filseg.data import data_root
from filseg.postprocess import rle_counts
from h6_features import MIN_AREA, mr_map, table
from h6_official import FALLBACK_AREA, TAU, unet_map
from h6_rule import make_model, target

import os
O = Path(os.environ.get("SOLARFIL_ASSET_ROOT", "/mnt/datos/solar-filament-20261106"))


def components(name: str, w: float, t: float):
    stem = Path(name).stem
    unet = unet_map(stem)
    with (O / "dets_official.pkl").open("rb") as fh:              # own cache from dump_official.py
        dets = pickle.load(fh)[name]
    mr = mr_map(dets)
    prob = w * unet + (1 - w) * mr
    lab, area, peak, _ = S.components(prob, t)
    return prob, unet, mr, lab, area, peak


def crops(a) -> None:
    data = pickle.loads((O / "h6/components.pkl").read_bytes())  # own cache from h6_features.py
    names, sites, w, t = list(data["names"]), data["sites"], data["w"], data["t"]
    cols = [i for i, n in enumerate(names) if not (a.no_site and n == "site")]
    X, y, sw = [], [], []
    for feats, records in data["rows"].values():
        if len(feats):
            X.append(feats[:, cols])
            y.append(target(len(feats), records))
            sw.append(np.full(len(feats), len(records), float))
    model = (HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
                                           l2_regularization=1.0, random_state=0) if a.no_site
             else make_model([names[i] for i in cols]))
    model.fit(np.concatenate(X), np.concatenate(y), sample_weight=np.concatenate(sw))
    with (O / "dets_official.pkl").open("rb") as fh:              # own cache from dump_official.py
        names_official = sorted(pickle.load(fh))
    images = data_root() / "test" / "test_images"
    report = []
    for name in names_official:
        stem = Path(name).stem
        prob, unet, mr, lab, area, peak = components(name, w, t)
        feats, _ = table(prob, unet, mr, [], t, sites.index(stem[-2]))
        big = np.nonzero(area >= MIN_AREA)[0]
        pred = model.predict(feats[:, cols]) if len(feats) else np.zeros(0)
        keep, score, source = big[pred > TAU], pred[pred > TAU], "rule"
        if keep.size == 0 and (area >= FALLBACK_AREA).any():
            keep = np.array([int(np.argmax(np.where(area >= FALLBACK_AREA, peak, -1)))])
            score, source = np.ones(1), "strongest"
        k = keep.size
        Xc = np.zeros((k, 4, C.CH, C.CW), np.uint8)
        Ms = np.zeros((k, 2, 3), np.float32)
        if k:
            flat = C.flat_field(cv2.imread(str(images / name), cv2.IMREAD_GRAYSCALE))
            p8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
            all_big = np.isin(lab, big + 1).astype(np.uint8) * 255
            size = (C.CW, C.CH)
            for i, c in enumerate(keep):
                seed = lab == c + 1
                ys, xs = np.nonzero(seed)
                M = C.frame(xs, ys)
                Ms[i] = M
                s8 = cv2.warpAffine(seed.astype(np.uint8) * 255, M, size, flags=cv2.INTER_NEAREST)
                Xc[i, 0] = cv2.warpAffine(flat, M, size, flags=cv2.INTER_LINEAR, borderValue=127)
                Xc[i, 1] = cv2.warpAffine(p8, M, size, flags=cv2.INTER_LINEAR)
                Xc[i, 2] = s8
                Xc[i, 3] = np.where(s8 > 0, 0, cv2.warpAffine(all_big, M, size, flags=cv2.INTER_NEAREST))
        np.savez_compressed(a.out / "crops" / f"{stem}.npz", X=Xc, M=Ms, comp=keep.astype(np.int64), score=score)
        report.append({"image": stem, "kept": int(k), "candidates": int(big.size), "source": source if k else "control"})
    (a.out / "crops_report.json").write_text(json.dumps({"no_site": a.no_site, "per_image": report}, indent=1),
                                             encoding="utf-8")
    print("images", len(report), "kept", sum(r["kept"] for r in report), flush=True)


def paste(a) -> None:
    data = pickle.loads((O / "h6/components.pkl").read_bytes())  # own cache from h6_features.py
    w, t = data["w"], data["t"]
    zs = [np.load(p) for p in a.probs]
    control: dict = {}
    with (O / "submission.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            control.setdefault(row["filament_id"].rsplit("_", 1)[0], []).append(row["segmentation_rle"])
    rows, report = [], []
    for path in sorted((a.out / "crops").glob("*.npz")):
        stem = path.stem
        z = np.load(path)
        comp, score, M = z["comp"], z["score"], z["M"]
        masks = []
        if comp.size:
            _, _, _, lab, _, _ = components(stem + ".jpeg", w, t)
            refined = np.mean([q[stem].astype(np.float32) for q in zs], 0).round().astype(np.uint8)
            seeds = [H.refined(lab == c + 1, refined[i], M[i]) for i, c in enumerate(comp)]
            final = H.assemble(seeds, np.argsort(-score, kind="stable"))
            masks = [rle_counts(final == k) for k in range(1, int(final.max()) + 1)]
        source = "refined" if masks else "control"
        if not masks:
            masks = control.get(stem, [])
        rows += [(f"{stem}_{k}", m) for k, m in enumerate(masks, 1)]
        report.append({"image": stem, "instances": len(masks), "source": source})
    sub = a.out / "submission_h9b.csv"
    with sub.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["filament_id", "segmentation_rle"])
        wr.writerows(rows)
    summary = {"rows": len(rows), "images": len(report), "sha256": hashlib.sha256(sub.read_bytes()).hexdigest(),
               "sources": {s: sum(r["source"] == s for r in report) for s in ("refined", "control")}}
    (a.out / "submission_h9b_report.json").write_text(json.dumps({**summary, "per_image": report}, indent=1),
                                                      encoding="utf-8")
    print(json.dumps(summary), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("crops", "paste"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-site", action="store_true")
    ap.add_argument("--probs", type=Path, nargs="+")
    a = ap.parse_args()
    (a.out / "crops").mkdir(parents=True, exist_ok=True)
    crops(a) if a.step == "crops" else paste(a)


if __name__ == "__main__":
    main()
