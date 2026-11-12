"""Score probability maps with the keep-rule grid of protocol P7 (CPU only, official counting).

Components of one threshold are pixel-disjoint, so the hit pairs between components and ground-truth
filaments do not depend on which components a rule keeps. They are computed once per image and
threshold, and every rule in the grid is then evaluated from that table.

    python scripts/score_maps.py --maps DIR --members resnet34_s0 [efficientnet_v2_s_s0 ...]
           --out DIR [--control DIR] [--csv submission.csv]

`--maps` holds `<block>_<member>/<group>/<stem>.png`. Members of one block are averaged. One global
rule is chosen on the calibration groups of F1, F2 and F3, then the test groups are scored once.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

ROOT = Path("D:/PROJECTS/SolarFilamentSegmentation2026")
sys.path.insert(0, str(ROOT / "pipeline"))
os.environ.setdefault("SOLARFIL_DATA", "D:/PROJECTS/196_NeuroPixel/kaggle/filament/data/MAGFiLO_1.0_Kaggle_2026")
from filseg.data import data_root, load_records           # noqa: E402
from filseg.metrics import gt_native, summarize           # noqa: E402
from filseg.postprocess import rle_counts                  # noqa: E402

SELECT = ("F1", "F2", "F3")
BLOCKS = ("F1", "F2", "F3", "F4")
T_LO = (0.25, 0.35, 0.45)
PEAK = (0.90, 0.95, 0.97, 0.99, 0.995)
MEAN = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90)
AREA = (100, 200, 300)
RULES = [dict(t_lo=a, peak=b, mean=c, area=d) for a, b, c, d in itertools.product(T_LO, PEAK, MEAN, AREA)]


def load_map(maps: Path, block: str, members, group: str, stem: str) -> np.ndarray:
    acc = None
    for m in members:
        p = cv2.imread(str(maps / f"{block}_{m}" / group / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if p is None:
            raise IOError(f"missing map {block}_{m}/{group}/{stem}")
        acc = p.astype(np.float32) if acc is None else acc + p
    return acc / len(members) / 255.0


def components(prob: np.ndarray, t_lo: float):
    n, lab = cv2.connectedComponents((prob > t_lo).astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    if n <= 1:
        return lab, np.zeros(0), np.zeros(0), np.zeros(0)
    idx = np.arange(1, n)
    area = np.bincount(lab.ravel(), minlength=n)[1:].astype(np.float64)
    mean = np.bincount(lab.ravel(), weights=prob.ravel(), minlength=n)[1:] / np.maximum(area, 1)
    peak = np.asarray(ndi.maximum(prob, lab, idx))
    return lab, area, peak, mean


def image_table(prob: np.ndarray, records) -> dict:
    """Per threshold: component statistics and, per record, the (component, IoU) hit pairs."""
    table = {}
    for t in T_LO:
        lab, area, peak, mean = components(prob, t)
        n = area.size
        per_record = []
        for gts in records:
            comp, ious = [], []
            for (y0, y1, x0, x1), m in gts:
                g_area = float(m.sum())
                if g_area == 0 or n == 0:
                    continue
                inter = np.bincount(lab[y0:y1, x0:x1][m], minlength=n + 1).astype(np.float64)[1:]
                iou = inter / np.maximum(g_area + area - inter, 1)
                for c in np.nonzero(iou > 0.5)[0]:
                    comp.append(int(c))
                    ious.append(float(iou[c]))
            per_record.append((len(gts), np.array(comp, dtype=np.int64), np.array(ious)))
        table[t] = (area, peak, mean, per_record)
    return table


def counts(table: dict, rule: dict) -> np.ndarray:
    area, peak, mean, per_record = table[rule["t_lo"]]
    keep = (area >= rule["area"]) & (peak > rule["peak"]) & (mean >= rule["mean"])
    kept = int(keep.sum())
    out = np.zeros(4)
    for n_gt, comp, ious in per_record:
        sel = keep[comp] if comp.size else np.zeros(0, bool)
        tp = int(sel.sum())
        out += (float(ious[sel].sum()), tp, kept - np.unique(comp[sel]).size, n_gt - tp)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--control", type=Path)
    ap.add_argument("--blocks", nargs="+", default=list(BLOCKS))
    ap.add_argument("--select", nargs="+", default=list(SELECT))
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--csv-block", default="F1")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _, records = load_records(data_root())
    wanted = {f.stem + ".jpeg" for b in a.blocks for g in ("calibration", "test")
              for f in (a.maps / f"{b}_{a.members[0]}" / g).glob("*.png")}
    polys: dict = {}
    for r in records:
        if r.file in wanted:                                  # rasterised per image: all of them would not fit in RAM
            polys.setdefault(r.file, []).append(r.polygons)

    tables: dict = {}
    for b in a.blocks:
        tables[b] = {}
        for group in ("calibration", "test"):
            folder = a.maps / f"{b}_{a.members[0]}" / group
            stems = sorted(p.stem for p in folder.glob("*.png"))
            tables[b][group] = {
                s: image_table(load_map(a.maps, b, a.members, group, s), [gt_native(p) for p in polys[s + ".jpeg"]])
                for s in stems}
            print(f"{b}/{group}: {len(stems)} images", flush=True)

    def pooled(block, group, rule):
        return summarize([counts(t, rule) for t in tables[block][group].values()])

    rows = []
    for rule in RULES:
        per = {b: pooled(b, "calibration", rule) for b in a.select if b in tables}
        rows.append({"rule": rule, "mean_pq": float(np.mean([v["PQ"] for v in per.values()])), "blocks": per})
    rows.sort(key=lambda r: -r["mean_pq"])
    best = rows[0]["rule"]
    print(json.dumps({"chosen": best, "top": [{"rule": r["rule"], "mean_pq": round(r["mean_pq"], 5)}
                                              for r in rows[:6]]}, indent=1), flush=True)
    summary = {"members": a.members, "select_blocks": a.select, "chosen": best,
               "calibration_mean_pq": rows[0]["mean_pq"], "blocks": {}}
    for b in a.blocks:
        per_image = {s + ".jpeg": counts(t, best).tolist() for s, t in tables[b]["test"].items()}
        split: dict = {}
        for f, c in per_image.items():
            split.setdefault(len(polys[f]), []).append(c)
        summary["blocks"][b] = {"test": summarize(list(per_image.values())),
                                "by_records": {str(k): {"images": len(v), **summarize(v)}
                                               for k, v in sorted(split.items())}}
        if a.control and (a.control / f"gate_postgrid_{b}.json").exists():
            ctrl = json.loads((a.control / f"gate_postgrid_{b}.json").read_text(encoding="utf-8"))["control"]
            (a.out / f"gate_{b}.json").write_text(json.dumps(
                {"block": b, "control": ctrl, "candidate": per_image}), encoding="utf-8")
    (a.out / "selection.json").write_text(json.dumps({**summary, "table": rows[:200]}, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "table"}, indent=1), flush=True)

    if a.csv:
        folder = a.maps / f"{a.csv_block}_{a.members[0]}" / "official"
        out_rows = []
        for p in sorted(folder.glob("*.png")):
            prob = load_map(a.maps, a.csv_block, a.members, "official", p.stem)
            lab, area, peak, mean = components(prob, best["t_lo"])
            keep = np.nonzero((area >= best["area"]) & (peak > best["peak"]) & (mean >= best["mean"]))[0]
            for k, c in enumerate(keep, 1):
                out_rows.append((f"{p.stem}_{k}", rle_counts(lab == c + 1)))
        with a.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["filament_id", "segmentation_rle"])
            w.writerows(out_rows)
        print(f"wrote {a.csv} with {len(out_rows)} rows", flush=True)


if __name__ == "__main__":
    main()
