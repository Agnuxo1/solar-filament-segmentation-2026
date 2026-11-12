"""H9, step three: paste the refined instances back and score them with the official counting (CPU only).

See experiments/PROTOCOL_20261106.md, section P20. With --identity each instance stays as its seed, which must give back
H6 image by image before any refined result is read.

    python scripts/h9_score.py --maps DIR --dets DIR --crops DIR --keep keep.pkl --h6 DIR --probs DIR --out DIR [--identity]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

import score_maps as S
from filseg.data import data_root, load_records
from filseg.metrics import gt_native, native_counts
from h6_features import BLOCKS, MIN_AREA, mr_map

T_PIXEL, COVERED = 0.40, 0.5
WORK: dict = {}


def refined(seed: np.ndarray, crop: np.ndarray, M: np.ndarray) -> np.ndarray:
    full = cv2.warpAffine(crop, M, (2048, 2048), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    mask = full > T_PIXEL * 255
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return seed
    touching = np.unique(lab[seed])
    touching = touching[touching > 0]
    return np.isin(lab, touching) if touching.size else seed


def assemble(seeds: list, order: np.ndarray) -> np.ndarray:
    """Instances by decreasing H6 prediction; one mostly covered by earlier ones is absorbed, the rest yield pixels."""
    out = np.zeros((2048, 2048), np.int32)
    k = 0
    for i in order:
        m = seeds[i]
        area = int(m.sum())
        taken = int((out[m] > 0).sum())
        if area == 0 or taken >= COVERED * area:
            continue
        k += 1
        out[m & (out == 0)] = k
    return out


def image(task):
    block, name = task
    cv2.setNumThreads(1)
    stem = Path(name).stem
    unet = S.load_map(WORK["maps"], block, WORK["member"].split(","), "test", stem)
    prob = WORK["w"] * unet + (1 - WORK["w"]) * mr_map(WORK["dets"][block]["test"][name])
    lab, area, _, _ = S.components(prob, WORK["t"])
    big = np.nonzero(area >= MIN_AREA)[0]
    keep, pred = WORK["keep"][(block, name)]
    assert big.size == keep.size, name
    frames = np.load(WORK["crops"] / f"{block}_test_{stem}.npz")["M"]
    crops = None if WORK["identity"] else WORK["probs"][block].get(name)
    masks, chosen = [], np.nonzero(keep)[0]
    for k in chosen:
        seed = lab == big[k] + 1
        masks.append(seed if WORK["identity"] else refined(seed, crops[k], frames[k]))
    order = np.argsort(-pred[chosen], kind="stable")
    final = assemble(masks, order) if masks else np.zeros((2048, 2048), np.int32)
    total = np.zeros(4)
    for polys in WORK["polys"][name]:
        total += native_counts(gt_native(polys), final)
    return task, total.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--dets", type=Path, required=True)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--keep", type=Path, required=True)
    ap.add_argument("--h6", type=Path, required=True)
    ap.add_argument("--probs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--members", nargs="+", default=[""], help="suffixes of probs_<block><suffix>.npz to average (P21)")
    ap.add_argument("--member", default="resnet34_s0")
    ap.add_argument("--w", type=float, default=0.7)
    ap.add_argument("--t", type=float, default=0.35)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _, records = load_records(data_root())
    polys: dict = {}
    for r in records:
        polys.setdefault(r.file, []).append(r.polygons)
    keep = pickle.loads(a.keep.read_bytes())                 # own cache from h9_keep.py
    dets, tasks, probs = {}, [], {}
    for b in BLOCKS:
        with (a.dets / f"dets_{b}.pkl").open("rb") as fh:       # own cache from dump_dets.py
            dets[b] = pickle.load(fh)["groups"]
        tasks += [(b, f) for (x, f) in keep if x == b]
        if not a.identity:
            zs = [np.load(a.probs / f"probs_{b}{m}.npz") for m in a.members]
            probs[b] = {k: np.mean([z[k].astype(np.float32) for z in zs], 0).round().astype(np.uint8) for k in zs[0].files}
    WORK.update(polys=polys, dets=dets, keep=keep, maps=a.maps, member=a.member, w=a.w, t=a.t, crops=a.crops,
                identity=a.identity, probs=probs)
    done: dict = {b: {} for b in BLOCKS}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for n, ((b, f), row) in enumerate(pool.imap_unordered(image, tasks), 1):
            done[b][f] = row
            if n % 100 == 0:
                print(n, len(tasks), flush=True)
    for b in BLOCKS:
        control = json.loads((a.h6 / f"gate_{b}.json").read_text(encoding="utf-8"))["candidate"]
        same = sum(bool(np.allclose(done[b][f], control[f], atol=1e-3)) for f in done[b])
        (a.out / f"gate_{b}.json").write_text(json.dumps({"block": b, "control": control, "candidate": done[b]}), encoding="utf-8")
        s, c = np.sum(list(done[b].values()), axis=0), np.sum(list(control.values()), axis=0)
        print(b, "images", len(done[b]), "same_as_h6", same, "pq", round(s[0] / (s[1] + 0.5 * s[2] + 0.5 * s[3]), 4),
              "h6", round(c[0] / (c[1] + 0.5 * c[2] + 0.5 * c[3]), 4), "tp/fp/fn", s[1:].astype(int).tolist(), flush=True)


if __name__ == "__main__":
    main()
