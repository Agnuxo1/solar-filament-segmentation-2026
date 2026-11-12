"""Data access for the MAGFiLO Kaggle 2026 release.

Images are 2048x2048 grayscale GONG H-alpha JPEGs. One physical image can carry
several annotation *records* (one per annotator); evaluation is per record, so
every split keeps all records of a physical image together.

The 1024-pixel cache reproduces the label rule of the NeuroPixel control
(2x2 area average > 0.3, earlier annotation wins on overlap) so that the
1024-resolution PQ is directly comparable with that control.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

NATIVE = 2048
RES = 1024
ANN_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def data_root() -> Path:
    root = Path(os.environ.get("SOLARFIL_DATA", "data/MAGFiLO_1.0_Kaggle_2026"))
    if not (root / "train" / ANN_NAME).exists():
        raise FileNotFoundError(f"MAGFiLO data not found under {root}")
    return root


def read_gray(path: Path, size: int = RES) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.shape != (NATIVE, NATIVE):
        raise ValueError(f"unexpected shape {img.shape} for {path}")
    return img if size == NATIVE else cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


@dataclass
class Record:
    record_id: str          # COCO image id (one annotator's view of a physical image)
    file: str               # physical JPEG name
    img: int                # row in the image cache
    polygons: list          # list of COCO polygon segmentations (native resolution)


def load_records(root: Path) -> tuple[list[str], list[Record]]:
    coco = json.loads((root / "train" / ANN_NAME).read_text(encoding="utf-8"))
    files = sorted({im["file_name"] for im in coco["images"]})
    fidx = {f: k for k, f in enumerate(files)}
    anns: dict = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a["segmentation"])
    records = [Record(str(im["id"]), im["file_name"], fidx[im["file_name"]], anns[im["id"]])
               for im in coco["images"] if im["id"] in anns]
    return files, records


def polygon_mask(poly, size: int = NATIVE) -> np.ndarray:
    return mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(poly, size, size)))


def build_cache(root: Path, cache: Path) -> None:
    """Write imgs.npy (N,1024,1024 uint8), labs.npy (R,1024,1024 uint8) and meta.json."""
    cache.mkdir(parents=True, exist_ok=True)
    files, records = load_records(root)
    imgs = np.lib.format.open_memmap(cache / "imgs.npy", "w+", np.uint8, (len(files), RES, RES))
    for k, f in enumerate(files):
        imgs[k] = read_gray(root / "train" / "train_images" / f)
    imgs.flush()
    labs = np.lib.format.open_memmap(cache / "labs.npy", "w+", np.uint8, (len(records), RES, RES))
    for j, r in enumerate(records):
        lab = np.zeros((RES, RES), np.uint8)
        for n, poly in enumerate(r.polygons, 1):
            m = polygon_mask(poly).astype(np.float32)
            small = m.reshape(RES, 2, RES, 2).mean(axis=(1, 3)) > 0.3
            lab[small & (lab == 0)] = n
        labs[j] = lab
    labs.flush()
    meta = {"files": files, "records": [{"id": r.record_id, "file": r.file, "img": r.img} for r in records]}
    (cache / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_cache(cache: Path):
    imgs = np.load(cache / "imgs.npy", mmap_mode="r")
    labs = np.load(cache / "labs.npy", mmap_mode="r")
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    return imgs, labs, meta


def split_from_manifest(meta: dict, manifest_path: Path) -> dict[str, list[int]]:
    """Record indices per group (train / calibration / test) from a file-level manifest."""
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out = {}
    for g in ("train", "calibration", "test"):
        files = set(man["files"][g])
        out[g] = [i for i, r in enumerate(meta["records"]) if r["file"] in files]
    seen = [set(meta["records"][i]["file"] for i in out[g]) for g in out]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2]), "file overlap between groups"
    return out


def split_all_train(meta: dict) -> dict[str, list[int]]:
    return {"train": list(range(len(meta["records"]))), "calibration": [], "test": []}
