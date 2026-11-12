"""Shared training utilities: seeding, augmentation, normalisation, run bookkeeping."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

MEAN, STD = 0.45, 0.225          # grey value statistics used to normalise all three input channels


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_input(img_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [B,H,W] -> normalised float [B,3,H,W]."""
    x = (img_u8.float() / 255.0 - MEAN) / STD
    return x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()


def augment_pair(img: np.ndarray, tgt: np.ndarray, rng: np.random.Generator):
    """Dihedral transform + mild photometric jitter (applied to the image only)."""
    k = int(rng.integers(4))
    img, tgt = np.rot90(img, k), np.rot90(tgt, k)
    if rng.random() < 0.5:
        img, tgt = img[:, ::-1], tgt[:, ::-1]
    img = img.astype(np.float32)
    if rng.random() < 0.8:
        gain = rng.uniform(0.8, 1.2)
        bias = rng.uniform(-20, 20)
        gamma = rng.uniform(0.8, 1.25)
        img = np.clip(255.0 * (np.clip(img * gain + bias, 0, 255) / 255.0) ** gamma, 0, 255)
    return np.ascontiguousarray(img).astype(np.uint8), np.ascontiguousarray(tgt)


class RunLog:
    def __init__(self, out: Path, args: dict):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.data = {"args": args, "log": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def add(self, **row):
        row["elapsed_s"] = round(time.time() - self.t0, 1)
        self.data["log"].append(row)
        print(json.dumps(row), flush=True)
        self.save()

    def save(self):
        (self.out / "result.json").write_text(json.dumps(self.data, indent=1), encoding="utf-8")
