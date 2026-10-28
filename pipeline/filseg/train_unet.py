"""Semantic U-Net (segmentation_models_pytorch) + connected-component instances.

Trains on 1024-pixel images with random crops, selects the checkpoint and the
post-processing on the calibration group only, then scores the held-out test
group once with the frozen choice.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .common import RunLog, augment_pair, autocast, load_state, make_scaler, save_state, seed_everything, to_input
from .data import data_root, load_cache, load_records, split_all_train, split_from_manifest
from .evaluate import SEMANTIC_GRID, GroupScorer

QUICK_GRID = {"thr": [0.4, 0.5, 0.6], "min_area": [60, 120], "close": [0], "link": [0, 3]}


class CropSet(Dataset):
    def __init__(self, imgs, labs, meta, idx, crop, pos_frac, length, seed, target, offset=0):
        self.imgs, self.labs, self.meta, self.idx = imgs, labs, meta, idx
        self.crop, self.pos_frac, self.length, self.seed = crop, pos_frac, length, seed
        self.offset = offset
        self.target = target
        by_file: dict = {}
        for i in idx:
            by_file.setdefault(meta["records"][i]["file"], []).append(i)
        self.by_file = by_file

    def __len__(self):
        return self.length

    def _target(self, i, rng):
        if self.target == "record":
            return (np.asarray(self.labs[i]) > 0).astype(np.float32)
        sibs = self.by_file[self.meta["records"][i]["file"]]
        return np.mean([np.asarray(self.labs[j]) > 0 for j in sibs], axis=0).astype(np.float32)

    def __getitem__(self, n):
        rng = np.random.default_rng((self.seed, n + self.offset))
        i = self.idx[int(rng.integers(len(self.idx)))]
        img = np.asarray(self.imgs[self.meta["records"][i]["img"]])
        tgt = self._target(i, rng)
        size, c = img.shape[0], self.crop
        if c < size:
            ys, xs = np.nonzero(tgt > 0) if rng.random() < self.pos_frac else ((), ())
            if len(ys):
                j = int(rng.integers(len(ys)))
                y0 = int(np.clip(ys[j] - rng.integers(c // 8, 7 * c // 8), 0, size - c))
                x0 = int(np.clip(xs[j] - rng.integers(c // 8, 7 * c // 8), 0, size - c))
            else:
                y0, x0 = (int(v) for v in rng.integers(0, size - c + 1, 2))
            img, tgt = img[y0:y0 + c, x0:x0 + c], tgt[y0:y0 + c, x0:x0 + c]
        img, tgt = augment_pair(img, tgt, rng)
        return torch.from_numpy(img), torch.from_numpy(tgt)


def build_model(arch: str, encoder: str, weights: str | None):
    cls = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "fpn": smp.FPN}[arch]
    return cls(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1)


@torch.no_grad()
def predict_files(model, imgs, file_index, files, device, tta=False) -> dict:
    model.eval()
    out = {}
    for f in files:
        x = to_input(torch.from_numpy(np.asarray(imgs[file_index[f]]))[None]).to(device)
        views = [(lambda t: t, lambda t: t)]
        if tta:
            views += [(lambda t: t.flip(-1), lambda t: t.flip(-1)), (lambda t: t.flip(-2), lambda t: t.flip(-2)),
                      (lambda t: t.flip(-1, -2), lambda t: t.flip(-1, -2))]
        p = 0
        with autocast(device):
            for fwd, inv in views:
                p = p + torch.sigmoid(inv(model(fwd(x))).float())
        out[f] = (p / len(views))[0, 0].cpu().numpy().astype(np.float16)
    return out


def dice_loss(logits, tgt):
    p = torch.sigmoid(logits)
    inter = (p * tgt).sum((1, 2, 3))
    return 1 - ((2 * inter + 1) / (p.sum((1, 2, 3)) + tgt.sum((1, 2, 3)) + 1)).mean()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--all-train", action="store_true", help="train on every record (final model)")
    ap.add_argument("--arch", default="unet", choices=["unet", "unetpp", "fpn"])
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--weights", default="imagenet")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--pos-frac", type=float, default=0.5)
    ap.add_argument("--target", default="record", choices=["record", "consensus"])
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--save-probs", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    seed_everything(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgs, labs, meta = load_cache(a.cache)
    groups = split_all_train(meta) if a.all_train else split_from_manifest(meta, a.manifest)
    _, records_full = load_records(data_root())
    file_index = {f: k for k, f in enumerate(meta["files"])}
    log = RunLog(a.out, vars(a) | {"cache": str(a.cache), "manifest": str(a.manifest), "out": str(a.out)})
    log.data["groups"] = {g: len(v) for g, v in groups.items()}

    model = build_model(a.arch, a.encoder, None if a.weights == "none" else a.weights).to(device)
    log.data["params"] = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.iters))))
    scaler = make_scaler(device)
    start, best = load_state(a.out / "state.pt", model, opt, sched, device)
    ds = CropSet(imgs, labs, meta, groups["train"], a.crop, a.pos_frac, max(a.iters - start, 0) * a.batch,
                 a.seed, a.target, offset=start * a.batch)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, shuffle=False, drop_last=True,
                    persistent_workers=a.workers > 0 and len(ds) > 0)
    cal = GroupScorer(groups["calibration"], meta, labs, records_full) if groups["calibration"] else None
    model.train()
    for step, (x, y) in enumerate(dl, start + 1):
        x, y = to_input(x).to(device, non_blocking=True), y[:, None].to(device, non_blocking=True)
        with autocast(device):
            logits = model(x).float()
        loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % a.eval_every == 0 or step == a.iters:
            row = {"it": step, "loss": round(float(loss), 4)}
            if cal is not None:
                preds = predict_files(model, imgs, file_index, cal.files(), device)
                cfg, s, _ = cal.select(preds, QUICK_GRID, top_native=0)
                row |= {"cal_PQ1024": round(s["r1024"]["PQ"], 4), "cfg": cfg}
                if s["r1024"]["PQ"] > best[0]:
                    best = (s["r1024"]["PQ"], step)
                    torch.save(model.state_dict(), a.out / "best.pt")
                model.train()
            log.add(**row)
            save_state(a.out / "state.pt", model, opt, sched, step, best)
    torch.save(model.state_dict(), a.out / "last.pt")
    if cal is None:
        log.data["final"] = "all-train model; no held-out evaluation"
        log.save()
        return
    model.load_state_dict(torch.load(a.out / "best.pt", map_location=device, weights_only=True))
    log.data["best_it"] = best[1]
    test = GroupScorer(groups["test"], meta, labs, records_full)
    cal_preds = predict_files(model, imgs, file_index, cal.files(), device, tta=a.tta)
    cfg, cal_s, table = cal.select(cal_preds, SEMANTIC_GRID)
    test_preds = predict_files(model, imgs, file_index, test.files(), device, tta=a.tta)
    test_s = test.score(test_preds, cfg, native=True)
    log.data |= {"selected_cfg": cfg, "calibration": cal_s, "test": test_s, "native_table": table}
    log.save()
    print("FINAL", {"cfg": cfg, "cal": cal_s, "test": test_s}, flush=True)
    if a.save_probs:
        np.savez_compressed(a.out / "probs.npz", **{f: (p.astype(np.float32) * 255).astype(np.uint8)
                                                     for f, p in (cal_preds | test_preds).items()})


if __name__ == "__main__":
    main()
