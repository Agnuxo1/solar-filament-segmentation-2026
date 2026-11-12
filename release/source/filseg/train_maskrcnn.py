"""Instance segmentation with torchvision Mask R-CNN v2 (COCO-pretrained).

Same split, selection policy and scoring as ``train_unet``: checkpoint and
post-processing chosen on calibration, held-out test scored once.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .common import RunLog, augment_pair, autocast, load_state, make_scaler, save_state, seed_everything
from .data import data_root, load_cache, load_records, split_all_train, split_from_manifest
from .evaluate import INSTANCE_GRID, GroupScorer

QUICK_GRID = {"score_thr": [0.5, 0.7], "mask_thr": [0.5], "min_area": [20, 60]}


class InstanceSet(Dataset):
    def __init__(self, imgs, labs, meta, idx, length, seed, min_px=6, offset=0):
        self.imgs, self.labs, self.meta, self.idx = imgs, labs, meta, idx
        self.length, self.seed, self.min_px, self.offset = length, seed, min_px, offset

    def __len__(self):
        return self.length

    def __getitem__(self, n):
        rng = np.random.default_rng((self.seed, n + self.offset))
        i = self.idx[int(rng.integers(len(self.idx)))]
        img = np.asarray(self.imgs[self.meta["records"][i]["img"]])
        lab = np.asarray(self.labs[i])
        img, lab = augment_pair(img, lab, rng)
        masks, boxes = [], []
        for k in np.unique(lab):
            if k == 0:
                continue
            m = lab == k
            if m.sum() < self.min_px:
                continue
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            masks.append(m)
        x = torch.from_numpy(img).float().div(255).expand(3, -1, -1).contiguous()
        tgt = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
               "labels": torch.ones(len(boxes), dtype=torch.int64),
               "masks": torch.from_numpy(np.stack(masks)) if masks else torch.zeros(0, *lab.shape, dtype=torch.uint8)}
        tgt["masks"] = tgt["masks"].to(torch.uint8)
        return x, tgt


def collate(batch):
    return tuple(zip(*batch))


def build_model(size: int = 1024, pretrained: bool = True):
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None, min_size=size, max_size=size,
                                     box_detections_per_img=100, box_score_thresh=0.05)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    return model


@torch.no_grad()
def predict_files(model, imgs, file_index, files, device) -> dict:
    model.eval()
    out = {}
    for f in files:
        img = torch.from_numpy(np.asarray(imgs[file_index[f]])).float().div(255).expand(3, -1, -1).to(device)
        with autocast(device):
            r = model([img])[0]
        dets = []
        size = img.shape[-1]
        for s, b, m in zip(r["scores"].float().cpu().numpy(), r["boxes"].float().cpu().numpy(), r["masks"][:, 0]):
            x0, y0 = max(int(math.floor(b[0])), 0), max(int(math.floor(b[1])), 0)
            x1, y1 = min(int(math.ceil(b[2])), size), min(int(math.ceil(b[3])), size)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = (m[y0:y1, x0:x1].float() * 255).to(torch.uint8).cpu().numpy()   # crop on GPU: small host RAM
            dets.append((float(s), y0, y1, x0, x1, crop))
        out[f] = dets
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--all-train", action="store_true")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
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

    model = build_model().to(device)
    log.data["params"] = sum(p.numel() for p in model.parameters())
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.iters))))
    scaler = make_scaler(device)
    start, best = load_state(a.out / "state.pt", model, opt, sched, device)
    ds = InstanceSet(imgs, labs, meta, groups["train"], max(a.iters - start, 0) * a.batch, a.seed,
                     offset=start * a.batch)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, collate_fn=collate, drop_last=True,
                    persistent_workers=a.workers > 0 and len(ds) > 0)
    cal = GroupScorer(groups["calibration"], meta, labs, records_full) if groups["calibration"] else None
    model.train()
    for step, (xs, ts) in enumerate(dl, start + 1):
        xs = [x.to(device, non_blocking=True) for x in xs]
        ts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in ts]
        with autocast(device):
            losses = model(xs, ts)
        loss = sum(v.float() for v in losses.values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % a.eval_every == 0 or step == a.iters:
            row = {"it": step, "loss": round(float(loss), 4)} | {k: round(float(v), 4) for k, v in losses.items()}
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
    cfg, cal_s, table = cal.select(predict_files(model, imgs, file_index, cal.files(), device), INSTANCE_GRID)
    test_s = test.score(predict_files(model, imgs, file_index, test.files(), device), cfg, native=True)
    log.data |= {"selected_cfg": cfg, "calibration": cal_s, "test": test_s, "native_table": table}
    log.save()
    print("FINAL", {"cfg": cfg, "cal": cal_s, "test": test_s}, flush=True)


if __name__ == "__main__":
    main()
