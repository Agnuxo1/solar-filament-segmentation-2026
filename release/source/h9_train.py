"""H9, step two: instance refiner, one model per block, trained and applied on rotated crops (GPU, short ticket).

See experiments/PROTOCOL_20261106.md, section P20. Input channels: flat field, fusion probability, seed, other candidates.
The target is the consensus of the records that drew the filament, so existence stays the job of the H6 rule.

    python scripts/h9_train.py --pack DIR --block F1 --out DIR       # also ALL, for the final model
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

STEPS, BATCH, LR, WARM, SEED = 4000, 32, 3e-4, 100, 0
PURGE_DAYS, RESERVED = 5, "F4"


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.body(x if skip is None else torch.cat([x, skip], 1))


class Refiner(nn.Module):
    """ResNet34 U-Net with four input channels; the first filters start from the mean ImageNet filter."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        conv = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv.weight.copy_(net.conv1.weight.mean(1, keepdim=True).repeat(1, 4, 1, 1))
        self.stages = nn.ModuleList([nn.Sequential(conv, net.bn1, net.relu), nn.Sequential(net.maxpool, net.layer1),
                                     net.layer2, net.layer3, net.layer4])
        self.u4, self.u3, self.u2, self.u1 = Up(512, 256, 256), Up(256, 128, 128), Up(128, 64, 64), Up(64, 64, 32)
        self.u0 = Up(32, 0, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        taps = []
        for stage in self.stages:
            x = stage(x)
            taps.append(x)
        y = self.u1(self.u2(self.u3(self.u4(taps[4], taps[3]), taps[2]), taps[1]), taps[0])
        return self.head(self.u0(y))


class RefinerEff(nn.Module):
    """EfficientNetV2-S U-Net with four input channels, the second member of H9b (P21)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = models.efficientnet_v2_s(weights="IMAGENET1K_V1" if pretrained else None).features
        old = net[0][0]
        conv = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            conv.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, 4, 1, 1))
        net[0][0] = conv
        self.stages = nn.ModuleList([nn.Sequential(net[0], net[1]), net[2], net[3], nn.Sequential(net[4], net[5]), net[6]])
        self.u4, self.u3, self.u2, self.u1 = Up(256, 160, 256), Up(256, 64, 128), Up(128, 48, 64), Up(64, 24, 32)
        self.u0 = Up(32, 0, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        taps = []
        for stage in self.stages:
            x = stage(x)
            taps.append(x)
        y = self.u1(self.u2(self.u3(self.u4(taps[4], taps[3]), taps[2]), taps[1]), taps[0])
        return self.head(self.u0(y))


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def day(name: str) -> float:
    return datetime.strptime(name[:14], "%Y%m%d%H%M%S").timestamp() / 86400


def split(index: list, block: str):
    """Row numbers for training and for the held-out test of `block`, with the separation of H6."""
    reserved = {r["file"] for r in index if r["block"] == RESERVED and r["group"] == "test"}
    if block == "ALL":
        return list(range(len(index))), []
    test = sorted(r["file"] for r in index if r["block"] == block and r["group"] == "test")
    days = np.sort([day(f) for f in test])
    train = []
    for i, r in enumerate(index):
        if block != RESERVED and r["file"] in reserved:
            continue
        d = day(r["file"])
        k = np.searchsorted(days, d)
        near = min(abs(days[min(k, len(days) - 1)] - d), abs(days[max(k - 1, 0)] - d))
        if near >= PURGE_DAYS:
            train.append(i)
    return train, [i for i, r in enumerate(index) if r["block"] == block and r["group"] == "test"]


def augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Brightness on the flat field, and a seed grown or shrunk by up to 3 px, per batch."""
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.5) * (1 + 0.2 * (torch.rand(1, generator=g).item() - 0.5)) + 0.5
    k = int(torch.randint(0, 4, (1,), generator=g).item())
    if k:
        pooled = F.max_pool2d(x[:, 2:3] if torch.rand(1, generator=g).item() < 0.5 else -x[:, 2:3], 2 * k + 1, 1, k)
        x[:, 2:3] = pooled.abs()
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encoder", default="resnet34", choices=("resnet34", "efficientnet_v2_s"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    index = json.loads((a.pack / "index.json").read_text(encoding="utf-8"))
    n = index[-1]["start"] + index[-1]["n"]
    X = np.memmap(a.pack / "X.bin", np.uint8, "r", shape=(n, 4, 128, 384))
    Y = np.memmap(a.pack / "Y.bin", np.uint8, "r", shape=(n, 128, 384))
    has = np.fromfile(a.pack / "has.bin", np.uint8).astype(bool)
    train_rows, test_rows = split(index, a.block)
    pool = np.concatenate([np.arange(index[i]["start"], index[i]["start"] + index[i]["n"]) for i in train_rows])
    pool = pool[has[pool]]
    print(json.dumps({"block": a.block, "train_rows": len(train_rows), "crops": int(pool.size), "test_rows": len(test_rows)}), flush=True)
    dev = torch.device(a.device)
    amp = dev.type == "cuda"
    if amp:
        torch.cuda.set_per_process_memory_fraction(8 / 24)      # the ticket declares 8 GB of a 24 GB card
    model = (Refiner() if a.encoder == "resnet34" else RefinerEff()).to(dev).train()
    tag = a.block if a.encoder == "resnet34" else f"{a.block}_{a.encoder}"
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    g = torch.Generator().manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0, seen = time.time(), 0.0
    for step in range(1, a.steps + 1):
        pick = np.sort(rng.choice(pool, BATCH, replace=False))
        x = torch.from_numpy(np.asarray(X[pick])).to(dev).float() / 255
        y = torch.from_numpy(np.asarray(Y[pick])).to(dev).float()[:, None] / 255
        if torch.rand(1, generator=g).item() < 0.5:
            x, y = x.flip(3), y.flip(3)
        if torch.rand(1, generator=g).item() < 0.5:
            x, y = x.flip(2), y.flip(2)
        x = augment(x, g)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        seen = 0.98 * seen + 0.02 * float(loss) if step > 1 else float(loss)
        if step % 200 == 0 or step == a.steps:
            print(json.dumps({"step": step, "loss": round(seen, 4), "s": round(time.time() - t0)}), flush=True)
    torch.save(model.state_dict(), a.out / f"refiner_{tag}.pt")
    if not test_rows:
        return
    model.eval()
    probs = {}
    with torch.no_grad():
        for i in test_rows:
            r = index[i]
            if r["n"] == 0:
                continue
            acc = []
            for s in range(0, r["n"], 64):
                x = torch.from_numpy(np.asarray(X[r["start"] + s:r["start"] + min(s + 64, r["n"])])).to(dev).float() / 255
                total = 0
                for dims in ((), (3,), (2,), (2, 3)):
                    v = x.flip(dims) if dims else x
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        p = torch.sigmoid(model(v).float())
                    total = total + (p.flip(dims) if dims else p)
                acc.append((total / 4)[:, 0].cpu().numpy())
            probs[r["file"]] = (np.clip(np.concatenate(acc), 0, 1) * 255 + 0.5).astype(np.uint8)
    np.savez_compressed(a.out / f"probs_{tag}.npz", **probs)
    print("DONE", a.block, len(probs), flush=True)


if __name__ == "__main__":
    main()
