"""H9b: apply a final refiner to the official crops, with the same four flips as in validation (GPU, short ticket).

    python scripts/h9_infer.py --weights FILE --encoder resnet34 --crops DIR --out FILE
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("h9_train", Path(__file__).with_name("h9_train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--encoder", default="resnet34", choices=("resnet34", "efficientnet_v2_s"))
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    amp = dev.type == "cuda"
    if amp:
        torch.cuda.set_per_process_memory_fraction(8 / 24)
    model = (T.Refiner(False) if a.encoder == "resnet34" else T.RefinerEff(False)).to(dev)
    model.load_state_dict(torch.load(a.weights, map_location=dev, weights_only=True))
    model.eval()
    probs = {}
    with torch.no_grad():
        for path in sorted(a.crops.glob("*.npz")):
            X = np.load(path)["X"]
            if len(X) == 0:
                probs[path.stem] = np.zeros((0, 128, 384), np.uint8)
                continue
            acc = []
            for s in range(0, len(X), 64):
                x = torch.from_numpy(X[s:s + 64].copy()).to(dev).float() / 255
                total = 0
                for dims in ((), (3,), (2,), (2, 3)):
                    v = x.flip(dims) if dims else x
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        p = torch.sigmoid(model(v).float())
                    total = total + (p.flip(dims) if dims else p)
                acc.append((total / 4)[:, 0].cpu().numpy())
            probs[path.stem] = (np.clip(np.concatenate(acc), 0, 1) * 255 + 0.5).astype(np.uint8)
    np.savez_compressed(a.out, **probs)
    print("DONE", len(probs), flush=True)


if __name__ == "__main__":
    main()
