#!/usr/bin/env python3
"""Freeze the original H9b no-site selector from the own TRAIN-only feature cache.

This writes only a fitted model and its provenance manifest, not the training
annotation records or their per-instance targets. Re-running it is deterministic
provided sklearn and NumPy versions match the original training runtime.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import pickle

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def target(n: int, records) -> np.ndarray:
    y = np.zeros(n, dtype=np.float64)
    for _, comp, ious in records:
        np.add.at(y, comp, ious)
    return y / max(len(records), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    cache_hash = sha256(args.source)
    with args.source.open("rb") as f:
        data = pickle.load(f)
    names = list(data["names"])
    assert names[-1] == "site"
    cols = [i for i, n in enumerate(names) if n != "site"]
    X, y, sw = [], [], []
    for feats, records in data["rows"].values():
        if len(feats):
            X.append(np.asarray(feats[:, cols], np.float32))
            y.append(target(len(feats), records))
            sw.append(np.full(len(feats), len(records), float))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    sw = np.concatenate(sw, axis=0)
    selector = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.05, max_depth=3,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0
    )
    selector.fit(X, y, sample_weight=sw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Refuse to silently overwrite a frozen model.")
    with args.output.open("xb") as f:
        pickle.dump(selector, f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "format": "h9b-nosite-selector-v1",
        "weight_sha256": sha256(args.output),
        "train_feature_cache_sha256": cache_hash,
        "train_feature_rows": int(len(X)),
        "train_feature_columns": [names[i] for i in cols],
        "excluded_inference_features": ["site", "date", "temporal_neighbors"],
        "w": float(data["w"]), "t": float(data["t"]),
        "sites": list(data["sites"]),
        "no_competition_test_annotations": True,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    manifest_path = args.output.with_suffix(".json")
    with manifest_path.open("x", encoding="utf8") as f:
        json.dump(manifest, f, sort_keys=True, indent=2)
    print("FROZEN_H9B_SELECTOR", json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
