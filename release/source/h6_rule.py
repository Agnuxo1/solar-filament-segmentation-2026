"""H6, step two: learned keep rule against the P11 rule, scored with the official counting (CPU only).

See experiments/PROTOCOL_20261106.md, section P16. Keeping a component adds `mean_r sum(IoU)` to the PQ numerator
per record and 0.5 to its denominator whether it matches or not, so it pays exactly when that mean exceeds PQ / 2.
The model predicts that mean and the rule keeps what clears the bar. Each test block is predicted by a model that
never saw its images or anything within the purge, and F1-F3 never see the labels of F4.

    python scripts/h6_rule.py --components components.pkl --p11 DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

BLOCKS = ("F1", "F2", "F3", "F4")
RESERVED = "F4"
PURGE_DAYS = 5


def day(name: str) -> float:
    return datetime.strptime(name[:14], "%Y%m%d%H%M%S").timestamp() / 86400


def target(n: int, records) -> np.ndarray:
    y = np.zeros(n)
    for _, comp, ious in records:
        np.add.at(y, comp, ious)
    return y / max(len(records), 1)


def make_model(names: list) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0,
                                         categorical_features=[names.index("site")])


def counts(records, keep: np.ndarray) -> list:
    kept, out = int(keep.sum()), np.zeros(4)
    for n_gt, comp, ious in records:
        sel = keep[comp] if comp.size else np.zeros(0, bool)
        tp = int(sel.sum())
        out += (float(ious[sel].sum()), tp, kept - np.unique(comp[sel]).size, n_gt - tp)
    return out.tolist()


def pq(rows) -> float:
    s = np.sum(list(rows), axis=0)
    return float(s[0] / max(s[1] + 0.5 * s[2] + 0.5 * s[3], 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", type=Path, required=True)
    ap.add_argument("--p11", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = pickle.loads(a.components.read_bytes())           # own cache from h6_features.py
    rows, names = data["rows"], list(data["names"])
    sel = json.loads((a.p11 / "selection.json").read_text(encoding="utf-8"))
    rule, tau = sel["fusion"], sel["calibration"]["fusion"] / 2
    assert abs(rule["w"] - data["w"]) < 1e-9 and abs(rule["t_lo"] - data["t"]) < 1e-9, "table built for another fusion"
    i_area, i_peak, i_mean = names.index("log_area"), names.index("peak_p"), names.index("mean_p")
    reserved = {f for (b, g, f) in rows if b == RESERVED and g == "test"}
    summary: dict = {"tau": tau, "rule": rule, "blocks": {}}
    for b in BLOCKS:
        test = sorted(f for (x, g, f) in rows if x == b and g == "test")
        days = np.sort([day(f) for f in test])
        X, y, w = [], [], []
        for (x, g, f), (feats, records) in rows.items():
            if b != RESERVED and f in reserved:
                continue
            d = day(f)
            k = np.searchsorted(days, d)
            near = min(abs(days[min(k, len(days) - 1)] - d), abs(days[max(k - 1, 0)] - d))
            if near < PURGE_DAYS or len(feats) == 0:
                continue
            X.append(feats)
            y.append(target(len(feats), records))
            w.append(np.full(len(feats), len(records), float))
        X, y, w = np.concatenate(X), np.concatenate(y), np.concatenate(w)
        model = make_model(names).fit(X, y, sample_weight=w)
        control = json.loads((a.p11 / f"gate_{b}.json").read_text(encoding="utf-8"))["candidate"]
        learned, mismatch = {}, 0
        for f in test:
            feats, records = rows[(b, "test", f)]
            area = np.round(np.exp(feats[:, i_area].astype(np.float64)))
            base = (area >= rule["area"]) & (feats[:, i_peak] > rule["peak"]) & (feats[:, i_mean] >= rule["mean"])
            mismatch += not np.allclose(counts(records, base), control[f], atol=1e-3)
            keep = model.predict(feats) > tau if len(feats) else np.zeros(0, bool)
            learned[f] = counts(records, keep)
        (a.out / f"gate_{b}.json").write_text(json.dumps({"block": b, "control": control, "candidate": learned}),
                                              encoding="utf-8")
        summary["blocks"][b] = {"train_rows": int(len(y)), "mismatch_with_p11": int(mismatch),
                                "p11": pq(control.values()), "learned": pq(learned.values())}
        print(b, summary["blocks"][b], flush=True)
    (a.out / "selection.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
