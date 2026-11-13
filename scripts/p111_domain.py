#!/usr/bin/env python3
"""P111 supplementary read-only domain check for pre-existing refiner archives."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    """Hash a frozen artifact without loading its bytes at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_archive(path: Path, expected_count: int) -> dict:
    """Measure the full numerical domain; never infer probability semantics from filenames."""
    result = {
        "sha256": sha256_file(path),
        "keys": 0,
        "elements": 0,
        "minimum": None,
        "maximum": None,
        "below_zero": 0,
        "above_one": 0,
        "nan_or_inf": 0,
    }
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != expected_count:
            raise ValueError(f"Unexpected archive key count: {path}")
        result["keys"] = len(archive.files)
        for key in sorted(archive.files):
            array = archive[key]
            result["elements"] += int(array.size)
            result["nan_or_inf"] += int(np.count_nonzero(~np.isfinite(array)))
            result["below_zero"] += int(np.count_nonzero(array < 0))
            result["above_one"] += int(np.count_nonzero(array > 1))
            current_min = float(np.min(array))
            current_max = float(np.max(array))
            result["minimum"] = current_min if result["minimum"] is None else min(result["minimum"], current_min)
            result["maximum"] = current_max if result["maximum"] is None else max(result["maximum"], current_max)
    return result


def main() -> None:
    root = Path("/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112")
    p103 = root / "p103_full_public_stage_coupling_20261113"
    p106 = root / "p106_public_clean_6843362_20261113" / "inference_cuda" / "refiners"
    output = root / "p111_stage_drift_20261113" / "P111_VALUE_DOMAIN_AUDIT.json"
    results = {
        "scope": "Numeric domain verification only; no ground-truth or PQ",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "archives": {},
    }
    for model in ("resnet34", "efficientnet_v2_s"):
        results["archives"][f"p103_{model}"] = describe_archive(p103 / f"{model}_new176.npz", 176)
        results["archives"][f"p106_{model}"] = describe_archive(p106 / f"{model}_probabilities.npz", 180)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
