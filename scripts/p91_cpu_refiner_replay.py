#!/usr/bin/env python3
"""P91: CPU-only numerical provenance of the original H9b crop refiners.

Independent, fixed-configuration rerun of the frozen refiners on exactly 180
unlabelled crop archives. Does not train, tune, query Kaggle, or load labels.
Each completed model is checkpointed to a per-stage receipt for recovery.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

BASE = pathlib.Path("/srv/almacen/nas-2tb-1/solar-filament-2026")
WORK = BASE / "paired-small-20261112"
RELEASE = BASE / "final-release-staging-20261112"
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
OUT = WORK / "p91_cpu_fp32_refiner_20261113"
PROTOCOL = REPO / "docs/submission/P91_CPU_FP32_REFINER_PREREG_20261113.json"
CHECKPOINTS = (
    ("resnet34", "refiner_ALL.pt", "probs_resnet34.npz"),
    ("efficientnet_v2_s", "refiner_ALL_efficientnet_v2_s.pt",
     "probs_efficientnet_v2_s.npz"),
)
REF_SHA = "67f02e4d54b78424306133840ec144b426c8b308f6fc659f90a7f67131d5a79e"


def sha256_file(path: pathlib.Path) -> str:
    """Compute a bounded-memory SHA-256 for the supplied immutable file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, data: dict) -> None:
    """Create an immutable JSON receipt by atomic rename; reject duplicates."""
    if path.exists():
        raise RuntimeError(f"Receipt already exists: {path}")
    pending = path.with_name(path.name + f".{os.getpid()}.tmp")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def compare(a: pathlib.Path, b: pathlib.Path) -> dict:
    """Compute exact image and element parity for complete uint8 crop maps."""
    images = pixels = different = abs_total = exact_images = 0
    biggest = 0
    by_image: dict[str, int] = {}
    with np.load(a, allow_pickle=False) as current, np.load(
        b, allow_pickle=False
    ) as historical:
        if set(current.files) != set(historical.files) or len(current.files) != 180:
            raise RuntimeError("Expected exactly 180 matching frozen image keys")
        for name in sorted(historical.files):
            x, y = current[name], historical[name]
            if x.shape != y.shape or x.dtype != np.uint8 or y.dtype != np.uint8:
                raise RuntimeError(f"Shape or dtype mismatch: {name}")
            d = np.abs(x.astype(np.int16) - y.astype(np.int16))
            count = int(np.count_nonzero(d))
            by_image[name] = count
            images += 1
            pixels += int(d.size)
            different += count
            abs_total += int(d.sum(dtype=np.int64))
            biggest = max(biggest, int(d.max()) if d.size else 0)
            exact_images += int(count == 0)
    return {
        "images": images,
        "pixels": pixels,
        "differing_pixels": different,
        "exact_pixels": pixels - different,
        "exact_fraction": (pixels - different) / pixels,
        "exact_images": exact_images,
        "max_byte_delta": biggest,
        "mean_abs_byte_delta": abs_total / pixels,
        "images_with_differences": images - exact_images,
        "top_image_difference_counts": sorted(by_image.values(), reverse=True)[:12],
    }


def self_test() -> None:
    """Check that the parity comparator detects both exact and modified maps."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        first = pathlib.Path(directory) / "a.npz"
        second = pathlib.Path(directory) / "b.npz"
        images = {f"im{i:03d}": np.full((1, 2, 3), i % 255, np.uint8)
                  for i in range(180)}
        np.savez_compressed(first, **images)
        np.savez_compressed(second, **images)
        assert compare(first, second)["different_pixels"] == 0
        images["im002"][0, 0, 0] += 1
        np.savez_compressed(second, **images)
        r = compare(first, second)
        assert r["different_pixels"] == 1
        assert r["exact_images"] == 179
        assert r["max_byte_delta"] == 1
    print("P91_SELF_TEST_PASS", flush=True)


def main() -> None:
    """Verify frozen inputs, rerun FP32 CPU inference and audit final CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET") != "":
        raise RuntimeError("P91 requires an explicit empty CUDA_VISIBLE_DEVICES")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("experiment") != "P91_CPU_FP32_REFINER_SOURCE_DIVERGENCE":
        raise RuntimeError("Preregistration identity mismatch")
    if sha256_file(RELEASE / "submission_58112557.csv") != REF_SHA:
        raise RuntimeError("The accepted Kaggle CSV is not immutable")
    crops = RELEASE / "h9/official/crops"
    if len(list(crops.glob("*.npz"))) != 180:
        raise RuntimeError("Frozen crop inventory is incomplete")
    for name, checkpoint, original in CHECKPOINTS:
        expected = protocol["preexisting_evidence"]["refiner_checkpoint_sha256"][name]
        if sha256_file(RELEASE / "models" / checkpoint) != expected:
            raise RuntimeError(f"Frozen checkpoint checksum differs: {name}")
        if not (RELEASE / "h9/official" / original).is_file():
            raise RuntimeError(f"Frozen probability reference missing: {name}")
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    results: dict[str, dict] = {}
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "SOLARFIL_ASSET_ROOT": str(RELEASE),
        "PYTHONPATH": os.pathsep.join(
            [str(RELEASE / "src"), str(RELEASE), env.get("PYTHONPATH", "")]
        ),
    })
    for encoder, checkpoint, original in CHECKPOINTS:
        result = OUT / original
        frozen = RELEASE / "h9/official" / original
        prior = WORK / "p70_refiner_replay" / original
        receipt = OUT / (encoder + "_receipt.json")
        if not result.is_file():
            temporary = OUT / (original + ".partial.npz")
            if temporary.exists():
                temporary.unlink()
            command = [
                sys.executable, str(WORK / "h9_infer.py"),
                "--weights", str(RELEASE / "models" / checkpoint),
                "--encoder", encoder, "--crops", str(crops),
                "--out", str(temporary), "--device", "cpu"
            ]
            subprocess.run(command, cwd=WORK, env=env, check=True)
            with np.load(temporary, allow_pickle=False) as data:
                if len(data.files) != 180:
                    raise RuntimeError(f"Incomplete CPU output: {encoder}")
            os.replace(temporary, result)
        result_data = {
            "encoder": encoder,
            "checkpoint_sha256": sha256_file(RELEASE / "models" / checkpoint),
            "candidate_sha256": sha256_file(result),
            "frozen_sha256": sha256_file(frozen),
            "against_original": compare(result, frozen),
            "against_P70_cuda_replay": compare(result, prior),
        }
        if not receipt.exists():
            atomic_json(receipt, result_data)
        else:
            original_receipt = json.loads(receipt.read_text())
            if original_receipt != result_data:
                raise RuntimeError(f"Resume checksum mismatch: {encoder}")
        results[encoder] = result_data
        print("P91_MODEL", json.dumps({
            "encoder": encoder,
            "exact_fraction": result_data["against_original"]["exact_fraction"],
            "different_pixels": result_data["against_original"]["differing_pixels"],
        }), flush=True)
    assembled = OUT / "assembled"
    assembled.mkdir(exist_ok=True)
    link = assembled / "crops"
    if not link.is_symlink():
        if link.exists():
            raise RuntimeError("Expected immutable crop directory symlink")
        link.symlink_to(crops, target_is_directory=True)
    if link.resolve() != crops.resolve():
        raise RuntimeError("Cropping provenance differs")
    candidate = assembled / "submission_h9b.csv"
    if not candidate.is_file():
        subprocess.run([
            sys.executable, str(RELEASE / "src/h9_official.py"), "paste",
            "--out", str(assembled),
            "--probs", *(str(OUT / row[2]) for row in CHECKPOINTS),
        ], cwd=RELEASE, env=env, check=True)
    csv_audit = OUT / "p91_vs_accepted.json"
    if not csv_audit.exists():
        subprocess.run([
            sys.executable,
            str(REPO / "scripts/audit_p90_submission_mismatch.py"),
            "--accepted", str(RELEASE / "submission_58112557.csv"),
            "--candidate", str(candidate),
            "--output", str(csv_audit),
            "--expected-accepted-sha256", REF_SHA,
        ], cwd=REPO, env=env, check=True)
    audit = json.loads(csv_audit.read_text())
    final = {
        "experiment": protocol["experiment"],
        "status": "P91_CPU_REFINER_DIAGNOSTIC_COMPLETE",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preregistration_sha256": sha256_file(PROTOCOL),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "models": results,
        "csv_audit_sha256": sha256_file(csv_audit),
        "rle_rows_differing_from_accepted": audit["different_rle_rows"],
        "csv_byte_identical": audit["byte_identical"],
        "labels_read": False,
        "cuda_processes_started": 0,
        "kaggle_submissions": 0,
        "conclusion_scope": "CPU FP32 numerical reproducibility; no PQ inference",
    }
    if not (OUT / "receipt.json").exists():
        atomic_json(OUT / "receipt.json", final)
    print("P91_COMPLETE", json.dumps({
        "rle_rows_differing_from_accepted": final["rle_rows_differing_from_accepted"],
        "csv_byte_identical": final["csv_byte_identical"],
    }), flush=True)


if __name__ == "__main__":
    main()
