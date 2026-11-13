#!/usr/bin/env python3
"""P103: recoverable 180-image follow-up to the preregistered P102 H9b pilot.

Reuses the four pilot crop archives and their FP32 refiner outputs by SHA-256,
then processes only the other 176 original unlabelled JPEGs. Published H3 maps,
public detector proposals, consensus, selector, and weights are the inference
inputs. Historical H9b caches and accepted CSV are audit references only.
No competition TEST annotation or CUDA is accessed.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p102_public_stage_coupling as P

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
OUT = WORK / "p103_full_public_stage_coupling_20261113"
PILOT = WORK / "p102_public_stage_coupling_20261113"
PREREG = REPO / "docs/submission/P103_FULL_PUBLIC_STAGE_COUPLING_PREREG_20261113.md"
FOLDS = ("F1", "F2", "F3", "F4")
EXPECTED_PILOT_COUNT = 4


def check_pilot(images: list[Path]) -> dict:
    """Verify that previously measured first-four outputs are immutable."""
    original = json.loads((PILOT / "P102_AUDIT.json").read_text(encoding="utf8"))
    if original["pilot_fixed_sorted_prefix"] != EXPECTED_PILOT_COUNT:
        raise RuntimeError("Wrong preregistered pilot prefix.")
    if original["status"] != "P102_PUBLIC_STAGE_COUPLING_AUDITED_NOT_COMPETITION_SUBMISSION":
        raise RuntimeError("Pilot not independently audited.")
    if original["final"]["exact_images"] != 3 or original["final"]["fallback_images"] != 0:
        raise RuntimeError("Unexpected original P102 scientific outcome.")
    for image in images[:EXPECTED_PILOT_COUNT]:
        data = original["crops"]["records"][image.name]
        path = PILOT / "crops" / f"{image.stem}.npz"
        if P.sha(image) != data["image"] or P.sha(path) != data["crops"]:
            raise RuntimeError(f"Pilot JPEG/crop provenance changed: {image.name}")
    for encoder in ("resnet34", "efficientnet_v2_s"):
        with np.load(PILOT / f"{encoder}.npz", allow_pickle=False) as archive:
            if set(archive.files) != {p.stem for p in images[:4]}:
                raise RuntimeError(f"Incomplete reused pilot refiner: {encoder}")
    return original


def verify_resume_receipt(path: Path, expected: dict) -> bool:
    """Audit a completed per-image checkpoint before skipping computation."""
    if not path.exists():
        return False
    existing = json.loads(path.read_text(encoding="utf8"))
    if existing != expected:
        raise RuntimeError(f"Input or checkpoint drift for {path}")
    return True


def prepare_overlay(images: list[Path]) -> dict:
    """Create the published detector consensus from 720 upstream model outputs."""
    evidence = OUT / "P103_OVERLAY_RECEIPT.json"
    detector_file = OUT / "assets/dets_official.pkl"
    if evidence.exists():
        audit = json.loads(evidence.read_text(encoding="utf8"))
        if audit["detector_sha256"] != P.sha(detector_file):
            raise RuntimeError("Resumed public detector consensus digest differs.")
        for fold in FOLDS:
            target = OUT / "assets/h3/maps" / f"{fold}_resnet34_s0/official"
            if not target.is_symlink() or target.resolve() != P.H3[fold].resolve():
                raise RuntimeError(f"Resumed H3 source changed: {fold}")
            if audit["h3_receipt_sha256"][fold] != P.sha(P.RECEIPT[fold]):
                raise RuntimeError(f"Upstream public H3 manifest changed: {fold}")
        print("P103_OVERLAY_RESUMED_HASH_VERIFIED", flush=True)
        return audit
    if detector_file.exists():
        raise RuntimeError("Unattested detector cache present; manual integrity review needed.")
    result = P.create_overlay(images)
    P.atomic_json(evidence, result)
    print("P103_OVERLAY_CREATED_SHA_VERIFIED", flush=True)
    return result


def ensure_pilot_crops(images: list[Path], pilot: dict) -> None:
    """Hardlink—not recompute—the four previously tested crop tensors."""
    crops = OUT / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    for image in images[:EXPECTED_PILOT_COUNT]:
        original = PILOT / "crops" / f"{image.stem}.npz"
        destination = crops / original.name
        expected = pilot["crops"]["records"][image.name]["crops"]
        if P.sha(original) != expected:
            raise RuntimeError(f"Pilot crop changed: {image.name}")
        if not destination.exists():
            os.link(original, destination)
        if P.sha(destination) != expected:
            raise RuntimeError(f"Hardlinked crop mismatch: {image.name}")


def compute_remaining_crops(images: list[Path], metadata: dict) -> dict:
    """Checkpoint every image independently, resuming without duplicate inference."""
    with P.SELECTOR.open("rb") as stream:
        selector = pickle.load(stream)
    folder = OUT / "crops"
    receipts = OUT / "crop_receipts"
    folder.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    all_records = {}
    fields = ("X", "M", "comp", "score")
    disagreements = {name: 0 for name in fields}
    for index, image in enumerate(images, 1):
        P.verify_space()
        destination = folder / f"{image.stem}.npz"
        reference = P.REFERENCES / destination.name
        checkpoint = receipts / f"{image.stem}.json"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as archive:
                actual = {name: archive[name] for name in fields}
        else:
            if checkpoint.exists():
                raise RuntimeError(f"Orphaned receipt without crop: {checkpoint}")
            actual = P.rebuild_one(image, selector, metadata)
            pending = destination.with_suffix(".partial")
            if pending.exists():
                raise RuntimeError(f"Interrupted write needs review: {pending}")
            with pending.open("xb") as sink:
                np.savez_compressed(sink, **actual)
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(pending, destination)
        with np.load(reference, allow_pickle=False) as archived:
            comparison = P.compare_arrays(actual, {k: archived[k] for k in fields})
        record = {
            "image": image.name,
            "image_sha256": P.sha(image),
            "crop_sha256": P.sha(destination),
            "historical_crop_sha256": P.sha(reference),
            "selector_sha256": P.sha(P.SELECTOR),
            "provenance": "public_H3_and_public_P88_P99_consensus",
            "component_count": int(len(actual["comp"])),
            "comparison": comparison,
        }
        if not verify_resume_receipt(checkpoint, record):
            P.atomic_json(checkpoint, record)
        all_records[image.name] = record
        for field in fields:
            disagreements[field] += int(not comparison[field]["exact"])
        if index == 1 or index % 20 == 0 or index == len(images):
            print(f"P103_CROPS {index}/{len(images)} reused_verified={checkpoint.exists()}", flush=True)
    return {"field_image_mismatches_new_176": disagreements, "records": all_records}


def infer_new_refiners(images: list[Path]) -> tuple[dict, dict]:
    """Run new crops on CPU in FP32, each refiner with its own recovery gate."""
    expected = {image.stem for image in images}
    crop_names = {path.stem for path in (OUT / "crops").glob("*.npz")}
    if crop_names != expected:
        raise RuntimeError("Only the 176 newly processed crops may enter new inference.")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "release/source"), str(REPO)])
    result = []
    for encoder, weight in (
        ("resnet34", "refiner_ALL.pt"),
        ("efficientnet_v2_s", "refiner_ALL_efficientnet_v2_s.pt"),
    ):
        output = OUT / f"{encoder}_new176.npz"
        receipt = OUT / f"{encoder}_new176_receipt.json"
        if not output.exists():
            if receipt.exists():
                raise RuntimeError("Completed inference receipt without model output.")
            partial = OUT / f"{encoder}.partial.npz"
            if partial.exists():
                raise RuntimeError(f"Interrupted refiner output needs review: {partial}")
            subprocess.run([
                sys.executable, "-u",
                str(REPO / "release/source/h9_infer.py"),
                "--weights", str(REPO / "release/models" / weight),
                "--encoder", encoder, "--crops", str(OUT / "crops"),
                "--out", str(partial), "--device", "cpu",
            ], check=True, cwd=str(REPO), env=env, timeout=11000)
            os.replace(partial, output)
        with np.load(output, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise RuntimeError(f"Incorrect refiner output image count: {encoder}")
            values = {name: archive[name] for name in archive.files}
        evidence = {
            "encoder": encoder, "output_sha256": P.sha(output),
            "checkpoint_sha256": P.sha(REPO / "release/models" / weight),
            "images": len(expected), "cpu_fp32": True
        }
        if not verify_resume_receipt(receipt, evidence):
            P.atomic_json(receipt, evidence)
        print(f"P103_REFINER {encoder} new={len(values)}", flush=True)
        result.append(values)
    return result[0], result[1]


def merge_pilot_probabilities(images: list[Path], newer: tuple[dict, dict]) -> tuple[dict, dict]:
    """Merge immutable four-image pilot predictions without rerunning models."""
    merged = []
    for encoder, new in zip(("resnet34", "efficientnet_v2_s"), newer):
        with np.load(PILOT / f"{encoder}.npz", allow_pickle=False) as archive:
            reused = {key: archive[key] for key in archive.files}
        if not set(reused).isdisjoint(new):
            raise RuntimeError("Recomputed pilot image in full run.")
        both = {**reused, **new}
        if set(both) != {image.stem for image in images}:
            raise RuntimeError(f"Incomplete merged refiner inventory: {encoder}")
        merged.append(both)
    return merged[0], merged[1]


def main() -> None:
    """Continue from publicly hashed pilot to 180 masks without recomputation."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "MISSING") != "":
        raise RuntimeError("P103 explicitly prohibits any CUDA-visible GPU.")
    if not PREREG.is_file():
        raise FileNotFoundError("P103 preregistration was not published.")
    cv2.setNumThreads(1)
    P.self_test()
    P.verify_space()
    images = sorted(P.IMAGES.glob("*.jpeg"))
    metadata, weights = P.verify_inputs(images)
    if len(images) != 180:
        raise RuntimeError("Official test image inventory is incomplete.")
    pilot = check_pilot(images)
    if (OUT / "P103_FINAL_AUDIT.json").exists():
        print("P103_ALREADY_COMPLETE_NOT_REPEATED", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    P.OUT = OUT
    os.environ["SOLARFIL_ASSET_ROOT"] = str(OUT / "assets")
    start = time.monotonic()
    upstream = prepare_overlay(images)
    new = images[EXPECTED_PILOT_COUNT:]
    crop_audit = compute_remaining_crops(new, metadata)
    newly_inferred = infer_new_refiners(new)
    ensure_pilot_crops(images, pilot)
    models = merge_pilot_probabilities(images, newly_inferred)
    final = P.final_audit(images, metadata, *models)
    provisional = OUT / "p102_pilot_submission.csv"
    submitted = OUT / "p103_180_reconstructed.csv"
    if not provisional.exists() or submitted.exists():
        raise RuntimeError("Unexpected full reconstructed CSV state.")
    os.replace(provisional, submitted)
    mismatch = crop_audit["field_image_mismatches_new_176"].copy()
    for key, prior in pilot["crops"]["field_image_mismatches"].items():
        mismatch[key] += prior
    payload = {
        "experiment": "P103", "status": "PUBLIC_STAGE_180_AUDIT_COMPLETE_NO_KAGGLE_SUBMISSION",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "images": len(images), "pilot_reused_without_recomputation": 4,
        "additional_images_inferred": 176,
        "pilot_receipt_sha256": P.sha(PILOT / "P102_AUDIT.json"),
        "prereg_sha256": P.sha(PREREG),
        "script_sha256": P.sha(Path(__file__)),
        "upstream": upstream,
        "published_weight_sha256": weights,
        "crop_mismatch_images": mismatch,
        "crop_receipts": {key: value["crop_sha256"]
                          for key, value in crop_audit["records"].items()},
        "final": final,
        "reconstructed_csv_sha256": P.sha(submitted),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "cpu_only": True, "test_annotations_read": False,
        "official_submission_made": False,
        "requires_separate_clean_install_reproduction": True,
    }
    P.atomic_json(OUT / "P103_FINAL_AUDIT.json", payload)
    print("P103_COMPLETED", json.dumps({
        "images": 180, "exact_images": final["exact_images"],
        "fallback_images": final["fallback_images"],
        "rows": final["generated_rows"], "crop_mismatch_images": mismatch,
        "submitted_csv_sha256": final["reference_sha256"],
        "rebuilt_csv_sha256": final["generated_csv_sha256"],
    }), flush=True)


if __name__ == "__main__":
    main()
