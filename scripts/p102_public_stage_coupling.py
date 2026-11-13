#!/usr/bin/env python3
"""P102: audit the coupling of publicly replayed H3 and Mask R-CNN stages to H9b.

This is an immutable CPU-only scientific experiment. Input consists exclusively
of official unlabelled JPEGs, published model outputs regenerated independently
from those JPEGs, and the frozen TRAIN-only no-site selector. Original H9b
caches are comparison references only, not model inputs. A historical
self-generated, non-ground-truth control CSV is used only when the original
H9b fallback policy requests it, and this dependency is disclosed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
STAGING = WORK.parent / "final-release-staging-20261112"
IMAGES = Path("/home/ai/pc/d/PROJECTS/196_NeuroPixel/kaggle/filament/data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
OUT = WORK / "p102_public_stage_coupling_20261113"
P88 = WORK / "p88_public_fp32_cpu_full_20261112"
FOLDS = ("F1", "F2", "F3", "F4")
H3 = {
    "F1": WORK / "p77_full_h3_public_20261112/F1",
    **{f: WORK / f"p79_full_h3_public_20261112/{f}" for f in ("F2", "F3", "F4")}
}
RECEIPT = {
    "F1": WORK / "p77_full_h3_public_20261112/F1_h3_replay_receipt.json",
    **{f: WORK / f"p79_full_h3_public_20261112/{f}_h3_replay_receipt.json"
       for f in ("F2", "F3", "F4")}
}
PREREG = REPO / "docs/submission/P102_PUBLIC_STAGE_COUPLING_PREREG_20261113.md"
MANIFEST = REPO / "release/model_metadata/h9b_nosite_selector.json"
SELECTOR = REPO / "release/model_metadata/h9b_nosite_selector.pkl"
WEIGHT_MANIFEST = REPO / "release/model_metadata/WEIGHTS_SHA256SUMS.txt"
REFERENCES = STAGING / "h9/official/crops"
ACTUAL = WORK / "p74_accepted_artifact_audit/accepted_h9b_nosite.csv"
CONTROL = STAGING / "submission.csv"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "release/source"))

from official_consensus import consensus
from p95_nosite_jpeg_crop_audit import rebuild_one, compare_arrays
from filseg.postprocess import rle_counts
import h9_score


def sha(path: Path) -> str:
    """SHA-256 using bounded reads, to protect immutable scientific inputs."""
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, item: dict) -> None:
    """Publish write-once receipts after flushing; reject accidental overwrites."""
    if path.exists():
        raise FileExistsError(f"Immutable receipt already exists: {path}")
    pending = path.with_name(path.name + ".partial")
    with pending.open("x", encoding="utf8") as sink:
        json.dump(item, sink, indent=2, sort_keys=True)
        sink.write("\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(pending, path)


def mask_rows(path: Path) -> dict[str, list[str]]:
    """Read own predictions, not labels, and group RLEs by source image stem."""
    grouped: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected CSV schema: {path}")
        ids: set[str] = set()
        for row in reader:
            ident = row["filament_id"]
            if ident in ids:
                raise ValueError(f"Duplicate mask identifier: {ident}")
            ids.add(ident)
            grouped.setdefault(ident.rsplit("_", 1)[0], []).append(row["segmentation_rle"])
    return grouped


def verify_weights() -> dict[str, str]:
    """Validate SHA-256 of every published model file before scientific inference."""
    records = {}
    for raw in WEIGHT_MANIFEST.read_text(encoding="utf8").splitlines():
        digest, filename = raw.split()
        if len(digest) != 64:
            raise ValueError("Malformed model manifest.")
        source = REPO / "release/models" / filename
        if not source.is_file() or sha(source) != digest:
            raise RuntimeError(f"Published model checksum failed: {filename}")
        records[filename] = digest
    if len(records) != 10:
        raise RuntimeError("Expected exactly ten released model checkpoints.")
    return records


def verify_inputs(images: list[Path]) -> tuple[dict, dict]:
    """Reject incomplete upstream receipts, unexpected imagery, or selector drift."""
    if not PREREG.exists():
        raise RuntimeError("P102 preregistration must exist before processing.")
    if len(images) != 180 or len({p.stem for p in images}) != 180:
        raise RuntimeError("Expected 180 distinct official test JPEG files.")
    metadata = json.loads(MANIFEST.read_text(encoding="utf8"))
    if sha(SELECTOR) != metadata["weight_sha256"]:
        raise RuntimeError("TRAIN-only selector SHA-256 mismatch.")
    if metadata["excluded_inference_features"] != ["site", "date", "temporal_neighbors"]:
        raise RuntimeError("Unexpected no-site selector exclusion policy.")
    detector = json.loads((P88 / "receipt.json").read_text(encoding="utf8"))
    if detector["status"] != "P88_PUBLIC_SOURCE_CPU_720_EXACT_PASS":
        raise RuntimeError("P88 reconstructed detector receipt not final.")
    for fold in FOLDS:
        if detector["folds"][fold]["exact_images"] != 180:
            raise RuntimeError(f"Missing verified P88 fold: {fold}")
        h3_report = json.loads(RECEIPT[fold].read_text(encoding="utf8"))
        if not h3_report.get("full_180_images"):
            raise RuntimeError(f"Incomplete public H3 replay: {fold}")
        if len(h3_report.get("image_sha256", [])) != 180:
            raise RuntimeError(f"Incomplete H3 manifest: {fold}")
    if not ACTUAL.is_file() or not CONTROL.is_file():
        raise RuntimeError("Missing frozen own prediction CSV used for auditing.")
    if sha(ACTUAL) != "24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60":
        raise RuntimeError("Genuine historical Kaggle CSV checksum mismatch.")
    weights = verify_weights()
    return metadata, weights


def verify_space() -> None:
    """Keep the RAID/HDD safety reserve; avoid the 97%-full SSD."""
    stat = os.statvfs(WORK)
    if stat.f_bavail * stat.f_frsize < 20 * 1024 ** 3:
        raise OSError("Less than 20 GiB free on research HDD.")


def create_overlay(images: list[Path]) -> dict[str, str]:
    """Build public-source predictions; never link the old H9b detector cache."""
    overlay = OUT / "assets"
    detector_file = overlay / "dets_official.pkl"
    overlay.mkdir(parents=True, exist_ok=True)
    h3_hashes = {}
    for fold in FOLDS:
        folder = overlay / "h3/maps" / f"{fold}_resnet34_s0"
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / "official"
        if not link.is_symlink():
            if link.exists():
                raise RuntimeError(f"Unexpected map source: {link}")
            link.symlink_to(H3[fold], target_is_directory=True)
        if link.resolve() != H3[fold].resolve():
            raise RuntimeError(f"Changed H3 map source: {fold}")
        for image in images:
            source = H3[fold] / f"{image.stem}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
        h3_hashes[fold] = sha(RECEIPT[fold])

    if detector_file.exists():
        raise FileExistsError("Do not replace a historical or already assembled P102 detector file.")
    aggregated: dict[str, list] = {}
    for index, image in enumerate(images, 1):
        views = []
        for fold in FOLDS:
            source = P88 / fold / f"{image.stem}.pkl"
            with source.open("rb") as stream:
                one_fold = pickle.load(stream)
            if len(one_fold) != 2:
                raise RuntimeError(f"Bad detector view count: {fold} {image.name}")
            views.extend(one_fold)
        aggregated[image.name] = consensus(views)
        if index == 1 or index == len(images):
            print(f"P102_CONSENSUS {index}/{len(images)}", flush=True)
    temp = detector_file.with_suffix(".partial")
    with temp.open("xb") as sink:
        pickle.dump(aggregated, sink, protocol=pickle.HIGHEST_PROTOCOL)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temp, detector_file)
    return {"detector_sha256": sha(detector_file), "h3_receipt_sha256": h3_hashes}


def reconstruct_crops(images: list[Path], metadata: dict) -> dict:
    """Run the frozen public H6 selector on independent published-model outputs."""
    with SELECTOR.open("rb") as stream:
        selector = pickle.load(stream)
    crop_dir = OUT / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    differences = {field: 0 for field in ("X", "M", "comp", "score")}
    hashes = {}
    for index, image in enumerate(images, 1):
        destination = crop_dir / (image.stem + ".npz")
        if destination.exists():
            raise FileExistsError(f"No implicit replay/overwrite: {destination}")
        prediction = rebuild_one(image, selector, metadata)
        reference = REFERENCES / destination.name
        with np.load(reference, allow_pickle=False) as z:
            comparisons = compare_arrays(prediction, {k: z[k] for k in differences})
        for field, info in comparisons.items():
            differences[field] += int(not info["exact"])
        temporary = destination.with_suffix(".partial")
        with temporary.open("xb") as sink:
            np.savez_compressed(sink, **prediction)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, destination)
        hashes[image.name] = {"image": sha(image), "crops": sha(destination),
                              "reference_crop": sha(reference),
                              "compare": comparisons,
                              "instances": int(len(prediction["comp"]))}
        print(f"P102_CROPS {index}/{len(images)} k={len(prediction['comp'])}", flush=True)
    return {"field_image_mismatches": differences, "records": hashes}


def infer_refiners(images: list[Path]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Replay the two published refiners on generated crops, with no CUDA."""
    names = ("resnet34", "efficientnet_v2_s")
    results = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "release/source"), str(REPO)])
    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    for encoder, checkpoint in zip(names, ("refiner_ALL.pt", "refiner_ALL_efficientnet_v2_s.pt")):
        target = OUT / (encoder + ".npz")
        if target.exists():
            raise FileExistsError(f"Refiner output already exists: {target}")
        temp = OUT / (encoder + ".partial.npz")
        if temp.exists():
            raise FileExistsError(f"Review interrupted inference before retrying: {temp}")
        subprocess.run([sys.executable, "-u", str(REPO / "release/source/h9_infer.py"),
                        "--weights", str(REPO / "release/models" / checkpoint),
                        "--encoder", encoder, "--crops", str(OUT / "crops"),
                        "--out", str(temp), "--device", "cpu"],
                       check=True, cwd=str(REPO), env=env, timeout=12600)
        with np.load(temp, allow_pickle=False) as source:
            if set(source.files) != {p.stem for p in images}:
                raise RuntimeError(f"Refiner prediction inventory mismatch: {encoder}")
        os.replace(temp, target)
        with np.load(target, allow_pickle=False) as source:
            results.append({k: source[k] for k in source.files})
        print(f"P102_REFINER {encoder} images={len(images)}", flush=True)
    return results[0], results[1]


def final_audit(images: list[Path], metadata: dict, maps_a: dict, maps_b: dict) -> dict:
    """Recompose public-stage final masks; audit against original upload, never fit."""
    import h9_official
    actual, control = mask_rows(ACTUAL), mask_rows(CONTROL)
    records = {}
    out_rows = []
    fallback_count = 0
    for image in images:
        stem = image.stem
        with np.load(OUT / "crops" / f"{stem}.npz", allow_pickle=False) as data:
            comp, scores, matrices = data["comp"], data["score"], data["M"]
        if len(comp):
            _, _, _, lab, _, _ = h9_official.components(image.name, metadata["w"], metadata["t"])
            merged = np.mean([maps_a[stem].astype(np.float32),
                              maps_b[stem].astype(np.float32)], axis=0).round().astype(np.uint8)
            if merged.shape[0] != len(comp):
                raise RuntimeError(f"Crop and refiner count mismatch for {stem}")
            seeds = [h9_score.refined(lab == int(c) + 1, merged[j], matrices[j])
                     for j, c in enumerate(comp)]
            labels = h9_score.assemble(seeds, np.argsort(-scores, kind="stable"))
            predictions = [rle_counts(labels == j) for j in range(1, int(labels.max()) + 1)]
            source = "public_pipeline"
        else:
            predictions = control.get(stem, [])
            source = "archived_self_generated_control_fallback"
            fallback_count += 1
        expected = actual.get(stem, [])
        exact = predictions == expected
        records[stem] = {"source": source, "predicted_instances": len(predictions),
                         "submitted_instances": len(expected),
                         "identical_rle_sequence": exact,
                         "identical_rle_count": sum(a == b for a, b in zip(predictions, expected)),
                         "different_rows_by_position": sum(a != b for a, b in zip(predictions, expected))
                                                       + abs(len(predictions) - len(expected))}
        out_rows.extend((f"{stem}_{i}", rle) for i, rle in enumerate(predictions, 1))
    csv_path = OUT / "p102_pilot_submission.csv"
    if csv_path.exists():
        raise FileExistsError("P102 pilot CSV already exists.")
    with csv_path.open("x", newline="", encoding="utf8") as sink:
        writer = csv.writer(sink, lineterminator="\n")
        writer.writerow(("filament_id", "segmentation_rle"))
        writer.writerows(out_rows)
    return {"comparison_by_image": records, "fallback_images": fallback_count,
            "independent_raw_to_csv": fallback_count == 0,
            "exact_images": sum(d["identical_rle_sequence"] for d in records.values()),
            "all_exact": all(d["identical_rle_sequence"] for d in records.values()),
            "generated_rows": len(out_rows), "generated_csv_sha256": sha(csv_path),
            "reference_sha256": sha(ACTUAL)}


def self_test() -> None:
    """Exercise grouping and a deliberately modified row in a temporary test CSV."""
    import tempfile
    with tempfile.TemporaryDirectory(dir=str(WORK)) as dirname:
        sample = Path(dirname) / "rows.csv"
        sample.write_text("filament_id,segmentation_rle\nA_1,3 1\nA_2,6 1\nB_1,7 2\n")
        assert mask_rows(sample) == {"A": ["3 1", "6 1"], "B": ["7 2"]}
        sample.write_text("filament_id,segmentation_rle\nA_1,3 1\nA_1,4 1\n")
        try:
            mask_rows(sample)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate-row corruption was not detected.")
    print("P102_SELF_TEST_PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "MISSING") != "":
        raise RuntimeError("P102 must have an explicitly empty CUDA_VISIBLE_DEVICES.")
    cv2.setNumThreads(1)
    self_test()
    if args.self_test:
        return
    if not 1 <= args.limit <= 180:
        parser.error("--limit must be between 1 and 180.")
    verify_space()
    images = sorted(IMAGES.glob("*.jpeg"))
    metadata, weight_hashes = verify_inputs(images)
    chosen = images[:args.limit]
    if OUT.exists():
        raise FileExistsError(f"Refuse duplicate experiment: {OUT}")
    OUT.mkdir(parents=True)
    start = time.monotonic()
    os.environ["SOLARFIL_ASSET_ROOT"] = str(OUT / "assets")
    upstream = create_overlay(chosen)
    crops = reconstruct_crops(chosen, metadata)
    resnet, efficient = infer_refiners(chosen)
    final = final_audit(chosen, metadata, resnet, efficient)
    receipt = {
        "experiment": "P102", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pilot_fixed_sorted_prefix": args.limit,
        "status": "P102_PUBLIC_STAGE_COUPLING_AUDITED_NOT_COMPETITION_SUBMISSION",
        "prereg_sha256": sha(PREREG), "script_sha256": sha(Path(__file__)),
        "selector_sha256": sha(SELECTOR), "published_model_sha256": weight_hashes,
        "P88_receipt_sha256": sha(P88 / "receipt.json"), "upstream": upstream,
        "crops": crops, "final": final,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "cpu_only": True, "test_annotations_read": False,
        "official_kaggle_submission_made": False,
        "verifies_stages_not_full_independent_training": True
    }
    atomic_json(OUT / "P102_AUDIT.json", receipt)
    print("P102_COMPLETE", json.dumps({k: v for k, v in final.items()
                                      if k != "comparison_by_image"}), flush=True)


if __name__ == "__main__":
    main()
