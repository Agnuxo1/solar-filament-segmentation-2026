#!/usr/bin/env python3
"""Reconstruct public H9b detector proposals from JPEGs and original FP32 weights.

This portable inference entry point needs no historical predictions, labels,
network access or training. It restores the exact original checkpoint bytes
from the public shards, computes original/reflected image views and records
input/output hashes. CPU is the default; CUDA requires one idle selected GPU.
Successful execution does not establish final-CSV parity or a Kaggle score.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

from public_replay_runtime import (
    check_gpu_idle, prepare_output, process_environment, publish_json,
    select_device, sha256_file,
)
from reassemble_maskrcnn_fp32 import restore_models, validate_manifest, verify_parts
from replay_public_detector_p81 import unreflect, verify_proposals

FOLDS = ("F1", "F2", "F3", "F4")
EXPECTED_FP32 = {
    "F1": "399b4bd790c5acf88f32da926dd00a9ac4b704040241b8b0899117c687756820",
    "F2": "9a50cc6d23d1689f8d9a69ac90d25c086d128d98682c77669366ce59caca1115",
    "F3": "e7f9ed980842c2a04ee5ff21fe3018d00eef0ed742aace9f9bb0e08af77f8952",
    "F4": "59e942afc969318681c7de5c748ed58bd29b79448243389e34db93a8fe54827f",
}


class OneImage:
    """Expose a single grayscale array to the released detector's input API."""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse portable paths, the numerical backend and bounded host resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weights", type=Path,
                        help="Restored FP32 directory; defaults to OUT/weights_fp32.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-uuid", help="Physical GPU UUID or index for CUDA.")
    parser.add_argument("--fold", choices=FOLDS, action="append", dest="folds")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--reserve-gib", type=float, default=20)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1 or args.reserve_gib < 0:
        parser.error("CPU threads must be positive and disk reserve non-negative.")
    if args.device == "cpu" and args.gpu_uuid:
        parser.error("--gpu-uuid is only valid with --device cuda.")
    args.folds = tuple(dict.fromkeys(args.folds)) if args.folds else FOLDS
    return args


def contract_hash(contract: dict) -> str:
    """Hash execution inputs to prevent mixing results across model/backend changes."""
    value = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def validate_saved_prediction(path: Path, receipt: Path, image_hash: str,
                              checkpoint_hash: str, run_hash: str) -> dict:
    """Verify a completed prediction before loading its own trusted pickle."""
    if not path.is_file() or not receipt.is_file():
        raise RuntimeError(f"Unattested detector output requires review: {path}")
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if (record.get("input_sha256") != image_hash
            or record.get("checkpoint_sha256") != checkpoint_hash
            or record.get("contract_sha256") != run_hash
            or record.get("sha256") != sha256_file(path)):
        raise RuntimeError(f"Saved detector output has changed inputs or bytes: {path}")
    return record


def infer_one(model, image: Path, size: int, device, detector, read_gray):
    """Predict the two fixed geometric views without accessing annotations."""
    import numpy as np
    import torch

    gray = read_gray(image, size)
    if gray.shape != (size, size):
        raise ValueError(f"Unexpected source image geometry: {image}")
    with torch.inference_mode():
        first = detector.predict_files(
            model, OneImage(gray), {image.name: 0}, [image.name], device,
        )[image.name]
        mirrored = np.ascontiguousarray(gray[:, ::-1])
        second = detector.predict_files(
            model, OneImage(mirrored), {image.name: 0}, [image.name], device,
        )[image.name]
    result = [first, [unreflect(item, size) for item in second]]
    verify_proposals(result, size)
    return result


def save_prediction(path: Path, value) -> None:
    """Commit an inference output atomically; never overwrite an earlier one."""
    if path.exists():
        raise FileExistsError(path)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict:
    """Restore original weights, run or verify four folds, and publish a receipt."""
    release = args.release.expanduser().resolve()
    root = release.parent
    images = sorted(args.images.expanduser().resolve().glob("*.jpeg"))
    if len(images) != 180 or len({image.stem for image in images}) != 180:
        raise ValueError("Expected exactly 180 distinct official test JPEGs.")
    models = validate_manifest(release / "models/fp32_shards/manifest.json")
    if any(models[fold]["original_sha256"] != EXPECTED_FP32[fold] for fold in FOLDS):
        raise ValueError("The public manifest does not describe the frozen original FP32 checkpoints.")
    out = prepare_output(args.out, args.reserve_gib)
    lock = out / ".public_detector_replay.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Another or interrupted replay owns {lock}; inspect it before restarting.") from exc
    os.close(descriptor)
    try:
        return run_locked(args, release, root, images, models, out)
    finally:
        lock.unlink(missing_ok=True)


def run_locked(args, release: Path, root: Path, images: list[Path], models: dict,
               out: Path) -> dict:
    """Execute one owner at a time, rechecking immutable artifacts on resume."""
    verify_parts(models, release / "models/fp32_shards", args.folds)
    weights = args.weights.expanduser().resolve() if args.weights else out / "weights_fp32"
    restore_models(models, release / "models/fp32_shards", weights, args.folds)
    selected, gpu_uuid = select_device(args.device, args.gpu_uuid)
    env = process_environment(selected, gpu_uuid, root, args.cpu_threads)
    for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = env[key]
    sys.path.insert(0, str(release / "source"))
    import numpy as np
    import torch
    import torchvision
    import filseg.train_maskrcnn as detector
    from filseg.data import RES, read_gray

    if selected == "cuda":
        check_gpu_idle(gpu_uuid)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA must expose exactly the selected physical GPU.")
    detector.autocast = lambda device: contextlib.nullcontext()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda:0" if selected == "cuda" else "cpu")
    image_hashes = {path.name: sha256_file(path) for path in images}
    source_paths = [
        Path(__file__), Path(__file__).with_name("public_replay_runtime.py"),
        Path(__file__).with_name("replay_public_detector_p81.py"),
        release / "source/filseg/train_maskrcnn.py", release / "source/filseg/data.py",
        release / "source/filseg/common.py",
    ]
    contract = {
        "schema": "PUBLIC_ORIGINAL_FP32_DETECTOR_V1", "images": image_hashes,
        "weights": {fold: EXPECTED_FP32[fold] for fold in args.folds},
        "source_sha256": {path.name: sha256_file(path) for path in source_paths},
        "device": selected, "gpu_uuid": gpu_uuid, "cpu_threads": args.cpu_threads,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": np.__version__, "inference_precision": "float32_no_autocast",
        "folds": list(args.folds),
    }
    run_hash = contract_hash(contract)
    contract_file = out / "run_contract.json"
    if contract_file.exists():
        if json.loads(contract_file.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Output directory belongs to a different execution contract; use a new directory.")
    else:
        publish_json(contract_file, contract)
    started = time.monotonic()
    completed_folds = {}
    for fold in args.folds:
        folder = out / fold
        folder.mkdir(exist_ok=True)
        entries = {}
        model = None
        try:
            for index, image in enumerate(images, 1):
                output = folder / f"{image.stem}.pkl"
                receipt = folder / f"{image.stem}.receipt.json"
                if output.exists() or receipt.exists():
                    entry = validate_saved_prediction(output, receipt, image_hashes[image.name], EXPECTED_FP32[fold], run_hash)
                else:
                    if model is None:
                        model = detector.build_model(size=RES, pretrained=False)
                        model.load_state_dict(torch.load(weights / f"maskrcnn_{fold}_fp32.pt", map_location="cpu", weights_only=True), strict=True)
                        model = model.eval().to(device)
                    result = infer_one(model, image, RES, device, detector, read_gray)
                    save_prediction(output, result)
                    entry = {
                        "input_sha256": image_hashes[image.name], "sha256": sha256_file(output),
                        "checkpoint_sha256": EXPECTED_FP32[fold], "contract_sha256": run_hash,
                        "counts": [len(view) for view in result],
                    }
                    publish_json(receipt, entry)
                entries[image.name] = entry
                if index == 1 or index % 20 == 0 or index == len(images):
                    print("PUBLIC_FP32_DETECTOR", fold, index, len(images), flush=True)
        finally:
            del model
            if selected == "cuda":
                torch.cuda.empty_cache()
        payload = {"status": "PUBLIC_FP32_FOLD_COMPLETE", "images": len(images),
                   "fold": fold, "contract_sha256": run_hash, "entries": entries}
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != payload:
                raise RuntimeError(f"Saved fold manifest changed: {fold}")
        else:
            publish_json(manifest_path, payload)
        completed_folds[fold] = {"images": len(images), "manifest_sha256": sha256_file(manifest_path)}
    final_path = out / "receipt.json"
    final = {
        "status": "PUBLIC_ORIGINAL_FP32_DETECTORS_COMPLETE_NOT_FINAL_CSV",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "images": len(images),
        "contract_sha256": run_hash, "device": selected, "gpu_uuid": gpu_uuid,
        "folds": completed_folds, "elapsed_seconds": round(time.monotonic() - started, 3),
        "test_annotations_used": False, "kaggle_submission_made": False,
        "historical_prediction_parity_asserted": False,
    }
    if final_path.exists():
        previous = json.loads(final_path.read_text(encoding="utf-8"))
        if (previous.get("contract_sha256") != run_hash or previous.get("folds") != completed_folds):
            raise RuntimeError("Final detector receipt differs from verified folds.")
        return previous
    publish_json(final_path, final)
    return final


def main(argv: list[str] | None = None) -> int:
    """Expose the complete portable detector inference workflow as a CLI."""
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
