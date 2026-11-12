#!/usr/bin/env python3
"""Restore byte-exact Solar Filament 2026 Mask R-CNN FP32 weights.

The four original FP32 checkpoint files exceed GitHub's 100 MB file limit.
Each original model was therefore split into two independently SHA-256-bound
pieces of no more than 95,000,000 bytes. This script uses only the Python
standard library, validates every piece, and assembles original checkpoints
without overwriting any existing file.

Example:
    python scripts/reassemble_maskrcnn_fp32.py --check-only
    python scripts/reassemble_maskrcnn_fp32.py --output-dir /mnt/hdd/solarfil-weights

The reconstruction is bit-for-bit, not a numerical FP16-to-FP32 conversion.
It does not load Kaggle labels, execute CUDA or submit any predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import tempfile

FOLDS = ("F1", "F2", "F3", "F4")
MAX_PART_BYTES = 95_000_000
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PART_PATTERN = re.compile(r"^maskrcnn_(F[1-4])_fp32\.pt\.part(0[01])$")


def digest_file(path: pathlib.Path) -> str:
    """Hash arbitrary files in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest_path: pathlib.Path) -> dict:
    """Validate the release manifest before reading any supplied part path."""
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    if obj.get("format") != "solar-filament-maskrcnn-shards-v1":
        raise ValueError("Unsupported model shard manifest format")
    models = obj.get("models")
    if not isinstance(models, dict) or set(models) != set(FOLDS):
        raise ValueError("Exactly four F1-F4 models are required")
    if obj.get("shard_max_bytes") != MAX_PART_BYTES:
        raise ValueError("The shard size bound does not match the release")
    for fold in FOLDS:
        entry = models[fold]
        if not isinstance(entry, dict) or not HASH_PATTERN.fullmatch(
                str(entry.get("original_sha256", ""))):
            raise ValueError(f"Missing original hash for {fold}")
        if not isinstance(entry.get("original_bytes"), int) or entry["original_bytes"] < 1:
            raise ValueError(f"Missing original length for {fold}")
        parts = entry.get("parts")
        if not isinstance(parts, list) or len(parts) != 2:
            raise ValueError(f"Exactly two shards are expected for {fold}")
        total = 0
        for index, part in enumerate(parts):
            expected = f"maskrcnn_{fold}_fp32.pt.part{index:02d}"
            if not isinstance(part, dict) or part.get("file") != expected:
                raise ValueError(f"Unexpected or unsafe filename for {fold} part {index}")
            if not PART_PATTERN.fullmatch(expected):
                raise ValueError("Invalid checkpoint shard filename")
            length = part.get("bytes")
            if not isinstance(length, int) or not 0 < length <= MAX_PART_BYTES:
                raise ValueError(f"Invalid shard length for {expected}")
            if not HASH_PATTERN.fullmatch(str(part.get("sha256", ""))):
                raise ValueError(f"Invalid shard hash for {expected}")
            total += length
        if total != entry["original_bytes"]:
            raise ValueError(f"Original and shard lengths disagree for {fold}")
    return models


def verify_parts(models: dict, shard_dir: pathlib.Path, folds: tuple[str, ...]) -> None:
    """Reject missing, altered or symlinked shard files before assembly."""
    for fold in folds:
        for part in models[fold]["parts"]:
            path = shard_dir / part["file"]
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"Missing or unsafe shard: {path}")
            if path.stat().st_size != part["bytes"]:
                raise ValueError(f"Incorrect shard byte count: {path}")
            if digest_file(path) != part["sha256"]:
                raise ValueError(f"SHA-256 mismatch in shard: {path}")
        print(f"SHARDS_VERIFIED {fold}", flush=True)


def restore_models(models: dict, shard_dir: pathlib.Path,
                   output_dir: pathlib.Path, folds: tuple[str, ...]) -> None:
    """Stream immutable verified pieces into atomic, hash-checked outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    if shutil.disk_usage(output_dir).free < sum(
            models[f]["original_bytes"] for f in folds) + 256 * 1024 * 1024:
        raise OSError("Insufficient output disk space for safe reconstruction")
    for fold in folds:
        entry = models[fold]
        destination = output_dir / f"maskrcnn_{fold}_fp32.pt"
        if destination.is_symlink():
            raise ValueError(f"Refusing to follow destination symlink: {destination}")
        if destination.exists():
            if (destination.stat().st_size != entry["original_bytes"]
                    or digest_file(destination) != entry["original_sha256"]):
                raise FileExistsError(f"Existing output has a different hash: {destination}")
            print(f"ALREADY_VERIFIED {fold} {entry['original_sha256']}", flush=True)
            continue
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".maskrcnn_{fold}.", suffix=".tmp", dir=output_dir
        )
        temporary = pathlib.Path(temporary_name)
        try:
            original_hash = hashlib.sha256()
            written = 0
            with os.fdopen(fd, "wb") as target:
                for part in entry["parts"]:
                    with (shard_dir / part["file"]).open("rb") as source:
                        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                            target.write(chunk)
                            original_hash.update(chunk)
                            written += len(chunk)
                target.flush()
                os.fsync(target.fileno())
            if written != entry["original_bytes"] or original_hash.hexdigest() != entry["original_sha256"]:
                raise ValueError(f"Assembled model checksum failure: {fold}")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Concurrent writer created {destination}")
            os.replace(temporary, destination)
            print(f"RESTORED {fold} {entry['original_sha256']}", flush=True)
        finally:
            if temporary.exists():
                temporary.unlink()


def main() -> None:
    """Run integrity verification and optional deterministic restoration."""
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=root / "release/models/fp32_shards/manifest.json")
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--fold", choices=FOLDS, action="append", dest="folds")
    args = parser.parse_args()
    if not args.check_only and args.output_dir is None:
        parser.error("--output-dir is required except with --check-only")
    selected = tuple(dict.fromkeys(args.folds)) if args.folds else FOLDS
    models = validate_manifest(args.manifest)
    verify_parts(models, args.manifest.resolve().parent, selected)
    if not args.check_only:
        restore_models(models, args.manifest.resolve().parent,
                       args.output_dir.resolve(), selected)
    print("FP32_RELEASE_ASSETS_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
