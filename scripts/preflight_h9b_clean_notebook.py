#!/usr/bin/env python3
"""Audit the public raw-JPEG notebook without launching GPU work.

This preflight validates source syntax, checkpoint inventory, dependency
availability, public-stage interfaces and declared experiment restrictions.
It does not run prediction or claim byte-identical Kaggle submission output.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

EXPECTED_FILES = (
    "release/model_metadata/WEIGHTS_SHA256SUMS.txt",
    "release/model_metadata/h9b_nosite_selector.json",
    "release/model_metadata/h9b_nosite_selector.pkl",
    "release/source/h9_infer.py",
    "release/source/h9_official.py",
    "release/source/h9_score.py",
    "scripts/verify_release_assets.py",
    "scripts/replay_h3_from_original_images.py",
    "scripts/replay_maskrcnn_from_original_images.py",
    "scripts/replay_public_detector_p81.py",
    "scripts/public_replay_runtime.py",
    "scripts/reassemble_maskrcnn_fp32.py",
    "release/models/fp32_shards/manifest.json",
    "scripts/p95_nosite_jpeg_crop_audit.py",
    "scripts/official_consensus.py",
)
PYTHON_MODULES = (
    "cv2", "numpy", "torch", "torchvision", "pycocotools", "PIL",
    "scipy", "sklearn",
)
EXPECTED_SYMBOLS = {
    "scripts/p95_nosite_jpeg_crop_audit.py": {"rebuild_one"},
    "scripts/official_consensus.py": {"consensus"},
    "release/source/h9_score.py": {"assemble", "refined"},
    "release/source/h9_official.py": {"components"},
}

def sha256_file(path: Path) -> str:
    """Read a file sequentially and return its SHA-256 digest."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()

def audit(root: Path, full_hash: bool) -> dict:
    """Validate the notebook's static execution contract without CUDA."""
    notebook_path = root / "notebooks/h9b_clean_raw_jpeg_to_csv.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise AssertionError("Expected a version 4 Jupyter notebook.")
    cells = notebook.get("cells", [])
    if not cells:
        raise AssertionError("The notebook is empty.")
    compiled = 0
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        raw = cell.get("source", "")
        source = "".join(raw) if isinstance(raw, list) else raw
        try:
            compile(source, f"notebook-cell-{index}", "exec")
        except SyntaxError as exc:
            raise AssertionError(f"Invalid Python cell {index}: {exc}") from exc
        compiled += 1
    if compiled < 5:
        raise AssertionError("The notebook has too few executable cells.")
    missing = [name for name in EXPECTED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Public source files are missing: {missing}")
    for name, required_names in EXPECTED_SYMBOLS.items():
        tree = ast.parse((root / name).read_text(encoding="utf-8"), name)
        functions = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing_functions = required_names - functions
        if missing_functions:
            raise AssertionError(f"{name} lacks {sorted(missing_functions)}")
    missing_modules = [
        name for name in PYTHON_MODULES
        if importlib.util.find_spec(name) is None
    ]
    if missing_modules:
        raise RuntimeError(f"Missing runtime packages: {missing_modules}")
    manifest_path = root / EXPECTED_FILES[0]
    entries = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split()
        if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
            raise AssertionError(f"Invalid SHA-256 manifest: {filename}")
        weight = root / "release/models" / filename
        if not weight.is_file():
            raise FileNotFoundError(f"Missing public weight: {filename}")
        if full_hash and sha256_file(weight) != digest:
            raise AssertionError(f"Public weight SHA-256 mismatch: {filename}")
        entries.append(filename)
    if len(entries) != 10 or len(set(entries)) != 10:
        raise AssertionError("Expected ten unique public checkpoints.")
    sys.path.insert(0, str(root / "scripts"))
    from reassemble_maskrcnn_fp32 import validate_manifest, verify_parts
    shard_manifest = root / "release/models/fp32_shards/manifest.json"
    originals = validate_manifest(shard_manifest)
    shard_files = [
        shard_manifest.parent / part["file"]
        for model in originals.values() for part in model["parts"]
    ]
    missing_shards = [str(path.relative_to(root)) for path in shard_files if not path.is_file()]
    if missing_shards:
        raise FileNotFoundError(f"Original FP32 checkpoint shards are missing: {missing_shards}")
    if full_hash:
        verify_parts(originals, shard_manifest.parent, tuple(originals))
    source_text = notebook_path.read_text(encoding="utf-8")
    for phrase in (
        "SOLARFIL_DEVICE", "SOLARFIL_GPU_UUID",
        "SOLARFIL_TEST_IMAGES", "SOLARFIL_REPLAY_OUT",
        "replay_maskrcnn_from_original_images.py", "original_fp32_detector_sha256",
        "NOT_OFFICIAL_SUBMISSION",
        "exact_historical_upload_parity_asserted",
    ):
        if phrase not in source_text:
            raise AssertionError(f"Notebook lacks critical guardrail: {phrase}")
    return {
        "audit": "PUBLIC_NOTEBOOK_STATIC_PREFLIGHT",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notebook": str(notebook_path.relative_to(root)),
        "notebook_sha256": sha256_file(notebook_path),
        "code_cells_compiled": compiled,
        "public_source_files_checked": len(EXPECTED_FILES),
        "interface_source_files_checked": len(EXPECTED_SYMBOLS),
        "checkpoints_present": len(entries),
        "checkpoints_fully_hashed": bool(full_hash),
        "original_fp32_shards_present": len(shard_files),
        "original_fp32_shards_fully_hashed": bool(full_hash),
        "package_imports_available": list(PYTHON_MODULES),
        "cuda_jobs_launched": 0,
        "model_inference_executed": False,
        "official_submission_claimed": False,
    }

def main() -> int:
    """Run the offline checks, return an explicit success or failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.full_hash)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        destination = args.receipt.resolve()
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite receipt: {destination}")
        temp = destination.with_suffix(destination.suffix + ".partial")
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
