#!/usr/bin/env python3
"""Verify the published Solar Filament H9b release without using a GPU.

This is an integrity preflight, not an end-to-end inference test. It checks
that independently downloaded public weights match the published SHA-256
manifest, that the released Python source parses, and that both audit
notebooks contain syntactically valid Python cells.

The script never loads competition labels or images and never downloads,
modifies, trains, or executes models. A successful result must not be
presented as reproduction of Kaggle submission 58112557.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = Path("release/model_metadata/WEIGHTS_SHA256SUMS.txt")
MODEL_DIR = Path("release/models")
SOURCE_DIR = Path("release/source")
NOTEBOOKS = (
    Path("notebooks/h9b_stage2_intermediate_replay.ipynb"),
    Path("release/notebooks/h9b_stage2_exact_replay.ipynb"),
)
EXPECTED_WEIGHTS = {
    "F1_resnet34_s0_fp16.pt",
    "F2_resnet34_s0_fp16.pt",
    "F3_resnet34_s0_fp16.pt",
    "F4_resnet34_s0_fp16.pt",
    "maskrcnn_F1_fp16.pt",
    "maskrcnn_F2_fp16.pt",
    "maskrcnn_F3_fp16.pt",
    "maskrcnn_F4_fp16.pt",
    "refiner_ALL.pt",
    "refiner_ALL_efficientnet_v2_s.pt",
}
MANIFEST_LINE = re.compile(r"^([a-f0-9]{64})  ([A-Za-z0-9_.-]+\\.pt)$")


def sha256_file(path: Path) -> str:
    """Hash binary files in bounded-memory blocks on the host CPU."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_weights(root: Path) -> list[dict]:
    """Check strict manifest completeness, filenames and all model hashes."""
    manifest = root / MANIFEST
    if not manifest.is_file():
        return [{"item": str(MANIFEST), "status": "MISSING"}]
    entries: dict[str, str] = {}
    errors: list[dict] = []
    for lineno, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = MANIFEST_LINE.fullmatch(raw)
        if not match:
            errors.append({"item": f"{MANIFEST}:{lineno}", "status": "INVALID_MANIFEST_LINE"})
            continue
        wanted_hash, filename = match.groups()
        if filename in entries:
            errors.append({"item": filename, "status": "DUPLICATE_MANIFEST_ENTRY"})
        entries[filename] = wanted_hash

    for name in sorted(EXPECTED_WEIGHTS - entries.keys()):
        errors.append({"item": name, "status": "MISSING_MANIFEST_ENTRY"})
    for name in sorted(entries.keys() - EXPECTED_WEIGHTS):
        errors.append({"item": name, "status": "UNEXPECTED_MANIFEST_ENTRY"})
    for filename, expected in sorted(entries.items()):
        path = root / MODEL_DIR / filename
        if not path.is_file():
            errors.append({"item": str(MODEL_DIR / filename), "status": "MISSING"})
            continue
        actual = sha256_file(path)
        errors.append({
            "item": str(MODEL_DIR / filename),
            "status": "OK" if actual == expected else "HASH_MISMATCH",
            "bytes": path.stat().st_size,
            "sha256": actual,
            "expected_sha256": expected,
        })
    return errors


def inspect_source(root: Path) -> list[dict]:
    """Parse released Python modules without importing them or allocating CUDA."""
    folder = root / SOURCE_DIR
    paths = sorted(folder.rglob("*.py")) if folder.is_dir() else []
    if not paths:
        return [{"item": str(SOURCE_DIR), "status": "NO_PYTHON_SOURCE"}]
    results: list[dict] = []
    for path in paths:
        rel = str(path.relative_to(root))
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            results.append({"item": rel, "status": "OK"})
        except (SyntaxError, UnicodeError, OSError) as exc:
            results.append({"item": rel, "status": "INVALID_SOURCE", "detail": str(exc)})
    return results


def inspect_notebooks(root: Path) -> list[dict]:
    """Check notebook JSON and each Python code cell, without running cells."""
    results: list[dict] = []
    for rel in NOTEBOOKS:
        path = root / rel
        if not path.is_file():
            results.append({"item": str(rel), "status": "MISSING"})
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = notebook["cells"]
            code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
            if not code_cells:
                raise ValueError("No code cells")
            for index, cell in enumerate(code_cells):
                source = cell["source"]
                source = "".join(source) if isinstance(source, list) else str(source)
                ast.parse(source, filename=f"{rel}:cell_{index}")
            results.append({"item": str(rel), "status": "OK", "code_cells": len(code_cells)})
        except (KeyError, TypeError, ValueError, SyntaxError, UnicodeError, OSError) as exc:
            results.append({"item": str(rel), "status": "INVALID_NOTEBOOK", "detail": str(exc)})
    return results


def main(argv: list[str] | None = None) -> int:
    """Generate a factual receipt and signal any release-integrity failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path, help="Optional destination for a JSON audit receipt")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    weights = inspect_weights(root)
    source = inspect_source(root)
    notebooks = inspect_notebooks(root)
    failures = [entry for entry in weights + source + notebooks if entry["status"] != "OK"]
    receipt = {
        "schema": "SOLAR_H9B_PUBLIC_RELEASE_PREFLIGHT_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "weights": weights,
        "source": source,
        "notebooks": notebooks,
        "status": "ASSETS_VERIFIED" if not failures else "INTEGRITY_FAILURE",
        "end_to_end_raw_image_to_accepted_csv": "NOT_ESTABLISHED",
        "accepted_submission_id": 58112557,
        "official_public_pq": None,
        "limitations": (
            "This verifies file integrity and Python syntax only. It does not "
            "run the notebooks, measure Panoptic Quality, or reproduce the "
            "accepted Kaggle CSV. Use a separate fully executed raw-image "
            "pipeline audit before asserting scientific reproducibility."
        ),
    }
    rendered = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.json_out:
        dest = args.json_out.expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered, encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
