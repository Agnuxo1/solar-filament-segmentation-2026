#!/usr/bin/env python3
"""P118: Frozen, CPU-only public release parity and provenance audit."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112")
OUT = ROOT / "p118_public_release_parity_20261114"
P106 = ROOT / "p106_public_clean_6843362_20261113"
CHECKOUT = ROOT / "p106_checkout_6843362"
RELEASE = ROOT / "release_checkout_20261112"
COMMIT = "97f1b08703c90196fcd6f89e9e9988ba7aeb979f"
PUBLIC_BASE = f"https://raw.githubusercontent.com/Agnuxo1/solar-filament-segmentation-2026/{COMMIT}"

EXPECTED = {
    "notebook": "317b1781101fd6652b21b3f86d297b27136bd5c61ef7addfc8a904a7776314ad",
    "r2_pdf": "8f75a47996fa7e6e5f37f5d8e85a6cf5b64033701507782a54a81d786e5ed9f5",
    "r2_zip": "6bee5b287bf6920ae8ef3bab82b7518a6366140da92de19b98e6b972b128955b",
    "h9b": "24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60",
    "p103": "8ae3160af9bddd5ac3ef914bc807da71e3e9cbad0444cc055d2ff0056cdfe2ba",
    "p106": "16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3",
    "p117_receipt": "cac6f6c26255bd80659d0629981b10b16ac4b67eb2ad79cef3e25ddba302827d",
}
PUBLIC = {
    "notebook": "notebooks/h9b_clean_raw_jpeg_to_csv.ipynb",
    "r2_pdf": "docs/submission/TECHNICAL_REPORT_4P_20261113_R2.pdf",
    "r2_zip": "docs/submission/TECHNICAL_REPORT_4P_20261113_R2_SOURCE.zip",
    "p117_receipt": "docs/submission/p117_evidence_20261114/P117_FINAL_RECEIPT.json",
}
LOCAL = {
    "notebook": CHECKOUT / PUBLIC["notebook"],
    "r2_pdf": RELEASE / PUBLIC["r2_pdf"],
    "r2_zip": RELEASE / PUBLIC["r2_zip"],
    "h9b": ROOT / "p74_accepted_artifact_audit/accepted_h9b_nosite.csv",
    "p103": ROOT / "p103_full_public_stage_coupling_20261113/p103_180_reconstructed.csv",
    "p106": P106 / "inference_cuda/submission_public_rebuilt.csv",
    "p117_receipt": ROOT / "p117_public_full_hash_20261114/P117_FINAL_RECEIPT.json",
}

def sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
            length += len(chunk)
    return digest.hexdigest(), length

def download_public(name: str, relpath: str, tmp: Path) -> Path:
    target = tmp / name
    request = urllib.request.Request(
        f"{PUBLIC_BASE}/{relpath}",
        headers={"User-Agent": "P118-solar-filament-reproducibility/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response, target.open("wb") as output:
        if response.status != 200:
            raise RuntimeError(f"{name}: unexpected HTTP status {response.status}")
        while chunk := response.read(1 << 20):
            output.write(chunk)
    return target

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    checks: dict[str, object] = {}
    errors: list[str] = []

    def check(label: str, condition: bool, detail: object) -> None:
        checks[label] = {"pass": bool(condition), "detail": detail}
        if not condition:
            errors.append(label)

    for name, path in LOCAL.items():
        try:
            actual, size = sha_file(path)
            check(f"local_{name}", actual == EXPECTED[name], {"sha256": actual, "bytes": size})
        except Exception as exc:
            check(f"local_{name}", False, f"{type(exc).__name__}: {exc}")

    with tempfile.TemporaryDirectory(prefix="p118_public_", dir=OUT) as folder:
        tmp = Path(folder)
        public_paths: dict[str, Path] = {}
        for name, relpath in PUBLIC.items():
            try:
                path = download_public(name, relpath, tmp)
                public_paths[name] = path
                actual, size = sha_file(path)
                check(f"public_{name}", actual == EXPECTED[name], {"sha256": actual, "bytes": size, "commit": COMMIT})
            except Exception as exc:
                check(f"public_{name}", False, f"{type(exc).__name__}: {exc}")

        if "r2_zip" in public_paths:
            try:
                with zipfile.ZipFile(public_paths["r2_zip"]) as archive:
                    corrupted_entry = archive.testzip()
                    check("source_zip_crc", corrupted_entry is None,
                          {"entries": len(archive.namelist()), "corrupted_entry": corrupted_entry})
            except Exception as exc:
                check("source_zip_crc", False, f"{type(exc).__name__}: {exc}")

        if "notebook" in public_paths:
            try:
                nb = json.loads(public_paths["notebook"].read_text(encoding="utf-8"))
                source_by_index = {
                    i: "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                    for i, cell in enumerate(nb["cells"]) if cell.get("cell_type") == "code"
                }
                receipt = json.loads((P106 / "P106_FINAL_RECEIPT.json").read_text(encoding="utf-8"))
                comparisons = []
                for cell in receipt["code_cells"]:
                    index = cell["cell_index"]
                    actual = hashlib.sha256(source_by_index[index].encode("utf-8")).hexdigest()
                    comparisons.append({"index": index, "matches": actual == cell["source_sha256"]})
                check("p106_cell_source_parity", len(comparisons) == 8 and all(x["matches"] for x in comparisons), comparisons)
                check("p106_receipt_metadata", receipt["status"] == "P106_ALL_EIGHT_NOTEBOOK_CELLS_COMPLETED"
                      and not receipt["test_annotations_used"] and not receipt["kaggle_submission_made"]
                      and receipt["notebook_sha256"] == EXPECTED["notebook"]
                      and receipt["csv_sha256"] == EXPECTED["p106"],
                      {"completed_code_cells": len(receipt["code_cells"]),
                       "p106_completed_utc": receipt["completed_utc"],
                       "new_official_pq": receipt["new_official_pq"]})
            except Exception as exc:
                check("p106_cell_source_parity", False, f"{type(exc).__name__}: {exc}")

        try:
            url = f"{PUBLIC_BASE}/docs/submission/REPORT_REVISION_MANIFEST_20261113_R2.json"
            with urllib.request.urlopen(url, timeout=30) as response:
                manifest = json.load(response)
            names = manifest["files"]
            check("r2_manifest_consistency",
                  manifest["revision"] == "R2" and manifest["page_count"] == 4
                  and names["TECHNICAL_REPORT_4P_20261113_R2.pdf"]["sha256"] == EXPECTED["r2_pdf"]
                  and names["TECHNICAL_REPORT_4P_20261113_R2_SOURCE.zip"]["sha256"] == EXPECTED["r2_zip"],
                  {"revision": manifest["revision"], "page_count_reported": manifest["page_count"]})
            url = f"{PUBLIC_BASE}/docs/submission/FINAL_SUBMISSION_RECEIPT_20261113.md"
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8")
            check("organizer_public_receipt_consistency",
                  EXPECTED["h9b"] in text and EXPECTED["r2_pdf"] in text
                  and "58112557" in text and "14:11:57" in text,
                  {"h9b_hash_documented": EXPECTED["h9b"] in text,
                   "r2_hash_documented": EXPECTED["r2_pdf"] in text,
                   "acknowledgment_documented": "14:11:57" in text})
        except Exception as exc:
            check("manifest_or_receipt_fetch", False, f"{type(exc).__name__}: {exc}")

    final = {
        "experiment": "P118",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": COMMIT,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "scope": "CPU-only public release parity and provenance; no GT or PQ",
        "status": "PASS" if not errors else "FAIL",
        "checks_total": len(checks),
        "checks_passed": sum(x["pass"] for x in checks.values()),
        "errors": errors,
        "checks": checks,
        "no_gpu_no_training_no_submissions": True,
        "final_selection_unchanged": "H9b Kaggle submission 58112557; no authenticated Kaggle read in P118",
    }
    target = OUT / "P118_FINAL_RECEIPT.json"
    target.write_text(json.dumps(final, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in final.items() if k != "checks"}, indent=2))
    print("RECEIPT_SHA256", sha_file(target)[0])
    return 0 if not errors else 2

if __name__ == "__main__":
    raise SystemExit(main())
