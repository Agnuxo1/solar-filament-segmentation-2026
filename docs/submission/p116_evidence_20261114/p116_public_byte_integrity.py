#!/usr/bin/env python3
"""P116: verify public release bytes independently, without GPU or test labels.

This is an artifact-availability and byte-integrity audit, NOT an accuracy
experiment. Public shard sampling does not establish whole-file integrity.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import pathlib
import re
import time
import urllib.request

REPO = "Agnuxo1/solar-filament-segmentation-2026"
COMMIT = "d1d88d6b16479af40dfcc4b734c193bcaa9a8203"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"
SHARD_DIRECTORY = pathlib.Path("release/models/fp32_shards")
SAMPLE_BYTES = 4096
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
NETWORK_TIMEOUT = 20
R2_SHA256 = "8f75a47996fa7e6e5f37f5d8e85a6cf5b64033701507782a54a81d786e5ed9f5"


def timestamp() -> str:
    """Return a timezone-aware audit timestamp in UTC."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def hash_stream(path: pathlib.Path, combined=None) -> str:
    """Hash a file in bounded chunks and optionally update a model hash."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            if combined is not None:
                combined.update(chunk)
    return digest.hexdigest()


def verify_local_shards(checkout: pathlib.Path, manifest: dict) -> list[dict]:
    """Verify every full local part and reconstructed checkpoint against manifest."""
    records = []
    for fold, spec in sorted(manifest["models"].items()):
        whole = hashlib.sha256()
        total_bytes = 0
        for item in spec["parts"]:
            local_file = checkout / SHARD_DIRECTORY / item["file"]
            actual_bytes = local_file.stat().st_size
            digest = hash_stream(local_file, combined=whole)
            total_bytes += actual_bytes
            passed = actual_bytes == item["bytes"] and digest == item["sha256"]
            records.append({
                "fold": fold, "name": item["file"], "size": actual_bytes,
                "sha256": digest, "expected_sha256": item["sha256"],
                "expected_size": item["bytes"], "pass": passed,
            })
        whole_pass = total_bytes == spec["original_bytes"] and whole.hexdigest() == spec["original_sha256"]
        records.append({
            "fold": fold, "type": "assembled_stream",
            "size": total_bytes, "sha256": whole.hexdigest(),
            "expected_sha256": spec["original_sha256"],
            "expected_size": spec["original_bytes"], "pass": whole_pass,
        })
    return records


def request_url(path: str, range_header: str | None = None):
    """Open a public HTTPS resource without credentials or authentication."""
    headers = {"User-Agent": "SolarFilament-P116-Public-Artifact-Audit"}
    if range_header is not None:
        headers["Range"] = range_header
    return urllib.request.urlopen(
        urllib.request.Request(f"{RAW}/{path}", headers=headers),
        timeout=NETWORK_TIMEOUT,
    )


def verify_public_range(checkout: pathlib.Path, item: dict, start: int) -> dict:
    """Compare one HTTP 206 byte range with the frozen local model part."""
    name = item["name"]
    total = item["size"]
    length = min(SAMPLE_BYTES, total)
    end = start + length - 1
    relpath = f"{SHARD_DIRECTORY.as_posix()}/{name}"
    with request_url(relpath, f"bytes={start}-{end}") as response:
        status = response.status
        content_range = response.headers.get("Content-Range", "")
        received = response.read(length + 1)
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
    actual_range = tuple(map(int, match.groups())) if match else None
    with (checkout / SHARD_DIRECTORY / name).open("rb") as handle:
        handle.seek(start)
        expected = handle.read(length)
    passed = (
        status == 206
        and actual_range == (start, end, total)
        and len(received) == length
        and received == expected
    )
    return {
        "shard": name, "range": f"{start}-{end}", "status": status,
        "content_range": content_range, "bytes": len(received),
        "local_slice_sha256": hashlib.sha256(expected).hexdigest(),
        "public_slice_sha256": hashlib.sha256(received).hexdigest(),
        "pass": passed,
    }


def verify_public_small_file(checkout: pathlib.Path, relpath: str, expected_digest: str | None) -> dict:
    """Download one whole small public file and verify its SHA-256."""
    with request_url(relpath) as response:
        status = response.status
        received = response.read(MAX_DOWNLOAD_BYTES + 1)
    actual_digest = hashlib.sha256(received).hexdigest()
    local_digest = (
        hash_stream(checkout / relpath) if expected_digest is None
        else expected_digest
    )
    passed = status == 200 and len(received) <= MAX_DOWNLOAD_BYTES and actual_digest == local_digest
    return {
        "resource": relpath, "status": status, "bytes": len(received),
        "local_or_recorded_sha256": local_digest,
        "public_sha256": actual_digest, "pass": passed,
    }


def run_audit(checkout: pathlib.Path, output: pathlib.Path) -> dict:
    """Run frozen P116 checks and retain failures without mutating model inputs."""
    start = time.monotonic()
    manifest_path = checkout / SHARD_DIRECTORY / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {
        "experiment": "P116", "started_utc": timestamp(),
        "public_ref": COMMIT, "project": REPO,
        "scope": "Local complete weights SHA-256, public short GET ranges and small full GET",
        "no_cuda_or_training": True, "no_test_labels": True,
        "no_kaggle_upload_or_selection_change": True, "official_PQ_P106": None,
        "local_part_and_model_checks": [], "public_range_checks": [],
        "public_small_file_checks": [], "errors": [],
    }
    try:
        result["local_part_and_model_checks"] = verify_local_shards(checkout, manifest)
        if not all(r["pass"] for r in result["local_part_and_model_checks"]):
            result["errors"].append("Full local shard integrity failed; no public probes launched")
        else:
            tasks = []
            for check in result["local_part_and_model_checks"]:
                if "name" in check:
                    tasks.append(("range", check, 0))
                    tasks.append(("range", check, check["size"] - min(SAMPLE_BYTES, check["size"])))
            small_resources = [
                ("requirements.txt", None),
                ("notebooks/h9b_clean_raw_jpeg_to_csv.ipynb", None),
                ("release/models/fp32_shards/manifest.json", None),
                ("docs/submission/TECHNICAL_REPORT_4P_20261113_R2.pdf", R2_SHA256),
            ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {}
                for kind, item, position in tasks:
                    futures[pool.submit(verify_public_range, checkout, item, position)] = ("range", item["name"])
                for path, digest in small_resources:
                    futures[pool.submit(verify_public_small_file, checkout, path, digest)] = ("full", path)
                for future in concurrent.futures.as_completed(futures):
                    kind, resource = futures[future]
                    try:
                        observation = future.result()
                        key = "public_range_checks" if kind == "range" else "public_small_file_checks"
                        result[key].append(observation)
                        print(f"{kind}: {resource}: {'PASS' if observation['pass'] else 'FAIL'}", flush=True)
                    except Exception as exc:
                        result["errors"].append(f"{kind} {resource}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        result["errors"].append(f"Local verification: {type(exc).__name__}: {exc}")
    result["public_range_checks"].sort(key=lambda record: (record["shard"], record["range"]))
    result["public_small_file_checks"].sort(key=lambda record: record["resource"])
    result["completed_utc"] = timestamp()
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    counts_match = (
        len(result["local_part_and_model_checks"]) == 12
        and len(result["public_range_checks"]) == 16
        and len(result["public_small_file_checks"]) == 4
    )
    all_valid = all(
        record["pass"] for field in
        ("local_part_and_model_checks", "public_range_checks", "public_small_file_checks")
        for record in result[field]
    )
    result["status"] = (
        "PASS_PUBLIC_SAMPLED_BYTES_AND_LOCAL_FULL_HASHES"
        if counts_match and all_valid and not result["errors"] and result["elapsed_seconds"] <= 180
        else "FAIL_OR_INCOMPLETE"
    )
    result["public_shard_full_hash_verified"] = False
    result["JEV_or_Goldfish_consulted"] = False
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "P116_FINAL_RECEIPT.json"
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "receipt": str(receipt), "status": result["status"],
        "local_checks": len(result["local_part_and_model_checks"]),
        "public_range_checks": len(result["public_range_checks"]),
        "full_public_file_checks": len(result["public_small_file_checks"]),
        "errors": result["errors"], "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    outcome = run_audit(arguments.checkout, arguments.output)
    return 0 if outcome["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
