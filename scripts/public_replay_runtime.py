"""Portable execution and integrity helpers for the public inference notebook.

CPU is the explicit default. CUDA execution must identify one idle GPU on the
current machine; no author's machine UUID, home directory or remote service is
required. This module imports only the Python standard library.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha256_file(path: Path) -> str:
    """Return a file digest while keeping host-memory use bounded."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def notebook_source_sha256(path: Path) -> str:
    """Fingerprint executable cells, ignoring Jupyter autosave and display state.

    Notebook sources may be strings or lists of lines. Both serialisations and
    CRLF/LF line endings are canonicalised. Outputs, execution counts, Markdown
    and notebook/cell metadata do not affect the execution contract.
    """
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Notebook cells must be a list.")
    sources = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Each notebook cell must be an object.")
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError("Code-cell source must be a string or list of strings.")
        sources.append(source.replace("\r\n", "\n").replace("\r", "\n"))
    if not sources:
        raise ValueError("The notebook has no executable code cells.")
    canonical = json.dumps(
        {"schema": "NOTEBOOK_CODE_SOURCES_V1", "code_cells": sources},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def repository_root(start: Path) -> Path:
    """Find the checkout whether Jupyter starts in its root or notebooks/."""
    origin = Path(start).expanduser().resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "release/model_metadata/WEIGHTS_SHA256SUMS.txt").is_file():
            return candidate
    raise FileNotFoundError("Run this notebook from the public repository checkout.")


def prepare_output(path: Path, reserve_gib: float = 20) -> Path:
    """Create output on the requested filesystem and check that filesystem."""
    if reserve_gib < 0:
        raise ValueError("The free-space reserve must be non-negative.")
    destination = Path(path).expanduser().resolve()
    ancestor = destination
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise NotADirectoryError(ancestor)
    if shutil.disk_usage(ancestor).free < reserve_gib * 1024**3:
        raise OSError(f"At least {reserve_gib:g} GiB is required on {ancestor}.")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _nvidia_output(arguments: list[str]) -> str:
    """Read GPU metadata, without creating any CUDA context."""
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments], capture_output=True, text=True,
            check=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError("CUDA was requested but nvidia-smi is unavailable.") from exc
    return result.stdout


def check_gpu_idle(gpu_uuid: str) -> None:
    """Refuse CUDA when the chosen physical device already has compute work."""
    output = _nvidia_output([
        "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ])
    for line in output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if fields and fields[0] == gpu_uuid:
            raise RuntimeError(
                f"Selected GPU {gpu_uuid} is busy. Preserve its job; use CPU or another idle GPU."
            )


def select_device(requested: str = "cpu", gpu_uuid: str | None = None,
                  environ: dict[str, str] | None = None) -> tuple[str, str | None]:
    """Resolve a CPU/CUDA request against current hardware without using Torch.

    An explicit UUID or numeric physical index is accepted. Otherwise a single
    CUDA_VISIBLE_DEVICES entry is honoured, or the only physical GPU is used.
    Multiple candidates are never selected silently. An existing visibility
    restriction is respected when checking an explicit selection.
    """
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Device must be 'cpu' or 'cuda'.")
    if requested == "cpu":
        if gpu_uuid:
            raise ValueError("A GPU selection is incompatible with --device cpu.")
        return "cpu", None
    env = os.environ if environ is None else environ
    lines = _nvidia_output(["--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    inventory = {}
    for line in lines.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2 or not fields[1].startswith("GPU-"):
            raise RuntimeError("Cannot interpret the physical NVIDIA GPU inventory.")
        inventory[fields[0]] = fields[1]
    if not inventory:
        raise RuntimeError("CUDA was requested but no physical NVIDIA GPU was found.")

    def resolve(value: str) -> str:
        if value in inventory:
            return inventory[value]
        if value in inventory.values():
            return value
        raise RuntimeError(f"Selected CUDA device is absent from nvidia-smi: {value}")

    visible = env.get("CUDA_VISIBLE_DEVICES")
    allowed = None
    if visible is not None:
        values = [value.strip() for value in visible.split(",") if value.strip()]
        if not values or values == ["-1"]:
            raise RuntimeError("CUDA_VISIBLE_DEVICES disables CUDA; select CPU or configure visibility explicitly.")
        allowed = {resolve(value) for value in values}
    if gpu_uuid:
        chosen = resolve(gpu_uuid.strip())
        if allowed is not None and chosen not in allowed:
            raise RuntimeError("Requested GPU is outside the existing CUDA visibility restriction.")
    else:
        candidates = allowed if allowed is not None else set(inventory.values())
        if len(candidates) != 1:
            raise RuntimeError("Multiple GPUs are visible; set SOLARFIL_GPU_UUID or --gpu-uuid explicitly.")
        chosen = next(iter(candidates))
    check_gpu_idle(chosen)
    return "cuda", chosen


def process_environment(device: str, gpu_uuid: str | None, root: Path,
                        threads: int = 2) -> dict[str, str]:
    """Build an isolated child environment without changing the caller's one."""
    if threads < 1 or device not in {"cpu", "cuda"}:
        raise ValueError("Positive CPU threads and a supported device are required.")
    if (device == "cuda") != bool(gpu_uuid):
        raise ValueError("CUDA requires one UUID; CPU requires none.")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid if device == "cuda" else ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = str(threads)
    env["PYTHONPATH"] = os.pathsep.join([str(root / "release/source"), str(root)])
    return env


def publish_json(path: Path, payload: dict) -> None:
    """Write an immutable receipt atomically, preserving prior evidence."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_h3_receipt(receipt_path: Path, target: Path, images: list[Path],
                      weight_sha256: str, device: str) -> dict:
    """Check saved H3 image/map hashes, weights and backend before reuse."""
    evidence = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (evidence.get("images_processed") != len(images)
            or not evidence.get("full_180_images")
            or evidence.get("weight_sha256") != weight_sha256
            or evidence.get("device") != device):
        raise RuntimeError("Saved H3 stage has a different inventory, model or device.")
    entries = {entry["file"]: entry for entry in evidence.get("image_sha256", [])}
    if len(entries) != len(images) or set(entries) != {path.name for path in images}:
        raise RuntimeError("H3 receipt does not cover the exact image inventory.")
    if {path.stem for path in target.glob("*.png")} != {path.stem for path in images}:
        raise RuntimeError("H3 output inventory changed.")
    for path in images:
        entry = entries[path.name]
        if (entry.get("input_sha256") != sha256_file(path)
                or entry.get("map_sha256") != sha256_file(target / f"{path.stem}.png")):
            raise RuntimeError(f"H3 input or output changed: {path.name}")
    return evidence


def materialize_map(source: Path, destination: Path) -> None:
    """Create a checked hard link or copy, without requiring symlink privileges."""
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"Overlay map differs from source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    if sha256_file(destination) != sha256_file(source):
        raise RuntimeError(f"Overlay map copy failed integrity verification: {destination}")
