#!/usr/bin/env python3
"""P88: Resume-safe public-source FP32 CPU detector parity on all 720 image/fold pairs.

Uses original unlabelled Kaggle test pixels and original released checkpoints.
Compares the CPU outputs with hash-verified, immutable historical P69 proposals.
No CUDA, Kaggle submission, training, calibration or competition labels.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import pathlib
import pickle
import shutil
import time

from replay_public_detector_p81 import sha256, unreflect, verify_proposals

EXPECTED = {
    "F1": "399b4bd790c5acf88f32da926dd00a9ac4b704040241b8b0899117c687756820",
    "F2": "9a50cc6d23d1689f8d9a69ac90d25c086d128d98682c77669366ce59caca1115",
    "F3": "e7f9ed980842c2a04ee5ff21fe3018d00eef0ed742aace9f9bb0e08af77f8952",
    "F4": "59e942afc969318681c7de5c748ed58bd29b79448243389e34db93a8fe54827f",
}


class OneImage:
    """Expose one unlabelled grayscale pixel array to the detector API."""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.value


def exact_views(candidate, reference):
    """Count exact ordered proposal tuples, including every float and crop byte."""
    import numpy as np

    if len(candidate) != len(reference):
        return False, 0
    verified = 0
    for newer, older in zip(candidate, reference):
        if (len(newer) != 6 or len(older) != 6 or newer[:5] != older[:5]
                or not np.array_equal(newer[5], older[5])):
            return False, verified
        verified += 1
    return True, verified


def self_test():
    """Prove that counts, scores and mask bytes are all part of the acceptance gate."""
    import numpy as np

    a = (0.4, 1, 3, 2, 4, np.ones((2, 2), dtype=np.uint8))
    b = (0.41, 1, 3, 2, 4, np.ones((2, 2), dtype=np.uint8))
    assert exact_views([a], [a]) == (True, 1)
    assert exact_views([a], [a, a])[0] is False
    assert exact_views([a], [b])[0] is False
    assert exact_views([a], [(0.4, 1, 3, 2, 4, np.zeros((2, 2), dtype=np.uint8))])[0] is False


def publish(path, value):
    """Atomically create a manifest without ever overwriting prior evidence."""
    if path.exists():
        raise FileExistsError(f"Immutable artifact exists: {path}")
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as writer:
        json.dump(value, writer, sort_keys=True, indent=2)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temp, path)


def event(root, kind, **fields):
    """Append a physically timestamped recovery and audit event to the HDD."""
    payload = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "kind": kind, **fields}
    with (root / "events.jsonl").open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(payload, sort_keys=True) + "\n")
        writer.flush()


def verify_reference(fold, reference_root, images):
    """Validate trusted P69 manifest and each source pickle before new inference."""
    baseline = reference_root / fold
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "CPU_BYTE_EXACT_PASS"
            or manifest.get("images") != 180
            or manifest.get("original_weights_sha256") != EXPECTED[fold]):
        raise ValueError(f"P69 reference manifest is not approved: {fold}")
    registered = manifest.get("image_outputs", {})
    if set(registered) != {p.name for p in images}:
        raise ValueError(f"P69 image manifest differs from frozen inventory: {fold}")
    for image in images:
        name = image.stem + ".pkl"
        if sha256(baseline / name) != registered[image.name]["sha256"]:
            raise ValueError(f"P69 reference hash changed: {fold}/{name}")
    return manifest


def infer_one(model, image, size, device, detector, read_gray):
    """Compute both geometric views without converting pixels to labels."""
    import numpy as np
    import torch

    gray = read_gray(image, size)
    if gray.shape != (size, size):
        raise ValueError(f"Unexpected test pixel geometry: {image.name}")
    with torch.inference_mode():
        first = detector.predict_files(
            model, OneImage(gray), {image.name: 0}, [image.name], device
        )[image.name]
        flipped = np.ascontiguousarray(gray[:, ::-1])
        second = detector.predict_files(
            model, OneImage(flipped), {image.name: 0}, [image.name], device
        )[image.name]
    views = [first, [unreflect(item, size) for item in second]]
    verify_proposals(views, size)
    return views


def run(args):
    """Verify each completed file before resuming; preserve failures untouched."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("P88 must be CPU-only: CUDA_VISIBLE_DEVICES must be empty.")
    if shutil.disk_usage(args.out.parent).free < 20 * (1024 ** 3):
        raise RuntimeError("Insufficient HDD free-space reserve.")
    images = sorted(args.images.glob("*.jpeg"))
    if len(images) != 180 or len({p.name for p in images}) != 180:
        raise ValueError("Expected precisely 180 distinct original test JPEGs.")
    for fold, expected_hash in EXPECTED.items():
        if sha256(args.weights / f"maskrcnn_{fold}_fp32.pt") != expected_hash:
            raise ValueError(f"FP32 release hash mismatch: {fold}")
        verify_reference(fold, args.reference, images)

    import numpy as np
    import torch
    import filseg.train_maskrcnn as detector
    from filseg.data import RES, read_gray

    if torch.cuda.is_available():
        raise RuntimeError("CUDA visibility was not isolated.")
    detector.autocast = lambda device: contextlib.nullcontext()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    device = torch.device("cpu")

    args.out.mkdir(parents=True, exist_ok=True)
    completed_receipt = args.out / "receipt.json"
    if completed_receipt.exists():
        record = json.loads(completed_receipt.read_text(encoding="utf-8"))
        if record.get("status") != "P88_PUBLIC_SOURCE_CPU_720_EXACT_PASS":
            raise RuntimeError("An unexpected immutable receipt already exists.")
        print("P88_ALREADY_COMPLETE", flush=True)
        return

    start = time.monotonic()
    reports = {}
    event(args.out, "resume_or_start", code_sha256=sha256(pathlib.Path(__file__)),
          image_count=180, cpu_only=True)
    for fold in EXPECTED:
        target = args.out / fold
        target.mkdir(exist_ok=True)
        manifest_path = target / "manifest.json"
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved.get("status") != "PUBLIC_SOURCE_CPU_EXACT_PASS" or saved.get("images") != 180:
                raise RuntimeError(f"Invalid completed manifest for {fold}")
            for image in images:
                stored = target / f"{image.stem}.pkl"
                if sha256(stored) != saved["entries"][image.name]["sha256"]:
                    raise RuntimeError(f"Previously completed output changed: {fold}/{image.name}")
            reports[fold] = saved
            event(args.out, "fold_already_verified", fold=fold)
            continue

        model = detector.build_model(size=RES, pretrained=False)
        model.load_state_dict(
            torch.load(args.weights / f"maskrcnn_{fold}_fp32.pt",
                       map_location="cpu", weights_only=True), strict=True
        )
        model.eval().to(device)
        entries = {}
        exact_images = exact_views_count = exact_instances = 0
        for idx, image in enumerate(images, 1):
            if time.monotonic() - start > 42000:
                raise TimeoutError("P88 twelve-hour budget reached: recoverable files preserved.")
            file = target / f"{image.stem}.pkl"
            temporary = target / f"{image.stem}.pkl.{os.getpid()}.tmp"
            if temporary.exists():
                raise RuntimeError(f"Unreconciled interrupted temporary file: {temporary}")
            if file.exists():
                current = pickle.loads(file.read_bytes())
            else:
                current = infer_one(model, image, RES, device, detector, read_gray)
                with temporary.open("xb") as stream:
                    pickle.dump(current, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, file)

            baseline = pickle.loads(
                (args.reference / fold / f"{image.stem}.pkl").read_bytes()
            )
            if len(current) != 2 or len(baseline) != 2:
                raise RuntimeError("Unexpected number of view arrays")
            flags = []
            for view in (0, 1):
                identical, count = exact_views(current[view], baseline[view])
                exact_instances += count
                exact_views_count += int(identical)
                flags.append(identical)
            is_exact = all(flags)
            if not is_exact:
                event(args.out, "exact_parity_failure", fold=fold, image=image.name)
                raise RuntimeError(f"Public CPU source differs from historical reference: {fold}/{image.name}")
            exact_images += 1
            entries[image.name] = {
                "input_sha256": sha256(image),
                "sha256": sha256(file),
                "both_views_bit_exact": True,
                "counts": [len(current[0]), len(current[1])]
            }
            if idx == 1 or idx % 20 == 0 or idx == 180:
                event(args.out, "progress", fold=fold, done=idx, exact=exact_images)
                print("P88_CPU_PROGRESS", fold, idx, "/180", flush=True)

        payload = {
            "status": "PUBLIC_SOURCE_CPU_EXACT_PASS",
            "fold": fold, "images": 180, "bit_exact_images": exact_images,
            "bit_exact_views": exact_views_count,
            "exact_instances": exact_instances,
            "checkpoint_sha256": EXPECTED[fold],
            "entries": entries, "source_sha256": sha256(pathlib.Path(__file__))
        }
        publish(manifest_path, payload)
        reports[fold] = payload
        event(args.out, "fold_complete", fold=fold, images=exact_images)
        del model

    final = {
        "status": "P88_PUBLIC_SOURCE_CPU_720_EXACT_PASS",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "source_sha256": sha256(pathlib.Path(__file__)),
        "cpu_only": True, "no_test_labels_used": True,
        "no_new_kaggle_submission": True,
        "folds": {
            fold: {"images": report["images"],
                   "exact_images": report["bit_exact_images"],
                   "exact_views": report["bit_exact_views"],
                   "checkpoint_sha256": report["checkpoint_sha256"]}
            for fold, report in reports.items()
        }
    }
    publish(completed_receipt, final)
    event(args.out, "complete", elapsed=final["elapsed_seconds"])
    print("P88_PUBLIC_SOURCE_CPU_COMPLETE", json.dumps(final), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=pathlib.Path, required=True)
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("P88_SELFTEST_OK", flush=True)
        return
    run(args)


if __name__ == "__main__":
    main()
