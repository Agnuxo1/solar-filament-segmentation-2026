#!/usr/bin/env python3
"""Offline integrity and failure-mode tests for public FP32 checkpoint assembly."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

MODULE = pathlib.Path(__file__).resolve().parent / "reassemble_maskrcnn_fp32.py"
spec = importlib.util.spec_from_file_location("maskrcnn_shard_restore", MODULE)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ShardRestoreTests(unittest.TestCase):
    """Validate exact reconstruction and refusal of corrupted release data."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="solarfil-shard-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.shards = self.root / "release"
        self.shards.mkdir()
        self.output = self.root / "restored"
        self.data = {}
        self.manifest = {"format": "solar-filament-maskrcnn-shards-v1",
                         "shard_max_bytes": mod.MAX_PART_BYTES, "models": {}}
        for fold in mod.FOLDS:
            body = (fold + "-reference-weights").encode() * 31
            parts = []
            midpoint = len(body) // 2
            for i, chunk in enumerate((body[:midpoint], body[midpoint:])):
                name = f"maskrcnn_{fold}_fp32.pt.part{i:02d}"
                (self.shards / name).write_bytes(chunk)
                parts.append({"file": name, "bytes": len(chunk),
                              "sha256": hashlib.sha256(chunk).hexdigest()})
            self.manifest["models"][fold] = {
                "original_sha256": hashlib.sha256(body).hexdigest(),
                "original_bytes": len(body), "parts": parts
            }
            self.data[fold] = body
        self.manifest_path = self.shards / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_exact_bytes_and_idempotent_second_verification(self) -> None:
        models = mod.validate_manifest(self.manifest_path)
        mod.verify_parts(models, self.shards, mod.FOLDS)
        mod.restore_models(models, self.shards, self.output, mod.FOLDS)
        for fold in mod.FOLDS:
            self.assertEqual(
                (self.output / f"maskrcnn_{fold}_fp32.pt").read_bytes(),
                self.data[fold]
            )
        mod.restore_models(models, self.shards, self.output, mod.FOLDS)

    def test_rejects_wrong_shard_hash(self) -> None:
        models = mod.validate_manifest(self.manifest_path)
        (self.shards / "maskrcnn_F1_fp32.pt.part00").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            mod.verify_parts(models, self.shards, ("F1",))

    def test_rejects_path_traversal(self) -> None:
        self.manifest["models"]["F1"]["parts"][0]["file"] = "../../etc/passwd"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            mod.validate_manifest(self.manifest_path)

    def test_rejects_existing_mismatched_output(self) -> None:
        models = mod.validate_manifest(self.manifest_path)
        self.output.mkdir()
        (self.output / "maskrcnn_F1_fp32.pt").write_bytes(b"other")
        with self.assertRaises(FileExistsError):
            mod.restore_models(models, self.shards, self.output, ("F1",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
