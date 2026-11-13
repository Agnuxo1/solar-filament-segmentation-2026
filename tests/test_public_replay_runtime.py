"""CPU-only regression tests for public replay portability and resume integrity."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import public_replay_runtime as runtime
import replay_maskrcnn_from_original_images as replay

GPU_A = "GPU-00000000-1111-2222-3333-444444444444"
GPU_B = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_never_queries_nvidia(self):
        with patch.object(runtime, "_nvidia_output", side_effect=AssertionError("Hardware touched")):
            self.assertEqual(runtime.select_device("cpu", environ={}), ("cpu", None))

    def test_single_independent_gpu_is_accepted(self):
        with patch.object(runtime, "_nvidia_output", side_effect=[f"0, {GPU_A}\n", ""]):
            self.assertEqual(runtime.select_device("cuda", environ={}), ("cuda", GPU_A))

    def test_multiple_gpu_selection_requires_explicit_choice(self):
        with patch.object(runtime, "_nvidia_output", return_value=f"0, {GPU_A}\n1, {GPU_B}\n"):
            with self.assertRaisesRegex(RuntimeError, "Multiple GPUs"):
                runtime.select_device("cuda", environ={})

    def test_existing_single_visible_index_is_respected(self):
        with patch.object(runtime, "_nvidia_output", side_effect=[f"0, {GPU_A}\n1, {GPU_B}\n", ""]):
            self.assertEqual(runtime.select_device("cuda", environ={"CUDA_VISIBLE_DEVICES": "1"}), ("cuda", GPU_B))

    def test_explicit_gpu_cannot_escape_visibility(self):
        with patch.object(runtime, "_nvidia_output", return_value=f"0, {GPU_A}\n1, {GPU_B}\n"):
            with self.assertRaisesRegex(RuntimeError, "visibility restriction"):
                runtime.select_device("cuda", GPU_B, {"CUDA_VISIBLE_DEVICES": "0"})

    def test_existing_compute_job_is_preserved(self):
        with patch.object(runtime, "_nvidia_output", side_effect=[f"0, {GPU_A}\n", f"{GPU_A}, 1234\n"]):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                runtime.select_device("cuda", environ={})

    def test_disabled_visibility_does_not_enable_gpu(self):
        with patch.object(runtime, "_nvidia_output", return_value=f"0, {GPU_A}\n"):
            with self.assertRaisesRegex(RuntimeError, "disables CUDA"):
                runtime.select_device("cuda", environ={"CUDA_VISIBLE_DEVICES": ""})

    def test_cpu_child_env_does_not_mutate_parent(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU_A}):
            result = runtime.process_environment("cpu", None, ROOT)
            self.assertEqual(result["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], GPU_A)

    def test_malformed_device_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            runtime.select_device("auto")
        with self.assertRaises(ValueError):
            runtime.select_device("cpu", GPU_A)
        with self.assertRaises(ValueError):
            runtime.process_environment("cuda", None, ROOT)


class ResumeIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_root_discovery_from_notebook_directory(self):
        manifest = self.root / "release/model_metadata/WEIGHTS_SHA256SUMS.txt"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("")
        nested = self.root / "notebooks"
        nested.mkdir()
        self.assertEqual(runtime.repository_root(nested), self.root)

    def test_jupyter_autosave_does_not_change_source_fingerprint(self):
        path = self.root / "run.ipynb"
        notebook = {
            "cells": [{"cell_type": "code", "source": "x = 1\nprint(x)\n",
                       "execution_count": None, "outputs": [], "metadata": {}}],
            "metadata": {}, "nbformat": 4,
        }
        path.write_text(json.dumps(notebook), encoding="utf-8")
        original = runtime.notebook_source_sha256(path)
        notebook["metadata"] = {"kernelspec": {"name": "different-kernel"}}
        code = notebook["cells"][0]
        code["source"] = ["x = 1\r\n", "print(x)\r\n"]
        code["execution_count"] = 4
        code["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "1\n"}]
        code["metadata"] = {"execution": {"iopub.status.busy": "2026-11-13"}}
        notebook["cells"].insert(0, {"cell_type": "markdown", "source": "Updated explanation"})
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        self.assertEqual(runtime.notebook_source_sha256(path), original)

    def test_executable_edit_changes_source_fingerprint(self):
        path = self.root / "run.ipynb"
        notebook = {"cells": [{"cell_type": "code", "source": "threshold = 0.4\n"}]}
        path.write_text(json.dumps(notebook), encoding="utf-8")
        original = runtime.notebook_source_sha256(path)
        notebook["cells"][0]["source"] = "threshold = 0.5\n"
        path.write_text(json.dumps(notebook), encoding="utf-8")
        self.assertNotEqual(runtime.notebook_source_sha256(path), original)

    def test_immutable_receipt_cannot_be_overwritten(self):
        path = self.root / "receipt.json"
        runtime.publish_json(path, {"version": 1})
        with self.assertRaises(FileExistsError):
            runtime.publish_json(path, {"version": 2})
        self.assertEqual(json.loads(path.read_text()), {"version": 1})

    def test_map_copy_without_symlink_privileges(self):
        source = self.root / "source.png"
        source.write_bytes(b"synthetic map bytes")
        destination = self.root / "overlay" / "map.png"
        with patch.object(runtime.os, "link", side_effect=OSError("links unavailable")):
            runtime.materialize_map(source, destination)
        self.assertEqual(source.read_bytes(), destination.read_bytes())
        destination.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "differs"):
            runtime.materialize_map(source, destination)

    def test_modified_detector_output_is_rejected_before_pickle_load(self):
        prediction = self.root / "image.pkl"
        prediction.write_bytes(b"trusted synthetic output")
        receipt = self.root / "image.receipt.json"
        payload = {"input_sha256": "input", "checkpoint_sha256": "weights",
                   "contract_sha256": "contract", "sha256": runtime.sha256_file(prediction)}
        runtime.publish_json(receipt, payload)
        self.assertEqual(replay.validate_saved_prediction(prediction, receipt, "input", "weights", "contract"), payload)
        prediction.write_bytes(b"corrupted")
        with self.assertRaisesRegex(RuntimeError, "changed inputs or bytes"):
            replay.validate_saved_prediction(prediction, receipt, "input", "weights", "contract")

    def test_different_backend_contract_cannot_reuse_prediction(self):
        prediction = self.root / "image.pkl"
        prediction.write_bytes(b"output")
        receipt = self.root / "image.receipt.json"
        runtime.publish_json(receipt, {"input_sha256": "input", "checkpoint_sha256": "weights",
                             "contract_sha256": "cpu", "sha256": runtime.sha256_file(prediction)})
        with self.assertRaises(RuntimeError):
            replay.validate_saved_prediction(prediction, receipt, "input", "weights", "cuda")

    def test_h3_reuse_validates_map_and_input_content(self):
        image = self.root / "test.jpeg"
        image.write_bytes(b"original pixels")
        target = self.root / "maps"
        target.mkdir()
        output = target / "test.png"
        output.write_bytes(b"prediction")
        receipt = self.root / "h3.json"
        runtime.publish_json(receipt, {
            "images_processed": 1, "full_180_images": True, "weight_sha256": "weight", "device": "cpu",
            "image_sha256": [{"file": image.name, "input_sha256": runtime.sha256_file(image),
                              "map_sha256": runtime.sha256_file(output)}],
        })
        runtime.verify_h3_receipt(receipt, target, [image], "weight", "cpu")
        output.write_bytes(b"altered prediction")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            runtime.verify_h3_receipt(receipt, target, [image], "weight", "cpu")


class PublicNotebookStageTests(unittest.TestCase):
    def setUp(self):
        self.nb = json.loads((ROOT / "notebooks/h9b_clean_raw_jpeg_to_csv.ipynb").read_text())

    def test_all_cells_compile_without_machine_uuid(self):
        code = [cell["source"] for cell in self.nb["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code), 8)
        for index, source in enumerate(code):
            compile(source, f"public-cell-{index}", "exec")
            self.assertNotIn("GPU-b2046dc8", source)
            self.assertNotIn("/srv/almacen", source)

    def test_h3_stage_cpu_does_not_require_gpu_or_uuid(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            image = folder / "test.jpeg"
            image.write_bytes(b"synthetic input")
            commands = []

            class FakeProcess:
                @staticmethod
                def run(command, **kwargs):
                    commands.append((command, kwargs))
                    fold = command[command.index("--fold") + 1]
                    target = folder / "h3" / fold
                    target.mkdir(parents=True)
                    output = target / "test.png"
                    output.write_bytes(b"synthetic map")
                    runtime.publish_json(folder / "h3" / f"{fold}_h3_replay_receipt.json", {
                        "images_processed": 1, "full_180_images": True, "device": "cpu", "weight_sha256": "weight",
                        "image_sha256": [{"file": image.name, "input_sha256": runtime.sha256_file(image),
                                          "map_sha256": runtime.sha256_file(output)}],
                    })

            variables = {
                "process_environment": runtime.process_environment, "DEVICE": "cpu", "GPU_UUID": None,
                "ROOT": ROOT, "SCRIPTS": ROOT / "scripts", "H3": folder / "h3", "IMAGES": folder,
                "FOLDS": ("F1",), "FILES": [image], "manifest": {"F1_resnet34_s0_fp16.pt": "weight"},
                "verify_h3_receipt": runtime.verify_h3_receipt, "sys": sys, "subprocess": FakeProcess,
                "check_gpu_idle": lambda _: self.fail("CPU stage attempted GPU access"),
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(self.nb["cells"][5]["source"], "h3-stage", "exec"), variables)
            self.assertEqual(len(commands), 1)
            command, options = commands[0]
            self.assertEqual(command[command.index("--device") + 1], "cpu")
            self.assertNotIn("--gpu-uuid", command)
            self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_detector_stage_uses_original_fp32_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            commands = []

            class FakeProcess:
                @staticmethod
                def run(command, **kwargs):
                    commands.append(command)
                    runtime.publish_json(folder / "receipt.json", {
                        "status": "PUBLIC_ORIGINAL_FP32_DETECTORS_COMPLETE_NOT_FINAL_CSV",
                        "folds": {fold: {} for fold in replay.FOLDS},
                    })

            variables = {
                "process_environment": runtime.process_environment, "DEVICE": "cpu", "GPU_UUID": None,
                "ROOT": ROOT, "SCRIPTS": ROOT / "scripts", "IMAGES": folder, "FP32_MODELS": folder / "weights",
                "DETECTORS": folder, "FOLDS": replay.FOLDS, "sys": sys, "json": json, "subprocess": FakeProcess,
                "check_gpu_idle": lambda _: self.fail("CPU stage attempted GPU access"),
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(self.nb["cells"][7]["source"], "detector-stage", "exec"), variables)
            self.assertEqual(Path(commands[0][2]).name, "replay_maskrcnn_from_original_images.py")
            self.assertIn("--weights", commands[0])
            self.assertNotIn("--gpu-uuid", commands[0])

    def test_cli_defaults_to_cpu_and_rejects_inconsistent_gpu_option(self):
        args = ["--release", "release", "--images", "test", "--out", "output"]
        parsed = replay.parse_args(args)
        self.assertEqual(parsed.device, "cpu")
        self.assertEqual(parsed.folds, replay.FOLDS)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            replay.parse_args(args + ["--gpu-uuid", GPU_A])

    def test_selector_metadata_change_invalidates_actual_notebook_contract(self):
        tree = ast.parse(self.nb["cells"][3]["source"])
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "run_contract"
                                  for target in node.targets))
        expression = compile(ast.Expression(assignment.value), "notebook-run-contract", "eval")
        with tempfile.TemporaryDirectory() as directory:
            selector_path = Path(directory) / "selector.pkl"
            selector_json = Path(directory) / "selector.json"
            selector_path.write_bytes(b"unchanged frozen selector")
            selector_json.write_text(json.dumps({"w": 0.7, "t": 0.35, "sites": ["B", "C"]}))
            variables = {
                "FILES": [], "manifest": {}, "fp32_hashes": {}, "source_paths": [],
                "ROOT": ROOT, "selector_path": selector_path, "selector_json": selector_json,
                "sha": runtime.sha256_file, "notebook_source_sha256": runtime.notebook_source_sha256,
                "DEVICE": "cpu", "GPU_UUID": None,
            }
            original = eval(expression, variables)
            selector_json.write_text(json.dumps({"w": 0.5, "t": 0.35, "sites": ["B", "C"]}))
            changed = eval(expression, variables)
            self.assertEqual(original["selector_sha256"], changed["selector_sha256"])
            self.assertNotEqual(original["selector_metadata_sha256"], changed["selector_metadata_sha256"])
            self.assertNotEqual(original, changed)

    def test_cli_help_needs_no_machine_learning_dependencies(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/replay_maskrcnn_from_original_images.py"), "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--device {cpu,cuda}", result.stdout)


if __name__ == "__main__":
    unittest.main()
