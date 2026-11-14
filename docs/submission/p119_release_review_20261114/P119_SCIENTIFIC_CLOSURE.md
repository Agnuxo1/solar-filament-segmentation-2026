# P119 - Independent public-release CPU contract verification

Public revision: `973f17da9ddd1a227a46047443d74995102956fa`.

**Scope and result: PASS.** P119 independently checked out the latest public GitHub tree, passed **23/23 CPU regression tests**, compiled all **8 notebook code cells**, and verified full SHA-256 hashes of **10 checkpoints** and **8 FP32 shards**. It checked 15 published source files. The notebook executable source digest was unchanged: `317b1781101fd6652b21b3f86d297b27136bd5c61ef7addfc8a904a7776314ad`.

The [preregistered protocol](P119_PROTOCOL.md) was hashed before running these tests: `aef10b83567ffb58e6e7b296a1d2b8fb78b632196d64c7c230da8f5cb5a1bba1`. Direct logs are retained in [unit tests](P119_UNIT_TESTS.log) and [preflight](P119_PREFLIGHT.log), and the [receipt](P119_FINAL_RECEIPT.json) contains exact machine-readable checks.

## Precisely what was reproduced

P119 used a fresh independent checkout from public GitHub to execute the source contract tests and the full-hash model inventory audit. It also recalculated local SHA-256 for the authentic H9b CSV and previously generated P106 CSV, which matched their frozen reference digests.

P119 **did not rerun 180-image inference or train a model**. Those are covered separately by P106: eight cells from 180 original JPEG images and released weights, 1,241 predicted masks. The P106 result is not byte-identical to the accepted H9b file. It has no official PQ score and must not inherit the accepted H9b **0.39 public PQ** (submission **58112557**). No new PQ, hidden labels, Kaggle upload, or form update resulted from P119.

Scientific dependencies reused the verified P106 environment; they were not installed independently on unrelated hardware. The P108 out-of-fold provenance gate continues to block the proposed TRAIN extra-context 1.15x ablation. The existence of test logs does not establish a Claude-to-Codex executor handoff.

## Reproduction

```bash
python -m unittest discover -s tests -p test_public_replay_runtime.py -v
python scripts/preflight_h9b_clean_notebook.py --root . --full-hash
```

**Status:** this independent CPU release gate is VERIFIED; the existing H9b entry is SUBMITTED_WITH_RECEIPT; a complete new agent-based end-to-end replay is not claimed.
