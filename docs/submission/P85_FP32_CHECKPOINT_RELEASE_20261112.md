# P85 — Byte-exact FP32 Mask R-CNN release and reproduction gate

**Date:** 12 November 2026 (UTC)
**Purpose:** Remove a proven source of numerical divergence in the public H9b reconstruction without introducing any new training, any new ground-truth access or a Kaggle submission.

## Documented scientific rationale

The previously released FP16 Mask R-CNN weights are useful for efficient inference, but are not byte-identical to the original FP32 checkpoints used by the accepted H9b workflow. The P82 full-domain GPU reconstruction consequently had only 3 exact proposal-mask pairs across the four 180-image sets, whereas the original CPU FP32 reconstruction P69 achieved 180/180 byte-identical two-view images in each of F1-F4.

**Hypothesis:** Publishing the original, hash-verified FP32 detector weights in safely reconstructable shards will remove the checkpoint-quantization difference and enable an independent CPU exact-proposal replay. This concerns source reproducibility, **not** improved PQ or a stronger model.

**Primary acceptance gate:** All eight published shards pass their individual SHA-256 checks; the four assembled model hashes equal the original P69 hashes and all files are under the GitHub 100 MB single-blob cap. The reconstruction script rejects tampering, path traversal, and unexpected existing outputs in offline tests. If any gate fails, withdraw the publication and do not claim exact source restoration.

## Original unmodified model hashes

| Fold | Original FP32 SHA-256 |
|---|---|
| F1 | `399b4bd790c5acf88f32da926dd00a9ac4b704040241b8b0899117c687756820` |
| F2 | `9a50cc6d23d1689f8d9a69ac90d25c086d128d98682c77669366ce59caca1115` |
| F3 | `e7f9ed980842c2a04ee5ff21fe3018d00eef0ed742aace9f9bb0e08af77f8952` |
| F4 | `59e942afc969318681c7de5c748ed58bd29b79448243389e34db93a8fe54827f` |

Each restored source is 183,895,165 bytes, split as 95,000,000 + 88,895,165 bytes. `release/models/fp32_shards/manifest.json` binds the exact parts to the original hashes. These files are not novel checkpoints, extra GPU training or manually altered models.

## Reproduce safely

From a clone of this public repository, with Python 3.10 or later:

```bash
python scripts/reassemble_maskrcnn_fp32.py --check-only
python scripts/reassemble_maskrcnn_fp32.py --output-dir /path/to/large/HDD/weights
python -m unittest discover -s scripts -p test_reassemble_maskrcnn_fp32.py -v
```

The first command only verifies the original shards, without constructing new files. The second creates four original FP32 checkpoints under a user-selected output folder using atomic writes. Existing correct checkpoint files are verified without rebuilding; existing altered checkpoint files are rejected. **Use a large HDD, never a nearly full system SSD.** The tests are synthetic, use only the standard library, and do not load challenge annotations.

## Independent evidence and limitations

P69 restored four sets of 180 original detector two-view predictions **bit-for-bit** on CPU from the original checkpoints; that prior verification is documented in the private research archive and is not rerun merely to obtain a new result. Publishing the source weights makes an equivalent public-source restoration possible. P82's approximate detector replay from FP16 remains valid as a separate, explicitly non-identical benchmark.

The full final H9b pipeline includes four U-Nets, trained selection logic, two crop refiners, and panoptic instance encoding. Detector checkpoint parity **alone does not** establish raw-image-to-final-CSV equality. The accepted CSV SHA-256 remains `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60` (Kaggle submission **58112557**, public PQ **0.39**). No new Kaggle prediction was submitted during this weight-release task.

The repository's MIT licence covers the original repository code. Third-party dependencies and any pretrained upstream components retain their respective licences; consult those licences before reusing the models.

## Remaining final-evaluation gates

The four-page English PDF is available in `docs/submission/`. A truly self-contained end-to-end notebook reproducing the accepted CSV from original image pixels is still **not verified**. The competition's mandatory Google Form submission is also **unconfirmed**. Those outstanding requirements take priority over further speculative experiments.

**Resource budget:** CPU verification and a one-fold 183.9 MB assembly smoke test; zero CUDA jobs, zero Kaggle submissions, eight binary shards totalling about 736 MB on the research HDD plus Git objects. GPU1 must remain untouched. No private test masks or Kaggle annotations are included.
