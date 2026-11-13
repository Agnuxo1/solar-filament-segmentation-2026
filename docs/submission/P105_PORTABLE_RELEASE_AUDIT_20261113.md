# P105 — Portable public inference release

Author: Francisco Angulo de Lafuente  
Date: 13 November 2026  
Parent repository commit: `83835e101bc7fa75479e680fc3585be27b530fc5`

## Delivered behavior

The public raw-JPEG notebook now runs with CPU as its default, portable input/output paths, and optional explicitly selected CUDA. It reconstructs the four original FP32 Mask R-CNN checkpoints from the published eight shards. The earlier quantized FP16 detector invocation and mandatory author-machine UUID have been removed from this workflow. Historical audit programs remain intact.

The new detector CLI processes both geometric views of every image and fold, saves per-image receipts, and validates prior output before reusing it. Hard links or verified copies replace a requirement for symbolic-link privileges.

Run contracts now bind the executable notebook sources, model weights, selector binary **and metadata**, every relevant imported inference helper, image hashes, and numerical backend. Jupyter output, cell execution counters and metadata are excluded from the canonical source fingerprint so a normal autosave does not block resumption.

## Independent review and validation

Two separate code-review passes compared the new paths with the published model interfaces. The second review identified and resolved:

1. Raw notebook-file hashes incorrectly invalidated resumes after autosave.
2. Selector metadata was absent from the input contract.
3. The imported proposal-reflection helper was absent from the source contract and preflight.

The final CPU regression suite passed **23/23 tests**. The primary agent independently reran the same suite: **23/23 passed in 0.070 seconds**. All eight executable notebook cells compile. Tests cover device selection, occupied GPUs, existing visibility restrictions, integrity invalidation, portable detector arguments, Jupyter autosave, selector changes and map-copy behavior.

Command:

```bash
python -m unittest discover -s tests -p test_public_replay_runtime.py -v
```

Test fixtures are synthetic and subprocess/model stages are simulated where stated in the test. No full model inference was performed by this audit.

## Scientific boundary

The accepted historical artifact remains submission **58112557**, public PQ **0.39**, SHA-256 `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`.

This correction does **not** establish exact JPEG-to-accepted-CSV equivalence, a new PQ score or an accuracy gain. It preserves explicit numerical-backend differences and a public semantic fallback different from the defined historical control fallback. The available evidence does not establish that the historical fallback was actually used by the genuine accepted CSV.

The full 180-image replay, the pending P103 result and any subsequent model promotion need their own executed evidence. The current audit does not infer GPU utilization from prior records.

## Final report

The revised four-page report uses ACM `acmart/sigconf`, corrects the H3 architecture to four ResNet34 U-Nets, links to the public repository, and distinguishes P94 crop-to-CSV equality from full raw-image reconstruction. PDF and complete source are versioned alongside this audit.

The official organizer form and Kaggle's final-selection state have separate evidence. Neither code publication nor a PDF alone proves submission of that form.

