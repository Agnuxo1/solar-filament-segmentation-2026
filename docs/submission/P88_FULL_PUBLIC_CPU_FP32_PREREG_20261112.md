# P88 — Full 4-fold public-source FP32 CPU detector replay (preregistration)

**Preregistered before execution:** 12 November 2026 UTC. This is a continuation of the positive P87 pilot, not a new model-tuning study.

## Scientific hypothesis and rationale

The accepted H9b detector used original FP32 Mask R-CNN source/inference on CPU. Public-release checkpoint shards have now been verified and all four original FP32 state dictionaries reassembled without modification. P87 matched every ordered box, float score and crop-mask byte on the first five original unlabelled JPEGs **for all four folds** (20/20 image-fold pairs, 40/40 views, 1,104/1,104 instances). In contrast P86 running the same FP32 states on CUDA reached 0/20 fully exact images. The full raw-pixel-to-detector stage using the released public source still requires comprehensive verification.

**Primary hypothesis:** All **720 image/fold pairs** (4 models × 180 images), each with original and horizontally reflected view, will exactly match the immutable P69 CPU detector outputs in ordered proposal counts, float scores, coordinates and binary crop-mask bytes.

## Frozen protocol

- Only use the 180 original unlabelled official test JPEGs, unchanged released public `filseg` source, four SHA-256-verified FP32 checkpoints, the public unreflection logic and read-only P69 original CPU reference predictions with signed per-image manifest hashes.
- Run CPU-only (`CUDA_VISIBLE_DEVICES=`, two Torch CPU threads). Never initialize CUDA, never alter GPU processes, and run at low systemd process priority with a CPU cap to avoid interfering with independent competitions.
- For each image/fold, store a distinct recoverable pickle on the NAS HDD, with an immutable hash in the per-fold manifest and an append-only timestamped event log. On restart, verify every existing output against the reference and hash; skip rather than recompute.
- A single unexpected mismatch is a **failure**: preserve the positive and negative evidence, stop without skipping or silently repairing it. No score tuning, GT, labels, leaderboard, Kaggle submissions or test-driven selection.
- Validate that `len(images)==180`, filenames unique, all 4 checkpoint and reference hashes correct. Require a minimum of 20 GiB free. Preserve all existing checkpoints; avoid `/mnt/datos` for new data.

## Budget and decision

Up to 12 wall-clock hours, a background low-priority CPU-only job on IA-Server, CPUQuota at most 150%, memory at most 6 GiB, HDD output expected below 1 GiB, no GPU. Monitor at each automation cycle and resume only from verified outputs on error. Do not repeat P69 or P87 for their own sake. The experiment is needed to certify **public** source equivalence on the full domain, not to increase a PQ score.

If complete, report 720/720 fully exact images or the actual failure count; then prioritize the still-unverified H6/crop-refiner/CSV reconstruction and the mandatory final Google Form. Freeze final delivery at 15 November 2026 00:00 UTC, six hours before the requested 06:00 UTC deadline. The accepted H9b submission remains 58112557, public PQ 0.39, with no new upload from P88.
