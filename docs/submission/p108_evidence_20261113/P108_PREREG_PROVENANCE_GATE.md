# P108 — preregistered provenance gate for the proposed 1.15× TRAIN context ablation

**Status before execution: audit protocol only.** Registered on 2026-11-13 UTC before the P108 evidence inventory. The locked hypothesis and prospective numerical intervention remain those in [NEXT_TRAIN_TEST_PROPOSAL_20261113.md](../p103_evidence_20261113/NEXT_TRAIN_TEST_PROPOSAL_20261113.md); P108 does **not** run inference or estimate accuracy.

## Question

Can the previously proposed 1.15× crop context / 50:50 native-space fusion ablation be evaluated on the frozen F1 TRAIN calibration/test cohorts **without any image appearing in the training or fitting inputs of any model, selector, or refiner contributing to its prediction**?

## Immutable evaluation gate

The only admissible outcome is `ELIGIBLE` if all of the following are evidenced at **per-physical-image** granularity, independently on all 117 calibration and 139 reserved F1 test images:

1. Frozen F1 TRAIN / calibration / test filename memberships are mutually disjoint; all annotators of one physical image stay together.
2. The P106 notebook execution receipt is complete (eight code cells) and the frozen released checkpoint manifests match its recorded model family. P106's competition-TEST predictions are *not* evaluation data.
3. Frozen baseline probabilities and upstream maps/objects are available for those F1 TRAIN-derived images, with source-image and code/checkpoint hashes.
4. Each H3 model, Mask R-CNN proposal model, **both crop refiners**, and the no-site selector has a verifiable training/fitting manifest excluding the exact scored image. Absence of an exclusion manifest is a failure; a model named `ALL` is **not** acceptable evidence of out-of-fold predictions.
5. The baseline and 1.15× treatment share the same eligible OOF upstream sources; no competition-TEST annotations, accepted CSV, or leaderboard outcome enter the metric.

No fallback to all-TRAIN checkpoints or ad hoc splits. File naming alone does not certify out-of-sample status. A failed gate must explicitly abort the 45-minute CPU-only ablation, not silently run an in-sample accuracy study.

## Audit procedure and budget

Inventory only project-controlled on-disk artifacts under the frozen `paired-small-20261112` directory and its two release checkouts. Independently hash the split and P106 receipt. Count filename membership and refiner checkpoint names (without publishing annotation content); read selector metadata. Record presence or absence of **image-exclusion evidence** rather than infer it from model naming. This is a finite read-only CPU/metadata audit, ≤5 minutes, 0 CUDA time, ≤100 MiB new HDD output. Do not access Kaggle TEST labels or create any new submission.

Write a machine-readable receipt with explicit gates, source SHA-256 values, and a fail-closed final status; publish the receipt with a concise interpretation. If any prerequisite is false or unverifiable, the next-context ablation is blocked and H9b is retained. If every gate is verifiably true, the intervention needs a separate frozen implementation, test and execution (maximum 45-minute CPU wall-time, 6 GiB RAM, 2 GiB HDD) as specified in the prior proposal; P108 alone is not the intervention.

## Promotion rule

P108 never promotes a model. The +0.005 paired official-PQ gate, 2,000 group bootstrap resamples, TP/FP constraints and Kaggle quota remain unchanged from the proposal; only a separately executed, eligible study can assess those rules.
