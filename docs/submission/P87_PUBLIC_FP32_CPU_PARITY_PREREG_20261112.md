# P87 — Public original-FP32 detector CPU source-equivalence pilot

**Preregistered:** 12 November 2026 UTC, before P87 execution.

## Motivation and falsifiable hypothesis

P86 tested four SHA-256-restored original FP32 checkpoints on RTX 3090 GPU0 and found 0/20 fully byte-exact two-view image/fold pairs against original CPU P69 output, despite exact ordered proposal counts and boxes. CPU and GPU floating-point paths can differ; this negative experiment does not invalidate the public weights. The accepted H9b detector was originally evaluated on CPU in FP32.

**Hypothesis:** With the unchanged *publicly released detector Python source* and restored original FP32 checkpoint states on CPU, outputs for all four folds and the first five original unlabelled test JPEGs will exactly match the immutable P69 CPU reference in ordered counts, every floating score, box coordinate and mask pixel byte.

## Frozen procedure and stop rules

- Inputs: the same first five lexically sorted of 180 officially provided, original **unlabelled** test JPEGs, and all four restored FP32 Mask R-CNN states. Verify every weight SHA-256 against the signed public manifest and the P69 reference status. Require `CUDA_VISIBLE_DEVICES` to be empty and `torch.cuda.is_available()` to be false. GPU0 and GPU1 must remain untouched.
- Each fold: infer original and horizontally reflected view with the released `filseg.train_maskrcnn` source and the original image preparation, invert reflection, validate proposal boxes and compare every proposal tuple and mask array with the frozen P69 CPU output.
- Primary outcome: 20 image/fold pairs with *both* views byte-exact. Report all failures, count and pixel mismatches. Reject any result based solely on box counts or geometric IoU.
- Budget: up to 600 seconds CPU, at most four host CPU threads and no GPU. One run only; write uniquely named immutable JSON receipt to the NAS HDD. No large files on `/mnt/datos`.
- Gate: if 20/20 are exact, a **new** pre-registered full 180-image/fold CPU source replay may be justified. Otherwise stop to diagnose public-source differences without repeating previously failed jobs. Never use test annotations, competition leaderboard, or Kaggle submissions in this comparison.

This is a reproducibility test, not an improvement claim about PQ. The four-page PDF, complete accepted H9b CSV notebook, and verified mandatory final Google Form remain priority deliverables.
