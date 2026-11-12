# P86 — Original FP32 public detector GPU parity: pilot completed (negative)

**Executed:** 12 November 2026 23:04:35–23:04:50 UTC.  
**Preregistration:** [P86 protocol](P86_PUBLIC_FP32_GPU_PARITY_PREREG_20261112.md).  
**Public executable:** [P86 script](../../scripts/p86_public_fp32_gpu_parity.py).

## Primary outcome

The four *original*, SHA-256-verified FP32 Mask R-CNN checkpoints were reconstructed from the public release shards on the research HDD. The GPU0 CUDA pilot then processed the first five lexically sorted official unlabeled test JPEGs for each fold F1–F4, producing 40 individual-view comparisons (20 image–fold pairs) against the frozen original **CPU FP32** P69 detector reference. The CUDA process ran under the existing supervisor lock with GPU0 UUID `GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14`, no other compute process on that GPU at preflight, and with CUDA TF32 and autocast disabled. GPU1 was untouched.

| Fold | Fully exact two-view images | Proposal counts (both implementations) | Same-index exact boxes | Same-index exact scores | Same-index exact crop masks | Exact full instances |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 0/5 | 266 | 266 | 21 | 187 | 18 |
| F2 | 0/5 | 287 | 287 | 13 | 212 | 9 |
| F3 | 0/5 | 289 | 289 | 11 | 210 | 9 |
| F4 | 0/5 | 262 | 262 | 13 | 187 | 9 |
| **Total** | **0/20** | **1,104** | **1,104** | **58** | **796** | **45** |

**Interpretation:** FP32 checkpoint restoration removes the FP16 weight mismatch but **does not restore bit-for-bit equivalence between CUDA and the original CPU inference**. All 1,104 indexed box coordinates and per-view proposal counts matched in this limited pilot, but exact score values and binary crop masks frequently differed. This does not imply that accuracy or PQ changed, and does not indicate an error in the published FP32 weights. Different CPU/GPU floating-point execution paths are a plausible source, but their precise contribution is not isolated here. The primary acceptance gate failed, so the preregistered **full-domain CUDA follow-up is not permitted**. No repeat job or Kaggle submission was triggered.

## Audit trail and constraints

- Runtime: **15.926 seconds**, one CUDA process on isolated GPU0; source code was syntax-checked and synthetic exact-comparison self-tests passed before execution.
- Full local receipt: `paired-small-20261112/p86_public_fp32_gpu_pilot_20261112/receipt.json`.
- Receipt SHA-256: `6bc151c4e69c6fa06be18e47853471b64666e4534f0ef5879a960c5e73d7b9d5`.
- All twenty original image SHA-256 values, fold checkpoint hashes, per-view counts and agreement metrics remain in the immutable receipt.
- Test **pixels only**, no test labels, no Kaggle API, no training, no leaderboard tuning. This is an inference-reproducibility result, **not an official PQ score**.

## Next scientifically justified step

Reconstruct the same pilot with **the public source and original FP32 weights on CPU** and compare every output against P69 byte-for-byte. Preregister and bound that separate experiment; if fully exact, consider running the full 180-image/fold CPU source replay, subject to the final notebook and submission deadlines. The accepted H9b source inference was CPU FP32; changing the device is not a valid shortcut to claiming exact reproducibility.

The final four-page report is already published; the end-to-end notebook and verified mandatory Google Form submission remain outstanding and retain priority over exploratory model training. The best accepted public PQ remains **0.39**, submission **58112557**; it was neither resubmitted nor updated by P86.
