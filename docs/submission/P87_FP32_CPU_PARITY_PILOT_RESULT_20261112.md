# P87 — Public original FP32 CPU source parity: 20/20 exact pilot

**Executed:** 12 November 2026, 23:08:48–23:12:25 UTC.  
**Preregistered protocol:** [P87 protocol](P87_PUBLIC_FP32_CPU_PARITY_PREREG_20261112.md).  
**Executable:** [P87 CPU replay script](../../scripts/p87_public_fp32_cpu_parity.py).

## Verified result

The public, released `filseg.train_maskrcnn` Python source, the original FP32 Mask R-CNN checkpoint states reassembled from eight verified public shards, and the publicly released reflection logic were run on **CPU only**, against the same five first lexically sorted original unlabelled test JPEGs in each of F1–F4. Every fold matched the immutable private P69 CPU evidence in ordered proposal counts, exact floating-point scores, integer coordinates, and **every crop mask byte**.

| Fold | Fully exact image/fold pairs | Full exact views | Exact proposal instances |
| --- | ---: | ---: | ---: |
| F1 | 5/5 | 10/10 | 266/266 |
| F2 | 5/5 | 10/10 | 287/287 |
| F3 | 5/5 | 10/10 | 289/289 |
| F4 | 5/5 | 10/10 | 262/262 |
| **Total** | **20/20** | **40/40** | **1,104/1,104** |

All 1,104 corresponding boxes, scores and masks are identical. The single controlled run returned exit code 0 after **217.463 seconds**. The script's independent synthetic exact-instance and unequal-list regression checks and Python syntax check passed prior to this run. No CUDA context was initialized; `CUDA_VISIBLE_DEVICES` was empty. There was no GPU usage attributable to P87.

## Comparison and interpretation

The corresponding preregistered P86 experiment used the **same original FP32 weights on CUDA**: it reproduced all 1,104 same-index boxes but only 58 scores and 796 crop masks exactly, producing 0/20 fully exact image/fold pairs. The platform path therefore materially affects byte-exact reproducibility in this limited pilot. An isolated test of every individual CPU-vs-GPU numerical source has **not** been performed. Neither pilot measures segmentation PQ or demonstrates final-mask CSV equality.

**P87 clears the preregistered gate for a separately preregistered full-domain CPU public-source replay** (four folds × 180 original unlabelled images). This next phase can verify full detector-stage reproducibility but would still leave the H6 selector, refinement and final RLE/CSV steps to be reconstructed. Do not replace the accepted H9b official submission based on this result.

## Integrity and boundary

- Immutable source receipt stored at `paired-small-20261112/p87_public_fp32_cpu_pilot_20261112/receipt.json` on the large HDD.
- Receipt SHA-256: `66e85b875ed77bf1c63180386752ce9cabd05f8f76d7c8ada8d7d0ab4eeb9450`.
- Every input file and restored original FP32 checkpoint SHA-256 is recorded in the receipt. Source script SHA-256, UTC timestamps and per-view exact-match metrics are included.
- Only unlabelled official test pixels were read; no competition test annotations, hidden leaderboard, new model training, or new Kaggle submission occurred.

The published four-page English technical PDF is unchanged. The accepted H9b CSV remains unproven end-to-end by the public notebook, and the final required Google Form has no verified submission receipt. The best public Kaggle score remains **0.39** on accepted submission **58112557**; preserve it until a scientifically validated replacement meets all gates.
