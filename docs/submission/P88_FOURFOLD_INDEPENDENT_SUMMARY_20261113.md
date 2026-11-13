# P88 — Completed independent four-fold FP32 CPU detector parity

**Audit date (UTC):** 13 November 2026. **Status:** four folds independently verified.

The original, unlabelled 180-image test-domain inventory was processed
by each of four released detectors on CPU, in two views per fold.
Each output was independently compared with archived P69 detector
outputs, including input and output SHA-256, ordered scores, boxes
and per-instance binary masks. No new training or GPU inference was run.

| Fold | Original images | Views | Exactly matching proposals |
| --- | ---: | ---: | ---: |
| F1 | 180 | 360 | 13,254 |
| F2 | 180 | 360 | 12,982 |
| F3 | 180 | 360 | 12,323 |
| F4 | 180 | 360 | 12,193 |
| **Total image-fold pairs** | **720** | **1440** | **50,752** |

**Important interpretation:** the test inventory consists of 180
distinct unlabelled original images, each evaluated under four
detector folds. Therefore, 720 is the number of image-fold pairs,
not the number of different images.

F1 and F2 receipts were compared with their already published immutable
results; F3 and F4 were evaluated after the CPU replay completed.
The independent auditor's self-test passed.

**Scope:** detector-stage CPU parity only. This does not reproduce
the full H9b stage-two selection, final-mask refinements or the
accepted Kaggle CSV. PQ cannot be inferred from exact detector parity.
The previously identified 325/1,241 RLE-row disagreement in the
stage-two replay has not been resolved by this audit.

**Competition:** retain baseline Kaggle submission 58112557,
public score 0.39, without a replacement until independent validation.
Do not use competition test annotations or hidden leaderboard feedback
for model selection.

**Unresolved mandatory gates:** exact original-image-to-final-CSV
notebook and proof of the official final Google Form submission.
Freeze all final deliverables by 15 November 2026 at 00:00 UTC.

**Machine-readable summary:** [`P88_FOURFOLD_INDEPENDENT_AUDIT_RECEIPT_20261113.json`](P88_FOURFOLD_INDEPENDENT_AUDIT_RECEIPT_20261113.json).
