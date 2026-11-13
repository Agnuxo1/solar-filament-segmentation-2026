# P88 ? Independent exact replay of the complete F1 detector domain

Completed on 2026-11-13T00:10:31.491968+00:00. Under the preregistered P88 protocol, the released public source and original FP32 detector checkpoint were replayed from 180 original unlabelled official JPEGs on CPU. A separate audit checked every input SHA-256, both persisted proposal views against the immutable P69 reference, every exact score/box/crop-mask byte, and all manifest totals.

**Independently confirmed:** 180/180 image-fold pairs; 360/360 views; 13254 exact detector proposals.

P88 F1 manifest SHA-256: 6c6f92be04f5d6579dd58e20011489b3e3c9a88bd08c322058e80fcfebb45034

Public executable: scripts/audit_p88_full_fold_parity.py. Machine-readable receipt: P88_F1_INDEPENDENT_EXACT_RECEIPT_20261113.json.

This confirms **detector-stage parity only**. It is not a reconstruction of the accepted H9b CSV nor a new PQ score. F2?F4 remain subject to their own completion and audits. The accepted H9b CSV differs from the reconstructed stage-two CSV in 325/1,241 RLE rows. The complete final notebook and mandatory Google Form receipt remain outstanding. Preserve Kaggle submission 58112557 (PQ 0.39) without new uploads until independently verified improvement.
