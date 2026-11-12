# P84 cross-fold matched-step Panoptic Quality audit

Status: **POST_HOC_CROSSFOLD_AUDIT_COMPLETE_NO_PROMOTION**.

This is an exploratory, post-hoc analysis of existing and previously examined held-out splits derived exclusively from Kaggle TRAIN. It is not a new blind validation set, an official Kaggle score, or a training improvement claim.

| Fold | Seed | 7,200-step control PQ | 7,200-step treatment PQ | Matched delta | 95% paired image bootstrap CI |
|---|---|---:|---:|---:|---|
| F1 | P59_original | 0.398796 | 0.394077 | -0.004719 | [-0.010617, +0.001062] |
| F1 | 20261113 | 0.398975 | 0.395253 | -0.003721 | [-0.009258, +0.001815] |
| F1 | 20261114 | 0.398163 | 0.387987 | -0.010176 | [-0.015180, -0.005473] |
| F1 | 20261115 | 0.403401 | 0.387933 | -0.015467 | [-0.022293, -0.008283] |
| F2 | 20261112 | 0.428724 | 0.424196 | -0.004528 | [-0.010414, +0.001215] |
| F2 | 20261113 | 0.424996 | 0.427163 | +0.002167 | [-0.002689, +0.007084] |
| F3 | 20261112 | 0.406631 | 0.403617 | -0.003014 | [-0.008943, +0.003319] |
| F3 | 20261113 | 0.404802 | 0.409670 | +0.004868 | [-0.001586, +0.010901] |
| F4 | 20261112 | 0.393369 | 0.377128 | -0.016241 | [-0.024924, -0.008260] |
| F4 | 20261113 | 0.385124 | 0.389723 | +0.004600 | [-0.007088, +0.015806] |

**Descriptive direction:** 3 positive and 7 negative effects across 10 previously executed reports.

All 10 official counting tuples were recomputed and matched their stored native PQ summaries within 1e-11. A fixed 4,096-replicate paired physical-image bootstrap was executed independently for each report. Repeated test images within a fold and correlated seeds preclude pooling these intervals as independent evidence.

**Decision: NO PROMOTION, NO NEW KAGGLE SUBMISSION.** Preserve the current official H9b result (historically 0.39 PQ, submission 58112557); prioritize reproducible source-image-to-final-CSV implementation and the mandatory final Google Form.

Source SHA-256, frozen fold-specific postprocessing policies, sample sizes and exact values are in [the JSON receipt](P84_CROSSFOLD_MATCHED_PQ_RECEIPT_20261112.json). The [analysis script](../../scripts/audit_p84_crossfold_matched_effects.py) includes a deterministic self-test.
