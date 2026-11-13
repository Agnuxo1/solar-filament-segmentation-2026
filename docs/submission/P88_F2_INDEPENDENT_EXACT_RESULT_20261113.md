# P88 — Independent exact CPU replay of the full F2 detector domain

**Date (UTC):** 13 November 2026. **Status:** independently audited, complete.

The public Mask R-CNN FP32 F2 checkpoint and released Python implementation were replayed on CPU against 180 original unlabelled Kaggle test JPEG files, using both the direct and mirrored detector views per image. The independent auditor checked the input JPEG SHA-256 values, per-image output pickle hashes, original checkpoint identity, exact ordered proposal scores and bounding boxes, and all binary crop-mask bytes against the previously archived P69 CPU reference. Each data item is compared in its original sequence, without relabelling or rematching predictions.

| Audit measurement | Verified F2 result |
| --- | ---: |
| Distinct original images | 180 / 180 |
| Detector views | 360 / 360 |
| Exactly matching proposals | 12,982 |
| Unexpected images or pickles | 0 |
| Audit exceptions | 0 |

**P88 F2 manifest SHA-256:** `775217530859dc14b99a48910c17bac2977eed099d7ed70b3257019e9063f25d`

**Historical P69 F2 reference manifest SHA-256:** `0346e3b3337f784afabe9991b6ad47c2837abb0b392757f39a0f9960d53f3547`

**Independent audit source SHA-256:** `fb9e757f7c5ed36e7ec0ea93f2e7f07cd7ad65f7ddfd0ddf93f34e80c8078ca6`

The executable audit is [`scripts/audit_p88_full_fold_parity.py`](../../scripts/audit_p88_full_fold_parity.py); the machine-readable proof is [`P88_F2_INDEPENDENT_EXACT_RECEIPT_20261113.json`](P88_F2_INDEPENDENT_EXACT_RECEIPT_20261113.json). The auditor's status construction was corrected to report the actual F1–F4 fold rather than a hard-coded F1 label. Its self-test and source compilation both succeeded before re-auditing F1 and F2 using the corrected source; the original underlying replay pickles and manifests were **not modified**.

**Limitations:** this is exact detector-stage reproduction versus an earlier recorded reference, not an independent PQ score, not final-mask equality, and not reproduction of the accepted H9b CSV. Kaggle submission **58112557** remains the last known accepted baseline, with public PQ **0.39**. No hidden/test labels, new Kaggle submissions, or GPU processes were used in this audit. The mandatory final Google Form confirmation and complete image-to-CSV notebook remain outstanding. F3/F4 must each pass a separate audit once the existing CPU replay completes.
