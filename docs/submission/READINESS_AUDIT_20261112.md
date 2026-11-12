# Solar Filament 2026 — final-delivery readiness audit

**Audit time:** 2026-11-12 14:46 UTC.  
**Scope:** Independent verification of research, release artefacts and missing final-delivery evidence. This document is not itself an official submission or a claim of winning.

## Verified competition baseline

- Last known accepted Kaggle public score: **0.39**, submission **58112557**, uploaded on 2026-11-08. No better official score is established in this audit.
- The original H9b pipeline remains the reference. Its reported **F1 calibration PQ of 0.462433** is a local, labelled-data measurement, not the Kaggle leaderboard score.
- The 7,200-step F1 small-filament experiment and all three predeclared independent seeds (20261113–20261115) completed their training with receipts, but the held-out **F1 test PQ remains pending**.
- The F1 native-resolution calibration evaluator completed at 2026-11-12 13:59 UTC, freezing a control-selected policy `thr=0.25, min_area=200, peak=0.95, mean=0.7` with **control calibration PQ 0.4133849797**. This is *not* a held-out test score, a victory over H9b, or a Kaggle result.
- Temporal fold F2 completed both preregistered 7,200-step seeds on 2026-11-12 (20261112 at 14:11 UTC and 20261113 at 14:42 UTC); fold F3 seed 20261112 began at 14:42 UTC. F3/F4 and all held-out reports are pending.
- Only original Kaggle TRAIN annotations may be used for supervised fitting; competition TEST annotations are never available. Separate calibration/test partitions and a predeclared promotion gate prevent leakage and opportunistic leaderboard tuning.

## Public-release evidence

- [Four-page technical report](TECHNICAL_REPORT_4P_20261112.pdf) exists. SHA-256 of the staged PDF: `68dafbb8feb7ebeadf8e3dc8273ce8c0bd749bcd3024b477ddef8e0694e9cdc0`. Inspect visual layout again before final submission.
- The [release directory](../../release/README.md) contains ten model weight files for four U-Nets, four Mask R-CNNs and two refiners, plus model builders and hash metadata. Individual weight files are under GitHub's 100 MB threshold.
- The [stage-two replay notebook](../../notebooks/h9b_stage2_intermediate_replay.ipynb) is intentionally **not** an end-to-end raw-image reproduction. Its archived intermediate SHA-256 (`67f02e4d54b78424306133840ec144b426c8b308f6fc659f90a7f67131d5a79e`) differs from the originally accepted Kaggle CSV SHA-256 (`24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`). Never imply byte-identical replay of the accepted submission.
- On this audit, the undefined `EXPECTED_SHA` variable in both published replay notebook copies was corrected to `INTERMEDIATE_SHA`; the broken link to the four-page report was also corrected. These changes remove deterministic failures but **do not** establish complete reproducibility from a clean clone.

## Unresolved completion gates (must not be marked done)

1. **Independent raw-H-alpha-to-final-CSV replay** on a clean checkout with published weights, model implementation, a pinned working environment, and legally obtainable original test images. Validate file schema, disjoint instances, image counts, and SHA; if the exact accepted CSV cannot be reproduced, report the discrepancy.
2. **F1 held-out PQ** and four-fold preregistered external replication with uncertainty intervals; retain all positive and negative outcomes. Do not promote the new model unless the gate and direct H9b comparison pass.
3. **Final Google Form confirmation**: upload the report, supply the public repository and reproducibility notebook URLs, submit, and preserve the returned response/receipt. An existing Kaggle score does not count as submitting this form.
4. **Final freeze and checksum archive**: reserve 6 hours before the stated 2026-11-15 06:00 UTC cutoff; preserve final model weights, code commit, configuration, CSV hash, submission ID, and proof of external acceptance.

**System resource contract:** use only the assigned GPU0; preserve GPU1 for other work. Leave `/mnt/datos` (97% used during this audit) for small metadata only; use the HDD for large checkpoints. Never submit a candidate based solely on loss or calibration PQ.

**Organizers' reference:** [Kaggle competition requirements](https://www.kaggle.com/competitions/filament-segmentation-2026/overview) · [Final Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform).
