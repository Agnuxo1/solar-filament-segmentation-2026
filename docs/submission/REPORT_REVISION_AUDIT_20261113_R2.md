# Technical report revision 2 — 13 November 2026

This is a minimal evidence update of the previously submitted four-page report. The preceding PDF and source ZIP remain byte-for-byte unchanged. This revision has not been published or submitted by this document-editing task.

## Changes

- Replaced the pending P103 description with its completed result, verified against the exact audit JSON included in the source package.
- P103 completed at 10:50:06 UTC: 176 newly inferred images and four reused pilot cases; 180 images, 1,241 rows, 1,101 position-wise exact RLE matches, 140 position-wise differences, 90 identical complete image predictions, and identical instance counts on all 180 images. All used `public_pipeline`; no historical fallback was used. The CPU stage-coupling runtime was 1,661.308 seconds.
- Clarified that P103 from reconstructed stages does not replace a clean execution of the revised portable notebook at commit `6843362`. P106 is being prepared on CUDA GPU 0 with public weights; no completed result is asserted.
- Replaced the missing-form-receipt statement with the stable verified acknowledgment at 14:00:23 UTC (15:00:23 Madrid), 13 November 2026. Submission receipts and report revision provenance are referenced to the public repository [9], without making an attachment-version claim that could become stale.
- Preserved the historical official PQ 0.39 and all existing measured figures/results. No new PQ or Kaggle score is claimed and no cause or accuracy effect is inferred from the RLE differences.

## Evidence

P103 audit JSON SHA-256: `d67fe349750d6da9fe1eb5bb31377a4c83357dcba8548db666552c9eff327701`.

P103 CSV recorded SHA-256: `8ae3160af9bddd5ac3ef914bc807da71e3e9cbad0444cc055d2ff0056cdfe2ba`.

The audit's `official_submission_made: false` concerns the P103 prediction experiment; it does not contradict the separate organizer-form acknowledgment.

## Verification

The revised PDF has exactly four pages in the same acmart sigconf 9-point format. Every page was rendered and visually inspected for R2; pages 1 and 4 were re-rendered and reviewed after the final acknowledgment and position-wise comparison edits. There are no overfull TeX boxes or unresolved references. The source package includes the exact P103 audit JSON, full source, original figures, class source and template provenance. No additional clarification was needed for this bounded update.

PDF SHA-256: `8f75a47996fa7e6e5f37f5d8e85a6cf5b64033701507782a54a81d786e5ed9f5`.

Source ZIP SHA-256: `6bee5b287bf6920ae8ef3bab82b7518a6366140da92de19b98e6b972b128955b`.
