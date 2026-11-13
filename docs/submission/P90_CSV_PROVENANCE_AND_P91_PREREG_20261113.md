> **CORRECTION — 13 NOVEMBER 2026 (P93):** This historical P90 report incorrectly calls the stage-two CSV with SHA-256 `67f02e4d...` the "Kaggle-accepted" or "uploaded" file. The **actual uploaded submission 58112557** has SHA-256 `24b3ea1c8e151c30...`, as preserved and verified in P74. The stage-two intermediate differs from the uploaded CSV in **326 RLE rows**; the P71 rerun differs from the uploaded CSV in **325 RLE rows**. The **58 differences** below compare P71 to the *stage-two intermediate only*. Read the [P93 correction](P93_ACTUAL_UPLOAD_PROVENANCE_CORRECTION_20261113.md) before interpreting the historical results. The original P90 observations are retained to preserve audit history.

# P90 — Independent RLE reproducibility-gap audit (13 November 2026)

## Scope and frozen evidence

This is a **read-only, CPU-only forensic audit**, not a new Kaggle score or a new competition submission. The [P90 auditor](../../scripts/audit_p90_submission_mismatch.py) validates the exact CSV schema, unique instance IDs, row ordering, byte hashes, mismatching RLE contents, missing/extra instances, and the number of affected source observations. Its synthetic self-tests passed before running on project data. No competition test labels were read, and no CUDA processes were started.

The historically **accepted** file for official Kaggle submission **58112557**, public PQ **0.39**, has **1,241 instance rows** and frozen SHA-256:

`67f02e4d54b78424306133840ec144b426c8b308f6fc659f90a7f67131d5a79e`

Two distinct reconstruction paths were examined; **do not conflate their levels of reproducibility**.

| Candidate replay | Row ID coverage | Exactly matching RLE rows | Differing RLE rows | Affected images | Byte-for-byte identical to accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| P71: frozen H6 crops → separately rerun P70 refiners → final H9b assembly | 1,241/1,241 | 1,183 | **58** | **49** | No |
| Staged notebook: cached model predictions → H6 selection → H9b assembly → final CSV | 1,241/1,241 | **1,241** | **0** | **0** | **Yes** |

**P71 SHA-256:** `bb6bba8db2c4cc007154b380b02629137b62e2cbc3c859aebca128a36d713515`.

**Cached-stage replay SHA-256:** `67f02e4d54b78424306133840ec144b426c8b308f6fc659f90a7f67131d5a79e`.

These results independently confirm the original P71 parity receipt and staged notebook receipt using a separate implementation and independently computed hashes. The previously reported **325/1,241** discrepancy concerns another public stage-two replay; it is *not* the P71 or staged notebook candidate compared here, and must not be presented as the result of this P90 test.

## Scientific interpretation

The cached-output stage-two reconstruction matches the submitted CSV **exactly**. The separate P71 crop-refiner re-execution retains all 1,241 IDs, but 58 final RLE masks differ. Thus an independently reproducible exact final CSV exists **only when starting from archived model outputs**. The remaining reproducibility gate lies upstream of those cached predictions. P88/P89 also established exact CPU FP32 detector-stage reproduction (720 image–fold combinations, 1,440 views), while the public H3 probability map stage has near-byte parity, not exact parity.

The precise origin of the 58 P71 mask differences is **not yet identified**. Potential sources include numerical precision, interpolation, preprocessing, output rounding or postprocessing, but none of these is asserted as the cause without a controlled experiment. Exact cached-stage CSV parity must **never** be presented as an end-to-end original-image replay.

## Next preregistered diagnostic: P91 (do not start new GPU jobs)

**Question:** Do independently rerun crop refiners, held at fixed input crop, checkpoint, transform, architecture and thread count, reproduce the archived H9b crop probabilities and RLE when executed on CPU FP32?

**Input population:** The 58 P71 RLE-discordant instance IDs are used strictly as a forensic **reproduction discrepancy set**, not for model selection or competition metric optimization. Include a deterministically chosen matched set of RLE-concordant crops as a negative control. Do not use GT annotations, validation labels, competition test labels or the public leaderboard to alter predictions.

**Protocol:** Verify each crop/checkpoint SHA-256 and input transform; compare intermediate tensor dimensions, exact values, max absolute deviation, segmentation thresholds, and final RLE in order. Stop at the first differing stage. Preserve the original accepted CSV. Run on CPU first; GPU0/GPU1 are occupied by other tasks and must not be interrupted. Freeze code/configuration and hash the preregistration before analysis.

**Budget and safety gates:** 3 hours CPU wall time, <= 6 GiB system RAM and <= 2 GiB of new HDD outputs; no use of the 97%-full `/mnt/datos` SSD. The experiment should not be run if the immutable required source crops or documented model checkpoints cannot be resolved. A successful diagnostic identifies a source of numerical divergence; it is not evidence for improved PQ.

## Outstanding judging requirements

The mandatory Google Form remains without a verifiable submission receipt. A direct HTTP access attempt returned **401** on 13 November 2026; no form data was submitted. The official four-page report is staged, but the public notebook still does not demonstrate original JPEG → all models → H6 selection → two crop refiners → exact accepted CSV. Complete those gates before the **15 November 2026, 00:00 UTC** freeze.

**Local primary machine-readable receipts:**
- `audit/P90_P71_VS_ACCEPTED_20261113.json`
- `audit/P90_STAGE2_VS_ACCEPTED_20261113.json`

Both are preserved on the NAS HDD. These contain checksums and RLE fingerprints, not the original competition images or labels.
