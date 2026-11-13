# P93 — Historical Kaggle-upload provenance correction (13 November 2026)

## Critical identity correction

**Actual Kaggle-uploaded H9b no-site file (submission 58112557, public PQ 0.39):**
`24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`

**Archived H9b stage-two intermediate, misleadingly named `submission_58112557.csv`:**
`67f02e4d54b78424306133840ec144b426c8b308f6fc659f90a7f67131d5a79e`

**P71 independently re-inferred intermediate:**
`bb6bba8db2c4cc007154b380b02629137b62e2cbc3c859aebca128a36d713515`

The Kaggle CLI description of submission **58112557** includes the uploaded file SHA-256 prefix **24b3ea1c8e151c30** and currently confirms `COMPLETE`, public PQ **0.39**. A copy of that historical submitted CSV is preserved at `paired-small-20261112/p74_accepted_artifact_audit/accepted_h9b_nosite.csv`, with full hash matching the description. P74 had already correctly identified this on 12 November. This new independent audit detects and corrects an inconsistent subsequent description in P90.

## Independently rerun P93 audit, with no ground truth

The published P90 CSV comparison program was independently re-executed on 13 November with the **actual uploaded CSV** as reference. Its synthetic tests passed. Every comparison has the same 1,241 `filament_id` values and no missing/extra rows.

| Comparison to real upload | Exact RLE rows | Differing RLE rows | Affected images | Byte-identical |
| --- | ---: | ---: | ---: | --- |
| Archived stage-two intermediate | **915** | **326** | 150 | No |
| P71 re-inferred intermediate | **916** | **325** | 145 | No |

**Local immutable evidence receipts:**
- `audit/P93_ACTUAL_UPLOAD_VS_STAGE2_20261113.json`, SHA-256 `1a52ff74fe04537d729fe35c1e4daff2f5b6ca3b2423b4f9963de8663b3adf34`
- `audit/P93_ACTUAL_UPLOAD_VS_P71_20261113.json`, SHA-256 `a3cbb1a1dc50fe397b725c26d4b005ecf401b22a714895f1aa7a84a7cd6060ad`

The **58** different rows in P90 describe P71 versus *stage two*, **not** P71 versus the actual Kaggle upload. Likewise, byte-for-byte parity of the cached stage-two notebook is **not parity with the real uploaded file**.

The original stage-two notebook already disclosed this difference. The later P90 narrative erroneously identified stage two as the uploaded artifact. This correction takes precedence over that identification, without deleting or changing prior results.

## P91 scope amendment (documented, not silently changed)

P91 was preregistered and launched using the stage-two file at `final-release-staging-20261112/submission_58112557.csv` as its hash-checked CSV comparison. That file **is not the uploaded artifact**. The ongoing CPU FP32 rerun is still scientifically useful as a repeatability experiment against the original crop probabilities and stage-two assembly, but its CSV results must be labelled **stage-two parity**; they cannot be called exact Kaggle-upload parity. The originally registered P91 JSON and running code have been preserved unchanged for auditability.

The actual-upload comparison must subsequently use the separately preserved `p74_accepted_artifact_audit/accepted_h9b_nosite.csv`. The remaining gap involves the original **no-site selector and full raw-image-to-upload pipeline**, not necessarily only crop-refiner numerical differences. Avoid causal claims until the no-site path is reproduced under controlled, frozen settings.

No competition labels, private leaderboard scores, or new Kaggle submissions were used. No GPUs were interrupted.

## Submission-critical next actions

1. Use the genuine uploaded artifact SHA **24b3ea1c...** for every final reproduction gate and evidence table.
2. Investigate the frozen no-site H6 selection and native paste step; label any candidate that only reproduces SHA **67f02e4d...** as intermediate.
3. Complete the runnable original-JPEG-to-real-upload notebook if exact reproduction is feasible, or disclose precisely the remaining numerical differences.
4. Finish the four-page report and public source review, and obtain an actual receipt from the mandatory organizer Google Form.

**Freeze target:** 15 November 2026, 00:00 UTC. **Competition closes:** 15 November 2026, 06:00 UTC.
