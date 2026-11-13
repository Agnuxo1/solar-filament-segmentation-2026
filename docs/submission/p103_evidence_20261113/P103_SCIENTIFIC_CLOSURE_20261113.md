# P103 scientific closure: numerical correspondence and remaining limits

**Status:** completed audit, with a separate post-hoc read-only geometry analysis on 13 November 2026. No model was trained or inferred for the post-hoc analysis. No competition annotation, raw test JPEG, leaderboard feedback, or remote file modification was used.

## Main result

P103 completed the public-stage coupling for all 180 official images and produced 1,241 instance rows. Every image retained the accepted submission's instance count. All predictions came from the public pipeline; the archived control fallback was used for zero images. The P103 CSV is **not byte-identical** to the genuine accepted CSV.

The original P103 audit compares RLE rows **by position**. It reports 1,101 identical rows and 140 different rows, with 90/180 completely identical image-level sequences. A subsequent comparison of the underlying masks establishes a more precise interpretation:

| Comparison | Result |
| --- | ---: |
| All image instance counts agree | 180/180 |
| Exact RLE rows at the same position | 1,101/1,241 |
| Different RLE rows at the same position | 140/1,241 |
| Rows whose best geometric match has a different position | 11 |
| Of those 11, geometrically identical masks | 9 |
| Of those 11, masks with an additional geometric difference | 2 |
| Identical masks after a verified bijective within-image match | 1,110/1,241 |
| Geometrically different masks after that match | 131/1,241 |
| Geometrically identical image-level mask sets, ignoring identifiers | 91/180 |
| Symmetric pixel difference over all matched mask pairs | 268 |
| Median / maximum pixel difference among the 131 changed masks | 1 / 22 |
| Minimum / median matched IoU among the 131 changed masks | 0.9831675593 / 0.9994192799 |

The independent match is the maximum-IoU reference for each predicted mask within its image. Its mapping was explicitly checked to be bijective over the full inventory. Consequently, no reference mask was counted twice. The 140 differing rows consist of 129 geometric changes without a position change and 11 reordered rows, two of which also have a geometric change. Ninety-seven of the 131 changed masks differ by exactly one pixel.

A standalone COCO compressed-RLE interval implementation first produced these results. An independent read using **pycocotools** reproduced the exact mask count, changed mask count, total pixel difference, maximum pixel difference, minimum IoU, median IoU, and bijectivity. This comparison uses two sets of the participant's own predictions; it is **not an evaluation against ground truth**.

Evidence: [original P103 audit](P103_FINAL_AUDIT.json), [mask correspondence diagnostic](P103_MASK_MATCHING_DIAGNOSTIC.json), and [independent NumPy/pycocotools check](P103_INDEPENDENT_NUMPY_COCO_CHECK.json).

## Where the discrepancies enter

The selected crop inventory contains 1,244 components on both sides, before final mask assembly. Every stored selected component identifier agrees. This does not imply that the corresponding binary seed pixels are identical.

| Stored crop field | Images with any difference | Different array elements |
| --- | ---: | ---: |
| Four-channel tensor X | 180 | 300,778 / 244,580,352 |
| Affine transform M | 9 | 20 / 7,464 |
| Selected component identifiers | 0 | 0 / 1,244 |
| Selector score | 11 | 12 / 1,244 |

The tensor difference is 0.1229772% of its stored elements. It is concentrated by channel as follows:

| Channel | Images affected | Different elements | Largest absolute byte difference |
| --- | ---: | ---: | ---: |
| Flat-field intensity | 9 | 264,495 | 28 |
| Fused probability | 180 | 35,934 | 70 |
| Candidate seed | 34 | 275 | 255 |
| Other candidates | 17 | 74 | 255 |

These values demonstrate why it would be inaccurate to describe every downstream change as a one-byte perturbation. Altered geometry changes interpolated image samples, and binary seed changes have a full 255-byte amplitude. In the nine images with different affine transforms, 283,334 tensor elements differ. Among the 162 images whose M, score, and component identifiers are all identical, only 15,967 X elements differ, yet 105 final RLE rows differ.

The 12 changed selector scores have a maximum absolute change of 0.0330781636. Their stable descending sort changes in **four images**, exactly the four images containing the 11 reordered output rows. Thus output numbering accounts for a measured part of the RLE discrepancy; it is not evidence of large displaced or missing objects.

Evidence: [aggregated immutable crop receipts](P103_CROP_RECEIPTS_AGGREGATE.json), [direct affine/score comparison](P103_CROP_GEOMETRY_SCORE_DIAGNOSTIC.json), [channel comparison](P103_INDEPENDENT_NUMPY_COCO_CHECK.json), and the 176 original per-image crop receipts included in this evidence collection. The four remaining image receipts are in the original P102 audit.

## What the prior controls establish

P94 reproduced the genuine uploaded CSV exactly from frozen historical crops and CPU-FP32 refinement. P95 established exact crop reconstruction when the historical H3 and detector intermediates were supplied. P96 then changed the H3 maps to independently regenerated public-source maps while holding the historical detector cache and the other crop inputs fixed. It observed the same field-level pattern as P103: X differs in 180 images, M in nine, score in eleven, and component identifiers in none. P99 separately verified exact public detector consensus against the historical consensus in all 180 images.

Together these controls support the conclusion that small upstream H3-map differences propagate through thresholded component pixels, features, geometry, interpolation, refinement, and final thresholding. They do not isolate a unique hardware or arithmetic cause of the historical/public H3 difference.

The P100 fixed four-image, four-fold repeatability test obtained 16/16 identical pairs between two repeated public-source runs and the earlier public maps. P101's two alternative cuDNN settings did not recover the historical F1 maps on the four pilot images: the recorded changed-pixel count rose from 68 for the public baseline to 533 for either tested intervention. These bounded tests do not justify saying that random run-to-run variation alone explains the mismatch, or that a particular backend switch fixes it.

Evidence: the included original [P100 audit](p100_h3_repeatability_20261113/P100_FINAL_AUDIT.json), [P101 audit](p101_cudnn_h3_20261113/P101_FINAL_AUDIT.json), [P102 audit](p102_public_stage_coupling_20261113/P102_AUDIT.json), and the prior public P94/P95/P96/P99 receipts under `prior_public_evidence/`.

## Limits and accurate claims

P103 reused the four immutable P102 pilot outputs and inferred the other 176 crop sets. It consumed previously regenerated H3 maps and P88 detector outputs, rebuilding their public consensus and downstream stages. Its 1,661.308-second elapsed time measures that CPU coupling operation; it is not complete original-JPEG-to-CSV inference latency and does not include the earlier upstream model runs. The flag `cpu_only` applies to P103's own execution, not to every prior upstream model execution.

The original audit explicitly records `requires_separate_clean_install_reproduction: true`. Its `independent_raw_to_csv` flag is computed from zero archived fallback use and must not be interpreted as proof of a fresh isolated installation. P106 is the separate notebook execution intended to address that operational question; its outcome is outside this note.

This evidence proves close correspondence, with an exact account of the remaining differences. It proves neither equal PQ nor improved PQ: even a one-pixel change can affect a borderline match, and there are no test labels here. No new Kaggle score or submission resulted from P103 or this post-hoc analysis. The historically accepted CSV remains the reference, not a target for fitting thresholds or matching its bytes.

## Immutable identifiers

| Artifact | SHA-256 |
| --- | --- |
| Genuine accepted CSV | `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60` |
| P103 reconstructed CSV | `8ae3160af9bddd5ac3ef914bc807da71e3e9cbad0444cc055d2ff0056cdfe2ba` |
| P103 original final audit | `d67fe349750d6da9fe1eb5bb31377a4c83357dcba8548db666552c9eff327701` |
| Original P103 runner | `2eda02fcffe1770d32409fc4c056bacf4964249583cce2fe03d9de7f7c895460` |
| Original P103 preregistration | `fac960bd3c3a86b4952747b052318d71563d6e19bb82c18992718632d6d25ef0` |

The exact original JSON copies were transported in compressed form and checked against their remote SHA-256 hashes after local decompression. `SHA256SUMS.txt` covers the publishable evidence collection, including the later diagnostic scripts and results. No raw JPEG, mask-image file, crop tensor archive, model checkpoint, or complete prediction CSV is included in the collection.

