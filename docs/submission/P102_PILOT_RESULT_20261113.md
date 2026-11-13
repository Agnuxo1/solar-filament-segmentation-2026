# P102 — Published-stage H9b replay on first four official test images

**Executed:** 13 November 2026, 10:16–10:17 UTC. **Status:** completed pilot; negative discrepancy preserved.

The fixed pilot combined four independently reconstructed H3 U-Net map sets, P88 public Mask R-CNN proposals, the standalone P99 consensus algorithm, the published frozen no-site selector (trained on Kaggle TRAIN labels only), and the two published H9b crop refiners in CPU FP32. Input images were the first four sorted original, unlabelled official TEST JPEGs. No historical H9b H3 or detector caches served as inference inputs. All ten model checkpoint SHA-256 values passed the release manifest.

## Predeclared acceptance tests and observations

| Quantity | Pilot result |
|---|---:|
| JPEG images successfully processed | 4/4 |
| Final images with identical accepted-upload RLE lists | **3/4** |
| Final RLE rows identical by position | **14/15** |
| Discrepant output RLE rows | **1/15** |
| Crop component IDs exactly preserved | 4/4 |
| Crop selection scores exactly preserved | 4/4 |
| Crop affine geometries exactly preserved | 4/4 |
| Crop X tensor byte-exact | 0/4 |
| Individual crop tensor elements changed | **179** |
| Historical control fallback images | **0** |
| Test annotations accessed | **none** |
| CUDA processes launched | **none** |
| Kaggle submissions made | **none** |

The first three images reproduce their accepted masks exactly. The fourth image (20110329082654Uh) has 9 predicted and 9 reference instances, with 8 identical RLEs and 1 different RLE. The changes in crop X are 21, 21, 27, and 110 array elements per respective image. This supports a narrow conclusion: small differences inherited from public H3 replay can survive the complete crop-to-mask pipeline, but did not alter 14 of the 15 RLEs in this pilot.

**No new official PQ exists**, and a four-image reproducibility audit cannot justify new Kaggle submission or claim an improvement in segmentation performance. The original accepted Kaggle upload remains submission 58112557, public PQ 0.39, SHA-256 24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60.

## Provenance, limitations, and recovery

[Preregistration](P102_PUBLIC_STAGE_COUPLING_PREREG_20261113.md) was published in commit 5604142 before any P102 comparison. [Source](../../scripts/p102_public_stage_coupling.py) is public. All input checkpoint hashes, the pilot CSV digest, per-image crop comparisons, and per-image RLE counts are recorded in [P102_PILOT_AUDIT_20261113.json](P102_PILOT_AUDIT_20261113.json). The historical accepted CSV was read **only for audit**; no labels or leaderboard feedback entered inference.

The initial service failed at input validation because its code referenced an intermediate CSV rather than the authentic archived upload. It produced no pilot output and launched no CUDA process. The reference was corrected in commit c96a78e; the actual authorized service then completed with exit status 0, 35.942 seconds reported by the Python run, and a CPU memory peak of 2.2 GiB measured by systemd.

This audit assembles model outputs already independently regenerated from public weights, but does not yet constitute a clean single-command end-to-end rebuild from a new environment or prove that 180/180 final images match the genuine upload. A full 180-image audit is scientifically motivated by the one divergence. Before the deadline, prioritise the final reproducibility notebook and the required organiser Google Form (still lacking a verified receipt).
