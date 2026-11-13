# P96 — Controlled public H3 replay → H6 crop coupling: full 180-image negative result

**Completed:** 2026-11-13 07:17:38 UTC.  
**Preregistration:** [P96 protocol](P96_H3_REPLAY_CROP_COUPLING_PREREG_20261113.md).  
**Runner:** [CPU-only, recoverable source code](../../scripts/p96_h3_replayed_crop_coupling.py).  
**Audit evidence:** [Immutable 180-image summary](P96_H3_REPLAY_CROP_COUPLING_RECEIPT_20261113.json).

## Scientific question

Previously, P79/P80 reproduced the four original-image H3 model outputs with 99.998913738% exact quantized pixels, and P95 reproduced 180/180 H6 crop archives exactly **when historical H3 maps and consolidated detector outputs were supplied**. P96 isolates the effect of replacing those historical H3 maps with independently regenerated public-source H3 maps, while freezing the original unlabelled JPEGs, historical consolidated Mask R-CNN detector cache, TRAIN-only 15-feature no-site selector, threshold logic and crop geometry implementation.

The experiment was preregistered before comparisons, with fixed four-image pilot followed by the full sorted inventory, hash-verified inputs, deliberate self-test failures, immutable per-image results, CPU-only execution and no test labels.

## Results

| Acceptance criterion | Images with different values | Images exactly matching |
|---|---:|---:|
| Crop tensor `X` (including interpolated H3 intensity channel) | **180/180** | 0/180 |
| Affine matrices `M` | **9/180** | 171/180 |
| Selected component identifiers `comp` | **0/180** | 180/180 |
| Continuous selector scores `score` | **11/180** | 169/180 |
| All four fields jointly bit-exact | **180/180 different** | **0/180 exact** |

All 180 JPEGs were processed and all reference archives, original images, selector and public H3 model-map receipts were hash-validated. The CPU-only systemd execution exited successfully with no crash. No new training, GPU use, Kaggle score or submission was involved.

**Final immutable receipt SHA-256:** `7eb28e7642442622ad55405f0cfc1050adbe7af27e593b60d67dcee726a372c5`.

**Frozen preregistration SHA-256:** `9ce0656cfd32a378a753f4b536ae9432e4e3b11a62886b00784d3b642b6e49aa`.

## Interpretation and limits

This is a **useful negative numerical-parity result**, not a failure of the P95 historical-cache reconstruction. The independently regenerated H3 predictions are near-exact at the input map level but not byte-identical. Even sparse one-byte map differences propagate through interpolation into the crop tensor for every image. In the controlled P96 comparison, component identifiers did remain stable in all 180 images, while nine geometric matrices and eleven scores changed.

P96 deliberately continues to consume the original consolidated detector cache rather than rebuild it from P88. P97 independently found that simply concatenating public detector views does not recreate the consolidated cache: 0/4 pilot matches and radically different proposal counts. Therefore **no clean-install JPEG → prediction → 1,241-row true-upload CSV exact parity is claimed**.

The previously demonstrated P94 exact replay of the true uploaded CSV from frozen H9b crops remains valid. P96 has no Panoptic Quality metric and provides no reason to change the competition submission: 58112557, public PQ 0.39.

## Next gates

1. Locate the original detector aggregation/filtering source, explicitly audit its model/view merging against original consolidated proposals, and never assume a direct concatenation.
2. Independently test a bounded numerical-parity hypothesis for the H3 quantization differences (CPU vs GPU/precision) only after preregistration; measure effects without test labels, model refitting or Kaggle leaderboard feedback.
3. Integrate a standalone raw-image processing notebook with publicly permitted weights and dependencies, noting any reproducibility gap rather than substituting frozen predictions silently.
4. Prioritize the mandatory organizer form and preserve its receipt. The English four-page technical report has already been checked to contain exactly four A4 pages; form completion remains unconfirmed.

**Operational deadline:** freeze by 2026-11-15 00:00 UTC, six hours before the stipulated close. No unsupported final-judging eligibility assertion is made.
