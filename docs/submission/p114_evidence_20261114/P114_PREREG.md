# P114 preregistration — mask-union and exclusive-pixel audit

Frozen before execution on 14 November 2026, before competition delivery freeze.

## Question
P113 proved both genuine H9b and public P106 COCO-RLE streams have valid per-instance areas, but did not verify whether distinct predicted instances overlap. The published inference contract assigns each pixel to at most one instance. Does that invariant hold for both complete 180-image prediction CSVs?

## Frozen inputs
- H9b: /srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112/p74_accepted_artifact_audit/accepted_h9b_nosite.csv
- H9b SHA-256: 24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60
- P106: /srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112/p106_public_clean_6843362_20261113/inference_cuda/submission_public_rebuilt.csv
- P106 SHA-256: 16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3
- Dimensions from 180 original JPEG headers only; no test labels.

## Method and predeclared measurements
Parse each compressed COCO RLE into column-major foreground intervals with a standalone Python decoder, computing per-image union by interval sweepline. Independently compute the union area through pycocotools.mask.merge and assert exact equality for every image. Record per-image sum of instance areas, union area, overlap-multiplicity excess (sum minus union), and number of images violating exclusivity. Use two-pointer interval intersection to compute per-image H9b-vs-P106 foreground-union symmetric difference, a prediction-reproducibility measure that is NOT PQ. Check 180-image inventory, 1241 rows each, fixed SHA-256, valid image IDs, valid RLE dimensions, and exact image-level instance counts. The hypothesis is zero overlap excess for both CSVs; nonzero results are preserved and reported, not hidden.

## Resources and decisions
CPU only (CUDA_VISIBLE_DEVICES empty), no training, no labels, no network inference, no Kaggle upload. Wall-time budget 240 seconds from program start; deterministic, no randomness/seed. Source code must be saved and hashed before execution. All outputs under the project HDD p114_union_exclusivity_20261114. Abort on hash mismatch, library disagreement, or exceeded budget; preserve an error log. No final-selection promotion based on this audit; retain H9b submission 58112557 with last official public PQ 0.39. Distinguish instance-summed area, union area and accuracy. No JEV/Goldfish consultation is required for deterministic validation; disclose nonconsultation.
