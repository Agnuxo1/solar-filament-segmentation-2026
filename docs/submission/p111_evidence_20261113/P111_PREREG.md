# P111 preregistration — stage-level drift localization (P103 vs P106)

Date: 2026-11-13 UTC. Study ID: P111. Scientific scope: reproducibility, NOT accuracy/PQ.

## Prior evidence and question
P103 reconstructed 180 images using 176 newly computed refiners and four previously frozen pilot cases. P106 independently executed eight public notebook cells for 180 JPEGs and computed fresh refiners. The pre-existing P107 audit reports 12 one-pixel mask differences across 11 images, without instance reassignment. The mechanism has not been localized.

## Frozen inputs
- P103 final CSV SHA-256: `8ae3160af9bddd5ac3ef914bc807da71e3e9cbad0444cc055d2ff0056cdfe2ba`
- P106 fresh CSV SHA-256: `16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3`
- Accepted H9b, deliberately NOT used as an accuracy reference: `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`
- P103 per-image crop archives and 176-image ResNet34/EfficientNetV2 probability archives.
- P106 per-image crop archives and 180-image corresponding probability archives.
- P107_VS_P103_AUDIT.json, only to identify image-level prediction drift.

## Falsifiable primary hypothesis
For each of the eleven images with a changed P103/P106 segmentation, at least one upstream crop array or matching refiner probability array differs. A changed final segmentation on an image with exactly identical upstream tensors falsifies this hypothesis and points to downstream assembly or data-order effects.

Secondary descriptive questions: How many images have exact crop arrays `X,M,comp,score`? How many of the 176 freshly computed P103 cases have exactly matching refiner arrays? Are any of the four reused pilot images among the eleven changed outputs? Which changes have finite numerical magnitude and which are exact?

## Method, controls and budget
1. Verify both frozen CSV SHA-256 digests and the P107 result (180 image groups, 12 changed masks, 11 changed images).
2. Verify that each of the 180 crop archives exists in P103 and P106, and compare each matching array for exact shape/dtype/content; count changed elements and maximum absolute numeric difference.
3. For each of the 176 keys in each P103 refiner archive, require corresponding P106 keys and compare exact shape/dtype/content, changed entries, maximal absolute difference, and, descriptively only, threshold crossings at 0.5. The four excluded P103 pilot images must be reported explicitly rather than being represented as independent repetitions.
4. Compare with P107 per-image changed mask counts. Produce one JSON evidence row for each image and summary; record all falsifying examples.
5. Run independent synthetic unit controls for equal arrays, a one-element change, and mismatched shape. No manual tuning and no lookup of competition labels.

Precommitted resource budget: CPU-only, `CUDA_VISIBLE_DEVICES=""`; up to 10 minutes wall time, 4 GiB RSS, one process, no training, no new Kaggle submission. Seed 20261113 (not needed for deterministic calculations). Read-only original inputs, output artifacts confined to the NAS project directory. No /mnt/datos writes. Abort on missing files, hash mismatch, nonfinite values, or schema mismatch. Never substitute unseen data.

## Promotion or termination condition
This study cannot promote a segmentation model, since no TRAIN holdout PQ is computed. Its only promotion is as a reproducibility finding after independent tests and hashes. Any future model modification requires a separate preregistration and valid out-of-sample TRAIN evidence (P108 provenance gate remains in force). Preserve selected H9b and final organizer form without alteration. JEV/Goldfish consultation only if genuinely available; record an explicit absence otherwise.
