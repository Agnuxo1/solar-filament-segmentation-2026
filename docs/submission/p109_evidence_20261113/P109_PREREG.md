# P109 preregistration — topology and bounding-box stability of the clean public replay

**Status at publication:** preregistered, not executed. **Date:** 2026-11-13 UTC.
**Study type:** read-only reproducibility/geometry audit, not a TRAIN-label accuracy experiment or a candidate for Kaggle promotion.

## Motivation and hypothesis

P107 found that the eight-cell P106 clean-output replay differs from the accepted H9b CSV in 133 optimally paired masks across 180 public competition test images, with only 270 differing pixels in total, while all image IDs and instance counts remain unchanged. This is prediction-to-prediction evidence; it contains no ground truth. The new hypothesis is that the residual differences primarily affect one-pixel boundaries, and leave matched-instance bounding boxes and connected-component counts stable. We will test the hypothesis, including failures, rather than assuming it.

## Frozen data contract

Only two already-produced prediction CSVs may be read:

- Accepted H9b SHA-256 `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`, Kaggle submission 58112557.
- Clean P106 SHA-256 `16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3`, without official PQ.

Expected P107 control: 180 images; 1,241 rows per CSV; 1,108 exactly paired masks and 133 changed pairs totalling 270 symmetric-difference pixels. Any mismatch in SHA-256 or control totals aborts the audit.

## Frozen analysis

1. Require unique IDs, identical image sets and instance counts and valid 2048 x 2048 compressed COCO RLE.
2. Match instances bijectively within each image by maximum total IoU (Hungarian assignment), using no annotations and no leaderboard scores.
3. For every changed pair, compare area in pixels, COCO bounding-box coordinates, number of connected foreground components with eight-way connectivity, and the symmetric-difference pixel count. Independent COCO area/intersection operations must reproduce the interval arithmetic.
4. Summarise changes at pair and image level; record median, maximum, total, unchanged-box proportion and unchanged-component proportion, including exceptions.
5. Test on a synthetic pair with one changed boundary pixel and another with an added isolated pixel. Record the expected difference in component counts. Output a SHA-256 receipt and per-image breakdown.

The hypothesis is falsified if any changed pair changes its bounding box or component count; regardless, report all such exceptions. The experiment cannot establish PQ, rank, scientific segmentation accuracy, or a higher-quality submission. Its **promotion rule is permanently disabled**: the original H9b file and selected final Kaggle submission must remain unchanged.

## Resources and safety

Strict CPU-only run; do not invoke CUDA; no inference, fitting, or training. Wall-time cap 15 minutes; RAM cap 4 GiB; new HDD output cap 10 MiB. Run at low priority and abort if inputs or dependencies are absent. Do not write to `/mnt/datos` and do not touch GPU1, the DronERL process on GPU0, Kaggle, or the organizer form. Publish both positive and negative outcomes and the exact code revision; no JEV/Goldfish claim without an actual consultation.
