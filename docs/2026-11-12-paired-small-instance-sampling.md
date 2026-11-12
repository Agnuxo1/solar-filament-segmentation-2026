# Paired small-filament sampling experiment — 12 November 2026

**Status:** Completed, independently scored, **HOLD**. This experiment is not the official Kaggle submission and did not improve the pre-existing baseline.

## Research question

The existing, separately validated H9b pipeline misses many small solar-filament instances. We tested whether preferring small annotated bounding boxes during training improves a standalone semantic segmentation branch without disclosing or using competition test labels.

Both ResNet34 U-Net arms started from the same SHA-256-pinned F1 checkpoint and were trained for 3,600 paired updates (batch size 8), on **446 official GONG F1-training images only**. Each arm used the same seed, optimizer family, BCE + soft-Dice loss and eight-view inference. Only the source of crop centres differed: control sampled all annotation boxes; treatment preferentially sampled boxes with area <=3,500 pixels. No external ground-truth annotations were used.

## Measured results

The scorer compared native 2048 x 2048 predictions against **117 F1 calibration images and 190 annotation records**, following the competition's record-wise Panoptic Quality formula with IoU > 0.5. The prediction policy was fixed in advance: connected components (8 connectivity), probability threshold 0.5, minimum area 60 pixels and 8 test-time geometric views. No F1 test labels were opened.

| Model | Local PQ | SQ | RQ | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Paired control | 0.359912 | 0.680065 | 0.529232 | 1,385 | 1,964 | 500 |
| Small-box treatment | 0.370934 | 0.682447 | 0.543535 | 1,364 | 1,770 | 521 |
| Frozen H9b pipeline | **0.462433** | **0.689052** | **0.671115** | 1,207 | 505 | 678 |

The treatment's paired advantage over its matched control was **+0.011022 PQ**, with a 95% physical-image grouped bootstrap confidence interval **[+0.004813, +0.016818]** (4,000 resamples; PCG64 seed 20261112). Relative to the stronger frozen H9b baseline, however, treatment had **-0.091499 PQ**, 95% interval **[-0.105707, -0.077204]**.

Thus, the crop-sampling hypothesis improved the controlled semantic segmentation system but failed to clear the objective baseline gate. It **must not be presented as a superior model** or submitted to Kaggle based on this result. The F1 calibration set has been used historically; the bootstrap interval does not establish independent out-of-distribution performance.

## Execution and provenance

- Dedicated physical GPU: NVIDIA RTX 3090, GPU 0 UUID `GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14`. GPU 1 remained unassigned.
- Training: 3,600 / 3,600 updates completed on 12 November 2026, 07:42:58 UTC.
- Measured training duration: 970.34 s; allocated tensor peak 1.793 GiB. This is not a whole-process memory or energy measurement.
- Eight-view native calibration inference: all 117 images completed in 124.27 s; allocated tensor peak 1.837 GiB.
- CPU-only PQ evaluation: all 190 annotation records completed in 33.16 s.
- Train-only initial checkpoint: SHA-256 `a4cc0686786ad2f9a253e2e38cfe37896d16a9d6f090e8429cfe27ae138c4d24`.
- Final paired checkpoint: SHA-256 `cc990330b99c21fc68e1d9d903ad6eda72ad34bb118ee8cf5b3126acf37f603e`.
- Prediction manifest: SHA-256 `d8b6893f96a4c4ccf7462da468be40ea4b6786483f058a7200881d9f6d5d52ed`.
- Frozen local scorer source: SHA-256 `982bfdf210ee0a48b723c963dd51a4523415484957d7b2470fe479c6d6f49781`.
- Original data and checkpoints are not redistributed. Research run receipts and artifacts remain on the authorized server: `/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112/`.

**Official Kaggle result remains 0.39 PQ**, submission `58112557` (`submission_h9b_nosite.csv`, submitted 8 November 2026), independently verified via the authenticated Kaggle CLI on 12 November 2026. The local values above are not the leaderboard score.

## Interpretation

The treatment produced 194 fewer false-positive components but also 21 fewer true positives and 21 additional false negatives than the matched control. The H9b baseline was substantially better calibrated and had far fewer false positives. A targeted high-precision auxiliary detector might be worth testing, but that requires a preregistered integration experiment and evidence beyond this single calibration fold.

**Competition rules:** only official H-alpha test images are allowed at model inference; no test annotations, recovered answers or auxiliary test observations were used. See [competition overview](https://www.kaggle.com/competitions/filament-segmentation-2026).
