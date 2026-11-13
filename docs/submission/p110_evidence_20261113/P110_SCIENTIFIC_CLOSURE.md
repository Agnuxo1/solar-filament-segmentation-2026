# P110 scientific closure — identifiability of instance matching

**Date:** 13 November 2026 (UTC). **Result:** completed, without GPU or annotations. **Status:** no submission and no model promotion.

## Motivation and preregistration
P107 and P109 compared the eight-cell P106 clean notebook output with the genuine H9b Kaggle submission using a maximum-sum COCO-IoU bijective assignment within each image. Before P110 ran, its [protocol](P110_PREREG_20261113.md) was frozen and hashed as SHA-256 `4979027ba33de052f26aa199bc8cf55f6cd4db0faea3ce5c2aa43682b5e40d32`. The independent [implementation](../../../scripts/p110_assignment_identifiability.py) was frozen as SHA-256 `72ff444dbc7eaa39bda7369648bf42cde9dcb4b994c5313738c1fe6a7625eb5e`.

## Source integrity
- Accepted H9b, Kaggle submission 58112557: SHA-256 `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`.
- Public P106 notebook replay, not a scored submission: SHA-256 `16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3`.
- Both full-length CSVs contain 180 physical images and 1,241 instance masks, with matching instance identifiers. No competition test annotations were accessed.

## Frozen method
For each image, the original COCO-RLE masks were compared to the clean replay using the official `pycocotools.mask.iou` primitive. The assignment maximizing total instance IoU was obtained through SciPy's linear-sum algorithm. **For every assigned edge, that edge was forbidden and an alternative optimal assignment was computed**; the largest runner-up score gives the second-best complete assignment. This edge-exclusion method certifies whether an equally good alternative exists. A transposed independent assignment reproduced the optimum; synthetic unique, tied and single-instance controls all passed.

## Results
| Audit quantity | Measured result |
| --- | ---: |
| Image groups | 180 |
| Structurally unique one-instance groups | 9 |
| Images tested for alternative matching | 171 |
| Matched masks | 1,241 |
| Geometrically identical masks | 1,108 |
| Geometrically changed masks | 133 |
| Reordered instance IDs in the optimal matching | 11 |
| Nonunique optimal assignments (margin <= 1e-12) | **0** |
| Near-tied optimal assignments (margin <= 1e-8) | **0** |
| Smallest multi-instance margin in summed IoU | **1.9831675592960973** |

**Outcome:** The preregistered uniqueness hypothesis was not falsified. All 171 nontrivial per-image bijections have strictly positive margins far beyond floating-point tie thresholds. P107/P109 counts cannot be attributed to an arbitrary choice between equally optimal assignments. This does **not** prove that any of the 133 genuine mask differences is correct or incorrect.

P110 ran under user systemd as `solarfil-p110-assignment-20261113.service` with CPU quota 100%, memory ceiling 4 GiB, 900-second runtime ceiling, and an explicitly empty CUDA device visibility mask. Recorded computation elapsed time was 0.071 s (excluding interpreter startup). Service result `success`, exit 0. The receipt has been preserved on the HDD and its SHA-256 is `0f852874adb79a62cbe1edefacb1dd75ddf8116387ff8a40d5fa0e5df37fb7b3`.

## Scientific and competition limitations
This is an **identifiability audit of prediction-to-prediction matching only**, not a Panoptic Quality evaluation. H9b remains the only selected final result, with previously confirmed public PQ 0.39. P106 has **no official PQ**. No training, GPU inference, Kaggle upload, leaderboard tuning, organizer form edit, or test-label access took place during P110. The P108 out-of-fold provenance gate still blocks the proposed 1.15x context ablation.

The machine-readable, per-image evidence is in [P110_FINAL_RECEIPT.json](P110_FINAL_RECEIPT.json). There was no JEV or Goldfish consultation during this audit; independent checks consisted of transposed assignment, exhaustive chosen-edge exclusion, and synthetic controls.
