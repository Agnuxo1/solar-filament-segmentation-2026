# P113 scientific closure - independent COCO RLE cross-decoder audit

**Executed:** 2026-11-14 07:03:15 UTC. **Status:** PASS_ALL_ROWS_BOTH_INDEPENDENT_AREA_DECODERS.

## Hypothesis and preregistration

The accepted H9b artifact and the clean P106 public replay both contain syntactically and geometrically valid compressed COCO RLE streams. To distinguish actual prediction discrepancies from invalid serialization, an independent pure-Python decoder must agree with the published pycocotools area decoder on **every row**.

The protocol was frozen before execution in `P113_PREREG.md` (SHA-256 `482ac4c97dbe5e051f66f502ddae87ba6698f48e3486f8ebbe344fa4a60ba280`). The complete standalone source is `p113_rle_cross_decoder.py` (SHA-256 `1f34e5cca47a68cd2cd5500db3ae7707ded5ed973b01a2792e96dbbaaef7cf88`). The machine-readable receipt is `P113_FINAL_RECEIPT.json` (SHA-256 `ccdbb45e59444f004963935b257af5ebb50b61508920816af71abb6840bf88dc`).

## Method and outcomes

The audit used original image **headers only** (180 JPEGs, each 2048 x 2048 pixels) and the two frozen prediction CSVs; no TEST annotations were loaded. The pure-Python decoder independently parsed each signed, differential COCO run-length stream, checked nonnegative runs and the exact pixel total, and computed alternating-run foreground area. `pycocotools.mask.area` supplied an independent library cross-check. Both accepted and replayed files passed all **1,241 rows each**, or **2,482 masks total**, with no absent image, duplicate filament ID, dimension mismatch, invalid stream, empty mask, or per-row area disagreement.

| Measurement | Genuine accepted H9b | Clean P106 replay |
| --- | ---: | ---: |
| Images | 180 | 180 |
| Valid masks | 1,241 | 1,241 |
| Decoded integer runs | 254,917 | 254,913 |
| Sum of per-instance foreground pixels | 2,481,790 | 2,481,766 |
| Smallest/largest instance area (px) | 223 / 20,758 | 223 / 20,762 |
| SHA-256 | `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60` | `16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3` |

The net difference is **-24 summed instance pixels** in P106. This is not a union-area or accuracy measurement. Comparing the same filament IDs by position, 1,099 RLE strings match and 142 differ; 136 same-ID areas differ, with 18,764 total absolute area change. **Instance numbering can be reordered**, so this 18,764 value is not a geometrically matched discrepancy and must not replace P107's verified pairing result (133 changed matched masks; 270 total symmetric-difference pixels).

**Conclusion:** Both CSV files are valid under two independent COCO RLE area decoders. The small P106-versus-H9b prediction differences cannot be explained by broken RLE structure or a malformed dimension contract. The test does not identify the numerical cause of those differences, infer ground truth, estimate Panoptic Quality, or justify a new upload.

## Release decision

No GPU, training, hidden TEST label, model selection, Kaggle submission, final-selection change, or organizer-form edit occurred. Preserve the genuine selected H9b, submission **58112557**, last officially confirmed public PQ **0.39**, and its unchanged CSV. The P108 out-of-fold provenance gate remains blocking the 1.15x context ablation; P113 provides **no exception**. JEV/Goldfish were not consulted for this deterministic test.

