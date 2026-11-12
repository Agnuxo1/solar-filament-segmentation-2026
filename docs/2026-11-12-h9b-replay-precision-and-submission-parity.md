# H9b prediction-replay precision audit — 12 November 2026

**Scope:** forensic reproducibility audit of the existing Solar Filament Segmentation Challenge 2026 H9b solution. These are comparisons between **predicted masks**, not comparisons with hidden labels, not new validation PQ scores, and not evidence of a new Kaggle submission.

## Artifact identities (do not conflate these files)

| Artifact | SHA-256 | Rows | Notes |
| --- | --- | ---: | --- |
| **Actual submitted** `submission_h9b_nosite.csv` | `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60` | 1,241 | Preserved original file accepted as Kaggle submission **58112557**, public PQ **0.39** |
| Frozen intermediate H9b stage-two `submission_58112557.csv` | `67f02e4d54b78424306133840ec144b426c8b308f6fc659f90a7f67131d5a79e` | 1,241 | **Not byte-identical** to the actual Kaggle upload; it is an intermediate artifact |
| Fresh stage-two reconstruction P71 | `bb6bba8db2c4cc007154b380b02629137b62e2cbc3c859aebca128a36d713515` | 1,241 | Independently regenerated final-refiner outputs; not byte-identical to either file |

All three files share the same **1,241 filament identifiers across 180 test images**; the test labels were never read in this audit.

## Independent GPU calculation and reproducibility

P70 reran both original H9b refinement models on the frozen crop input, using **physical GPU0** on IA-Server (RTX 3090 UUID `GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14`). Against archived pre-submission crop probabilities:

- ResNet34: **61,103,920 of 61,145,088 probability bytes identical** (99.9327%). Of 41,168 different bytes, the maximum absolute difference was **1**.
- EfficientNetV2-S: **61,092,358 of 61,145,088 bytes identical** (99.9138%). Of 52,730 different bytes, the maximum absolute difference was **2**.

P72 preregistered deterministic CuDNN, disabled TF32, and deterministic CUDA algorithms before execution. The resulting probability tensors were **byte-for-byte identical to the P70 rerun for both models**. Therefore, these settings did not remove the small discrepancy relative to the archived originals. That discrepancy may depend on environment or arithmetic details; the exact cause has not been established.

## COCO-RLE mask geometry

P73 measured the difference between the intermediate stage-two file and its fresh replay. **58 of 1,241 RLE rows differ**, representing only **64 binary-mask pixel disagreements** in aggregate. Across these 58 changed masks, the **minimum IoU is 0.997050**, median **0.999539** and mean **0.999445**. The remaining **1,183 rows are byte-identical**.

P74 then checked the **actual uploaded Kaggle file**, using its original SHA-256, against both intermediate files:

| Versus actual uploaded file | Identical RLE rows | Differing rows | Pixel disagreements | Min IoU of differing masks | Median IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen intermediate stage two | 915 | 326 | 455 | 0.995238 | 0.999386 |
| Newly reconstructed stage two | 916 | 325 | 451 | 0.995238 | 0.999374 |

These differences are **real** even though the masks are nearly geometrically identical. Do **not** describe either stage-two CSV as an exact byte-for-byte reproduction of the accepted submission. Nor can this mask comparison establish the same hidden-label PQ without an official evaluation.

## Research and release integrity

- The accepted Kaggle artifact and all original checkpoints remain unchanged.
- No external test labels, annotations, adjacent-frame test inference, or new leaderboard submissions were used.
- All experiment inputs, code, preregistrations, original and generated output hashes, and JSON receipts are preserved on IA-Server at `/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112`, specifically P70, P71, P72, P73, and P74.
- This audit **does not replace** the challenge's mandatory four-page PDF, public source and weight release, self-contained runnable notebook, pinned dependencies, and final Google Form receipt. End-to-end exact submission reproduction remains a release requirement to be addressed explicitly or disclosed as a numerical-reproduction limitation.

**Lead:** Francisco Angulo de Lafuente · [ORCID 0009-0001-1634-7063](https://orcid.org/0009-0001-1634-7063).
