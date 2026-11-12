# Solar Filament 2026 — H9b scientific release

Author: Francisco Angulo de Lafuente ([ORCID 0009-0001-1634-7063](https://orcid.org/0009-0001-1634-7063)). The best accepted [Kaggle submission 58112557](https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard), dated 8 November 2026, received **public PQ 0.39**.

[Four-page technical report](../docs/TECHNICAL_REPORT_4P_20261112.pdf) · [Official final submission form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform?usp=dialog)

**New audit (12 November 2026):** [SFPAIR: 7,200-step small-filament sampler continuation and independent seed replications](../docs/SFPAIR-EXT-REPL-20261112.md). This training is unvalidated for PQ and has not been promoted or submitted. Its research ID is distinct from the existing P59–P62 studies.

## What has been independently checked

The submitted H9b no-site pipeline consists of four independently trained ResNet34 U-Nets, four Mask R-CNN proposal models, a calibrated instance selection rule, and two crop refiners. Inference uses only the supplied H-alpha test image, without temporal neighbours or externally annotated labels.

The preserved **stage-two** replay (from previously generated own model outputs to H9b instance masks and CSV) covers 180 official test images, 1,241 disjoint RLE instances and 2048-by-2048 masks. The rebuilt CSV hash is `67f02e4d54b78424306133840ec144b426c8b308f6fc659f90a7f67131d5a79e`; the independently preserved original accepted file has hash `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`. **These files have different digests**, so do not claim byte-exact identity or a complete raw-image replay.

A separate 12 November 2026 raw-to-U-Net audit regenerated **720 maps** (four trained U-Nets × 180 images) from the original weights on one RTX 3090. Per-block pixel equality with our own historical probability maps was F1 **99.9960%**, F2 **99.9956%**, F3 **99.9930%**, F4 **99.9931%**. The maximum uint8 difference was 1, 1, 18 and 6 respectively. None of the 720 images were entirely bit-exact. These tiny differences can affect thresholding and must be disclosed; this is not a verification of an exactly identical final submission.

## Verified research results (F1 calibration, not Kaggle)

The local native-PQ scorer evaluates **117 physical images and 190 annotator records** with grouped paired bootstrap intervals. This F1 calibration has been reused historically and is not an untouched final test set.

| Candidate | F1 native PQ | Decision |
| --- | ---: | --- |
| Frozen H9b | 0.462433 | **Preserved baseline** |
| P58 small-object sampler | 0.370934 | HOLD; despite +0.0110 vs weak paired control |
| P60 Tversky precision loss | 0.355080 | HOLD; −0.10735 vs H9b |
| P61 topology soft-clDice | 0.364443 | HOLD; −0.09799 vs H9b |
| P62 postprocess threshold 0.75 | 0.375876 | HOLD; +0.01143 vs P61 at 0.50 but −0.08656 vs H9b |

No new Kaggle submission was justified or made. A favorable paired within-family result is not evidence of improvement against the current stronger reference.

## Reproducibility and licensing policy

This directory holds the release workflow and documented measurements. A cache-dependent notebook, even if executable, is **not** sufficient for fully reproducible raw-image-to-CSV predictions. A final end-to-end implementation must run from competition-supplied pixels, own released inference weights, fixed preprocessing, fixed model and instance decisions, and a pinned environment. Every weight file has to be smaller than GitHub's 100 MB file limit or be handled as correctly reconstructable separate shards. Any fp16 conversion is numerically non-identical unless validated. Do not redistribute the competition's test images or annotations.

Code is MIT licensed under [the repository LICENSE](../LICENSE). Pretrained components and the official dataset may carry separate license obligations. Completion of the [organizers' final Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform?usp=dialog) must be verified separately from any Kaggle upload.
