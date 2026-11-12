# Solar Filament 2026 — H9b scientific release

Author: Francisco Angulo de Lafuente ([ORCID 0009-0001-1634-7063](https://orcid.org/0009-0001-1634-7063)). The best accepted [Kaggle submission 58112557](https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard), dated 8 November 2026, received **public PQ 0.39**.

[Four-page technical report](../docs/submission/TECHNICAL_REPORT_4P_20261112.pdf) · [Official final submission form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform?usp=dialog)

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

## Independently verified public-weight integrity (12 November 2026)

A fresh, sparse public Git checkout on IA-Server's HDD verified **all ten published model weights** against the repository's `release/model_metadata/WEIGHTS_SHA256SUMS.txt` with `sha256sum -c`. The released Python modules also passed compilation, and both five-code-cell audit notebooks passed Python syntax checks. The follow-up read-only, standard-library-only preflight returned `ASSETS_VERIFIED` (exit code 0).

Reproduce the integrity check from a checkout containing `release/`, `notebooks/` and `scripts/`:

```bash
python scripts/verify_release_assets.py --json-out release-integrity-receipt.json
```

The preflight **does not execute model inference or either notebook**, and the notebooks themselves start from intermediate cached predictions. It therefore does **not** establish source-image-to-final-CSV reproducibility or byte identity with the Kaggle-accepted submission. A complete, independently executed raw-image replay remains an open final-delivery requirement.


## Public source-to-H3 reproducibility test (12 November 2026)

The exact original, **one-channel native ResNet34 U-Net** model definition is
published at [release/source/native_model.py](source/native_model.py), alongside
the original [F1-F4 weights](models/) and their SHA-256 manifest.
The public [stage-one reconstruction CLI](../scripts/replay_h3_from_original_images.py)
and [notebook](../notebooks/h3_public_raw_image_replay.ipynb) use the
original 180 unlabeled competition test pixels to regenerate native-resolution
probability maps. For a complete run, set \`SOLARFIL_TEST_IMAGES\` and
\`SOLARFIL_H3_OUTPUT_DIR\` to a valid Kaggle image folder and a large output HDD.

The [P76 protocol](../docs/submission/P76_PUBLIC_SOURCE_H3_SMOKETEST_PREREG_20261112.md)
and [executed smoke-test report](../docs/submission/P76_PUBLIC_SOURCE_H3_RESULT_20261112.md)
document the first-image public-source F1 replay. That PNG differs from its historical
reference by 32 pixels (of 4,194,304), no difference exceeding one quantized
intensity level. **This is a one-image validation, not a claim of full
reproducibility or improved PQ**. A separate [held-out negative F1 result
report](../docs/submission/F1_HELDOUT_P75_NEGATIVE_20261112.md) records that
the small-object sampler did not surpass its paired control on withheld groups.
The accepted H9b submission remains at public PQ 0.39 (ID 58112557).

The full detector, H6, crop-refinement and final accepted-CSV pipeline still
requires an independently executed end-to-end reproduction. The official
Google Form must be submitted and its receipt verified separately.

**Full-domain public H3 source audit (12 Nov 2026):** [P77/P78 F1 180-image replay and byte-parity report](../docs/submission/P77_P78_FULL_F1_H3_REPRO_AUDIT_20261112.md). This is source-to-semantic-map reproducibility; final H9b CSV parity remains unverified.
