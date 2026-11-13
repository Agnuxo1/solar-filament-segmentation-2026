# P112 scientific closure — resolving apparent 255-level crop drift

**Completed: 2026-11-13 UTC.** Preregistered before reading the paired crop pixel values for this hypothesis. This is an independent, CPU-only reproducibility audit of the frozen P103/P106 public replay artifacts, not a segmentation-accuracy experiment.

## Provenance and integrity

- Preregistration SHA-256: `c1b67cbc7fc63138be73f4ae050bddf438ef2271177484e4eec5ec915ea79069`.
- Analysis source SHA-256: `449691a68818ab7348bb721916a4844d8df43f90b9a0500567c91fce64e3256a`.
- P112 receipt SHA-256: `8b7b9c94b0ccd76b6495fa23c065f09c6e917f8b00575caf67c9fd975b467823`.
- P103 CSV SHA-256: `8ae3160af9bddd5ac3ef914bc807da71e3e9cbad0444cc055d2ff0056cdfe2ba`.
- P106 CSV SHA-256: `16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3`.
- P111 frozen receipt, P107 image-level changes, and all 180 image IDs were checked. Source tensors were required to be aligned `uint8` arrays. Four synthetic positive/negative controls passed.

## Results

| Metric | Observed |
| --- | ---: |
| Images | 180 |
| Crop-tensor values compared | 244,580,352 |
| Different values, cross-checked with P111 | 1,577 |
| Absolute change exactly 1 | **1,573** |
| Absolute change 2–254 | **0** |
| Absolute change exactly 255 | **4** |
| Of those four: 0 → 255 | 2 |
| Of those four: 255 → 0 | 2 |
| Large changes that are *not* 0↔255 endpoint swaps | **0** |

Only about **0.000645%** of compared crop values differ. Every change is either one intensity level or an exact swap between the two uint8 endpoints.

The four extreme differences are distributed over three image IDs. One extreme difference is in a group with a changed final P103/P106 mask (11 images), and three are in unchanged-final-mask images (169 images). The final-mask-change groups contain 160 and 1,417 different crop values, respectively. This is a descriptive grouping; it is **not** evidence of causation or of a precision difference.

## Preregistered hypothesis

H stated that all absolute changes at least 128 were 0↔255 swaps. It **survived the exact test (4/4)**. An alternative non-endpoint high-magnitude value pair would have refuted it. However, the data cannot distinguish uint8 conversion, clipping/wrapping, quantization, interpolation, floating-point kernel variation, or another upstream source. Do not label 255 as an arbitrary huge continuous-valued model error: these are uint8 endpoint values.

These findings clarify the apparently anomalous P111 maximum of 255 without claiming bit-identical complete replay. P106's 12 one-pixel final-mask differences against P103 remain; P106 does not inherit the Kaggle score from H9b.

## Competition decision and safeguards

No labels, hidden leaderboard, Kaggle API, training, submission, or CUDA operations were used. GPU0/GPU1 workloads were preserved. P108 still blocks the proposed 1.15× test without genuine out-of-sample provenance. **No model or CSV promotion is justified by P112.** H9b submission 58112557, public PQ 0.39 (previously confirmed), selected final artifact, and the existing organizer form/R2 receipts remain unchanged.

No JEV or Goldfish consultation was made; independent controls were the frozen SHA-256 evidence, prior P111 counts, strict image-ID matching and synthetic positive/negative comparisons.

## Further work

Preserve the preregistration, source and final receipt. A deeper causal study would require paired *prequantization* pixel values and fixed operators; do not guess which operator caused the four endpoint swaps. Before 2026-11-15 00:00 UTC, prioritize checking the final report, repository, exact original H9b CSV and form receipts. Do not consume Kaggle submissions for reproducibility-only candidates.
