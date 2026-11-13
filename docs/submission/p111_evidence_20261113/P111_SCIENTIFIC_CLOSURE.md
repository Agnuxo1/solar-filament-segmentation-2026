# P111 scientific closure — isolating clean-replay numerical drift

**Completed on 13 November 2026 UTC.** This study was preregistered and
SHA-256 frozen before execution. It audits the existing P103/P106 predictions;
it does not perform any Kaggle scoring, labels-based evaluation, training, or
new submission.

## Integrity and conditions
- Preregistration SHA-256: `cd8a2716c6293e1550ca3cb6e256e14f4f1ad98c9d8f55b89ba6a873eeec468b`.
- Audit code SHA-256: `e79990b8e9f24e506a311cdbe3a7e91ada73750652deb4d2fd7214aee6b613a5`.
- Full receipt SHA-256: `7b0ad18bebf50fb436cd89e447c02c632d04ed7f098fe3306372cf8ee4a278fc`.
- Frozen P103 CSV SHA-256: `8ae3160af9bddd5ac3ef914bc807da71e3e9cbad0444cc055d2ff0056cdfe2ba`.
- Frozen P106 CSV SHA-256: `16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3`.
- P107 independent per-image baseline verified: 180 images, 12 final
  mask differences in 11 images, with no matched-instance reassignment.
- Three synthetic controls passed, including the FP32 one-ULP half-threshold case.
- CPU only, CUDA invisible, no labels or generated submission. Original inputs
  remained read-only and outputs remained on the project HDD.

## Measured differences

| Quantity | Result |
| --- | ---: |
| Images with changed P103/P106 output | 11/180 |
| Final masks with one-pixel difference | 12/1,241 |
| Images with changed crop pixels X | 166/180 |
| Output-changed images with different X | 11/11 |
| Output-unchanged images with different X | 155/169 |
| Different crop X values | 1,577 / 244,580,352 compared positions |
| Largest absolute crop pixel drift | 255 |
| Images with exactly identical M, comp, score | 180/180 |
| P103 pilot images lacking new refiner probability files | 4 |
| Output-changed images among those pilots | 0 |

The different-X count is the count of positions unequal under exact comparison;
unchanged masks often arise despite upstream numerical differences. The crop
tensor total across all images was 244,580,352 values; the number differing was
1,577 values.

| Quantized refiner archive (176 comparable images) | Exact images | Different uint8 values | Max absolute level drift | Raw zero/nonzero transitions* |
| --- | ---: | ---: | ---: | ---: |
| ResNet34 | 18/176 | 4,945/60,407,808 | 10 | 880 |
| EfficientNetV2-S | 16/176 | 13,095/60,407,808 | 17 | 2,824 |

*The predeclared archive name “probabilities” is misleading: an independent
[domain audit](P111_VALUE_DOMAIN_AUDIT.json) (SHA-256 `ce7430180aab0bb75245fda5bfddef69591dcc8d0db42d27dc6e62a0eac0ea2d`) confirms **uint8 values 0–255**, not floating-point probabilities in [0,1].
A numerical 0.5 crossing in uint8 means only zero versus nonzero, **not** the
actual segmentation threshold; it cannot be interpreted as PQ or confidence.
The maximum differences above are raw integer intensity levels. The P103 and
P106 refiner inference used FP32 computation, but the archived arrays are quantized.

The four P103 pilot image IDs omitted from the fresh refiner comparison are
`20110120105534Ch, 20110130110334Ch, 20110214175414Mh, 20110329082654Uh`. None had a changed final mask.

## Predeclared hypothesis and scientific interpretation
The primary falsifiable hypothesis survived this test: all 11 images with one or
more changed final masks had different upstream crop pixel tensors. This is an
association, not identification of the causal operator. The test cannot attribute a
specific threshold crossing to any particular final mask pixel. The archived, quantized uint8 refiner outputs also show additional numerical
drift, which often has no downstream effect. Reusing four P103 pilots limits like-for-like refiner comparison to 176 images.
No test ground truth, official PQ calculation, performance gain, or model promotion
is supported by these data.

The strict provenance gate established by P108 continues to block the proposed
1.15x additional-context evaluation. Selected Kaggle H9b submission 58112557
(public PQ 0.39) is unchanged. P106 remains a separately reproducible,
unscored CSV. No new PQ value is inferred.

No JEV or Goldfish consultation was performed. The independent controls were
cross-input SHA-256 integrity checks, frozen P107 per-image evidence, direct
NumPy comparisons and positive/negative synthetic tests.

## Recommended next step
Preserve the evidence and the already-confirmed organizer submission. Further
scientific work should only target a demonstrably new hypothesis with a frozen
protocol and valid TRAIN out-of-sample provenance, or audit final artifact
integrity before the 15 November 00:00 UTC freeze. Do not rerun known-negative
experiments or launch redundant CUDA work.
