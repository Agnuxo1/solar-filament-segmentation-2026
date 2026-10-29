# Solar Filament Segmentation Challenge 2026

![Solar filament header](assets/header.png)

An open-source preparation and evaluation pipeline for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026). The project focuses on precise, instance-aware masks for H-alpha observations and audits fragmentation, over-merging, and runtime because the challenge uses Panoptic Quality rather than a plain pixel-overlap score.

![Segmentation pipeline](assets/segmentation-pipeline.png)

## 28 October: real training results and an applied memory fix

The executable NeuroPixel-compatible pipeline is now published under
[`pipeline/`](pipeline), including the temporal split, training, native and
1024-resolution PQ, calibration-only model selection and submission inference.
The comparison control is NeuroPixel (calibration mean PQ1024 0.4052, held-out
test 0.4015), rather than the older ridge baseline below.

| U-Net ResNet34, ImageNet initialisation | Calibration PQ1024 | Test PQ1024 | Test native PQ |
|---|---:|---:|---:|
| Seed 0, recovered checkpoint | 0.415397 | 0.399921 | 0.387129 |
| Seed 1 | 0.419251 | **0.403124** | 0.386943 |
| Seed 2 | 0.421140 | 0.398986 | 0.382741 |

Calibration improves for all three trained seeds. Seed1 also improves the
held-out control by +0.001624; seed2 regresses by -0.002514. Both outcomes
are preserved. The completed three-seed mean is **0.400677 ±0.002170**
(sample SD), compared with the reported control0.4015; the mean delta is
-0.000823. This does not pass the replacement gate; no new leaderboard
submission is claimed. The official API still returns five entries with best
public PQ **0.35**. [Actual run records](pipeline/results/unet-20261028).

**Applied evaluator improvement:** NumPy bounding-box slices kept a complete
2048-square mask alive for each polygon. Crops now own their small buffers,
ground truth is rasterised as needed, and prediction caches are released after
the last annotation of each physical image. On the first three distinct real
annotation records, retained polygon buffers fall from **156 MiB to 0.424 MiB**;
PQ1024, native PQ, TP, FP and FN are exactly unchanged. This measures retained
array buffers, not whole-process RSS or model accuracy.
[Checked equivalence and byte counts](pipeline/results/memory_fix_verification.json).

The original seed0 log ends in `Killed` after8000training steps; that log does
not prove the operating system's reason. A recovery command evaluates its
already-saved best checkpoint without retraining and preserves the original
incomplete record. It explicitly matches the original T4 float16 inference
precision when run on the managed RTX3090. That recovery has now succeeded,
without retraining; the [checkpoint hash, complete evaluation and three-seed
summary](pipeline/results/unet-20261028) are published. Seeds1/2 were evaluated
on T4 and seed0 on RTX3090, so matched precision is not a claim of identical
hardware. All three seeds improve calibration over the reported control0.4052,
while only seed1 improves the held-out test. These gains and regressions are
retained together. The published control is rounded and the three seeds share
one split; no confidence interval across independent test datasets is inferred.

## Repository contents

- `PIPELINE_DESIGN.md` — reproducible design and assumptions.
- `RULES.md` — competition constraints and final-deliverable checklist.
- `scripts/` — inspection, validation, and submission helpers.
- `tests/` — contract tests that run without private competition data.
- `experiments.csv` — an auditable record of local experiments.

![Instance quality](assets/instance-quality.png)

## Recommended workflow

1. Download official data from Kaggle into an external ignored data directory.
2. Establish a deterministic train/validation split and preserve provenance.
3. Compare a strong segmentation baseline against morphology-aware post-processing.
4. Tune instance separation on Panoptic Quality, not Dice alone.
5. Validate submission format and runtime before upload.

![Robustness view](assets/robustness.png)

## Scientific limits

This is a competition engineering scaffold, not a solar-physics validation study. Generated figures are explanatory artwork, not observations. No competition images or labels are redistributed here.

## License

MIT for the original code. See [LICENSE](LICENSE). Dataset and observatory material remain under their source terms.
