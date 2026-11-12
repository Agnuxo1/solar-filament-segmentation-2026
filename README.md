# Solar Filament Segmentation Challenge 2026

![Solar filament segmentation research cover.](docs/figures/cover.gif)


**Project citation:** [Francisco Angulo de Lafuente](https://orcid.org/0009-0001-1634-7063) · [CITATION.cff](CITATION.cff) · [Software metadata](codemeta.json).


**Verified competition status (12 November 2026):** the highest accepted public Kaggle score is **0.39 Panoptic Quality**, submission [58112557](https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard), using the pixel-only H9b pipeline (8 November 2026). The earlier 0.35 figure in the historical 28 October results below is not the current best. A new 3,600-step single-GPU small-filament sampling ablation completed with **+0.0110 local PQ** versus its matched semantic-model control (95% CI +0.0048 to +0.0168), but **−0.0915 local PQ** versus the stronger H9b baseline and therefore **did not pass the promotion gate**. [Full negative-result and evidence report](docs/2026-11-12-paired-small-instance-sampling.md). No higher official score is claimed.


![Conceptual three-stage solar filament workflow](assets/filament-workflow-concept.gif)

*Conceptual, animated workflow illustration only: synthetic curves show observation → candidate mask → separated instances. This is **not** a telescope image, model output, measured metric or leaderboard result. [Static PNG alternative](assets/filament-workflow-concept.png) · [Generation script](scripts/generate_readme_animation.py).*

![Solar filament header](assets/header.png)

An open-source preparation and evaluation pipeline for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026). The project focuses on precise, instance-aware masks for H-alpha observations and audits fragmentation, over-merging, and runtime because the challenge uses Panoptic Quality rather than a plain pixel-overlap score.

![Segmentation pipeline](assets/segmentation-pipeline.png)

<!-- impact-note:start -->
<sub><strong>Efficiency and practical value.</strong> The repository reports an evaluator fix reducing retained polygon buffers from 156 MiB to 0.424 MiB on three real annotation records while preserving PQ, TP, FP and FN exactly. This avoids retaining full image arrays for small crops and supports more memory-conscious evaluation. The <a href="pipeline/results/memory_fix_verification.json">recorded comparison</a> concerns array buffers, not whole-process memory, energy consumption or improved model accuracy.</sub>
<!-- impact-note:end -->

## 28 October: real training results and an applied memory fix

![Held-out Panoptic Quality at 1024 px: NeuroPixel control: 0.4015; U-Net seed 0: 0.399921; U-Net seed 1: 0.403124; U-Net seed 2: 0.398986](docs/figures/fig-01.gif)

*Held-out Panoptic Quality at 1024 px.* [Static figure](docs/figures/fig-01.png).



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

![From design to a submission contract. Design: Document assumptions; Data rules: Verify permitted inputs; Scripts: Inspect and validate; Tests: Exercise file contracts](docs/figures/fig-02.gif)

*From design to a submission contract.* [Static figure](docs/figures/fig-02.png).



- `PIPELINE_DESIGN.md` — reproducible design and assumptions.
- `RULES.md` — competition constraints and final-deliverable checklist.
- `scripts/` — inspection, validation, and submission helpers.
- `tests/` — contract tests that run without private competition data.
- `experiments.csv` — an auditable record of local experiments.

![Instance quality](assets/instance-quality.png)

## Recommended workflow

![Measure split and merge errors. Held-out groups: Freeze assignments; Fit baseline: Keep inputs outside Git; Tune separation: Use Panoptic Quality; Validate format: Record runtime](docs/figures/fig-03.gif)

*Measure split and merge errors.* [Static figure](docs/figures/fig-03.png).



1. Download official data from Kaggle into an external ignored data directory.
2. Establish a deterministic train/validation split and preserve provenance.
3. Compare a strong segmentation baseline against morphology-aware post-processing.
4. Tune instance separation on Panoptic Quality, not Dice alone.
5. Validate submission format and runtime before upload.

![Robustness view](assets/robustness.png)

## Scientific limits

![Separate observed evidence from diagrams. Illustration: Explanatory only; Local tests: Checks file contracts; Scientific evidence: Needs real observations](docs/figures/fig-04.gif)

*Separate observed evidence from diagrams.* [Static figure](docs/figures/fig-04.png).



This is a competition engineering scaffold, not a solar-physics validation study. Generated figures are explanatory artwork, not observations. No competition images or labels are redistributed here.

## License

![Reuse and citation. Original code: MIT notice applies; Observatory data: Keep source terms; Cite the work: Use CITATION.cff](docs/figures/fig-05.gif)

*Reuse and citation.* [Static figure](docs/figures/fig-05.png).



MIT for the original code. See [LICENSE](LICENSE). Dataset and observatory material remain under their source terms.

## Bibliography and research directions

![How the references inform follow-up. Primary sources: Resolve DOI or arXiv ID; Study design: Predefine comparisons; Evidence to report: Separate hypotheses and results](docs/figures/fig-06.gif)

*How the references inform follow-up.* [Static figure](docs/figures/fig-06.png).



The following primary sources inform proposed checks. Their text, code, figures, and data are not included in this documentation update.

- [Kirillov et al. (2019), Panoptic Segmentation](https://arxiv.org/abs/1801.00868). Report segmentation and recognition components alongside Panoptic Quality to diagnose merging and fragmentation.
- [Ronneberger et al. (2015), U-Net](https://arxiv.org/abs/1505.04597). Use a fixed encoder–decoder baseline before attributing gains to instance-aware post-processing.

**Attribution and reuse.** Please credit [Francisco Angulo de Lafuente](https://orcid.org/0009-0001-1634-7063) in academic and software reuse and do not present this work as your own. When distributing copies or substantial portions of the original MIT software, preserve the copyright and permission notice. Upstream material keeps its own terms. [License](LICENSE) · [Figure methods and provenance](docs/figures/README.md).

<!-- research-provenance: project=solar-filament-segmentation-2026; date=2026-11-04; author=Francisco Angulo de Lafuente; orcid=https://orcid.org/0009-0001-1634-7063; original-software-license=MIT -->

## November 2026 submission evidence and report

The verified best official submission is **58112557**, which scored **public Panoptic Quality 0.39** on 8 November 2026.

- [Four-page technical report (PDF)](docs/submission/TECHNICAL_REPORT_4P_20261112.pdf) — an evidence-based scientific report, not proof that the final organizer form was submitted.
- [H9b GPU reproducibility and true uploaded CSV parity audit](docs/2026-11-12-h9b-replay-precision-and-submission-parity.md) — distinguishes the accepted Kaggle upload from intermediate stage-two artifacts, and quantifies small numerical differences.
- The full release still requires a self-contained end-to-end notebook with all legally distributable weights and a clean-checkout verification. Do not claim exact byte-identical reproduction of the accepted artifact from intermediate files.

