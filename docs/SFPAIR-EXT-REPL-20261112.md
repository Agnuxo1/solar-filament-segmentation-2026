# SFPAIR-EXT-REPL-20261112: independently seeded small-filament sampling study

**Status as of 12 November 2026, 12:13 UTC: training in progress; not a Kaggle submission.**

This addendum documents a paired ResNet34 U-Net crop-sampling experiment and
its independently seeded replications using only the supplied MAGFiLO/GONG
training images. It does not replace our verified best official
[Kaggle submission 58112557](https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard),
which scored public PQ **0.39** on 8 November.

**Scientific namespace:** This run is called `SFPAIR-EXT-REPL-20261112`.
The unrelated P59 (reliability classifier), P60 (Tversky), P61 (soft-clDice)
and P62 (postprocessing) studies in the main release retain their existing
identifiers and results. We deliberately avoid assigning those numbers to
these paired sampler runs.

## Hypothesis and experiment

The controlled intervention is preferentially centred sampling of small
human-annotated filament regions when extracting 512x512 training crops.
The control uses standard annotation-box-centred sampling. Both models share
their initial label-independent weights and optimizer family and train on
the same 446 physical F1 training images, excluding all F1 calibration
and test labels. Both arm outputs are retained at the preregistered step
budgets of 3,600 and 7,200 updates (batch 8).

The original paired run, seed 20261112, **completed 7,200 steps** on
12 November. Its final checkpoint SHA-256 is

`0a2bab35a6b84b79c3d2711ed74a3c40b73789b6a83ee3a76461ce42756b681b`.

Its step 3,600 checkpoint SHA-256 is

`cc990330b99c21fc68e1d9d903ad6eda72ad34bb118ee8cf5b3126acf37f603e`.

Three independent-seed paired replications (20261113, 20261114,
20261115) are preregistered with the same input split and
7,200-step budget. They test sensitivity to stochastic sampling.
Completion of a run or a low training loss is not a PQ improvement.

## Leak-free evaluation

An independent read-only evaluator derives official 2048x2048 instance
masks from the competition's published **training** COCO annotations,
separating annotators of the same physical observation only at the
*evaluation* step. The frozen F1 split contains:

| Partition | Physical images | Annotation records | Used to train this pair? |
|:--|--:|--:|:--|
| F1 train | 446 | 732 | Yes, mask-derived foreground only |
| F1 calibration | 117 | 190 | No |
| F1 held-out test | 139 | 223 | No |

A five-day temporal exclusion boundary is preserved. The complete
source data include other purged observations deliberately excluded
from these three groups.

The postprocessing policy is selected **only on calibration**
using the step-3600 control arm, then frozen before reporting any
comparisons against the F1 held-out test. The evaluator counts
IoU matches over 0.5, TP, FP, FN and summed IoU separately per
annotator record at native resolution; uncertainty is resampled
by physical image.

**Promotion is not automatic:** an SFPAIR candidate must not only
improve upon its own control (delta > +0.005 with paired bootstrap
95% interval excluding zero), but also beat the independently
frozen and substantially stronger H9b baseline. Evidence across
several temporal folds, a verified single-image inference pipeline,
an official-format 2048px COCO RLE CSV and remaining submission quota
are additional requirements.

**No SFPAIR PQ is claimed yet**. In particular this entry must not
be represented as an official Kaggle score.

## Checkpoint and delivery policy

The run preserves independent checkpoints, signed manifests, logs and
source snapshots. New experiments only address preregistered hypotheses;
checkpoint existence is never treated as model validation. The final
four-page report, complete runnable notebook, authorized weight
distribution and organizer's Google Form receipt have priority over
additional GPU work before the 15 November 2026 deadline.
