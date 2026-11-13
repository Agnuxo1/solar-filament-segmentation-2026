# Recommended single follow-up: fixed extra-context refinement on TRAIN

**Status: proposal only. Not executed or registered as a completed experiment.** A protocol identifier and immutable commit/hash must be assigned before execution. It is to run only after the clean P106 notebook gate, if suitable out-of-sample TRAIN artifacts already exist.

## Rationale

P103's remaining discrepancies are small reproducibility differences, not labelled prediction errors: 131 matched masks differ by a combined 268 pixels. They do not justify tuning against the accepted CSV. Separately, the documented TRAIN-derived H6 diagnostic found 11.2% of reference instances with best IoU in (0.3, 0.5], while permissive extra-proposal policies produced many false positives. One bounded experiment should therefore test an improvement to existing mask geometry without proposing extra candidates.

The hypothesis is that an additional, modestly wider context for the existing crop refiners can improve boundary placement and recovery of thin continuations, including candidates near the IoU matching threshold. P103's affine sensitivity makes a fixed context ablation reasonable, but supplies no evidence that the wider context is more accurate. The test may fail.

Reference for the labelled diagnostic: the released four-page technical report, Morphology and uncertainty / Controlled ablations sections, at public repository commit `6843362b964dd62a53f19c975e777664c8ea2245`.

## One locked intervention

The only candidate is **1.15 times the current crop context**, keeping input dimensions 128 x 384, orientation, four channels, both refiner architectures, original weights, and four reflection views unchanged. No training is performed.

For an existing source-to-crop affine transform M, scale its crop coordinates by 1/1.15 about the crop centre to obtain the expanded-context transform. Re-extract all four channels from the same frozen image/probability/seed inputs. Infer both refiners once for this extra context. Inverse-warp its averaged probability and the original-context averaged probability into the native image using the frozen interpolation and zero fill. Blend these native probability maps 50/50, then apply the unchanged 0.40 threshold, seed-touching connected-component rule, original score order, and final disjoint-mask assembler.

There is no scale search, threshold search, architecture change, new selector fit, extra object proposal, or retuning after viewing results. The baseline is the corresponding frozen H9b out-of-sample prediction on exactly the same images. Pipeline precision and all code/configuration hashes must be recorded.

## Data and leakage gate

Use only the existing, frozen TRAIN-derived F1 calibration cohort (117 physical images, 190 annotation records) and its reserved 139-image cohort. Keep all annotators of a physical image together. These cohorts have been examined in previous experiments, so this is an explicitly exploratory, preregistered ablation on previously inspected data, not a new blind test.

Before inference, verify physical-image membership against the training manifests of **every** model and selector used for that image. Both arms must use genuine held-out predictions/checkpoints. The released `ALL` refiners or an ensemble containing a model trained on a scored image are not acceptable substitutes. Required baseline outputs, upstream maps, and suitable refiner checkpoints must already exist and have auditable hashes. If they do not, abort the proposed short experiment; do not silently train a new family or report in-sample PQ as validation.

The competition's 180 test images, their accepted prediction CSV, and leaderboard scores are excluded from the experiment. No selection of cases is based on P103's disagreement list.

## Metric and stopping rule

Compute the organizer's global, record-wise PQ with IoU strictly greater than 0.5, pooling the numerator and TP/FP/FN counts across annotator-image records. Reuse the same verified scorer for both arms. Report paired differences, matched-IoU changes, TP, FP, FN, and the existing area strata. The primary outcome is the aggregate PQ change; do not substitute mean image PQ.

First evaluate the single frozen candidate on the calibration cohort, without changing parameters. Stop for nonpositive calibration delta or any invalid provenance. Otherwise evaluate the unchanged candidate once on all 139 reserved physical images.

The candidate qualifies for a promotion review only if all of the following hold on that reserved cohort:

- Absolute aggregate PQ improvement is at least **+0.005**.
- The lower endpoint of a **95% paired bootstrap interval** for PQ difference is greater than zero, using 2,000 resamples of physical images with all annotation records kept together and a fixed seed.
- TP does not decrease and FP increases by no more than 1% relative to the frozen baseline.
- All images and rows are valid, the resource cap is met, and no settings were selected from the reserved-cohort results.

Report all results if the gate fails; do not try a second scale. The bootstrap interval is conditional on these previously inspected cohorts and does not eliminate the effect of repeated historical experimentation. A passed local gate still does not establish a Kaggle improvement.

## Short budget

Hard ceiling: **45 minutes of additional CPU wall time, 6 GiB RAM, 2 GiB of new HDD outputs, zero GPU time, zero training**. Reuse existing baseline and upstream outputs. Include a small fixed-shape CPU check before spending the main budget. If the full paired comparison cannot finish within the cap, record it as incomplete and do not promote. Any later official-test inference or Kaggle submission is a separate action after review of the completed evidence.

## Deliverables

One immutable input/model manifest, one frozen intervention configuration, per-image paired scorer sufficient statistics, an aggregate result with the bootstrap interval and all gate outcomes, runtime/resource evidence, and a short interpretation. Preserve the accepted H9b submission unless the independent promotion review approves a tested candidate.

