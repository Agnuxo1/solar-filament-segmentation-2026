# P96 — Preregistered H3 replay to H6 crop coupling (2026-11-13 UTC)

## Hypothesis, rationale and causal boundary
P80 found 32,804 one-byte differences among 3,019,898,880 independently
replayed H3 pixel values, although all four public H3 fold models passed the
predefined near-parity gate. P88 independently proved exact detector proposal
replay for 720 fold/image pairs, and P95 reproduced all 180 H6 no-site crops
exactly when historical detector/H3 caches were supplied. P96 measures whether
substituting publicly regenerated H3 maps changes crop selection, geometry,
candidate score or raster input before the published CPU-FP32 refiners run.

## Design fixed before first P96 evaluation
- Use exactly 180 original unlabeled test JPEGs in lexicographic order.
- Replace historical H3 maps with four original-model, public-source map
  reconstructions from P77 and P79 (each 180 images), validate every source
  PNG against its immutable inference receipt.
- Continue to use the original immutable aggregate Mask R-CNN proposal cache
  for this *controlled single-factor perturbation*. P88's published exact
  detector replay receipt is audited, but P96 is NOT a newly reconstructed
  detector-aggregation experiment.
- Use unchanged H6 no-site selector, frozen metadata, JPEG pixels, crops
  algorithm, fallback rule, thresholds, and crop archives.
- Compare exact X, M, comp and score values and their dtypes/shapes for
  every original JPEG, allowing negative outcomes without post-hoc tuning.
- First 4 sorted images are a predetermined pilot; after successful unit
  tests and asset-integrity checks, evaluate the remaining 176.
- All intermediate/success and error JSON receipts are immutable and each
  references SHA-256 of its original JPEG, reference crop archive and four
  H3 reproduction manifests. Resume existing receipts only after revalidation.
- Label exclusion: never access test annotations or held-out GT, never score
  Panoptic Quality, never optimize from Kaggle public/hidden leaderboard.
- Success criterion for *exact* H3-to-H6 parity: 180/180 identical images,
  all four fields identical. A failure is reported quantitatively, not hidden.
- Resources: CPU only; CUDA_VISIBLE_DEVICES empty; <=150% CPU quota, <=6 GiB
  memory, <=3 hours wall time, <=2 GiB HDD output. Never use SSD /mnt/datos.
- Original assets and reference SHA-256 are read-only; no files are overwritten.
- Do not treat P96 as a complete raw-JPEG-to-CSV regeneration. Even 180/180
  matched crops leaves detector-cache repacking and refiner chain integration.

## After P96
Publish exact/negative results, all hashes and code with a clear limitation
statement. If H3 numerical differences change selected crops, characterize
their effects on the final CSV without fitting post-hoc thresholds or using
test labels. Prioritize the competition form and self-contained notebook
before the freeze at 2026-11-15T00:00:00Z.

Experiment authority: this is a new deterministic, CPU-only, preregistered
reproducibility experiment; it does not create a new Kaggle submission.
