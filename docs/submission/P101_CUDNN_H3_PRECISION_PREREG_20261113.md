# P101 — CuDNN mode sensitivity in H3 map quantization (preregistered)

Frozen 2026-11-13 UTC, before intervention inference.

P100 showed exact current-environment CUDA H3 repeatability (16/16
image-model pairs, zero changed pixels), while the previously published
public F1 H3 images remain up to one byte different at a few pixels from
the original P63 historical reference. The cause has not been identified.

**Question:** Do CuDNN benchmarking or deterministic convolution algorithms
cause the sparse output-byte difference seen in the H3 public replay?

**Hypotheses:**
- H0: both proposed CuDNN variants exactly reproduce the current public
  map and cannot explain historical differences.
- H1: a variant changes one or more output pixels; determine whether
  its difference against historical P63 maps decreases or increases,
  without selecting an official Kaggle candidate based on such parity.

**Frozen design:** first four lexically sorted original Kaggle TEST JPEGs,
fold F1 only; original unchanged released ResNet34 model and exact checkpoint.
Original preprocessing, four-view D4 TTA, AMP and output byte rounding are
identical to `native_model.predict`. Change **only**:
1. `benchmark_off`: `cudnn.benchmark=False`; retain
   `cudnn.deterministic=False` and default TF32 settings.
2. `deterministic`: `cudnn.benchmark=False` and
   `cudnn.deterministic=True`; retain default TF32 settings.

Primary endpoints: uint8 pixel disagreements per image versus frozen
public P100 baseline; secondary: disagreements versus P63 original
historical F1 maps, with hashes, maxima, per-image metrics and totals.

Only one GPU0 UUID `GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14` CUDA
process at a time. Verify nvidia-smi, lock `supervisor.lock`, never
touch GPU1, abort if another compute process occupies GPU0. Two separate
process invocations, maximum 150 seconds each; 10 minutes total; RAM
8 GiB, CUDA VRAM fraction 0.75; HDD output under 150 MiB; idempotent
completed-stage skips. Preserve all outputs and provenance. End before
2026-11-15 00:00 UTC freeze.

Use only original **unlabelled** test JPEGs and public checkpoint. Never
access Kaggle TEST ground truth, tune a model, submit, or infer a PQ score.
Do not promote results based on map-parity alone. Validate deliberate
single-pixel perturbation and metadata checks before drawing conclusions.

If both variants exactly match the public baseline, conclude only that
these CuDNN settings do not explain the observed historical discrepancy
in this four-image pilot. If not, investigate differences under a new
registered hypothesis rather than post-hoc parameter sweeping.
