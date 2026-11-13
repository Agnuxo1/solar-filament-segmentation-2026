# P94 - Genuine Kaggle-upload mask parity diagnostic (preregistered 13 Nov 2026)

## Provenance and hypothesis
The accepted submission 58112557 has SHA-256
`24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`.
The frozen stage-two replay (`67f02e4d...`) is **not** the accepted upload.
P91 finished on 13 November at 03:19 UTC; its registered comparison targeted
the stage-two file, not the accepted upload, and showed 326 distinct RLE rows.

Hypothesis: some differing compressed RLE strings encode nearly identical
binary masks; if so, deviations are likely numerical/threshold-related rather
than wholesale replacement. This is a *descriptive provenance test*, not a
PQ improvement experiment or a causal attribution.

## Frozen protocol (before execution)
- Compare the immutable **genuine upload** with the P91 CPU FP32 assembled
  result by filament_id; reject nonmatching IDs, duplicate rows and invalid RLE.
- Decode or compare each mask through pycocotools using exactly 2048 x 2048
  pixels; report paired IoU, area differences, image counts and exact coverage.
- No label images, MAGFiLO test annotations, hidden leaderboard scores,
  selector re-fitting, Kaggle submission, model inference or GPU use.
- Verify reference SHA-256 and retain a machine-readable receipt on HDD.
- The self-test must prove mask matching, one-pixel modifications and rejection
  of duplicates. Publish negative as well as positive observations.
- Stop after the audit; any follow-on experiment requires a separate preregistered
  hypothesis and budget.

## Budget and decision rules
CPU <= 20 minutes, RAM <= 2 GiB, HDD <= 10 MiB. A near-unity mask IoU
distribution supports *numerical closeness* only; low IoU suggests material
geometry differences. Neither permits promoting a new Kaggle submission.
