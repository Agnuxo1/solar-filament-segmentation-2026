# P97 — Preregistered public detector aggregation parity

Registered 2026-11-13 UTC before running the aggregation comparison.

Question: Can four independently reconstructed public-source FP32 CPU
Mask R-CNN detector outputs (P88: 180 original test images, 2 views, F1-F4)
reconstitute the exact historical unlabelled official aggregate
dets_official.pkl without any historical prediction cache as an input?

Freeze two natural, source-independent flattenings:
(a) fold-major then view-major and (b) view-major then fold-major.
For the first four sorted filenames, compare original proposal counts and
exact indexed (score, box, uint8 mask pixel array). Do not sort by score,
fit parameters, permute proposals, compare labels or use leaderboard.
If neither simple flattening matches, record the negative outcome and
inspect source construction separately before new experiments. If one
matches all four pilot images, continue full 180 in a new preregistered
phase, keeping original detector cache read-only and validating P88 status.

Resources: CPU only, <=6 GiB RAM, <=2 GiB HDD, <=2 hours, no CUDA;
no changes to GPU1 or other tasks. Output is a diagnostic, not a Kaggle
submission and not a fully verified original-image-to-final-CSV run.
