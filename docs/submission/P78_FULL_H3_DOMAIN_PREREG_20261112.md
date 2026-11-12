# P78: full-domain F1 H3 byte-parity audit (preregistered)

Registered on 12 November 2026, before comparing the complete 180-image P77 output with the earlier private P63 F1 output.

Purpose: test whether the *public-source* F1 H3 U-Net stage-one probability maps reproduce the earlier private P63 stage-one maps at native 2048 x 2048 resolution across **all 180 preselected test images**, without using any labels, calibration data, leaderboard information, or changing the model.

Input A: immutable P77 public-source F1 180-map replay and its per-image SHA-256 receipt. Input B: immutable P63 F1 180-map private-source baseline, same original unlabelled test images. Strictly require exact filename sets and 180 unique grayscale uint8 maps. The test reports all pixel counts and per-image deviations.

Preregistered descriptive consistency gate: at least 99.99 percent byte-identical pixels across all 180 images AND maximum absolute byte difference no larger than 1. This is a *quantization parity* gate, not a guarantee of end-to-end H9b CSV reproduction or unchanged PQ. A failure triggers investigation rather than output rewriting or promotion.

Budget: CPU-only, less than 2 GiB working memory, 15-minute runtime cap, outputs and logs on NAS HDD only; GPU0/1 unchanged. One run, no repeated tuning, SHA256 integrity check on P77 files, immutable result receipt. Independently verify parity math with a self-test before the full run.

Expected output: P78_F1_H3_180_PARITY_RECEIPT_20261112.json containing all 180 per-image errors, aggregate exact equality, maximum absolute byte error, source hashes, and an explicit no-label/no-official-score/no-final-CSV statement. If successful, publish a succinct English report and verifier in the public GitHub repository. Do not claim JEV/Goldfish consultation unless actually performed.
