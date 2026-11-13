# P112 preregistration — source of large crop-pixel drift
Date: 2026-11-13 UTC. Frozen before reading the paired crop values for this hypothesis.

## Scientific question
P111 found 1,577 unequal crop-tensor X values across P103/P106 (180 images), with maximum absolute difference 255. Are large changes simply exact 0-versus-255 endpoint swaps, as might result from a uint8 wrapping or boundary artifact, rather than arbitrary perturbations?

## Predeclared test
Use the frozen P103 and P106 crop NPZ archives, plus the P107 and P111 audit records. Match 180 files strictly by image ID. Validate original CSV SHA-256 values: P103 8ae3160af9bddd5ac3ef914bc807da71e3e9cbad0444cc055d2ff0056cdfe2ba; P106 16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3. No labels will be loaded.

Hypothesis H: **every** change with absolute uint8 magnitude >=128 is an exact endpoint swap (0 to 255 or 255 to 0). Refutation: even one large difference has another value pair. The analysis is descriptive and exploratory regarding causal source; this is not an evaluation of segmentation quality.

Before any analysis freeze this document by SHA-256 and record it in the audit. Require both X inputs to be numeric uint8 arrays with exactly the same shape. Check expected total X element comparisons and difference count (244,580,352 and 1,577) against P111. Abort if frozen source identities, IDs or counts differ.

Report fixed mutually exclusive absolute-difference bins [1], [2-4], [5-31], [32-127], [128-254], [255], directional 0->255 / 255->0, high magnitude non-endpoint pairs with no personally identifying filenames or labels, and image-level high-difference counts stratified by changed-final-image flag from frozen P107. Also report max magnitude and 180-image coverage. Include synthetic positive and negative controls.

## Computational contract
CPU only; CUDA_VISIBLE_DEVICES=-1; max 15 minutes, no training, no external network, no access to Kaggle/private labels, no use of GPU0 or GPU1, write only to project NAS HDD. Sources read-only. Seed 20261113 (no stochastic component). Budget <1 GiB working memory; parse one image at a time.

## Promotion and interpretation
Even if H survives, it does not identify the exact interpolation, dtype conversion, CUDA operator or threshold, and does not establish a PQ improvement. If refuted, preserve that negative result. This audit **cannot** promote a CSV, change submission H9b 58112557, or trigger a Kaggle upload. Promotion requires separate TRAIN-only grouped OOF PQ evidence and authorization under competition rules. Keep P108's OOF provenance gate blocking the proposed 1.15x test.

## Independent checks
Compare observed X diff count to P111; independently verify hashes, full ID coverage and synthetic array examples. Do not represent JEV/Goldfish as consulted unless an actual supported invocation succeeds.
