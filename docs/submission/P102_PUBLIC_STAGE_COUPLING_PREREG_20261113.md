# P102 — Preregistered public-stage coupling into H9b output

**Registration:** 2026-11-13 UTC, before P102 replay.

## Hypothesis and falsification
Independently replayed ResNet34 H3 maps (P79/P80), exact public Mask R-CNN proposals (P88), independently implemented consensus (P99), and the published frozen TRAIN-only H6 no-site selector should generate coherent crop tensors from official unlabeled JPEGs without using historical H3, detector or crop caches. Exact final-mask parity is not predicted: P96 documents numerical H3 differences. Record deviations, never tune to the original upload.

The fixed pilot takes the first four lexicographically sorted original Kaggle TEST JPEGs. Rebuild consensus and no-site crops, independently check crops against archived references, and run both published refiner models in CPU FP32 to compare RLE output against the actual H9b upload as an audit only. The accepted upload is not a tuning target. If validated, expand to all 180 without altering thresholds or algorithms.

## Data and integrity
Use competition TEST pixels for inference only; no TEST annotation access. Frozen selector trained using Kaggle TRAIN labels only, validated by SHA-256. Validate published weight manifest and upstream receipts. Upstream predictions must originate from already executed published-model runs rather than H9b historical output caches.

An optional historical control fallback may be used solely to diagnose historical CSV parity. Any dependency on that prior prediction must be explicitly counted; do not claim fully independent raw-to-CSV reproducibility when fallback is required. Never hide negative results.

## Budget and safety
CPU-only; CUDA_VISIBLE_DEVICES empty; CPU <=150%; RAM <=6 GiB; runtime <=4h; new HDD data <=4GiB, HDD free >=20GiB. GPU0/GPU1 untouched. No Kaggle submission, no TEST labels. All outputs are immutable and resume-safe on /srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112/p102_public_stage_coupling_20261113.

Success: hashes, 4-image crop+RLE diagnostics, correct IDs, reporting of all differences and fallback counts. If an upstream artifact or executable is missing, stop and document it. Explicitly maintain prioritization of notebook and final Google Form.