# P101 — Controlled CuDNN mode sensitivity: benchmark-off and deterministic

**Date:** 2026-11-13 UTC. **Status:** completed with immutable audit.

[Preregistered protocol](P101_CUDNN_H3_PRECISION_PREREG_20261113.md) ·
[Reproduction runner](../../scripts/p101_cudnn_h3_variants.py) ·
[Independent auditor](../../scripts/p101_cudnn_h3_audit.py) ·
[Full machine-readable audit](P101_FINAL_AUDIT_20261113.json).

## Results

The predefined first four official TEST JPEGs were inferred by the original
released F1 ResNet34 model under two controlled CuDNN settings. P100's
public-source CUDA maps (`cudnn.benchmark=True`) served as a frozen baseline,
and P63 original historical outputs were the external numerical reference.

| Variant / comparison (4 × 2,048² maps) | Changed pixels | Exactly matched images | Maximum byte change |
|---|---:|---:|---:|
| Public default vs historical P63 | **68** | 0/4 | 1 |
| CuDNN benchmark off vs public default | **539** | 0/4 | 1 |
| CuDNN benchmark off vs historical P63 | **533** | 0/4 | 1 |
| CuDNN deterministic vs public default | **539** | 0/4 | 1 |
| CuDNN deterministic vs historical P63 | **533** | 0/4 | 1 |

The two proposed variants yielded the **same aggregate statistics**.
A separate bytewise variant-to-variant equality test has not been included in
the primary receipt and is not claimed here; the stored per-image hashes permit
that additional check.

**Interpretation:** disabling CuDNN algorithm benchmarking measurably
changes the GPU AMP H3 output at a small number of pixel quantization
boundaries, but the interventions **increase**, rather than reduce,
disagreements relative to the original P63 maps on this fixed pilot.
Enabling deterministic CuDNN simultaneously did not improve either
aggregate metric. The underlying historical H3 discrepancy remains unresolved.

This is an accuracy-independent *numerical reproducibility* experiment:
there is no segmentation ground truth, no local or official PQ, no Kaggle
submission and no evidence of a better competition model.

## Execution and controls

Preregistration was pushed as commit `762b2d3` before intervention.
The first systemd launch failed instantly with code 69 because the new
runner lacked its executable permission for `flock` re-execution;
**no CUDA workload started** from that attempt. File permissions were fixed,
committed (`0a6596e`), and a new, correctly bounded systemd service
`solarfil-p101-gpu0-retry-20261113.service` executed successfully.

The successful service used only GPU0 UUID
`GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14`; GPU1 was left untouched.
It ran one CUDA process per variant in sequence, with explicit nvidia-smi
occupancy checks, original data SHA-256 checks, a supervisor flock,
10-minute service timeout, 8 GiB RAM cap, and 75% GPU0 VRAM cap.
Both variants completed four images each and the CPU-only independent
auditor passed synthetic corruption checks.

**Final audit SHA-256:**
`8ef420de820ab1ecf671c94df9aa631d0b6f0f3eb88b0ddcaa95d6475d911325`.

## Limitations and delivery priorities

The four-image pilot does not establish all-180-image behavior, prove
historical model fidelity, or show exact raw-JPEG-to-submission-CSV parity.
Do not repeat model training or optimize against leaderboard output from
this result. Further GPU interventions require a new, preregistered causal
hypothesis; the priority is now the raw-input notebook and mandatory
organizer Google Form, neither of which is verified as delivered.
