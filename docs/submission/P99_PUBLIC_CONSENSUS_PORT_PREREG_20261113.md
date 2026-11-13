# P99 — Public deterministic consensus port (preregistration)

**Registered:** 2026-11-13 UTC before executing this study.

## Motivation

P98 established exact 180/180 parity for the archived, non-public Mask R-CNN consensus implementation, rebuilding the frozen detector aggregate without any private prediction cache as a generation input. The published P98 audit script still imports an archived original script, so it is not independently executable on a clean checkout. A self-contained public implementation is required for the end-to-end inference notebook.

## Frozen hypothesis and method

- Reimplement only the original four-stage rules with frozen values: prefilter score >= 0.30; uint8 soft mask binarized at >127; support by a proposal from a different view with mask IoU >= 0.50; unsupported score multiplied by 0.75; descending-score greedy mask-NMS with IoU >= 0.50.
- The public port must preserve original ordering, tuple values and numeric comparison semantics. No approximation, sorting tie-breaks or parameter tuning.
- Pilot exactly the first 4 lexicographically ordered unlabelled test JPEGs; if all 4 match the archived original consensus and historical aggregate, audit all 180 without modifying the port.
- Require exact candidate-original proposal sequence including scalar fields and byte-identical mask arrays. Store SHA-256 hashes for archived reference, public port and P88 receipt, plus per-image discrepancies.
- Test positive and deliberately corrupted masks/scalars. If the pilot fails, record the result and stop; no post-hoc fixes based on hidden test labels or leaderboard.

## Limits and scientific controls

CPU only, no CUDA; <= 6 GiB RAM, 150% CPU, <= 2 GiB HDD, <= 3 hours. Do not touch GPU1 or processes belonging to other projects. No labelled test data, no hidden leaderboard optimization. Reuse only the documented P88 generated detector view predictions, not the historical aggregate as a generation input.

## Acceptance and limitations

Successful 180/180 exact parity authorizes publishing `scripts/official_consensus.py` as a standalone library used in the final inference code. It does not establish whole-JPEG-to-submission equivalence because the P96 H3 numerical issue and end-to-end cold notebook test still need independent verification. Prioritize final report and required form before the competition deadline.
