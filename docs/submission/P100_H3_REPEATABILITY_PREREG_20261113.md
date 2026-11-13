# P100 — Preregistered GPU0 H3 inference repeatability audit

Date: 2026-11-13 UTC, frozen before new inference.

## Motivation and hypothesis

P80 reports 8,726 one-byte pixel discrepancies over 180 F1 images between the
original historical H3 maps and the released public inference replay. P96
demonstrates that small H3 differences propagate into no-site crop tensors
although selected component IDs remain identical. P100 asks whether the
released CUDA/AMP pipeline is repeatable on the same assigned GPU, before
experimenting with alternate precision or backend settings.

**H0:** repeat A and repeat B are pixel-identical for all preregistered inputs,
and match P77/P79 public replay baselines. Differences from the original
historical maps would then not be explained by ordinary run-to-run variation
under the released environment.

**H1:** at least one pixel differs between A and B or the older public baseline,
suggesting runtime nondeterminism that needs further controlled investigation.

## Fixed experiment plan

- Inputs: first four original, unique sorted official Kaggle TEST JPEGs;
  image IDs selected by lexical order only, no labels.
- Models: F1, F2, F3, F4 released ResNet34 U-Nets, exact SHA-256 manifest.
- Conditions: A and B are separate invocations using the **unchanged** public
  `replay_h3_from_original_images.py`; one process at a time.
- Device: exclusively GPU0 UUID
  `GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14`.
- Guard: non-overlapping GPU0 compute processes verified using nvidia-smi
  immediately before each process; `supervisor.lock` held by flock; GPU1
  unmodified. Refuse to run if GPU0 is already occupied.
- Historical comparison: prior public replay outputs in P77 (F1) and P79
  (F2–F4). Do not use competition labels or hidden scoring.
- Primary endpoint: exact uint8 map parity count and number of changed pixels
  across 16 image/model pairs, A vs B.
- Secondary endpoint: A/B vs older public replay, with maximum byte difference,
  SHA-256 metadata and per-image receipts.
- Advancement: if exact, investigate remaining historical discrepancy via a
  separately preregistered precision/backend study. If not exact, document
  nondeterminism and test fixed deterministic backends next.
- Storage: NAS HDD only, outputs at most 200 MiB, CPU RAM limit 6 GiB;
  GPU VRAM cap 75% of GPU0. Deadline: 20 minutes wallclock.
- Failure policy: fail closed, do not overwrite existing receipts, do not
  relaunch a completed condition, do not touch GPU1 or other CUDA work.
- No Kaggle submission; no new PQ is inferred; no model promotion.

Acceptance: both condition receipts exist for each F1–F4 fold, exactly four
source images each, 16 auditable independent map comparisons, and summary
contains unique signatures of all compared files. Deliberate mismatch
self-tests must pass before publishing a scientific conclusion.

This study is an inference reproducibility diagnostic, not evidence of final
JPEG-to-CSV parity. Final form and standalone notebook are higher priority
than further speculative retraining.
