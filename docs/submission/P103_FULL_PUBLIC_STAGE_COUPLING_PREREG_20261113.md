# P103 — Full 180-image independent-public-stage H9b coupling, registered before run

## Prior evidence and locked hypothesis
P102 on the first four fixed official test JPEGs reproduced 3/4 image RLE lists (14/15 rows), with a single true numerical mismatch; every component ID, score, and crop geometry matched, while 179 crop tensor bytes differed. No test labels or leaderboard tuning were used.
The P103 follow-up is the **full-domain expansion explicitly permitted in the P102 preregistration**, without modifications to model hyperparameters, no-site selector, RLE assembler, threshold or inference precision. Hypothesis: the prevalence of final-mask disagreement across all 180 images can be measured and separated from component selection instability. No presumption of exact output equivalence or PQ improvement.

## Method and split discipline
Use official 180 TEST JPEGs strictly without annotation access. Consume the P77/P79 independently replayed public H3 maps (4 x 180), P88 publicly reexecuted Mask R-CNN proposals (4 x 180 x 2 views), published P99 consensus, frozen selector trained on Kaggle TRAIN data, and two released refiners in CPU FP32. The first four P102 crops and refiner outputs must be reused by immutable hashes and hard links; do not recompute them. Compute remaining 176 images and compare against the genuine archived upload only as a reproducibility diagnostic.
Compare each generated crop with the frozen historical crop as an **audit only**, not an inference input. The genuine upload is never a training/tuning target. Count all fallback rows separately: if archival control predictions are needed, the result cannot claim complete independent end-to-end reproducibility. Report unchanged all negative results.

## Checkpoints and safety budget
CPU-only; CUDA_VISIBLE_DEVICES empty; CPU quota 150%; RAM <=6GiB; 4h wall-clock; <=4GiB incremental HDD; preserve >=20GiB HDD reserve. Write only to /srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112/p103_full_public_stage_coupling_20261113. Use per-image write-once receipts, immutable hashes, interruption recovery, and a single flock lock. Do not touch GPU1 or spawn CUDA on GPU0. No Kaggle submission, no TEST annotations, no leaderboard optimization.

## Acceptance criteria
Complete 180 unique images, correct CSV schema and instance identifiers, hash-audited upstream and model weights, compare RLE lists per image to authentic upload with counts of exact rows, component/score/geometry/crop mismatches, explicit fallback count and timing, publication of negative outcomes. Keep the original Kaggle baseline 0.39 unless independently verified improvement exists. Mandatory final four-page report, notebook, and Google Form remain separate tasks.