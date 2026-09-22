# Preparation report — 2026-09-21

## Scope

Prepared `filament-segmentation-2026` for the first JEV-selected experiment. No
model training and no Kaggle submission were performed.

## Commands and evidence

- Confirmed the competition slug with `kaggle competitions list -s filament`.
- Confirmed the account has zero previous submissions with
  `kaggle competitions submissions -c filament-segmentation-2026`.
- Downloaded the official archive with `kaggle competitions download`.
- Archive SHA-256: `56d8cd2927859fa7df72bd28ad1586f53c5814df6f81db02e755223e430657df`.
- Pulled the public self-evaluation notebook from
  `azimahmadzadeh/self-evaluation-notebook`.
- Added `scripts/evaluate_pq.py` and `scripts/audit_submission.py`.
- Added synthetic tests; result: `2 passed`.
- Evaluated the empty local baseline: `PQ=0.0`, `8199` false negatives,
  `0` false positives.
- Verified 707 train JPEGs and 180 test JPEGs, all `2048x2048` grayscale.
- Verified 1,154 COCO image records, 8,199 annotations, and 447 repeated
  physical file names; grouped validation must use the physical file name.
- GPU detected: NVIDIA GeForce RTX 3090, approximately 24.4 GB free VRAM.

## JEV gate

JEV was consulted twice at the preparation-to-experiment transition using the
subscription-backed Codex CLI. Both calls returned
`MODEL BACKENDS TEMPORARILY UNAVAILABLE` after the 30-second CLI timeout.
Telemetry:

- `E:/Rescate-C-2026-09-21/Documents/JEV-Orchestrator/telemetry/20260921-192442-14668-072730000-cli.jsonl`
- `E:/Rescate-C-2026-09-21/Documents/JEV-Orchestrator/telemetry/20260921-193809-52864-366552100-cli.jsonl`

No hypothesis, model, or Kaggle submission was selected without JEV. The next
action is to retry JEV once the Codex subscription backend is available.

The requested `utilidades_herramientas` file was not found on the searched
project, Documents, D:, or E: paths. The attached instructions and installed
local tools were used instead. The named `typesafe-ai` skill is not available
in the current skill catalog.
