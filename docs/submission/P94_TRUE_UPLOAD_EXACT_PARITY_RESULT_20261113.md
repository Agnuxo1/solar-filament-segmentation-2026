# P94+ßuÁ‚ùÁT Exact reproduction of the actual uploaded Kaggle CSV (13 November 2026)

## Verified breakthrough: 1,241 of 1,241 rows are byte-identical

The P91 CPU-FP32 refiner replay of the **frozen H9b no-site crop files**
generated the CSV with SHA-256:

`24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`.

This **exactly equals** the independently archived genuine Kaggle upload
(submission ID 58112557, public PQ 0.39), including CSV byte ordering and
newline encoding. The direct `cmp` check succeeded. P94 independently
compared all 1,241 filament IDs and 1,241 COCO RLE strings and found
**zero discrepancies across zero images**. Neither labels nor Kaggle's
hidden feedback were used.

The audit receipt is `P94_TRUE_UPLOAD_EXACT_PARITY_RECEIPT_20261113.json`.
The immutable local receipt has SHA-256:
`6ef088a39f3120cb38d6abded347ee1008e805a8fd699acd14cc1fdbec56f824`.

## Why the earlier P91 report incorrectly implied a failure

P91's frozen script compared the newly produced, correct CSV against
`final-release-staging-20261112/submission_58112557.csv`,
SHA-256 `67f02e4d...`. That file was a **stage-two intermediate**,
not the genuine upload. Thus its reported 326 differing RLE strings
were real *relative to the intermediate* but were incorrectly interpreted
as a failure to reproduce the genuine submission.

The P91 log at 03:19:17 UTC already recorded `sha256=24b3ea1c...`
for the newly generated CSV. P94 confirmed this evidence with a fresh
independent source-to-source comparison at 04:04:46 UTC.

This is a correction of **reference identity**, not a new model score or
a Kaggle submission. We retain the historical protocols and receipts
unchanged to avoid retroactive alteration of the preregistered record.

## Precisely bounded reproducibility claim

**Proven:** frozen H9b no-site crop archives + the two original refiner
checkpoints, executed on CPU in FP32 via the public `h9_infer.py` and
`h9_official.py paste`, produce the exact Kaggle-uploaded CSV.

**Also previously proven:** P88 independently reproduced the frozen
original detector proposals in CPU FP32 across the 720 paired model/images.

**Not yet proven end-to-end:** rebuilding the exact frozen crop archives
and no-site H6 selection *from the original JPEG images* in a clean,
publicly reproducible environment, without relying on any precomputed
inference-stage input. This remains the next final-notebook gate.

The public result warrants preserving submission 58112557; it does not
justify promoting experimental F1-F4 treatments, whose local effects
were inconsistent.

## Reproduction commands, on the authorized research server

Using the archived originals under `paired-small-20261112`:

```bash
sha256sum p74_accepted_artifact_audit/accepted_h9b_nosite.csv
sha256sum p91_cpu_fp32_refiner_20261113/assembled/submission_h9b.csv
cmp p74_accepted_artifact_audit/accepted_h9b_nosite.csv p91_cpu_fp32_refiner_20261113/assembled/submission_h9b.csv
python scripts/audit_p94_true_upload_masks.py --self-test
```

Run the last command from the public repository checkout. The original
P91 script and its preregistration document the precise CPU inference,
checkpoint validation and RLE generation. Model weights are not retrained.
No test annotations are required.

## Submission status and priority

The four-page report exists. The required notebook must still demonstrate
original-JPEG-to-upload regeneration independently. The mandatory organizer
Google Form has **no verified submission receipt**. Finalization of both
is more urgent than another training experiment. Freeze target:
15 November 2026 00:00 UTC, six hours before the stated competition close.
