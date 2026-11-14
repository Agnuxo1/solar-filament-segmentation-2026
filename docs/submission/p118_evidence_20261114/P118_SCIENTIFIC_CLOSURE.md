# P118 — Public release parity and pre-freeze provenance closure

**Executed:** 14 November 2026 at 12:10:15 UTC (Europe/Madrid 13:10:15).  
**Result:** PASS — 16/16 deterministic gates, 0 failures, 2.274 seconds.  
**Frozen source:** `97f1b08703c90196fcd6f89e9e9988ba7aeb979f`.

## Scientific question

Do the publicly downloadable notebook, R2 report, source archive and published P117 audit retain the exact bytes documented in the successful clean P106 replay and the acknowledged organizer submission, while the three reference CSVs remain unmodified?

P118 was preregistered before downloading or analyzing the frozen release files. The protocol, exact executable source and machine-readable receipt are archived alongside this closure. These tests are deterministic **supply-chain and provenance checks**, not segmentation-accuracy tests.

## Observed evidence

| Gate | Observation |
| --- | --- |
| Seven frozen local SHA-256 references | 7/7 exact |
| Four downloaded public SHA-256 references | 4/4 exact |
| ZIP source integrity | 10 entries; CRC integrity passes |
| P106 notebook cell source hashes | All 8/8 code-cell hashes match P106 execution receipt |
| P106 final metadata | All required execution and no-TEST/no-upload flags match |
| R2 public revision manifest | R2, four pages as **reported by manifest**, expected PDF and ZIP digests |
| Public organizer receipt | H9b digest, R2 digest and update acknowledgment referenced consistently |
| Overall outcome | **16/16 checks PASS**, no exceptions |

The eight notebook cells have positions `1, 3, 5, 7, 9, 11, 13, 15`. The released notebook's SHA-256 is `317b1781101fd6652b21b3f86d297b27136bd5c61ef7addfc8a904a7776314ad`, matching the notebook used by P106. P106's documented successful completion on 13 November at 14:39:48 UTC remains distinct from P118's CPU-only source verification. The PDF's actual page count was **not** newly independently rendered or counted by P118; it was checked against the existing R2 manifest.

The accepted H9b CSV remains `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`. The P103 and P106 CSVs retain their documented digests `8ae3160af9bddd5ac3ef914bc807da71e3e9cbad0444cc055d2ff0056cdfe2ba` and `16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3`. No equality of predicted masks should be interpreted as ground-truth accuracy or as a Panoptic Quality score.

## Evidence chain

- Frozen preregistration SHA-256: `ba605467dd80ba1628f0980cec43b130c431e9f740c5fb7b1f40e82b02f09596`.
- Executed Python source SHA-256: `4f53d1356304f19248c6ecd6d90b9d09545b3cf28673fb2aa054b24de824a52c`.
- Final receipt SHA-256: `101767caae7bb185984e469146e6c84912baddee0556ef763c1645b9739f201d`.
- The three published files were each re-downloaded from the public repository and independently verified to match the original server-side SHA-256 after correcting a missing terminal newline in the first publication pass. No scientific values changed in that correction.

## Decision and limits

**Keep H9b** (Kaggle submission `58112557`, previously verified public PQ `0.39`) as the saved final candidate. The score is historical and P118 produced no new Kaggle metric. P118 did not access an authenticated Kaggle leaderboard, submit a CSV, edit Google Forms, alter the original or R2 reports, run model inference, train, touch either GPU, access TRAIN/TEST annotations, or consult JEV/Goldfish. The P108 out-of-fold provenance block on the proposed 1.15x TRAIN ablation remains in force. No independent reviewer consultation is claimed.

This supplementary release proof strengthens the integrity and reproducibility evidence already supplied, but does not amend the organizer's recorded response. Keep submission changes frozen at 00:00 UTC on 15 November, six hours before the 06:00 UTC competition deadline.
