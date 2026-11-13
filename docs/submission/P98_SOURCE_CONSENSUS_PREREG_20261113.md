# P98 — Historical source-based detector aggregation parity (preregistered)

**Registered:** 2026-11-13 UTC, before executing any P98 comparisons.

## Research question

P97 rejected naïve concatenation of eight independently replayed detector views. The archived and timestamped original `testinf/dump_official.py` has now been located: it calls `make_submission.consensus` on the four fold outputs, each with original and horizontally reversed views, to produce the 180-image consolidated detector cache. Can this exact documented consensus operation, using the independently verified P88 CPU FP32 outputs, reproduce the historical unlabeled official detector cache byte-equivalent at proposal level?

## Frozen methods and hypotheses

- **H0:** At least one image differs in ordered proposal tuples (score, box, mask shape/dtype and every mask value).
- **H1:** All images match exactly under the archived `make_submission.consensus`, with fold-major F1, F2, F3, F4 and original then unreflected view.
- **Pilot:** First four names in lexicographic order, frozen before comparison. If the pilot is 4/4 exact, conduct the 180-image audit **without changing the algorithm**.
- **Stop condition:** If any pilot image differs, preserve the negative receipt and investigate source/version provenance; do not tweak thresholds or train using the hidden leaderboard.
- **Reproducibility:** Verify P88 receipt requires 180/180 for each fold; record SHA-256 of source, manifest and original cache; create immutable, resumable receipts on HDD, and require explicit all-image audit before claiming end-to-end parity.
- **Isolation:** Labels from Kaggle test and calibration partitions remain inaccessible. No PQ optimization on leaderboard. This is a source-code provenance audit, not a new predictive evaluation.

## Budget and safety

CPU-only with `CUDA_VISIBLE_DEVICES=''`; RAM <= 6 GiB, disk <= 2 GiB, compute budget <= 3 hours. GPU0 and GPU1 remain untouched. Preserve every prior checkpoint and original artifact. Write only to `/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112`, never bulk-write to the 97%-full `/mnt/datos` SSD.

## Release gate

Publish source-based reproduction and receipts with a clear separation between the detector-aggregation stage, the independently verified P95 H6 crops and the exact final P94 CSV replay. Report any remaining H3 numerical differences, and **never claim independent original-JPEG-to-CSV exact parity** unless the entire notebook demonstrably establishes it. Prioritize the four-page PDF and mandatory final form.
