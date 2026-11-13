# P95 — Independent full-inventory audit of JPEG-to-H6 crop parity

**Date:** 2026-11-13 UTC  
**Experiment:** P95 (preregistered before any JPEG comparisons)  
**Status:** PASS — 180/180, independently rechecked at the SHA-256 and field-receipt levels  
**Code:** `scripts/p95_nosite_jpeg_crop_audit.py` and `scripts/audit_p95_full_crop_parity.py`  
**Machine-readable proof:** `docs/submission/P95_FULL_180_INDEPENDENT_RECEIPT_20261113.json`

## Objective and protocol

Independently rebuild the frozen H9b / H6 **no-site crop stage** using the official 180 raw, unlabeled Kaggle test JPEGs, the previously frozen TRAIN-only selector, historical H3 probability maps, and Mask R-CNN detection caches. Compare the regenerated arrays `X`, `M`, `comp`, and `score` against the original crop archives without changing either side.

P95 was executed as a CPU-only, recovery-safe service; every JPEG had a separate immutable JSON receipt. After the service completed, a **new independent audit script** checked the inventory, validated all recorded field comparisons, and freshly recomputed both the original JPEG SHA-256 and historical crop-archive SHA-256 for every receipt. It also rehashed the selector against its frozen manifest. This audit did not rerun the expensive inference; it rigorously audits its immutable outputs against their unmodified inputs.

## Results

| Check | Verified result |
| --- | ---: |
| Official JPEG inventory | 180 / 180 |
| Frozen reference crop inventory | 180 / 180 |
| Per-image P95 receipt inventory | 180 / 180 |
| All-field-exact per-image receipts | **180 / 180** |
| Inventory, hash, schema, or field errors | **0** |
| Elements compared in `X` | 244,580,352 |
| Elements compared in `M` | 7,464 |
| Elements compared in `comp` | 1,244 |
| Elements compared in `score` | 1,244 |
| Test annotation/ground-truth access | None |
| CUDA used | No |

**Audit receipt SHA-256:** `4e3379ebd775e92244ee4bc684c5813a23e88b9ccae4a4440d00ef457b0774a7`.

The independent audit's synthetic negative tests passed: changed input/reference/selector hashes, altered exactness values, a modified array-field count, test annotation use, GPU use, and a false cache-dependency declaration were correctly rejected.

## Interpretation and limitations

This proves exact agreement **only for the JPEG + frozen H3/detector predictions → H6 no-site crop stage**. It does **not** prove that a new, clean-room invocation of the H3 and Mask R-CNN models reproduces the upstream frozen caches, nor does it individually regenerate all 1,241 final RLE masks from an untouched dataset. The public H9b last-stage notebook and P94 attest to the final CPU-FP32 crop-to-CSV parity, and P79/P88 document earlier model replay results. These remain separate stage-specific validations, not yet a single fully self-contained end-to-end run.

P95 does not measure Panoptic Quality. The most recent authenticated Kaggle CLI check (2026-11-13) still reports public PQ 0.39, submission ID **58112557**, status COMPLETE. No new submission was made.

## Remaining work before final judging

1. Link the public raw-image H3 replay and public FP32 Mask R-CNN replay into an independently executable crop-generation stage; verify that it matches the original frozen caches without loading test annotations.
2. Chain that stage to the publicly documented CPU FP32 refinement and exact final CSV SHA-256 gate. Record explicit data-download, environment, and runtime requirements. Do not claim the resulting notebook is fully self-contained unless it passes from the original JPEGs.
3. Verify the four-page English PDF and public repository license; obtain the official final form receipt and archive its confirmation.
4. Keep H9b (PQ 0.39) as the competition submission unless a scientifically supported, quota-compliant improvement is proven on TRAIN-only held-out data.

**Deadline policy:** freeze the final artifacts by 2026-11-15 00:00 UTC, six hours ahead of the stipulated 06:00 UTC competition cutoff.
