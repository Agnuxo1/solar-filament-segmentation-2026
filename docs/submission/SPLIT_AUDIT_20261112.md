# Independent F1-F4 data-partition integrity audit - 12 November 2026

**Status:** FOUR_FOLD_SPLIT_AND_CACHE_INTEGRITY_VERIFIED at 16:08 UTC.

A read-only, CPU-only audit was executed against Kaggle TRAIN-derived local folds on IA-Server. No CUDA process was started or interrupted.

| Temporal fold | Training images | Calibration images | Held-out test images |
|---|---:|---:|---:|
| F1 | 446 | 117 | 139 |
| F2 | 449 | 109 | 146 |
| F3 | 449 | 113 | 139 |
| F4 | 454 | 108 | 143 |

## Checks actually performed

1. Exact SHA-256 of each frozen split, checked against the evaluator's fixed protocol.
2. No duplicate source names within partitions and no overlap between TRAIN, calibration and test.
3. Exact membership parity between calibration/test split lists and the prepared ground-truth manifests.
4. Recomputed SHA-256 of every per-image holdout annotation NPZ, compared with its declared value.
5. For F2-F4, matching TRAIN-pack provenance: fold, split SHA-256, image count and train-only preparation flag.

The synthetic self-test also passed for rejection of an incorrect image set and a tampered cache file. Both the self-test and the real four-fold audit exited with status 0.

**Verifier:** [scripts/audit_fold_split_integrity.py](../../scripts/audit_fold_split_integrity.py)

**Machine receipt:** [SPLIT_INTEGRITY_RECEIPT_20261112.json](SPLIT_INTEGRITY_RECEIPT_20261112.json). Original immutable local receipt: paired-small-20261112/audit/split_integrity_20261112T160806933948Z.json.

## Reproduction using locally prepared permitted data

Run python3 scripts/audit_fold_split_integrity.py --self-test, followed by python3 scripts/audit_fold_split_integrity.py --root /path/to/paired-small-20261112.

## Interpretive restrictions

These checks establish integrity of prepared fold manifests and holdout cache files, not mathematical proof of every previous model-training I/O operation. They do not access hidden competition TEST labels, demonstrate model quality, compute PQ, submit to Kaggle or complete the organisers' Google Form. The F1 retained-test PQ and F2-F4 outcomes remain independent pending work. The last verified official public PQ remains 0.39 (submission 58112557, accepted 8 November 2026).
