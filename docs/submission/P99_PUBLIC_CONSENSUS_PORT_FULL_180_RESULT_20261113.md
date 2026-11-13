# P99 — Full exact verification of a clean-checkout consensus implementation

**Completed:** 13 November 2026 (UTC)

## Purpose

P98 found and used the archived `make_submission.consensus` source to recreate the true 180-image detector aggregate. This historical source file was not shipped in the public repository. P99 removes the dependency by publishing a deterministic, **self-contained NumPy consensus implementation** in `scripts/official_consensus.py`, then comparing the new implementation against both archived original source and original unlabelled detector aggregate.

## Verified results

- **Preregistered pilot:** 4/4 images exactly identical.
- **Full study:** **180/180** images exactly identical, against **both** the archived consensus implementation and historical detector aggregate.
- Equality includes counts, proposal order, score and bounding-box tuples, mask shape/dtype and every individual soft-mask byte.
- The immutable full-machine receipt records input/source/port SHA-256 hashes and all 180 per-image comparison results.
- **Full receipt SHA-256:** `8871d6eddbebadef53631e6c1861a5503963b5ce3d513283ec9c618377673b22`.
- The test suite includes positive cases and deliberate changes of scores and masks.
- `solarfil-p99-full-20261113.service` finished with `Result=success` and `ExecMainStatus=0`.

The reference archived source is needed only by the *P99 provenance audit*, not for the standalone `scripts/official_consensus.py` inference function. The port itself requires NumPy only and no CUDA or ground-truth test annotations.

## Frozen inference semantics

The port uses the original score gate of 0.30, mask binarization `>127`, cross-view support at mask IoU 0.50, unsupported-score multiplier 0.75, stable descending-score sort, and greedy mask-NMS at IoU 0.50. The eight view groups retain fold-major F1-F4 ordering with normal and unflipped horizontal views, just as in the archived code.

## Verification

With the P88 CPU outputs and archived source/aggregate available on an authorized research computer:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/p99_public_consensus_port_audit.py --self-test
CUDA_VISIBLE_DEVICES='' python scripts/p99_public_consensus_port_audit.py --limit 4
CUDA_VISIBLE_DEVICES='' python scripts/p99_public_consensus_port_audit.py --limit 180
```

The public `scripts/official_consensus.py` can be imported independently with `from scripts.official_consensus import consensus`. The full P99 experiment is a **local provenance audit**, not an independent run using a clean Kaggle notebook session.

## Impact and remaining obligations

This closes the historical consensus implementation gap identified by the negative P97 experiment. P88 verifies per-detector proposals; P98 and P99 verify aggregation; P95 validates original-JPEG-derived no-site crops **when historical intermediate H3/detector artifacts are available**; P94 verifies final CPU FP32 masks from historical crops against the actual uploaded CSV.

**The complete clean-checkout JPEG-to-upload verification remains open**, primarily because P96 found numerical differences when the original H3 cache was independently regenerated. Do not claim that the whole accepted CSV is already independently reproducible. No fresh Kaggle score or submission has been obtained by P99. The mandatory organizer form must still be submitted and receipt verified.
