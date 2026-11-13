# P88 — Independent exact CPU detector replay, F3

**Audit date (UTC):** 13 November 2026. **Status:** P88_F3_INDEPENDENT_FULL_DOMAIN_PASS.

The public Mask R-CNN FP32 F3 weights and public implementation were
replayed from 180 original, unlabelled Kaggle test JPEG images on CPU.
The independent auditor compared each original JPEG SHA-256, every output
pickle SHA-256, the historical P69 reference, original checkpoint identity,
ordered confidence scores, bounding boxes and binary crop-mask bytes.
Both direct and horizontally mirrored detector views were checked.

| Verified measure | Result |
| --- | ---: |
| Original unlabelled images | 180/180 |
| Detector views | 360/360 |
| Exactly matching proposals | 12,323 |
| Divergent output image files | 0 |
| Test annotations accessed | No |
| New Kaggle submissions | No |

**Replay manifest SHA-256:** `d8c2fee0305a1fa016928b02d87991f83308f3d3b7ca447cdf5221d2fbeffa0b`

**P69 reference manifest SHA-256:** `861af4022631fac7fc20a57790266051dacea0998f1ec590c1fa46359a4b9827`

**Auditor source SHA-256:** `fb9e757f7c5ed36e7ec0ea93f2e7f07cd7ad65f7ddfd0ddf93f34e80c8078ca6`

The executable independent check is
[`scripts/audit_p88_full_fold_parity.py`](../../scripts/audit_p88_full_fold_parity.py).
The accompanying JSON receipt is
[`P88_F3_INDEPENDENT_EXACT_RECEIPT_20261113.json`](P88_F3_INDEPENDENT_EXACT_RECEIPT_20261113.json).

**Strict scientific limitation:** this demonstrates detector-stage equality
with an archived local reference, not equality of final masks or of the
accepted H9b CSV, and not a new PQ measurement. Original competition test
labels were never used. The accepted Kaggle baseline remains submission
58112557 (public PQ 0.39); this audit is not evidence for promoting a model.
The end-to-end reproduction notebook and final submission form receipt
remain distinct, unfulfilled final-delivery gates.
