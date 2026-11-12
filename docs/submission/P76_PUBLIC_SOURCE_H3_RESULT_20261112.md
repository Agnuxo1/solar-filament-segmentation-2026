# P76 result: public H3 model source and raw-image replay

**Executed 12 November 2026, 17:19 UTC.** This result follows the frozen P76 public-source smoke-test protocol. It tests one predetermined test image, without its labels. It is not a new Kaggle submission, an end-to-end H9b reconstruction, or evidence that scientific PQ improved.

## Reproduced public stage-one U-Net inference

- The original ResNet34 one-channel NativeUNet implementation is included verbatim at [release/source/native_model.py](../../release/source/native_model.py). SHA-256: `4adf71184470abe334aece8ad9abd98d730866f13c27a10700f83bbae1e85572`.
- [Python CLI](../../scripts/replay_h3_from_original_images.py) verifies all 180 source-image filenames and the previously published F1 weight SHA-256 `a4cc0686786ad2f9a253e2e38cfe37896d16a9d6f090e8429cfe27ae138c4d24`, and reconstructs exactly the first 2048x2048 H3 probability PNG from the supplied test H-alpha image `20110120105534Ch.jpeg`. Original input SHA-256: `7190e95661c142b194c3d2afddfedfee3382c2b3ded2f2f1b023d13e3bd58baf`.
- Output PNG SHA-256: `1b231f058abb482340831c4863334c8de618c40aa0b50ef7704d0c21ef4e7e26`; peak GPU memory 2.058 GiB, model runtime 3.57 s; one CUDA process on assigned GPU0 and no GPU1 compute process, verified by nvidia-smi.
- Immutable private receipt: `paired-small-20261112/p76_public_release_smoke/F1_h3_replay_receipt.json`.

## Independent quantized-map parity

A read-only comparison decoded three native 2048-square uint8 prediction maps and compared all 4,194,304 image pixels. It did not use ground-truth masks, target labels or CSV submissions.

| Compared maps | Identical pixels | Differing pixels | Maximum absolute byte difference |
| --- | ---: | ---: | ---: |
| Public-source P76 vs private P63 replay | 99.9997139% | 12 | 1 |
| Public-source P76 vs historical H3 output | 99.9992371% | 32 | 1 |
| Private P63 replay vs historical H3 output | 99.9993801% | 26 | 1 |

Thus public-source inference **loads and generates a valid native probability map from original pixels**, with small but nonzero numerical differences from the historical output. This is an observed single-image result; do not extrapolate to all 180 images. Full H9b inference additionally depends on Mask R-CNN proposals, H6 instance selection and H9b crop refiners, which are *not* covered by this smoke test. Byte-exact reproduction of the accepted submission 58112557 remains unverified.

Private parity receipt `paired-small-20261112/p76_public_release_smoke/h3_parity_receipt.json` includes all three input map file hashes and pairwise error counts. The public [parity audit script](../../scripts/audit_h3_map_parity.py) accepts explicit paths and writes an immutable JSON receipt.

**Decision: PASS the narrow public-source smoke-test gate; HOLD all stronger final-CSV and scoring claims.** The user-facing final report must distinguish these levels of evidence and must not claim that the required Google Form was submitted unless the form receipt is independently verified.
