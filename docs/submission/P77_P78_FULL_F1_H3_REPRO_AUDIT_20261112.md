# P77/P78 - 180-image public H3 replay and byte-parity audit

**Executed:** 12 November 2026 UTC. **Scope:** Stage-one F1 ResNet34 U-Net semantic probability maps. **Not** the full H9b final-CSV inference chain.

## P77 - original image to public H3 model

- Replayed the **entire predefined set of 180 original, unlabelled Kaggle TEST H-alpha JPEG images**, with the public `release/source/native_model.py`, frozen public F1 checkpoint and `scripts/replay_h3_from_original_images.py`.
- Reconstructed exactly **180 native-resolution 2048 x 2048 uint8 PNG maps** on the dedicated RTX 3090 GPU0. One CUDA process under the same exclusive lock used by the scientific evaluation supervisor; GPU1 was not allocated.
- Runtime **89.398 s**, observed peak allocated CUDA VRAM **2.058 GiB**. Model/source/script SHA256, original input image SHA256 and output map SHA256 recorded individually in [P77 receipt](P77_F1_H3_180_REPLAY_RECEIPT_20261112.json).
- F1 model SHA256: `a4cc0686786ad2f9a253e2e38cfe37896d16a9d6f090e8429cfe27ae138c4d24`; model source SHA256: `4adf71184470abe334aece8ad9abd98d730866f13c27a10700f83bbae1e85572`.
- Frozen design [P77 preregistration](P77_FULL_PUBLIC_H3_REPLAY_PREREG_20261112.md). No annotations or Kaggle hidden ground-truth labels were accessed.

## P78 - strict map parity against archived P63 F1 output

The [P78 protocol](P78_FULL_H3_DOMAIN_PREREG_20261112.md) was recorded before all-image comparison; the [public verifier](../../scripts/audit_h3_full_domain_parity.py) passed syntax and arithmetic self-tests. It compared all 180 pairs of quantized 2048 x 2048 grayscale maps, each bound to a SHA-256-verified P77 receipt. The archived P63 reference maps are a private first-stage research artifact, not a downloadable Kaggle submission.

| Read-only outcome | Verified value |
| --- | ---: |
| Maps compared | 180 |
| Pixels compared | 754,974,720 |
| Exactly identical pixels | 754,966,749 |
| Pixels differing by one quantization level | 7,971 |
| Byte-level agreement | 99.998944203% |
| Maximum absolute byte difference | 1 |
| Mean absolute byte error | 0.000010557969 |
| Frozen descriptive gate (>=99.99% exact and <=1 max difference) | **PASS** |

Read the [complete P78 receipt](P78_F1_H3_180_PARITY_RECEIPT_20261112.json) for all 180 image-level SHA256 values and deviations. The source and reference maps were not modified.

## Reproduction and limits

Execute the public `notebooks/h3_public_raw_image_replay.ipynb` or the command-line program `scripts/replay_h3_from_original_images.py` with the original competition test JPEG directory and published F1 checkpoint. An isolated GPU may be selected by its exact physical UUID. P78 comparison additionally requires the author's archived P63 F1 probability PNGs; it is not independently rerunnable from the public repo alone without those historical reference maps.

**Important limitation:** Near byte equality of one semantic stage does **not** demonstrate equivalence of Mask R-CNN proposals, H6 instance selection, H9b crop refiners or final instance RLE CSV. The archived accepted Kaggle submission **58112557** has a public PQ score of **0.39**. Neither P77 nor P78 has a new Kaggle score, validates hidden Kaggle TEST labels, demonstrates a PQ improvement, or proves the accepted CSV is byte-identically reconstructed. The organisers' final Google Form still requires an independently verified submission receipt.

These experiments support source release reliability and are not evidence that crop sampler F1-F4 research has passed its preregistered promotion gate.
