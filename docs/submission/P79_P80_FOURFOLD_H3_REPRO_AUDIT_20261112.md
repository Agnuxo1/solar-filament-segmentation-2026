# P79/P80: complete 720-map public U-Net source replay and independent byte-parity audit

**Executed 12 November 2026 UTC. Scope:** All four F1-F4 public H3 semantic U-Net models, 180 original unlabelled official competition test JPEGs per model (720 probability maps). This completes only the **semantic model** stage of the historic H9b pipeline, **not** Mask R-CNN proposal generation, H6 candidate selection, two crop refiners, or the original accepted final CSV.

## Frozen, preregistered source-reconstruction experiment

P77 independently replayed all 180 F1 maps. [P79 protocol](P79_FULL_PUBLIC_FOURFOLD_H3_PREREG_20261112.md) froze the next three folds F2/F3/F4 before execution. Each uses the unmodified public `release/source/native_model.py`, the original unlabelled 2048x2048 solar H-alpha JPEG inputs, all source/model hashes in published [F1](P77_F1_H3_180_REPLAY_RECEIPT_20261112.json), [F2](P79_F2_H3_180_REPLAY_RECEIPT_20261112.json), [F3](P79_F3_H3_180_REPLAY_RECEIPT_20261112.json) and [F4](P79_F4_H3_180_REPLAY_RECEIPT_20261112.json) receipts, and each model's publicly released FP16 checkpoint.

Execution on the assigned RTX 3090 GPU0 was serialized by the existing `supervisor.lock` and verified with `nvidia-smi`; no GPU1 CUDA process was launched. All PNG maps, receipts and logs were written to the NAS HDD. Each of 720 source filenames has a SHA-256-bound quantized 2048x2048 probability PNG. Total fourfold GPU inference elapsed time across all runs: **351.488 seconds**. Each fold recorded 2.058 GiB peak PyTorch CUDA allocated VRAM.

## Independent preregistered map parity vs preserved private P63 maps

The [P80 preregistration](P80_F2_F4_H3_DOMAIN_PARITY_PREREG_20261112.md) and tested public [CPU-only parity script](../../scripts/audit_h3_full_domain_parity.py) fixed the same descriptive gate for every fold: at least 99.99% exactly identical quantized pixels, and maximum absolute byte error <=1. The historical P63 reference cache is an unpublished research artifact; researchers can reproduce public maps but cannot independently replicate this *pairwise historical cache audit* without the reference images.

| Fold | Images | GPU s | Peak CUDA GiB | Unequal pixels | Exactly equal | Max absolute byte difference | Preregistered gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F1 | 180 | 89.398 | 2.058 | 7,971 | 99.998944203% | 1 | PASS_STAGE_ONE_NEAR_PARITY |
| F2 | 180 | 87.246 | 2.058 | 8,726 | 99.998844200% | 1 | PASS_STAGE_ONE_NEAR_PARITY |
| F3 | 180 | 87.439 | 2.058 | 8,341 | 99.998895195% | 1 | PASS_STAGE_ONE_NEAR_PARITY |
| F4 | 180 | 87.405 | 2.058 | 7,766 | 99.998971356% | 1 | PASS_STAGE_ONE_NEAR_PARITY |

**Fourfold aggregate:** 3,019,898,880 compared quantized pixels, **32,804** one-byte differing values, **3,019,866,076** byte-identical pixels, **99.998913738%** exact pixel equality. All four folds passed their originally frozen, *stage-one-only* map-parity thresholds without changes to inference parameters.

Source receipts: [F1 P78](P78_F1_H3_180_PARITY_RECEIPT_20261112.json), [F2 P80](P80_F2_H3_180_PARITY_RECEIPT_20261112.json), [F3 P80](P80_F3_H3_180_PARITY_RECEIPT_20261112.json), [F4 P80](P80_F4_H3_180_PARITY_RECEIPT_20261112.json). These include per-image outputs and reference SHA-256 values plus all counts. No segmentation labels were accessed in P77-P80 and no Kaggle competition scoring was performed by these reproducibility experiments.

## Required limits and next steps

Stage-one near equality of **all four** published semantic models cannot by itself establish historical **final H9b CSV** equivalence. The public [raw-image H3 notebook](../../notebooks/h3_public_raw_image_replay.ipynb) can rerun this stage. Full image-to-final-instance reproducibility additionally requires public original-image-to-Mask R-CNN proposal generation, deterministic H6 instance selection and both H9b crop-refiner stages, then comparison of actual final CSV bytes to accepted submission **58112557** (public PQ **0.39**, original 8 November 2026). The current independently reassembled *intermediate* CSV has a different SHA-256 from the accepted upload, so do not claim end-to-end fidelity, new PQ improvement or final delivery completion.

The competition's mandatory final Google Form still has **no independently verified submission receipt**. This report cannot substitute for that form, the official four-page English report or a working end-to-end competition notebook.
