# H9b model metadata and research implementation archive

Updated: 2026-11-12 UTC. This directory contains the published inference-related source files and an independently frozen H9b no-site component selector. It is **not yet** an end-to-end reproduction of accepted Kaggle submission 58112557.

## Included artifacts

- `model_metadata/h9b_nosite_selector.pkl`: serialized, fitted HistGradientBoostingRegressor generated from competition TRAIN annotations only. SHA-256: `5db6a9a9787a7cbdef6f35a115c9515364b0e2bc31e63fccf546ff0a7ca98a2c`.
- `model_metadata/h9b_nosite_selector.json`: source feature names, frozen preprocessing parameters, provenance and hash.
- `model_metadata/WEIGHTS_SHA256SUMS.txt`: identifiers and SHA-256 digests for ten **publicly hosted** FP16 U-Net, Mask R-CNN and crop-refiner checkpoints in [`../models/`](../models/). All ten staged sources passed SHA-256 verification before upload. These are numerical FP16 exports, not evidence of byte-identical original FP32 inference. The original Kaggle-uploaded CSV is preserved separately.
- `source/`: archived Python research modules for H6/H9 selection, instance construction, evaluation, and selector freezing. The modules pass Python syntax compilation, but the full H9b reconstruction still depends on additional upstream raw-image inference, historic model-output caches, and nonpublished original FP32 numerical state.

## Audit boundary and limitations

The stage-two notebook in `notebooks/` reproduced 1,241 instance rows from cached model outputs across 180 competition test images. Reconstructed CSV SHA-256 `67f02e4d54b78424306133840ec144b426c8b308f6fc659f90a7f67131d5a79e` differs from original accepted submission SHA-256 `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`. This is a successful intermediate replay, **not exact Kaggle-file parity**.

The earlier U-Net raw-image audit regenerated 720 maps (4 models x 180 images). Its pixel comparisons are not byte-identical. Changes in model precision and thresholding may change instance RLEs. Full raw-image-to-CSV reproducibility therefore remains an unresolved release gate.

The serialized selector uses Python pickle. Load it **only in an isolated environment if you trust the repository and exact SHA-256**; pickle is not a safe format for arbitrary untrusted artifacts. Never substitute this for validation of the complete pipeline.

## New scientific status on 12 November

The independent F1 seed replications 20261113, 20261114 and 20261115 each finished 7,200 training steps with checkpoint receipts. They are **not yet measured on held-out F1 test**.

At 13:59:48 UTC, the separately managed CPU evaluator froze a calibration-only parameter policy (`thr=0.25, min_area=200, peak=0.95, mean=0.7`) selected from the **P59 control model at step 3,600**. Native PQ on the 117-image F1 calibration sample was `0.4133849796583869` **for that policy-selection control**. It is not a measured F1 test result, not an official Kaggle score, and not proof of treatment improvement. The prior frozen H9b F1 calibration benchmark is `0.462433`, but comparisons across different evaluations must be interpreted cautiously.

No upload of a new Kaggle candidate is warranted until the frozen policy has been used on the separate F1 test and temporal F2-F4 holdouts without leakage. No external labels, hidden leaderboard optimization, or official test annotations were used in this audit.

## Completion gates

All ten staged FP16 checkpoint files are now publicly available with verifiable checksums. To achieve an independently reproducible *accepted* Kaggle submission, the raw-image preprocessing, U-Net and Mask R-CNN inference, instance selector, refiners and final CSV writer must still be executed as one pinned pipeline; the known FP16 numerical differences must be resolved or quantified, and exact or tolerance-qualified CSV parity must be checked. The separate competition final-form receipt must also be independently verified.

Do not redistribute Kaggle images, annotator masks, credentials, or private datasets. See [the main technical report](../../docs/TECHNICAL_REPORT_4P_20261112.pdf), [the release README](../README.md), and the repository LICENSE.
