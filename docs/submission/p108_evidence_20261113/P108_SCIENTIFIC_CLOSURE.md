# P108 OOF provenance gate: completed audit and ablation decision

**Completed:** 2026-11-13 17:06:48 UTC. **Outcome:** `BLOCKED_NO_VERIFIED_OOF_PROVENANCE`. This is a successful *audit execution*, **not** an accuracy experiment, and it provides no new PQ or Kaggle score.

## Locked protocol and reproducibility

- Preregistration: [P108 protocol](P108_PREREG_PROVENANCE_GATE.md) committed before the audit, commit `e06d83776cf0bd9ed985b74660f689ac08b38f7e`.
- Deterministic read-only [P108 checker](../../../scripts/p108_oof_provenance_gate.py), final tested revision `d8cf1a1857128a7c18e75b6dc06007e0f34710a5`.
- [Full machine-readable receipt](P108_FINAL_RECEIPT.json), SHA-256 `06277cbeac54cc6cb6964aa7ec003a9ca8f8d2996f90fe82ddd5a7448034946f`. The SHA-256 of the public GitHub bytes was verified identical to the project-HDD receipt.
- P106 input receipt SHA-256 `a4bf132c3ba4ca0464bcdbc822f1b1448bb64080b0dcc004afaab4686fe78257`; source split SHA-256 `4f30e3b8f6d84e03bee1cfc6a65c03261116f725a0ff621ddfbd4de49e88cbda`.

## Gate results

| Preregistered criterion | Outcome |
| --- | --- |
| F1 TRAIN/calibration/test physical-image memberships distinct | **Pass**: 446 / 117 / 139; zero duplicates or pairwise overlaps |
| Clean P106 8/8 code-cell receipt and genuine H9b reference hash | **Pass** |
| Image-exclusion certificate for every semantic-model and Mask R-CNN checkpoint | **Not demonstrated** |
| Image-exclusion certificate for ResNet34 and EfficientNetV2-S crop refiners | **Not demonstrated** |
| Image-exclusion certificate for the fitted no-site selector | **Not demonstrated** |
| Fully traceable, held-out F1 baseline and treatment upstream maps | **Not demonstrated** |

The known P59 inference-cache inventory contains 446 NPZ files matching the **446 TRAIN split image stems**, and **none** matching the F1 calibration or reserved test image stems. This establishes file presence only, not model training provenance. In the frozen public release, the two available refiner weights are `refiner_ALL.pt` and `refiner_ALL_efficientnet_v2_s.pt`; no non-ALL refiner checkpoint was found within the audited folds/seeds/release scope. The selector metadata records 11,449 fitting feature rows but no per-image TRAIN-exclusion certificate. Merely using release checkpoint names or a disjoint ground-truth split does not make fitted predictions out-of-sample.

**Interpretation:** missing or unverified training-exclusion metadata is not proof of leakage. It is, however, sufficient to forbid a claim of independently validated TRAIN PQ under the strict preregistered gate. This is especially consequential for the all-TRAIN refiners and the fitted selector.

## Decision and boundaries

The proposed single 1.15× extra-context / 50:50 fusion ablation is **not authorised to run on scored TRAIN labels** with the current artifacts. No new inference, fitting, model promotion, official Kaggle submission, test-label use, or leaderboard-guided tuning was performed. GPU0 was not used by P108. The already submitted and selected genuine H9b artifact (SHA-256 `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`, Kaggle submission `58112557`, public PQ `0.39`) remains untouched.

A future implementation of the 45-minute ablation requires per-image OOF model/selector/refiner evidence, frozen baseline/upstream input hashes, and the unchanged calibration-first / held-out-test promotion protocol. Without those artifacts, the scientifically valid outcome of this exploration is **stop**, not an in-sample score. P106 and P107 remain reproducibility studies; their near-parity RLEs are not ground-truth accuracy results.

This audit uses project-controlled files on the HDD and cannot certify the absence of alternate artifacts elsewhere. It records no JEV/Goldfish consultation.
