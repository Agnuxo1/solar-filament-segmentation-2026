# P83 independent-seed F1 native-PQ replication audit

Status: **COMPLETE_DESCRIPTIVE_NO_PROMOTION**. Completed 3/3; pending: [].

The frozen policy is threshold 0.25, minimum area 200, peak 0.95, mean 0.7. The 139-image F1 test partition derives exclusively from competition TRAIN. It has already been examined in P59; these are exploratory repeated holdout evaluations, not new blind validation or Kaggle scores.

| Seed | 3600 control | 3600 treatment | 7200 control | 7200 treatment | Treatment-control at 7200 |
|---|---:|---:|---:|---:|---:|
| 20261113 | 0.396036 | 0.392521 | 0.398975 | 0.395253 | -0.003721 |
| 20261114 | 0.404530 | 0.391219 | 0.398163 | 0.387987 | -0.010176 |
| 20261115 | 0.398171 | 0.386408 | 0.403401 | 0.387933 | -0.015467 |
| Descriptive mean | 0.399579 | 0.390049 | 0.400179 | 0.390391 | — |

Each native evaluator additionally reports image-cluster bootstrap intervals versus its own 3600-step control. Full per-seed CIs, source file SHA-256, and selected policies are in the companion JSON receipt; the small number of seeds does not justify a confident cross-seed improvement claim.

**Decision: HOLD/NO PROMOTION.** Do not claim a Kaggle improvement, change the accepted H9b submission, or optimize the previously examined F1 test. No hidden test labels were used.
